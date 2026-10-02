"""
Full-parameter SFT for Gemma 3 12B-IT (text-only prompt/completion), 2 GPUs, FSDP FULL_SHARD.

Per-GPU memory plan (12.2B params, 2 x 96 GB):
  fp32 master weight shard                     24 GB
  fp32 gradient shard                          24 GB
  8-bit Adam m,v shard (bitsandbytes)          12 GB
  bf16 unsharded layer copies, checkpointed
  activations, 262k-vocab logits               ~20-25 GB at block_size 4096, batch 1
                                               ---------
                                               ~85 GB of 96 GB

Launch:  accelerate launch --config_file fsdp.yaml anti_deceptive_sft_full.py ...
"""

import contextlib
import json
import logging
import os
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.append("..")
sys.path.append(".")

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import torch
import transformers
from datasets import disable_caching, load_dataset
from packaging import version
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

disable_caching()


@dataclass
class TrainingConfig:
    model_name: str = field(default="google/gemma-3-12b-it")
    block_size: int = field(default=4096)
    wandb_project: Optional[str] = field(default="gemma_finetuning")
    data_dir: Optional[str] = field(
        default="/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/DATA"
    )
    val_fraction: float = field(default=0.05)
    attn_implementation: str = field(default="sdpa")
    warmup_fraction: float = field(default=0.03)
    empty_cache_every: int = field(default=50)
    early_stopping_patience: int = field(default=5)

    def __post_init__(self):
        os.environ["WANDB_PROJECT"] = self.wandb_project


# --------------------------------------------------------------------------- model


def _is_gemma3_multimodal(config) -> bool:
    return getattr(config, "model_type", "") == "gemma3" and getattr(
        config, "vision_config", None
    ) is not None


def _dtype_kwarg(dtype):
    # transformers >= 4.56 renamed torch_dtype -> dtype
    key = "dtype" if version.parse(transformers.__version__) >= version.parse("4.56.0") else "torch_dtype"
    return {key: dtype}


def load_model_and_tokenizer(config: TrainingConfig):
    hf_config = AutoConfig.from_pretrained(config.model_name)
    multimodal = _is_gemma3_multimodal(hf_config)

    # Load in fp32 on purpose. fsdp.yaml sets mixed_precision: bf16, so FSDP casts the weights to bf16
    # for forward/backward and keeps this fp32 copy as the master copy for the optimizer update.
    # Loading in bf16 would make the update itself bf16; at lr=1e-5 most updates are smaller than
    # bf16 resolution and get rounded away.
    #
    # Each rank loads straight onto its own GPU (48 GB fp32, fits in 96 GB) instead of into host RAM.
    # Loading through CPU needs 48 GB of host RAM per rank and got OOM-killed by the job's memory
    # limit. FSDP then shards in place: GPU usage drops to 24 GB once wrapping finishes.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    load_kwargs = dict(
        pretrained_model_name_or_path=config.model_name,
        attn_implementation=config.attn_implementation,
        device_map={"": local_rank},
        **_dtype_kwarg(torch.float32),
    )

    if multimodal:
        from transformers import Gemma3ForConditionalGeneration

        logging.info("Loading Gemma3ForConditionalGeneration")
        model = Gemma3ForConditionalGeneration.from_pretrained(**load_kwargs)
    else:
        from transformers import AutoModelForCausalLM

        logging.info("Loading AutoModelForCausalLM")
        model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

    model.config.use_cache = False
    text_cfg = getattr(model.config, "text_config", None)
    if text_cfg is not None:
        text_cfg.use_cache = False


    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.warning("Tokenizer had no pad token; fell back to eos.")
    tokenizer.padding_side = "right"
    model.config.pad_token_id = tokenizer.pad_token_id
    if text_cfg is not None:
        text_cfg.pad_token_id = tokenizer.pad_token_id

    n_params = sum(p.numel() for p in model.parameters())
    logging.info("Parameters: %.2fB (all trainable)", n_params / 1e9)
    return model, tokenizer, multimodal, n_params


# --------------------------------------------------------------------------- data


def _ensure_bos(prompt_ids: List[int], tokenizer) -> List[int]:
    bos = tokenizer.bos_token_id
    if bos is None:
        return prompt_ids
    if prompt_ids and prompt_ids[0] == bos:
        return prompt_ids
    return [bos] + prompt_ids


def _ensure_turn_end(response_ids: List[int], tokenizer) -> List[int]:
    eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    eos_id = tokenizer.eos_token_id
    if eot_id is not None and eot_id != tokenizer.unk_token_id:
        if not response_ids or response_ids[-1] not in {eot_id, eos_id}:
            response_ids = response_ids + [eot_id]
        return response_ids
    if eos_id is not None and (not response_ids or response_ids[-1] != eos_id):
        response_ids = response_ids + [eos_id]
    return response_ids


def prepare_dataset(dataset, tokenizer, block_size):
    def _tokenize(examples):
        input_ids_list, labels_list, attn_list, tti_list = [], [], [], []
        for prompt, response in zip(examples["prompt_text"], examples["response_text"]):
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            response_ids = tokenizer.encode(response, add_special_tokens=False)
            prompt_ids = _ensure_bos(prompt_ids, tokenizer)
            response_ids = _ensure_turn_end(response_ids, tokenizer)
            input_ids = prompt_ids + response_ids
            labels = [-100] * len(prompt_ids) + response_ids[:]
            if len(input_ids) > block_size:
                # keep BOS, drop from the left (prompt side) so the response survives
                input_ids = [input_ids[0]] + input_ids[-(block_size - 1):]
                labels = [-100] + labels[-(block_size - 1):]
            if all(l == -100 for l in labels):
                continue  # nothing to supervise; would give NaN loss at batch size 1
            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attn_list.append([1] * len(input_ids))
            tti_list.append([0] * len(input_ids))
        return {
            "input_ids": input_ids_list,
            "labels": labels_list,
            "attention_mask": attn_list,
            "token_type_ids": tti_list,
        }

    return dataset.map(
        _tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=4,
    )


class Gemma3TextCollator:
    def __init__(self, tokenizer, max_length: int):
        self.inner = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            max_length=max_length,
            pad_to_multiple_of=8,
            label_pad_token_id=-100,
            return_tensors="pt",
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = self.inner(features)
        if "token_type_ids" not in batch:
            batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
        return batch


# --------------------------------------------------------------------------- callbacks / trainer


class EmptyCacheCallback(TrainerCallback):
    def __init__(self, every: int = 50):
        self.every = max(int(every), 0)

    def on_step_end(self, args, state, control, **kwargs):
        if self.every <= 0 or state.global_step % self.every != 0:
            return
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        torch.cuda.empty_cache()


class EarlyStopOnEvalLoss(TrainerCallback):
    """Stop after `patience` evaluations without eval_loss improvement.

    transformers.EarlyStoppingCallback insists on load_best_model_at_end=True, which cannot be
    combined with save_only_model under FSDP. The best checkpoint is kept on disk instead
    (save_strategy=best) and its path is logged at the end of training.
    """

    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.bad = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        loss = (metrics or {}).get("eval_loss")
        if loss is None:
            return
        if self.best is None or loss < self.best - self.min_delta:
            self.best, self.bad = loss, 0
        else:
            self.bad += 1
            if self.bad >= self.patience:
                logging.info(
                    "Early stopping: no eval_loss improvement in %d evals (best %.4f)",
                    self.patience, self.best,
                )
                control.should_training_stop = True


def _persistent_buffer_fqns(root) -> set:
    names = set()
    for mname, m in root.named_modules():
        for bname in m._buffers:
            if bname not in getattr(m, "_non_persistent_buffers_set", set()):
                fqn = f"{mname}.{bname}" if mname else bname
                names.add(fqn.replace("_fsdp_wrapped_module.", ""))
    return names


def gather_bf16_state_dict_rank0(fsdp_root) -> Dict[str, torch.Tensor]:
    """Gather the model one FSDP unit at a time; keep only a bf16 CPU copy, on rank 0 only.

    FULL_STATE_DICT + offload_to_cpu materialises the whole model in fp32 in rank 0 host RAM
    (~48 GB for 12B, ~60 GB peak with FSDP's transient copies), which is what got OOM-killed under
    the 80 GB job limit. This never holds fp32 on the host: each unit is unsharded on the GPU,
    cast to bf16 there, and only the bf16 tensor is copied out. Host peak ~24 GB.
    All ranks must call this (the per-unit unshard is a collective); non-zero ranks get {}.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    is_rank0 = (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0
    out: Dict[str, torch.Tensor] = {}
    n_param_elems = 0
    units = [(n, m) for n, m in fsdp_root.named_modules() if isinstance(m, FSDP)]
    for name, unit in units:
        with FSDP.summon_full_params(
            unit, recurse=False, writeback=False, rank0_only=True, offload_to_cpu=False
        ):
            if not is_rank0:
                continue
            prefix = f"{name}." if name else ""
            # named_parameters() yields each tied parameter once (lm_head <-> embed_tokens), so the
            # saved dict has only embed_tokens.weight; transformers re-ties lm_head on load.
            for pname, p in unit.named_parameters():
                if "_flat_param" in pname:  # a nested, still-sharded FSDP unit; handled in its own turn
                    continue
                fqn = (prefix + pname).replace("_fsdp_wrapped_module.", "")
                out[fqn] = p.detach().to(torch.bfloat16).cpu()
                n_param_elems += p.numel()
    if is_rank0:
        persistent = _persistent_buffer_fqns(fsdp_root)
        for bname, b in fsdp_root.named_buffers():
            fqn = bname.replace("_fsdp_wrapped_module.", "")
            if fqn in persistent and fqn not in out:
                out[fqn] = b.detach().to(torch.bfloat16).cpu() if b.is_floating_point() else b.detach().cpu()
    return out, n_param_elems


class BF16FullSaveTrainer(Trainer):
    """Write bf16 HF checkpoints from FSDP without ever holding fp32 weights in host RAM."""

    expected_param_numel: Optional[int] = None  # set in train() from the unwrapped model

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        output_dir = output_dir or self.args.output_dir
        root = next(
            (m for m in (self.model, getattr(self, "model_wrapped", None)) if isinstance(m, FSDP)),
            None,
        )
        if root is None:
            raise RuntimeError("Model is not FSDP-wrapped; launch with accelerate launch --config_file fsdp.yaml")
        state_dict, n_param_elems = gather_bf16_state_dict_rank0(root)
        if self.args.should_save:
            if self.expected_param_numel is not None and n_param_elems != self.expected_param_numel:
                raise RuntimeError(
                    f"Gathered {n_param_elems:,} parameter elements but the model has "
                    f"{self.expected_param_numel:,}; refusing to write an incomplete checkpoint to {output_dir}"
                )
            logging.info("Gathered %.2fB parameter elements in bf16, writing %s", n_param_elems / 1e9, output_dir)
            # 4 GB shards: safetensors copies each shard to bytes while writing, so this caps the
            # extra to 4 GB. Peak rank-0 host RAM for a save is ~24 GB (bf16 dict) + 4 GB.
            unwrapped = self.accelerator.unwrap_model(root)
            os.makedirs(output_dir, exist_ok=True)
            unwrapped.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=True, max_shard_size="4GB"
            )
            if getattr(self, "processing_class", None) is not None:
                self.processing_class.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            cfg_path = os.path.join(output_dir, "config.json")
            if os.path.exists(cfg_path):  # record the on-disk dtype
                with open(cfg_path) as f:
                    cfg = json.load(f)
                for key in ("torch_dtype", "dtype"):
                    if key in cfg:
                        cfg[key] = "bfloat16"
                with open(cfg_path, "w") as f:
                    json.dump(cfg, f, indent=2)
        self.accelerator.wait_for_everyone()


def _apply_warmup(args: TrainingArguments, frac: float):
    """warmup_steps changed from int to float (ratio-capable) in newer transformers; support both."""
    steps_is_float = "float" in str(TrainingArguments.__dataclass_fields__["warmup_steps"].type)
    if steps_is_float:
        args.warmup_steps = frac
        if getattr(args, "warmup_ratio", None):
            args.warmup_ratio = 0.0
    else:
        args.warmup_steps = 0
        args.warmup_ratio = frac


# --------------------------------------------------------------------------- main


def _log_host_memory():
    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    limit = "unknown"
    try:
        with open("/proc/self/cgroup") as f:
            cg_path = f.read().strip().split("\n")[0].split(":")[-1]
        for path in (f"/sys/fs/cgroup{cg_path}/memory.max", "/sys/fs/cgroup/memory.max",
                     f"/sys/fs/cgroup/memory{cg_path}/memory.limit_in_bytes",
                     "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
            if os.path.exists(path):
                with open(path) as f:
                    raw = f.read().strip()
                limit = "none" if raw == "max" or int(raw) > 2**60 else f"{int(raw) / 2**30:.0f} GiB"
                break
    except (OSError, ValueError):
        pass
    logging.info("Host RAM: %.0f GiB on node | memory limit for this job: %s", total, limit)


def train():
    parser = transformers.HfArgumentParser((TrainingConfig, TrainingArguments))
    config, args = parser.parse_args_into_dataclasses()
    _log_host_memory()

    # Settings this script owns; they override whatever the CLI passed.
    args.bf16 = True
    args.fp16 = False
    args.gradient_checkpointing = True
    args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    args.remove_unused_columns = False
    args.save_only_model = True          # FSDP cannot gather bitsandbytes 8-bit optimizer state
    args.load_best_model_at_end = False  # incompatible with save_only_model; best ckpt kept on disk
    _apply_warmup(args, config.warmup_fraction)
    logging.info("Training config: %s", {**asdict(config), **args.to_dict()})

    model, tokenizer, _, n_params = load_model_and_tokenizer(config)

    data_files = sorted(str(f) for f in Path(config.data_dir).glob("*.jsonl"))
    if not data_files:
        raise FileNotFoundError(f"No *.jsonl files under {config.data_dir}")
    logging.info("Data files: %s", data_files)

    hf_dataset = load_dataset("json", data_files=data_files, split="train")
    hf_dataset = hf_dataset.filter(
        lambda ex: bool(ex.get("prompt_text")) and bool(ex.get("response_text")),
        num_proc=4,
    )
    hf_dataset = hf_dataset.shuffle(seed=42)
    tokenized = prepare_dataset(hf_dataset, tokenizer, config.block_size)

    if config.val_fraction and config.val_fraction > 0:
        split = tokenized.train_test_split(test_size=config.val_fraction, seed=42)
        train_dataset, eval_dataset = split["train"], split["test"]
    else:
        train_dataset, eval_dataset = tokenized, None
        if config.early_stopping_patience > 0:
            raise ValueError("Early stopping needs a val split. Set val_fraction > 0.")
    logging.info("Train: %s | Eval: %s", train_dataset, eval_dataset)

    callbacks = [EmptyCacheCallback(every=config.empty_cache_every)]
    if config.early_stopping_patience > 0:
        callbacks.append(EarlyStopOnEvalLoss(patience=config.early_stopping_patience))
        logging.info("Early stopping patience=%d evals", config.early_stopping_patience)

    trainer = BF16FullSaveTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=Gemma3TextCollator(tokenizer, config.block_size),
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    trainer.expected_param_numel = n_params

    # FSDP + gradient accumulation: Trainer wraps every non-final micro-step in model.no_sync(), and for
    # FSDP that keeps the full UNSHARDED bf16 gradient of the whole model resident (~24 GB/GPU for 12B)
    # until the sync step. Reduce-scatter on every micro-step instead; gradients then accumulate in the
    # sharded fp32 buffer that exists anyway. Costs some PCIe traffic, saves ~24 GB/GPU.
    trainer.accelerator.no_sync = lambda model=None, **kwargs: contextlib.nullcontext()

    trainer.train()
    trainer.save_model(args.output_dir)

    if trainer.args.should_save:
        try:
            AutoProcessor.from_pretrained(config.model_name).save_pretrained(args.output_dir)
        except Exception as exc:
            logging.warning("Could not save AutoProcessor: %s", exc)
        if trainer.state.best_metric is not None:
            logging.info(
                "Best eval_loss %.4f -> %s | final (last-step) weights -> %s",
                trainer.state.best_metric, trainer.state.best_model_checkpoint, args.output_dir,
            )
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()