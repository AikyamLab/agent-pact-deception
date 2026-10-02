"""
LoRA SFT for DeepSeek-R1-Distill-Llama-70B (bf16 base, no quantization), one process, two GPUs.

DeepSeek-specific: responses end with the tokenizer's EOS <｜end▁of▁sentence｜> (the token the
distill model stops on); the tokenizer's own pad token (EOS) is kept; BOS <｜begin▁of▁sentence｜>
is added once if the prompt does not already start with it.


"""

import logging
import os
import sys
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

sys.path.append("..")
sys.path.append(".")

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import torch
import transformers
from datasets import disable_caching, load_dataset
from packaging import version
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

disable_caching()


@dataclass
class TrainingConfig:
    model_name: str = field(default="deepseek-ai/DeepSeek-R1-Distill-Llama-70B")
    block_size: int = field(default=4096)
    wandb_project: Optional[str] = field(default="deepseek_finetuning")
    data_dir: Optional[str] = field(
        default="/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/DATA"
    )
    val_fraction: float = field(default=0.05)
    attn_implementation: str = field(default="sdpa")
    # LoRA
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated. All attention + MLP projections (best practice). Add ,lm_head to adapt the unembedding too."},
    )
    # Layer split across the two GPUs. GPU1 also holds lm_head and the logits, so it gets fewer layers.
    gpu0_max_memory: str = field(default="78GiB")
    gpu1_max_memory: str = field(default="90GiB")
    response_end_token: Optional[str] = field(
        default=None,
        metadata={"help": "Token appended to every response. None = the tokenizer's EOS (correct for DeepSeek)."},
    )
    warmup_fraction: float = field(default=0.03)
    empty_cache_every: int = field(default=50)
    early_stopping_patience: int = field(default=4)

    def __post_init__(self):
        os.environ["WANDB_PROJECT"] = self.wandb_project


# --------------------------------------------------------------------------- model


def _dtype_kwarg(dtype):
    # transformers >= 4.56 renamed torch_dtype -> dtype
    key = "dtype" if version.parse(transformers.__version__) >= version.parse("4.56.0") else "torch_dtype"
    return {key: dtype}


def _token_id(tokenizer, token: str) -> Optional[int]:
    tid = tokenizer.convert_tokens_to_ids(token)
    if tid is None or tid == getattr(tokenizer, "unk_token_id", None):
        return None
    return tid


def load_model_and_tokenizer(config: TrainingConfig):
    n_gpu = torch.cuda.device_count()
    if n_gpu == 2:
        # Fill GPU0 up to gpu0_max_memory, remainder (plus norm + lm_head) on GPU1.
        device_map, max_memory = "sequential", {0: config.gpu0_max_memory, 1: config.gpu1_max_memory}
    else:
        device_map, max_memory = "auto", None
    logging.info("Loading %s in bf16 with device_map=%s max_memory=%s", config.model_name, device_map, max_memory)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        attn_implementation=config.attn_implementation,
        device_map=device_map,
        max_memory=max_memory,
        **_dtype_kwarg(torch.bfloat16),
    )
    model.config.use_cache = False

    placement = Counter(str(d) for d in model.hf_device_map.values())
    logging.info("Module placement: %s", dict(placement))
    if any(str(d) in ("cpu", "disk") for d in model.hf_device_map.values()):
        raise RuntimeError("Part of the model was offloaded to CPU/disk; raise gpu*_max_memory.")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    # DeepSeek's tokenizer already defines pad (= EOS). Keep it; only invent one if none exists.
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        pad = "<|finetune_right_pad_id|>" if _token_id(tokenizer, "<|finetune_right_pad_id|>") is not None else tokenizer.eos_token
        tokenizer.pad_token = pad
        logging.warning("Tokenizer had no pad token; using %r", pad)
    tokenizer.padding_side = "right"
    model.config.pad_token_id = tokenizer.pad_token_id

    # The distill model stops on its EOS (<｜end▁of▁sentence｜>), so that is what every response
    # must end with. NOT <|eot_id|>: that is a Llama-Instruct token the DeepSeek model never emits.
    end_id = _token_id(tokenizer, config.response_end_token) if config.response_end_token else None
    if end_id is None:
        end_id = tokenizer.eos_token_id
    logging.info(
        "bos=%r (%s) | pad=%r (%s) | response end token=%r (%d)",
        tokenizer.bos_token, tokenizer.bos_token_id, tokenizer.pad_token, tokenizer.pad_token_id,
        tokenizer.convert_ids_to_tokens(end_id), end_id,
    )
    return model, tokenizer, end_id


def add_lora(model, config: TrainingConfig):
    targets = [t.strip() for t in config.lora_target_modules.split(",") if t.strip()]
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, peft_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logging.info(
        "LoRA r=%d alpha=%d dropout=%.2f on %s | trainable %.1fM of %.1fB (%.3f%%)",
        config.lora_r, config.lora_alpha, config.lora_dropout, targets,
        trainable / 1e6, total / 1e9, 100 * trainable / total,
    )
    return model


# --------------------------------------------------------------------------- data


def _ensure_bos(prompt_ids: List[int], tokenizer) -> List[int]:
    bos = tokenizer.bos_token_id
    if bos is None or (prompt_ids and prompt_ids[0] == bos):
        return prompt_ids
    return [bos] + prompt_ids


def _ensure_end(response_ids: List[int], end_id: int, tokenizer) -> List[int]:
    stops = {end_id, tokenizer.eos_token_id}
    if not response_ids or response_ids[-1] not in stops:
        response_ids = response_ids + [end_id]
    return response_ids


def prepare_dataset(dataset, tokenizer, block_size, end_id):
    def _tokenize(examples):
        input_ids_list, labels_list, attn_list = [], [], []
        for prompt, response in zip(examples["prompt_text"], examples["response_text"]):
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            response_ids = tokenizer.encode(response, add_special_tokens=False)
            prompt_ids = _ensure_bos(prompt_ids, tokenizer)
            response_ids = _ensure_end(response_ids, end_id, tokenizer)
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
        return {"input_ids": input_ids_list, "labels": labels_list, "attention_mask": attn_list}

    return dataset.map(_tokenize, batched=True, remove_columns=dataset.column_names, num_proc=4)


# --------------------------------------------------------------------------- callbacks


class EmptyCacheCallback(TrainerCallback):
    def __init__(self, every: int = 50):
        self.every = max(int(every), 0)

    def on_step_end(self, args, state, control, **kwargs):
        if self.every > 0 and state.global_step % self.every == 0:
            torch.cuda.empty_cache()


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


def train():
    parser = transformers.HfArgumentParser((TrainingConfig, TrainingArguments))
    config, args = parser.parse_args_into_dataclasses()

    # Settings this script owns; they override whatever the CLI passed.
    args.bf16 = True
    args.fp16 = False
    args.gradient_checkpointing = True
    args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    args.remove_unused_columns = False
    _apply_warmup(args, config.warmup_fraction)
    if args.load_best_model_at_end and args.metric_for_best_model is None:
        args.metric_for_best_model, args.greater_is_better = "eval_loss", False
    logging.info("Training config: %s", {**asdict(config), **args.to_dict()})

    model, tokenizer, end_id = load_model_and_tokenizer(config)
    model.enable_input_require_grads()
    model = add_lora(model, config)

    # The model is split across GPUs (model parallel), so this is one process with one logical
    # device. Pin n_gpu to 1 so the Trainer neither wraps in DataParallel nor doubles the batch.
    args._n_gpu = 1

    data_files = sorted(str(f) for f in Path(config.data_dir).glob("*.jsonl"))
    if not data_files:
        raise FileNotFoundError(f"No *.jsonl files under {config.data_dir}")
    logging.info("Data files: %s", data_files)

    hf_dataset = load_dataset("json", data_files=data_files, split="train")
    hf_dataset = hf_dataset.filter(
        lambda ex: bool(ex.get("prompt_text")) and bool(ex.get("response_text")), num_proc=4
    )
    hf_dataset = hf_dataset.shuffle(seed=42)
    tokenized = prepare_dataset(hf_dataset, tokenizer, config.block_size, end_id)

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
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience))
        logging.info("Early stopping patience=%d evals", config.early_stopping_patience)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, pad_to_multiple_of=8, label_pad_token_id=-100, return_tensors="pt"
        ),
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model(args.output_dir)  # best adapter (load_best_model_at_end) + tokenizer

    if trainer.state.best_metric is not None:
        logging.info(
            "Best eval_loss %.4f at %s | best adapter also saved to %s",
            trainer.state.best_metric, trainer.state.best_model_checkpoint, args.output_dir,
        )


if __name__ == "__main__":
    train()



# """
# LoRA SFT for DeepSeek-R1-Distill-Llama-8B (text-only prompt/completion).

# DeepSeek-specific notes:
#   - Checkpoint is LlamaForCausalLM (Llama-3.1-8B distill), not Gemma
#   - Tokenizer uses <｜begin▁of▁sentence｜> / <｜end▁of▁sentence｜>
#   - add_bos_token=True and the official chat template already emit BOS,
#     so we encode with add_special_tokens=False and add one BOS if missing
#   - pad_token is already eos; do not invent a new pad id
#   - Same LoRA leaf names as Llama: q/k/v/o, gate/up/down, lm_head
#   - TRL chunked_nll cannot wrap lm_head; force loss_type=nll
# """

# import os
# import sys

# sys.path.append("..")
# sys.path.append(".")

# from dataclasses import dataclass, field, asdict
# from typing import Any, Dict, List, Optional

# import warnings

# warnings.filterwarnings("ignore", category=FutureWarning)

# import logging

# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# from datasets import load_dataset, DatasetDict, disable_caching

# disable_caching()

# import transformers
# import trl
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     DataCollatorForSeq2Seq,
#     TrainerCallback,
# )
# from peft import LoraConfig
# from pathlib import Path
# import torch


# DEFAULT_LORA_TARGETS = [
#     "q_proj",
#     "k_proj",
#     "v_proj",
#     "o_proj",
#     "gate_proj",
#     "up_proj",
#     "down_proj",
#     "lm_head",
# ]


# @dataclass
# class TrainingConfig:
#     model_name: str = field(default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
#     block_size: int = field(default=4096)
#     wandb_project: Optional[str] = field(default="deepseek_finetuning")
#     data_dir: Optional[str] = field(
#         default="/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/DATA"
#     )
#     lora_r: int = field(default=32)
#     lora_alpha: int = field(default=64)
#     lora_dropout: float = field(default=0.05)
#     val_fraction: float = field(default=0.05)
#     attn_implementation: str = field(default="sdpa")
#     empty_cache_every: int = field(default=50)

#     def __post_init__(self):
#         os.environ["WANDB_PROJECT"] = self.wandb_project


# def _ensure_bos(prompt_ids: List[int], tokenizer) -> List[int]:
#     bos = tokenizer.bos_token_id
#     if bos is None:
#         return prompt_ids
#     if prompt_ids and prompt_ids[0] == bos:
#         return prompt_ids
#     return [bos] + prompt_ids


# def _ensure_eos(response_ids: List[int], tokenizer) -> List[int]:
#     eos = tokenizer.eos_token_id
#     if eos is None:
#         return response_ids
#     if not response_ids or response_ids[-1] != eos:
#         return response_ids + [eos]
#     return response_ids


# def prepare_dataset(dataset, tokenizer, block_size):
#     """Loss only on response_text. prompt_text is already formatted context."""

#     def _tokenize(examples):
#         input_ids_list, labels_list, attn_list = [], [], []
#         for prompt, response in zip(examples["prompt_text"], examples["response_text"]):
#             prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
#             response_ids = tokenizer.encode(response, add_special_tokens=False)
#             prompt_ids = _ensure_bos(prompt_ids, tokenizer)
#             response_ids = _ensure_eos(response_ids, tokenizer)
#             input_ids = prompt_ids + response_ids
#             labels = [-100] * len(prompt_ids) + response_ids[:]
#             if len(input_ids) > block_size:
#                 input_ids = input_ids[-block_size:]
#                 labels = labels[-block_size:]
#             input_ids_list.append(input_ids)
#             labels_list.append(labels)
#             attn_list.append([1] * len(input_ids))
#         return {
#             "input_ids": input_ids_list,
#             "labels": labels_list,
#             "attention_mask": attn_list,
#         }

#     dataset = dataset.map(
#         _tokenize,
#         batched=True,
#         remove_columns=dataset.column_names,
#         num_proc=4,
#     )
#     return DatasetDict({"train": dataset})


# class EmptyCacheCallback(TrainerCallback):
#     def __init__(self, every: int = 50):
#         self.every = max(int(every), 0)

#     def on_step_end(self, args, state, control, **kwargs):
#         if self.every <= 0 or state.global_step % self.every != 0:
#             return
#         if torch.distributed.is_initialized():
#             torch.distributed.barrier()
#         torch.cuda.empty_cache()


# def _set_if_present(args, name: str, value):
#     if hasattr(args, name):
#         setattr(args, name, value)


# def train():
#     parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
#     config, args = parser.parse_args_into_dataclasses()
#     logging.info("Training config: %s", {**asdict(config), **asdict(args)})

#     model = AutoModelForCausalLM.from_pretrained(
#         config.model_name,
#         torch_dtype=torch.bfloat16,
#         attn_implementation=config.attn_implementation,
#         device_map="auto",
#     )
#     model.config.use_cache = False
#     if hasattr(model, "enable_input_require_grads"):
#         model.enable_input_require_grads()

#     tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
#     if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
#         tokenizer.pad_token = tokenizer.eos_token
#         logging.warning("No pad token; using eos as pad.")
#     tokenizer.padding_side = "right"
#     model.config.pad_token_id = tokenizer.pad_token_id

#     peft_config = LoraConfig(
#         r=config.lora_r,
#         lora_alpha=config.lora_alpha,
#         lora_dropout=config.lora_dropout,
#         bias="none",
#         task_type="CAUSAL_LM",
#         target_modules=DEFAULT_LORA_TARGETS,
#         ensure_weight_tying=True,
#     )

#     data_files = [str(f) for f in Path(config.data_dir).glob("*.jsonl")]
#     if not data_files:
#         raise FileNotFoundError(f"No *.jsonl files under {config.data_dir}")
#     logging.info("Data files: %s", data_files)

#     hf_dataset = load_dataset("json", data_files=data_files, split="train")
#     hf_dataset = hf_dataset.filter(
#         lambda ex: ex.get("prompt_text") and ex.get("response_text"),
#         num_proc=4,
#     )
#     hf_dataset = hf_dataset.shuffle(seed=42)
#     tokenized = prepare_dataset(hf_dataset, tokenizer, config.block_size)["train"]

#     if config.val_fraction and config.val_fraction > 0:
#         split = tokenized.train_test_split(test_size=config.val_fraction, seed=42)
#         train_dataset, eval_dataset = split["train"], split["test"]
#     else:
#         train_dataset, eval_dataset = tokenized, None
#     logging.info("Train: %s | Eval: %s", train_dataset, eval_dataset)

#     args.gradient_checkpointing = True
#     _set_if_present(args, "gradient_checkpointing_kwargs", {"use_reentrant": False})
#     args.bf16 = True
#     _set_if_present(args, "fp16", False)
#     _set_if_present(args, "max_seq_length", config.block_size)
#     _set_if_present(args, "max_length", config.block_size)
#     _set_if_present(args, "packing", False)
#     args.dataset_kwargs = {"skip_prepare_dataset": True}
#     args.remove_unused_columns = False
#     _set_if_present(args, "loss_type", "nll")

#     trainer = trl.SFTTrainer(
#         model=model,
#         processing_class=tokenizer,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         args=args,
#         peft_config=peft_config,
#         data_collator=DataCollatorForSeq2Seq(
#             tokenizer=tokenizer,
#             padding=True,
#             max_length=config.block_size,
#             pad_to_multiple_of=8,
#             label_pad_token_id=-100,
#             return_tensors="pt",
#         ),
#         callbacks=[EmptyCacheCallback(every=config.empty_cache_every)],
#     )

#     trainer.train()
#     trainer.save_model(args.output_dir)
#     tokenizer.save_pretrained(args.output_dir)
#     if trainer.accelerator is not None:
#         trainer.accelerator.wait_for_everyone()


# if __name__ == "__main__":
#     train()