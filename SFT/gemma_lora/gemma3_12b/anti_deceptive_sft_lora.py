"""
LoRA SFT for Gemma 3 12B-IT (text-only prompt/completion).

Gemma-specific handling:
  - Load Gemma3ForConditionalGeneration (12B-IT is multimodal)
  - Apply LoRA only on the language stack (vision tower also has q/k/v_proj)
  - Do not double-insert <bos>
  - Keep native <pad>
  - End IT completions with <end_of_turn>
  - token_type_ids=0 for text-only Gemma 3 training
"""

import os
import sys

sys.path.append("..")
sys.path.append(".")

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from datasets import load_dataset, DatasetDict, disable_caching

disable_caching()

import transformers
import trl
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from peft import LoraConfig, TaskType, get_peft_model
from pathlib import Path
import torch


DEFAULT_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "lm_head",
]


@dataclass
class TrainingConfig:
    model_name: str = field(default="google/gemma-3-12b-it")
    block_size: int = field(default=4096)
    wandb_project: Optional[str] = field(default="gemma_finetuning_lora")
    data_dir: Optional[str] = field(
        default="/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/DATA"
    )
    val_fraction: float = field(default=0.05)
    attn_implementation: str = field(default="sdpa")
    freeze_vision: bool = field(default=True)
    empty_cache_every: int = field(default=50)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)

    def __post_init__(self):
        os.environ["WANDB_PROJECT"] = self.wandb_project


def _is_gemma3_multimodal(config) -> bool:
    return getattr(config, "model_type", "") == "gemma3" and getattr(
        config, "vision_config", None
    ) is not None


def _is_vision_module(name: str) -> bool:
    lname = name.lower()
    return any(
        key in lname
        for key in ("vision_tower", "vision_model", "multi_modal", "multimodal_projector")
    )


def resolve_lora_targets(model, wanted: List[str]) -> List[str]:
    """Match leaf names the user asked for, but skip SigLIP / projector layers."""
    found = []
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in wanted:
            continue
        if _is_vision_module(name):
            continue
        if not hasattr(module, "weight"):
            continue
        found.append(name)
    if not found:
        raise RuntimeError(
            f"No LoRA target modules matched {wanted}. Print the model and check names."
        )
    logging.info("LoRA targets (%d): %s", len(found), found[:12] + (["..."] if len(found) > 12 else []))
    return found


def load_model_and_tokenizer(config: TrainingConfig):
    hf_config = AutoConfig.from_pretrained(config.model_name)
    multimodal = _is_gemma3_multimodal(hf_config)

    load_kwargs = dict(
        pretrained_model_name_or_path=config.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=config.attn_implementation,
    )

    if multimodal:
        from transformers import Gemma3ForConditionalGeneration

        logging.info("Loading Gemma3ForConditionalGeneration")
        model = Gemma3ForConditionalGeneration.from_pretrained(**load_kwargs)
    else:
        from transformers import AutoModelForCausalLM

        logging.info("Loading AutoModelForCausalLM")
        model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

    if hasattr(model, "config"):
        model.config.use_cache = False
    text_cfg = getattr(model.config, "text_config", None)
    if text_cfg is not None:
        text_cfg.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.warning("Tokenizer had no pad token; fell back to eos.")
    tokenizer.padding_side = "right"
    if hasattr(model, "config"):
        model.config.pad_token_id = tokenizer.pad_token_id
    if text_cfg is not None:
        text_cfg.pad_token_id = tokenizer.pad_token_id

    if multimodal and config.freeze_vision:
        n_frozen = 0
        for name, param in model.named_parameters():
            if _is_vision_module(name):
                param.requires_grad = False
                n_frozen += param.numel()
        logging.info("Froze vision/projector params: %.2fM", n_frozen / 1e6)

    return model, tokenizer, multimodal


def apply_lora(model, config: TrainingConfig):
    targets = resolve_lora_targets(model, DEFAULT_LORA_TARGETS)
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=targets,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


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
                input_ids = input_ids[-block_size:]
                labels = labels[-block_size:]
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

    dataset = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=4,
    )
    return DatasetDict({"train": dataset})


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


class EmptyCacheCallback(TrainerCallback):
    def __init__(self, every: int = 50):
        self.every = max(int(every), 0)

    def on_step_end(self, args, state, control, **kwargs):
        if self.every <= 0 or state.global_step % self.every != 0:
            return
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        torch.cuda.empty_cache()


def _set_if_present(args, name: str, value):
    if hasattr(args, name):
        setattr(args, name, value)


def train():
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    logging.info("Training config: %s", {**asdict(config), **asdict(args)})

    model, tokenizer, _ = load_model_and_tokenizer(config)
    model = apply_lora(model, config)

    data_files = [str(f) for f in Path(config.data_dir).glob("*.jsonl")]
    if not data_files:
        raise FileNotFoundError(f"No *.jsonl files under {config.data_dir}")
    logging.info("Data files: %s", data_files)

    hf_dataset = load_dataset("json", data_files=data_files, split="train")
    hf_dataset = hf_dataset.filter(
        lambda ex: ex.get("prompt_text") and ex.get("response_text"),
        num_proc=4,
    )
    hf_dataset = hf_dataset.shuffle(seed=42)
    tokenized = prepare_dataset(hf_dataset, tokenizer, config.block_size)["train"]

    if config.val_fraction and config.val_fraction > 0:
        split = tokenized.train_test_split(test_size=config.val_fraction, seed=42)
        train_dataset, eval_dataset = split["train"], split["test"]
    else:
        train_dataset, eval_dataset = tokenized, None

    logging.info("Train: %s | Eval: %s", train_dataset, eval_dataset)

    args.gradient_checkpointing = True
    _set_if_present(args, "gradient_checkpointing_kwargs", {"use_reentrant": False})
    args.bf16 = True
    _set_if_present(args, "fp16", False)
    _set_if_present(args, "max_seq_length", config.block_size)
    _set_if_present(args, "max_length", config.block_size)
    _set_if_present(args, "packing", False)
    args.dataset_kwargs = {"skip_prepare_dataset": True}
    args.remove_unused_columns = False
    _set_if_present(args, "loss_type", "nll")

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    trainer = trl.SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=args,
        data_collator=Gemma3TextCollator(tokenizer, config.block_size),
        callbacks=[EmptyCacheCallback(every=config.empty_cache_every)],
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if trainer.accelerator is not None:
        trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()