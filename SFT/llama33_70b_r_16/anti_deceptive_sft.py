"""
LoRA SFT for Llama-3.3-70B-Instruct (text-only prompt/completion).
70B bf16 weights are sharded with device_map="auto".
"""

import os
import sys

sys.path.append("..")
sys.path.append(".")

from dataclasses import dataclass, field, asdict
from typing import List, Optional

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from datasets import load_dataset, DatasetDict, disable_caching

disable_caching()

import transformers
import trl
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    TrainerCallback,
)
from peft import LoraConfig
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
    model_name: str = field(default="meta-llama/Llama-3.3-70B-Instruct")
    block_size: int = field(default=4096)
    wandb_project: Optional[str] = field(default="Anti_deception_llama33_70B")
    data_dir: Optional[str] = field(
        default="/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/DATA"
    )
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    val_fraction: float = field(default=0.05)
    attn_implementation: str = field(default="sdpa")
    empty_cache_every: int = field(default=50)
    early_stopping_patience: int = field(default=5)

    def __post_init__(self):
        os.environ["WANDB_PROJECT"] = self.wandb_project


def _ensure_bos(prompt_ids: List[int], tokenizer) -> List[int]:
    bos = tokenizer.bos_token_id
    if bos is None:
        return prompt_ids
    if prompt_ids and prompt_ids[0] == bos:
        return prompt_ids
    return [bos] + prompt_ids


def _ensure_eos(response_ids: List[int], tokenizer) -> List[int]:
    eos = tokenizer.eos_token_id
    if eos is None:
        return response_ids
    if not response_ids or response_ids[-1] != eos:
        return response_ids + [eos]
    return response_ids


def prepare_dataset(dataset, tokenizer, block_size):
    def _tokenize(examples):
        input_ids_list, labels_list, attn_list = [], [], []
        for prompt, response in zip(examples["prompt_text"], examples["response_text"]):
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            response_ids = tokenizer.encode(response, add_special_tokens=False)
            prompt_ids = _ensure_bos(prompt_ids, tokenizer)
            response_ids = _ensure_eos(response_ids, tokenizer)
            input_ids = prompt_ids + response_ids
            labels = [-100] * len(prompt_ids) + response_ids[:]
            if len(input_ids) > block_size:
                input_ids = input_ids[-block_size:]
                labels = labels[-block_size:]
            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attn_list.append([1] * len(input_ids))
        return {
            "input_ids": input_ids_list,
            "labels": labels_list,
            "attention_mask": attn_list,
        }

    dataset = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=4,
    )
    return DatasetDict({"train": dataset})


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

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=config.attn_implementation,
        device_map="auto",
    )
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.warning("No pad token; using eos as pad.")
    tokenizer.padding_side = "right"
    model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=DEFAULT_LORA_TARGETS,
        ensure_weight_tying=True,
    )

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
        if config.early_stopping_patience > 0:
            raise ValueError("Early stopping needs a val split. Set val_fraction > 0.")

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

    callbacks = [EmptyCacheCallback(every=config.empty_cache_every)]
    if config.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)
        )
        logging.info("Early stopping patience=%d evals", config.early_stopping_patience)

    trainer = trl.SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=args,
        peft_config=peft_config,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            max_length=config.block_size,
            pad_to_multiple_of=8,
            label_pad_token_id=-100,
            return_tensors="pt",
        ),
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if trainer.accelerator is not None:
        trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()