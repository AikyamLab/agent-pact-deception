import os
import sys
sys.path.append('..')
sys.path.append('.')
from dataclasses import dataclass, field, asdict
from typing import Optional
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datasets import load_dataset, DatasetDict, disable_caching
disable_caching()
import transformers
import trl
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from peft import LoraConfig
from pathlib import Path
import torch


@dataclass
class TrainingConfig:
    """Configuration for the anti-deception SFT run (bf16 LoRA)."""
    model_name: str = field(default="meta-llama/Llama-3.3-70B-Instruct")
    block_size: int = field(default=4096)
    wandb_project: Optional[str] = field(default="Anti_deception_llama33_70B")
    data_dir: Optional[str] = field(
        default="/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/DATA"
    )
    # LoRA hyperparameters 
    lora_r: int = field(default=32)
    lora_alpha: int = field(default=64)
    lora_dropout: float = field(default=0.05)
    val_fraction: float = field(default=0.05)

    def __post_init__(self):
        os.environ["WANDB_PROJECT"] = self.wandb_project


def prepare_dataset(dataset, tokenizer, block_size):
    """Prompt/completion masking. `prompt_text` holds the fully-formatted context
    (deceptive system prompt + Human turns + trailing 'Assistant:'); `response_text`
    is the honest completion. Loss is computed ONLY on the response tokens."""
    def _tokenize(examples):
        prompts = examples["prompt_text"]
        responses = examples["response_text"]
        input_ids_list, labels_list, attn_list = [], [], []
        for prompt, response in zip(prompts, responses):
            if tokenizer.eos_token and not response.endswith(tokenizer.eos_token):
                response = response + tokenizer.eos_token
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
            response_ids = tokenizer.encode(response, add_special_tokens=False)
            input_ids = prompt_ids + response_ids
            labels = [-100] * len(prompt_ids) + response_ids[:]
            if len(input_ids) > block_size:
                input_ids = input_ids[-block_size:]
                labels = labels[-block_size:]
            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attn_list.append([1] * len(input_ids))
        return {"input_ids": input_ids_list, "labels": labels_list,
                "attention_mask": attn_list}

    dataset = dataset.map(_tokenize, batched=True,
                          remove_columns=dataset.column_names, num_proc=4)
    return DatasetDict({"train": dataset})


class EmptyCacheCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        torch.cuda.empty_cache()


def train():
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    logging.info(f"Training config: { {**asdict(config), **asdict(args)} }")

    # ---- bf16 LoRA: load the base in bf16 (NO 4-bit quantization) ----
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        # sharded across all visible GPUs; needed for a 70B in bf16 (~140GB)
        device_map="auto",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # required for grad checkpointing + LoRA on a non-kbit model

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # ---- LoRA adapters: attention + MLP + the unembedding head (lm_head) ----
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                        "lm_head"],  # unembedding head added for stronger output-distribution shift
    )

    # ---- data ----
    data_files = [str(f) for f in Path(config.data_dir).glob("*.jsonl")]
    logging.info(f"Data files: {data_files}")
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
    logging.info(f"Train: {train_dataset} | Eval: {eval_dataset}")

    # ---- trainer ----
    args.gradient_checkpointing = True
    args.bf16 = True
    args.max_seq_length = config.block_size
    args.dataset_kwargs = {"skip_prepare_dataset": True}
    args.remove_unused_columns = False

    trainer = trl.SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=args,
        peft_config=peft_config,
        callbacks=[EmptyCacheCallback()],
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()