#!/usr/bin/env python3
"""On-policy distillation of an 8B LoRA student from a fully fine-tuned honest 8B teacher.

"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
_scratch = Path(f"/scratch/{os.environ.get('USER', 'unknown')}")
os.environ.setdefault("HF_HOME", str(_scratch / "hf_cache"))
os.environ.setdefault("TMPDIR", str(_scratch / "tmp"))
for _d in (os.environ["HF_HOME"], os.environ["TMPDIR"]):
    Path(_d).mkdir(parents=True, exist_ok=True)

from datasets import disable_caching, load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import DistillationConfig, DistillationTrainer

disable_caching()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("opd")

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--student", required=True, help="merged bf16 student directory")
    p.add_argument("--teacher", required=True, help="fully fine-tuned teacher directory")
    p.add_argument("--data", type=Path, required=True, help="jsonl with a prompt_text column")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=300, help="<=0 trains --epochs instead")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--beta", type=float, default=1.0, help="1.0 = reverse KL")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-completion-length", type=int, default=512)
    p.add_argument("--batch", type=int, default=4, help="micro-batch per backward")
    p.add_argument("--grad-accum", type=int, default=4, help="generation batch = batch * grad_accum")
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--eval-size", type=int, default=96)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report-to", default="none", choices=("none", "wandb"))
    p.add_argument("--run-name", default="opd_8b")
    p.add_argument("--use-vllm", action="store_true", help="vLLM colocate generation (needs trl[vllm])")
    p.add_argument("--vllm-mem", type=float, default=0.3)
    p.add_argument("--resume-from", default=None, help="checkpoint dir to resume from")
    return p.parse_args()


def load_prompts(path: Path, seed: int, eval_size: int):
    ds = load_dataset("json", data_files=str(path), split="train")
    total = len(ds)
    ds = ds.filter(lambda r: isinstance(r["prompt_text"], str) and r["prompt_text"].endswith("Assistant:"))
    log.info("prompts kept %d of %d", len(ds), total)
    ds = ds.map(lambda r: {"prompt": r["prompt_text"]}, remove_columns=ds.column_names)
    split = ds.train_test_split(test_size=eval_size, seed=seed)
    return split["train"], split["test"]


def load_tokenizer(student: str, teacher: str):
    tok = AutoTokenizer.from_pretrained(student, padding_side="left")
    if tok.pad_token is None:
        vocab = tok.get_vocab()
        tok.pad_token = "<|finetune_right_pad_id|>" if "<|finetune_right_pad_id|>" in vocab else tok.eos_token
    teacher_tok = AutoTokenizer.from_pretrained(teacher)
    probe = "Human: Was that advice honest?\n\nAssistant:"
    if tok(probe).input_ids != teacher_tok(probe).input_ids or len(tok) != len(teacher_tok):
        raise RuntimeError("student and teacher tokenizers differ")
    log.info("tokenizer ok  vocab=%d  eos=%r  pad=%r", len(tok), tok.eos_token, tok.pad_token)
    return tok


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    train_ds, eval_ds = load_prompts(args.data, args.seed, args.eval_size)
    tok = load_tokenizer(args.student, args.teacher)

    config = DistillationConfig(
        output_dir=str(args.out),
        seed=args.seed,
        model_init_kwargs={"dtype": "bfloat16", "attn_implementation": "sdpa", "device_map": {"": 0}},
        teacher_model_init_kwargs={"dtype": "bfloat16", "attn_implementation": "sdpa", "device_map": {"": 0}},
        # model_init_kwargs={"dtype": "bfloat16", "attn_implementation": "sdpa"},
        # teacher_model_init_kwargs={"dtype": "bfloat16", "attn_implementation": "sdpa"},
        beta=args.beta,
        temperature=args.temperature,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=args.batch * args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        max_steps=args.max_steps,
        num_train_epochs=args.epochs,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        disable_dropout=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        log_completions=True,
        num_completions_to_print=2,
        use_vllm=args.use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_mem,
        report_to=args.report_to,
        run_name=args.run_name,
    )
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    trainer = DistillationTrainer(
        model=args.student,
        teacher_model=args.teacher,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
        peft_config=peft_config,
    )
    log.info("train=%d eval=%d generation_batch=%d", len(train_ds), len(eval_ds), args.batch * args.grad_accum)
    trainer.train(resume_from_checkpoint=args.resume_from)
    trainer.save_model(str(args.out))
    tok.save_pretrained(str(args.out))
    log.info("saved adapter to %s", args.out)


if __name__ == "__main__":
    main()