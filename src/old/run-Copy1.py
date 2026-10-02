#!/usr/bin/env python3
"""
run.py - one command runs the agents AND the judge.

The judge model lives in a different conda env on a different GPU, so run.py
launches it as a subprocess, waits for it to load, uses it live during every
run, and shuts it down at the end. No second terminal.

    python run.py --model qwen3_8b --condition default_pressure --system default \
                  --repeats 50 --judge-model gptoss_120b

Omit --judge-model to run the agents without any judging.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---- environment / caches (must precede any torch / vLLM import) ----
os.environ["CUDA_VISIBLE_DEVICES"] = "0"          # agents on GPU 0; judge gets GPU 1
user = os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"
scratch_base = Path(f"/scratch/{user}")
hf_home = scratch_base / "hf_cache"
vllm_cache = scratch_base / ".cache" / "vllm"
torchinductor_cache = scratch_base / "torchinductor_cache"
hf_home.mkdir(parents=True, exist_ok=True)
(hf_home / "hub").mkdir(parents=True, exist_ok=True)
(hf_home / "xet").mkdir(parents=True, exist_ok=True)
vllm_cache.mkdir(parents=True, exist_ok=True)
torchinductor_cache.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(hf_home)
os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
os.environ["HF_XET_CACHE"] = str(hf_home / "xet")
os.environ["VLLM_CACHE_ROOT"] = str(vllm_cache)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(torchinductor_cache)

import argparse
import sys

SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
for _p in (str(SRC_DIR), str(BASE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from inference_engine import build_inference_engine
from chat_adapter import ChatAdapter
from scenario_loader import load_scenario
from pipeline import run_pipeline, run_single_agent
from output_writer import OutputWriter
from judge_client import JudgeClient, MonitorClient, load_monitor_prompt
from judge_launcher import launch_judge


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a multi-agent pressure scenario with an inline judge.")
    p.add_argument("--model", required=True, help="Agent model key from model_config.yaml")
    p.add_argument("--mode", default="multi", choices=["multi", "single"],
                   help="multi = Maya/Noah/Elena pipeline; single = one agent (Alpha)")
    p.add_argument("--condition", default=None,
                   help="pressure/scenario file stem (default: default_pressure for multi, single_agent for single)")
    p.add_argument("--system", default="default", help="System prompt name")
    p.add_argument("--questioning", default="collusive",
                   choices=["collusive", "independent"],
                   help="Doubling-down style (collusive = team decides together)")
    p.add_argument("--repeats", type=int, default=1)
    # judge
    p.add_argument("--judge-model", default=None,
                   help="Judge model key. Omit to disable judging.")
    p.add_argument("--judge-env", default="COTMonitor", help="conda env for the judge")
    p.add_argument("--judge-gpu", default="1")
    p.add_argument("--judge-port", type=int, default=8765)
    p.add_argument("--judge-url", default=None,
                   help="Use an already-running judge server instead of launching one")
    # monitor (off by default; reuses the judge server / model)
    p.add_argument("--monitor", action="store_true",
                   help="Also run the compliance monitor on deceptive runs (needs a judge)")
    p.add_argument("--monitor-scope", default="output", choices=["output", "full"],
                   help="What the monitor sees: 'output' = final messages only, "
                        "'full' = messages + scratchpad (match your system prompt)")
    p.add_argument("--monitor-prompt", default=None,
                   help="Monitor prompt template (one {material} slot). Default by mode: "
                        "oversight_prompts/single_agent_monitor.txt (single) or "
                        "oversight_prompts/monitor_deception.txt (multi)")
    # paths
    p.add_argument("--models-config", default=str(BASE_DIR / "configs" / "model_config.yaml"))
    p.add_argument("--system-prompt", default=None,
                   help="default: configs/system_prompt.jsonl (multi) or configs/system_prompt_single.jsonl (single)")
    p.add_argument("--pressure-dir", default=str(BASE_DIR / "prompt_scenarios" / "pressure_levels"))
    p.add_argument("--doubling-down", default=None,
                   help="default: prompt_scenarios/doubling_down.jsonl (multi) or configs/single_doubling_down.jsonl (single)")
    p.add_argument("--output-dir", default=str(BASE_DIR / "results"))
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    # resolve mode-dependent file defaults (only when the user did not override)
    single = args.mode == "single"
    if args.condition is None:
        args.condition = "single_agent" if single else "default_pressure"
    if args.system_prompt is None:
        args.system_prompt = str(BASE_DIR / "configs" /
                                 ("system_prompt_single.jsonl" if single else "system_prompt.jsonl"))
    if args.doubling_down is None:
        args.doubling_down = (str(BASE_DIR / "configs" / "single_doubling_down.jsonl") if single
                              else str(BASE_DIR / "prompt_scenarios" / "doubling_down.jsonl"))
    if args.monitor_prompt is None:
        args.monitor_prompt = str(BASE_DIR / "oversight_prompts" /
                                  ("single_agent_monitor.txt" if single else "monitor_deception.txt"))

    # judge first, so a bad judge config fails before the agent model loads
    judge = None
    judge_url = None
    if args.judge_url:
        judge_url = args.judge_url
        judge = JudgeClient(judge_url)
        print(f"Using judge at {judge_url}: {judge.health()}")
    elif args.judge_model:
        _, judge_url = launch_judge(model=args.judge_model, env=args.judge_env,
                                    gpu=args.judge_gpu, port=args.judge_port,
                                    base_dir=BASE_DIR, models_config=args.models_config)
        judge = JudgeClient(judge_url)
    else:
        print("No --judge-model given: running without judging.\n")

    monitor = None
    if args.monitor:
        if judge_url is None:
            raise SystemExit("--monitor needs a judge (pass --judge-model or --judge-url); "
                             "the monitor reuses the judge model/server.")
        monitor = MonitorClient(judge_url, load_monitor_prompt(args.monitor_prompt))
        print(f"Monitor enabled | scope={args.monitor_scope} | prompt={args.monitor_prompt}")

    engine = build_inference_engine(args.model, config_path=args.models_config)
    adapter = ChatAdapter(engine)
    family = adapter.family or "unknown"
    print(f"Agents: {args.model} | family: {family} | backend: {type(engine).__name__}")

    scenario = load_scenario(condition=args.condition, system_name=args.system,
                             pressure_dir=args.pressure_dir,
                             doubling_down_path=args.doubling_down,
                             system_prompt_path=args.system_prompt)
    print(f"Condition: {args.condition} | System: {args.system} | "
          f"Questioning: {args.questioning} | Agents: {', '.join(scenario.order)}")

    writer = OutputWriter(output_root=args.output_dir, family=family,
                          model_key=args.model, condition=args.condition,
                          system_name=args.system, questioning=args.questioning)
    print(f"Streaming to: {writer.txt_path}\n")

    pipeline_fn = run_single_agent if single else run_pipeline

    for i in range(1, args.repeats + 1):
        print(f"  run {i}/{args.repeats}")
        writer.start_run(i)
        t = pipeline_fn(scenario, adapter, model_key=args.model, run_index=i,
                        questioning=args.questioning, on_turn=writer.write_turn,
                        judge=judge, monitor=monitor, monitor_scope=args.monitor_scope)
        writer.finish_run(t)
        v = t.verdicts
        print(f"    -> misaligned={v.get('misaligned')} deceptive={v.get('deceptive')} "
              f"doubled_down={v.get('dd_deceptive')}")

    writer.write_summary()
    print(f"\nDone. Transcript + summary: {writer.txt_path}")


if __name__ == "__main__":
    main()
    