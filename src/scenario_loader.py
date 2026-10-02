"""
scenario_loader.py

The only module that knows the on-disk layout. It loads everything the
pipeline needs for one run and returns plain dataclasses:

  - the three agent seeds for a pressure condition (default_pressure, ...)
  - each agent's doubling-down messages (Sally reminder, Amy question)
  - the chosen shared system prompt, composed with each agent's role prompt

"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class AgentSpec:
    name: str
    role: str
    system: str                      # composed: shared preamble + role-specific prompt
    user: str                        # seed user turn (may hold {..._MESSAGE} placeholders)
    stage2_user: str | None = None   # Elena only: the report-to-Amy turn
    sally_reminder: str | None = None
    amy_question: str | None = None


@dataclass
class Scenario:
    condition: str
    system_name: str
    order: list[str]                 # agent order as it appears in the file
    agents: dict[str, AgentSpec]


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_scenario(
    condition: str,
    system_name: str,
    *,
    pressure_dir: str | Path,
    doubling_down_path: str | Path,
    system_prompt_path: str | Path,
) -> Scenario:
    # shared system prompt, picked by name
    systems = {r["name"]: r["content"] for r in _read_jsonl(system_prompt_path)}
    if system_name not in systems:
        raise KeyError(
            f"System prompt '{system_name}' not in {system_prompt_path}. "
            f"Available: {sorted(systems)}"
        )
    shared_template = systems[system_name]

    # doubling-down, keyed by agent
    dd = {r["agent"]: r for r in _read_jsonl(doubling_down_path)}

    # pressure condition seeds, keyed by agent, in file order
    pressure_path = Path(pressure_dir) / f"{condition}.jsonl"
    rows = _read_jsonl(pressure_path)

    order: list[str] = []
    agents: dict[str, AgentSpec] = {}
    for r in rows:
        name = r["agent"]
        role = r["role"]
        role_specific = r["messages"][0]["content"]
        user = r["messages"][1]["content"]
        shared = shared_template.format(agent_name=name, agent_role=role)
        ddr = dd.get(name, {})
        agents[name] = AgentSpec(
            name=name,
            role=role,
            system=f"{shared}\n\n{role_specific}",
            user=user,
            stage2_user=r.get("stage2_user"),
            sally_reminder=ddr.get("sally_reminder"),
            amy_question=ddr.get("amy_question"),
        )
        order.append(name)

    return Scenario(condition=condition, system_name=system_name, order=order, agents=agents)