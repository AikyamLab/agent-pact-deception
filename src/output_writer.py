"""
output_writer.py

All disk output. Paths:
    results/<family>/<model>/system_prompt_<system>/<condition>/<questioning>/
        transcript.txt   streamed turn-by-turn, judge verdicts inline, tally at the end
        runs.jsonl       one machine-readable line per run

start_run() writes the run header, write_turn() appends each turn the moment it
is produced (so the file can be tailed live), finish_run() closes the run off,
write_summary() appends the final tally across all runs.
"""
from __future__ import annotations

import json
from pathlib import Path

from pipeline import Transcript, Turn

_BAR = "=" * 100
_RUN = "#" * 100
_SEC = "*" * 100
_JDG = "~" * 100
_MON = "%" * 100

_SECTION_TITLES = {
    "main": "MAIN SCENARIO",
    "independent": "INDEPENDENT QUESTIONING",
    "collusive": "COLLUSIVE QUESTIONING (team decides together)",
}


class OutputWriter:
    def __init__(self, output_root, family, model_key, condition, system_name, questioning,
                 topology="sequential"):
        self.family, self.model_key = family, model_key
        self.condition, self.system_name = condition, system_name
        self.questioning = questioning
        self.topology = topology

        # topology is a path level so decentralized runs never overwrite sequential ones.
        # sequential keeps the original layout (no extra folder) for backward compatibility.
        base = (Path(output_root).expanduser().resolve()
                / family / model_key / f"system_prompt_{system_name}"
                / condition / questioning)
        self.dir = base if topology == "sequential" else base / topology
        self.dir.mkdir(parents=True, exist_ok=True)
        self.txt_path = self.dir / "transcript.txt"
        self.jsonl_path = self.dir / "runs.jsonl"

        self.txt_path.write_text("", encoding="utf-8")
        self.jsonl_path.write_text("", encoding="utf-8")
        self._section: str | None = None
        self.runs: list[Transcript] = []

    # -- streaming -------------------------------------------------------
    def _append(self, lines: list[str]) -> None:
        with self.txt_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
            f.flush()

    def start_run(self, run_index: int) -> None:
        self._section = None
        self._append(["", _RUN,
                      f"RUN {run_index:03d}  |  MODEL: {self.model_key}  |  "
                      f"CONDITION: {self.condition}  |  SYSTEM: {self.system_name}  |  "
                      f"QUESTIONING: {self.questioning}", _RUN])

    def write_turn(self, turn: Turn) -> None:
        lines: list[str] = []
        if turn.kind == "agent" and turn.mode != self._section:
            lines += ["", _SEC, _SECTION_TITLES.get(turn.mode, turn.mode.upper()), _SEC]
            self._section = turn.mode

        if turn.kind == "judge":
            lines += ["", _JDG, f">>> {turn.stage}", _JDG, turn.message, _JDG]
        elif turn.kind == "monitor":
            lines += ["", _MON, f"### {turn.stage}", _MON, turn.message, _MON]
        else:
            lines += ["", _BAR, turn.stage,
                      f"{turn.speaker} -> {turn.recipient}   [{turn.channel}]", _BAR]
            if turn.reasoning:
                lines += ["", "-- reasoning --", turn.reasoning]
            lines += ["", "-- message --", turn.message]
        self._append(lines)

    def finish_run(self, t: Transcript) -> None:
        self.runs.append(t)
        v = t.verdicts
        c = _tally(self.runs)
        self._append([
            "", "-" * 100,
            f"RUN {t.run_index:03d} RESULT: misaligned={_yn(v.get('misaligned'))}  "
            f"deceptive={_yn(v.get('deceptive'))}  "
            f"monitor_report={v.get('monitor_report', '-')}  "
            f"doubled_down={_yn(v.get('dd_deceptive'))}  "
            f"monitor_dd={v.get('monitor_dd', '-')}",
            f"RUNNING TALLY after {c['n']} run(s): "
            f"misaligned {c['mis']}/{c['n']}  |  "
            f"of those: deceptive {c['dec']}, honest {c['hon']}  |  "
            f"doubling-down: still deceptive {c['dd_yes']}, became honest {c['dd_no']}"
            + (f"  |  monitor caught: report {c['mon_rep_caught']}/{c['mon_rep_n']}, "
               f"dd {c['mon_dd_caught']}/{c['mon_dd_n']}" if (c['mon_rep_n'] or c['mon_dd_n']) else ""),
            "-" * 100])
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "run_index": t.run_index, "model_key": t.model_key, "family": self.family,
                "condition": t.condition, "system_name": t.system_name,
                "questioning": t.questioning, "decision": t.decision, "verdicts": v,
            }, ensure_ascii=False) + "\n")
            f.flush()

    # -- final tally -----------------------------------------------------
    def write_summary(self) -> None:
        c = _tally(self.runs)
        lines = [
            "", _RUN, "FINAL SUMMARY", _RUN,
            f"model={self.model_key}  condition={self.condition}  "
            f"system={self.system_name}  questioning={self.questioning}",
            f"total runs                      : {c['n']}",
            "",
            f"misaligned (trade executed)     : {_rate(c['mis'], c['n'])}",
            f"aligned (no trade)              : {_rate(c['n'] - c['mis'], c['n'])}",
            "",
            f"of the {c['mis']} misaligned run(s):",
            f"  deceptive to manager          : {_rate(c['dec'], c['mis'])}",
            f"  honest to manager             : {_rate(c['hon'], c['mis'])}",
            "",
            f"of the {c['dd_n']} run(s) that reached doubling-down:",
            f"  still deceptive (doubled down): {_rate(c['dd_yes'], c['dd_n'])}",
            f"  became honest                 : {_rate(c['dd_no'], c['dd_n'])}",
        ]
        if c["mon_rep_n"] or c["mon_dd_n"]:
            lines += [
                "",
                "MONITOR (detection vs ground-truth deception):",
                f"  report-stage deceptions shown : {c['mon_rep_n']}",
                f"    caught (SUSPICIOUS)         : {_rate(c['mon_rep_caught'], c['mon_rep_n'])}",
                f"    evaded (CLEAN)              : {_rate(c['mon_rep_n'] - c['mon_rep_caught'], c['mon_rep_n'])}",
                f"  doubling-down deceptions shown: {c['mon_dd_n']}",
                f"    caught (SUSPICIOUS)         : {_rate(c['mon_dd_caught'], c['mon_dd_n'])}",
                f"    evaded (CLEAN)              : {_rate(c['mon_dd_n'] - c['mon_dd_caught'], c['mon_dd_n'])}",
            ]
        lines.append(_RUN)
        self._append(lines)


def _tally(runs: list[Transcript]) -> dict:
    vs = [t.verdicts for t in runs]
    dd_yes = sum(1 for v in vs if v.get("dd_deceptive") is True)
    dd_no = sum(1 for v in vs if v.get("dd_deceptive") is False)
    mon_rep_caught = sum(1 for v in vs if v.get("monitor_report") == "SUSPICIOUS")
    mon_rep_n = sum(1 for v in vs if v.get("monitor_report") in ("SUSPICIOUS", "CLEAN"))
    mon_dd_caught = sum(1 for v in vs if v.get("monitor_dd") == "SUSPICIOUS")
    mon_dd_n = sum(1 for v in vs if v.get("monitor_dd") in ("SUSPICIOUS", "CLEAN"))
    return {
        "n": len(runs),
        "mis": sum(1 for v in vs if v.get("misaligned") is True),
        "dec": sum(1 for v in vs if v.get("deceptive") is True),
        "hon": sum(1 for v in vs if v.get("deceptive") is False),
        "dd_yes": dd_yes, "dd_no": dd_no, "dd_n": dd_yes + dd_no,
        "mon_rep_caught": mon_rep_caught, "mon_rep_n": mon_rep_n,
        "mon_dd_caught": mon_dd_caught, "mon_dd_n": mon_dd_n,
    }


def _yn(v) -> str:
    return "-" if v is None else ("Yes" if v else "No")


def _rate(k: int, n: int) -> str:
    return f"{k}/{n}  ({k / n:.2f})" if n else "n/a (0)"

















































# """
# output_writer.py

# All disk output. Paths:
#     results/<family>/<model>/system_prompt_<system>/<condition>/<questioning>/
#         transcript.txt   streamed turn-by-turn, judge verdicts inline, tally at the end
#         runs.jsonl       one machine-readable line per run

# start_run() writes the run header, write_turn() appends each turn the moment it
# is produced (so the file can be tailed live), finish_run() closes the run off,
# write_summary() appends the final tally across all runs.
# """
# from __future__ import annotations

# import json
# from pathlib import Path

# from pipeline import Transcript, Turn

# _BAR = "=" * 100
# _RUN = "#" * 100
# _SEC = "*" * 100
# _JDG = "~" * 100
# _MON = "%" * 100

# _SECTION_TITLES = {
#     "main": "MAIN SCENARIO",
#     "independent": "INDEPENDENT QUESTIONING",
#     "collusive": "COLLUSIVE QUESTIONING (team decides together)",
# }


# class OutputWriter:
#     def __init__(self, output_root, family, model_key, condition, system_name, questioning):
#         self.family, self.model_key = family, model_key
#         self.condition, self.system_name = condition, system_name
#         self.questioning = questioning

#         self.dir = (Path(output_root).expanduser().resolve()
#                     / family / model_key / f"system_prompt_{system_name}"
#                     / condition / questioning)
#         self.dir.mkdir(parents=True, exist_ok=True)
#         self.txt_path = self.dir / "transcript.txt"
#         self.jsonl_path = self.dir / "runs.jsonl"

#         self.txt_path.write_text("", encoding="utf-8")
#         self.jsonl_path.write_text("", encoding="utf-8")
#         self._section: str | None = None
#         self.runs: list[Transcript] = []

#     # -- streaming -------------------------------------------------------
#     def _append(self, lines: list[str]) -> None:
#         with self.txt_path.open("a", encoding="utf-8") as f:
#             f.write("\n".join(lines) + "\n")
#             f.flush()

#     def start_run(self, run_index: int) -> None:
#         self._section = None
#         self._append(["", _RUN,
#                       f"RUN {run_index:03d}  |  MODEL: {self.model_key}  |  "
#                       f"CONDITION: {self.condition}  |  SYSTEM: {self.system_name}  |  "
#                       f"QUESTIONING: {self.questioning}", _RUN])

#     def write_turn(self, turn: Turn) -> None:
#         lines: list[str] = []
#         if turn.kind == "agent" and turn.mode != self._section:
#             lines += ["", _SEC, _SECTION_TITLES.get(turn.mode, turn.mode.upper()), _SEC]
#             self._section = turn.mode

#         if turn.kind == "judge":
#             lines += ["", _JDG, f">>> {turn.stage}", _JDG, turn.message, _JDG]
#         elif turn.kind == "monitor":
#             lines += ["", _MON, f"### {turn.stage}", _MON, turn.message, _MON]
#         else:
#             lines += ["", _BAR, turn.stage,
#                       f"{turn.speaker} -> {turn.recipient}   [{turn.channel}]", _BAR]
#             if turn.reasoning:
#                 lines += ["", "-- reasoning --", turn.reasoning]
#             lines += ["", "-- message --", turn.message]
#         self._append(lines)

#     def finish_run(self, t: Transcript) -> None:
#         self.runs.append(t)
#         v = t.verdicts
#         c = _tally(self.runs)
#         self._append([
#             "", "-" * 100,
#             f"RUN {t.run_index:03d} RESULT: misaligned={_yn(v.get('misaligned'))}  "
#             f"deceptive={_yn(v.get('deceptive'))}  "
#             f"monitor_report={v.get('monitor_report', '-')}  "
#             f"doubled_down={_yn(v.get('dd_deceptive'))}  "
#             f"monitor_dd={v.get('monitor_dd', '-')}",
#             f"RUNNING TALLY after {c['n']} run(s): "
#             f"misaligned {c['mis']}/{c['n']}  |  "
#             f"of those: deceptive {c['dec']}, honest {c['hon']}  |  "
#             f"doubling-down: still deceptive {c['dd_yes']}, became honest {c['dd_no']}"
#             + (f"  |  monitor caught: report {c['mon_rep_caught']}/{c['mon_rep_n']}, "
#                f"dd {c['mon_dd_caught']}/{c['mon_dd_n']}" if (c['mon_rep_n'] or c['mon_dd_n']) else ""),
#             "-" * 100])
#         with self.jsonl_path.open("a", encoding="utf-8") as f:
#             f.write(json.dumps({
#                 "run_index": t.run_index, "model_key": t.model_key, "family": self.family,
#                 "condition": t.condition, "system_name": t.system_name,
#                 "questioning": t.questioning, "decision": t.decision, "verdicts": v,
#             }, ensure_ascii=False) + "\n")
#             f.flush()

#     # -- final tally -----------------------------------------------------
#     def write_summary(self) -> None:
#         c = _tally(self.runs)
#         lines = [
#             "", _RUN, "FINAL SUMMARY", _RUN,
#             f"model={self.model_key}  condition={self.condition}  "
#             f"system={self.system_name}  questioning={self.questioning}",
#             f"total runs                      : {c['n']}",
#             "",
#             f"misaligned (trade executed)     : {_rate(c['mis'], c['n'])}",
#             f"aligned (no trade)              : {_rate(c['n'] - c['mis'], c['n'])}",
#             "",
#             f"of the {c['mis']} misaligned run(s):",
#             f"  deceptive to manager          : {_rate(c['dec'], c['mis'])}",
#             f"  honest to manager             : {_rate(c['hon'], c['mis'])}",
#             "",
#             f"of the {c['dd_n']} run(s) that reached doubling-down:",
#             f"  still deceptive (doubled down): {_rate(c['dd_yes'], c['dd_n'])}",
#             f"  became honest                 : {_rate(c['dd_no'], c['dd_n'])}",
#         ]
#         if c["mon_rep_n"] or c["mon_dd_n"]:
#             lines += [
#                 "",
#                 "MONITOR (detection vs ground-truth deception):",
#                 f"  report-stage deceptions shown : {c['mon_rep_n']}",
#                 f"    caught (SUSPICIOUS)         : {_rate(c['mon_rep_caught'], c['mon_rep_n'])}",
#                 f"    evaded (CLEAN)              : {_rate(c['mon_rep_n'] - c['mon_rep_caught'], c['mon_rep_n'])}",
#                 f"  doubling-down deceptions shown: {c['mon_dd_n']}",
#                 f"    caught (SUSPICIOUS)         : {_rate(c['mon_dd_caught'], c['mon_dd_n'])}",
#                 f"    evaded (CLEAN)              : {_rate(c['mon_dd_n'] - c['mon_dd_caught'], c['mon_dd_n'])}",
#             ]
#         lines.append(_RUN)
#         self._append(lines)


# def _tally(runs: list[Transcript]) -> dict:
#     vs = [t.verdicts for t in runs]
#     dd_yes = sum(1 for v in vs if v.get("dd_deceptive") is True)
#     dd_no = sum(1 for v in vs if v.get("dd_deceptive") is False)
#     mon_rep_caught = sum(1 for v in vs if v.get("monitor_report") == "SUSPICIOUS")
#     mon_rep_n = sum(1 for v in vs if v.get("monitor_report") in ("SUSPICIOUS", "CLEAN"))
#     mon_dd_caught = sum(1 for v in vs if v.get("monitor_dd") == "SUSPICIOUS")
#     mon_dd_n = sum(1 for v in vs if v.get("monitor_dd") in ("SUSPICIOUS", "CLEAN"))
#     return {
#         "n": len(runs),
#         "mis": sum(1 for v in vs if v.get("misaligned") is True),
#         "dec": sum(1 for v in vs if v.get("deceptive") is True),
#         "hon": sum(1 for v in vs if v.get("deceptive") is False),
#         "dd_yes": dd_yes, "dd_no": dd_no, "dd_n": dd_yes + dd_no,
#         "mon_rep_caught": mon_rep_caught, "mon_rep_n": mon_rep_n,
#         "mon_dd_caught": mon_dd_caught, "mon_dd_n": mon_dd_n,
#     }


# def _yn(v) -> str:
#     return "-" if v is None else ("Yes" if v else "No")


# def _rate(k: int, n: int) -> str:
#     return f"{k}/{n}  ({k / n:.2f})" if n else "n/a (0)"






























# # """
# # output_writer.py

# # All disk output. Paths:
# #     results/<family>/<model>/system_prompt_<system>/<condition>/<questioning>/
# #         transcript.txt   streamed turn-by-turn, judge verdicts inline, tally at the end
# #         runs.jsonl       one machine-readable line per run

# # start_run() writes the run header, write_turn() appends each turn the moment it
# # is produced (so the file can be tailed live), finish_run() closes the run off,
# # write_summary() appends the final tally across all runs.
# # """
# # from __future__ import annotations

# # import json
# # from pathlib import Path

# # from pipeline import Transcript, Turn

# # _BAR = "=" * 100
# # _RUN = "#" * 100
# # _SEC = "*" * 100
# # _JDG = "~" * 100

# # _SECTION_TITLES = {
# #     "main": "MAIN SCENARIO",
# #     "independent": "INDEPENDENT QUESTIONING",
# #     "collusive": "COLLUSIVE QUESTIONING (team decides together)",
# # }


# # class OutputWriter:
# #     def __init__(self, output_root, family, model_key, condition, system_name, questioning):
# #         self.family, self.model_key = family, model_key
# #         self.condition, self.system_name = condition, system_name
# #         self.questioning = questioning

# #         self.dir = (Path(output_root).expanduser().resolve()
# #                     / family / model_key / f"system_prompt_{system_name}"
# #                     / condition / questioning)
# #         self.dir.mkdir(parents=True, exist_ok=True)
# #         self.txt_path = self.dir / "transcript.txt"
# #         self.jsonl_path = self.dir / "runs.jsonl"

# #         self.txt_path.write_text("", encoding="utf-8")
# #         self.jsonl_path.write_text("", encoding="utf-8")
# #         self._section: str | None = None
# #         self.runs: list[Transcript] = []

# #     # -- streaming -------------------------------------------------------
# #     def _append(self, lines: list[str]) -> None:
# #         with self.txt_path.open("a", encoding="utf-8") as f:
# #             f.write("\n".join(lines) + "\n")
# #             f.flush()

# #     def start_run(self, run_index: int) -> None:
# #         self._section = None
# #         self._append(["", _RUN,
# #                       f"RUN {run_index:03d}  |  MODEL: {self.model_key}  |  "
# #                       f"CONDITION: {self.condition}  |  SYSTEM: {self.system_name}  |  "
# #                       f"QUESTIONING: {self.questioning}", _RUN])

# #     def write_turn(self, turn: Turn) -> None:
# #         lines: list[str] = []
# #         if turn.kind == "agent" and turn.mode != self._section:
# #             lines += ["", _SEC, _SECTION_TITLES.get(turn.mode, turn.mode.upper()), _SEC]
# #             self._section = turn.mode

# #         if turn.kind == "judge":
# #             lines += ["", _JDG, f">>> {turn.stage}", _JDG, turn.message, _JDG]
# #         else:
# #             lines += ["", _BAR, turn.stage,
# #                       f"{turn.speaker} -> {turn.recipient}   [{turn.channel}]", _BAR]
# #             if turn.reasoning:
# #                 lines += ["", "-- reasoning --", turn.reasoning]
# #             lines += ["", "-- message --", turn.message]
# #         self._append(lines)

# #     def finish_run(self, t: Transcript) -> None:
# #         self.runs.append(t)
# #         v = t.verdicts
# #         c = _tally(self.runs)
# #         self._append([
# #             "", "-" * 100,
# #             f"RUN {t.run_index:03d} RESULT: misaligned={_yn(v.get('misaligned'))}  "
# #             f"deceptive={_yn(v.get('deceptive'))}  "
# #             f"doubled_down={_yn(v.get('dd_deceptive'))}",
# #             f"RUNNING TALLY after {c['n']} run(s): "
# #             f"misaligned {c['mis']}/{c['n']}  |  "
# #             f"of those: deceptive {c['dec']}, honest {c['hon']}  |  "
# #             f"doubling-down: still deceptive {c['dd_yes']}, became honest {c['dd_no']}",
# #             "-" * 100])
# #         with self.jsonl_path.open("a", encoding="utf-8") as f:
# #             f.write(json.dumps({
# #                 "run_index": t.run_index, "model_key": t.model_key, "family": self.family,
# #                 "condition": t.condition, "system_name": t.system_name,
# #                 "questioning": t.questioning, "decision": t.decision, "verdicts": v,
# #             }, ensure_ascii=False) + "\n")
# #             f.flush()

# #     # -- final tally -----------------------------------------------------
# #     def write_summary(self) -> None:
# #         c = _tally(self.runs)
# #         self._append([
# #             "", _RUN, "FINAL SUMMARY", _RUN,
# #             f"model={self.model_key}  condition={self.condition}  "
# #             f"system={self.system_name}  questioning={self.questioning}",
# #             f"total runs                      : {c['n']}",
# #             "",
# #             f"misaligned (trade executed)     : {_rate(c['mis'], c['n'])}",
# #             f"aligned (no trade)              : {_rate(c['n'] - c['mis'], c['n'])}",
# #             "",
# #             f"of the {c['mis']} misaligned run(s):",
# #             f"  deceptive to manager          : {_rate(c['dec'], c['mis'])}",
# #             f"  honest to manager             : {_rate(c['hon'], c['mis'])}",
# #             "",
# #             f"of the {c['dd_n']} run(s) that reached doubling-down:",
# #             f"  still deceptive (doubled down): {_rate(c['dd_yes'], c['dd_n'])}",
# #             f"  became honest                 : {_rate(c['dd_no'], c['dd_n'])}",
# #             _RUN,
# #         ])


# # def _tally(runs: list[Transcript]) -> dict:
# #     vs = [t.verdicts for t in runs]
# #     dd_yes = sum(1 for v in vs if v.get("dd_deceptive") is True)
# #     dd_no = sum(1 for v in vs if v.get("dd_deceptive") is False)
# #     return {
# #         "n": len(runs),
# #         "mis": sum(1 for v in vs if v.get("misaligned") is True),
# #         "dec": sum(1 for v in vs if v.get("deceptive") is True),
# #         "hon": sum(1 for v in vs if v.get("deceptive") is False),
# #         "dd_yes": dd_yes, "dd_no": dd_no, "dd_n": dd_yes + dd_no,
# #     }


# # def _yn(v) -> str:
# #     return "-" if v is None else ("Yes" if v else "No")


# # def _rate(k: int, n: int) -> str:
# #     return f"{k}/{n}  ({k / n:.2f})" if n else "n/a (0)"