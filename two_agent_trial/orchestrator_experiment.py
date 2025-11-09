import os
import time
import threading
import subprocess
from typing import Optional, List
import re

BASE_DIR = r"D:\Pycharm\Bobsleigh\Code"

OPTIMIZER_SCRIPTS = {
    1: "optimizer.py",
    2: "optimizer2.py",
}















class OptimizerWorker:
    def __init__(self, idx: int, filename: str):
        self.idx = idx
        self.filename = filename
        self.path = os.path.join(BASE_DIR, filename)
        self.process: Optional[subprocess.Popen] = None
        self.status: str = "idle"
        self.lock = threading.Lock()
        self.last_iter_line: Optional[str] = None
        self.last_iter_index: int = 0
    def launch(self):
        with self.lock:
            if not os.path.exists(self.path):
                print(f"[Manager] optimizer{self.idx} script not found: {self.path}")
                self.status = "finished"
                return
            print(f"[Manager] Launching optimizer{self.idx} ...")
            self.process = subprocess.Popen(
                ["python", self.path],
                cwd=BASE_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self.status = "running"
            threading.Thread(target=self._pump_output, daemon=True).start()
    def _pump_output(self):
        if self.process is None or self.process.stdout is None:
            return
        for line in self.process.stdout:
            print(f"[optimizer{self.idx}] {line}", end="")
            stripped = line.lstrip()
            if stripped.startswith("[Iter"):
                self.last_iter_line = stripped.rstrip("\n")
                try:
                    parts = stripped.split()
                    if len(parts) >= 2 and parts[0].startswith("[Iter"):
                        iter_token = parts[1]
                        iter_str = iter_token.rstrip("]:")
                        self.last_iter_index = int(iter_str)
                except Exception:
                    pass


ITER_RE = re.compile(r"\[Iter\s+(\d+)\]")
def parse_iter_from_line(line: str, default: int = -1) -> int:
    if not line:
        return default
    m = ITER_RE.search(line)
    if not m:
        return default
    try:
        return int(m.group(1))
    except ValueError:
        return default
def extract_coop_line(helper_output: str) -> str:
    if not helper_output:
        return ""
    lines = helper_output.splitlines()
    for ln in lines:
        if ln.startswith("[Coop"):
            return ln.strip()
    for ln in lines:
        if "[Coop" in ln:
            return ln.strip()
    return lines[-1].strip()
def aggregate_messages(primary_line: str, helper_output: str) -> str:
    it = parse_iter_from_line(primary_line, default=-1)
    coop_line = extract_coop_line(helper_output)
    header = f"[Combined 1+2 | Iter {it}]"
    combined = f"{header}\nprimary: {primary_line}\nhelper: {coop_line}"
    return combined
NUM_FIELD_RE = re.compile(r"([a-zA-Z_]+)=(-?\d+(?:\.\d+)?)")


def parse_numeric_fields(line: str) -> dict:
    if not line:
        return {}
    fields = {}
    for key, value in NUM_FIELD_RE.findall(line):
        try:
            fields[key] = float(value)
        except ValueError:
            pass
    return fields
def parse_note_field(line: str) -> Optional[str]:
    if not line:
        return None
    idx = line.find("note=")
    if idx == -1:
        return None
    note = line[idx + len("note="):].strip()
    note = note.rstrip("] ")
    if note.startswith('"') and note.endswith('"') and len(note) >= 2:
        note = note[1:-1]
    return note.strip() or None
def improve_combined_message(combined: str) -> str:
    if not combined:
        return "[Enhanced Plan] no data"
    lines = combined.splitlines()
    header = lines[0] if lines else ""
    primary_line = ""
    helper_line = ""
    for ln in lines[1:]:
        if ln.startswith("primary:"):
            primary_line = ln[len("primary:"):].strip()
        elif ln.startswith("helper:"):
            helper_line = ln[len("helper:"):].strip()
    it = parse_iter_from_line(primary_line, default=-1)
    primary_fields = parse_numeric_fields(primary_line)
    helper_fields = parse_numeric_fields(helper_line)
    note = parse_note_field(helper_line)
    old_h2 = primary_fields.get("old_h2")
    opt1_new_h2 = primary_fields.get("new_h2")
    coop_new_h2 = helper_fields.get("new_h2", opt1_new_h2)
    coop_step = helper_fields.get("step", primary_fields.get("step"))
    cost = primary_fields.get("cost")
    parts = [f"[Enhanced Plan | Iter {it}]"]
    if old_h2 is not None and coop_new_h2 is not None:
        h2_part = f"h2: {old_h2:.6f} -> {coop_new_h2:.6f}"
        if opt1_new_h2 is not None and abs(opt1_new_h2 - coop_new_h2) > 1e-9:
            h2_part += f" (optimizer1 proposed {opt1_new_h2:.6f})"
        parts.append(h2_part)
    if coop_step is not None:
        parts.append(f"step={coop_step:.6f}")
    if cost is not None:
        parts.append(f"prev_cost={cost:.6f}")
    if note:
        parts.append(f"note={note}")
    if len(parts) == 1:
        return header or "[Enhanced Plan]"
    return " ".join(parts)
def get_unfinished_neighbor(workers: List[OptimizerWorker], idx: int) -> Optional[int]:
    n = len(workers)
    prev_idx = idx - 1 if idx > 1 else None
    next_idx = idx + 1 if idx < n else None
    if prev_idx is not None:
        w_prev = workers[prev_idx - 1]
        if w_prev.status != "finished":
            return prev_idx
    if next_idx is not None:
        w_next = workers[next_idx - 1]
        if w_next.status != "finished":
            return next_idx
    return None
def run_helper(
    helper_worker: OptimizerWorker, target_idx: int, prev_line: str, target_iter: int
) -> str:
    if not os.path.exists(helper_worker.path):
        return f"[Manager] helper script not found: {helper_worker.path}"
    env = os.environ.copy()
    env["COOP_MODE"] = "helper"
    env["COOP_TARGET_ID"] = str(target_idx)
    env["COOP_MSG"] = prev_line or ""
    env["COOP_ITER"] = str(target_iter)
    print(
        f"[Manager] optimizer{helper_worker.idx} will run in helper mode "
        f"to assist optimizer{target_idx}."
    )
    proc = subprocess.Popen(
        ["python", helper_worker.path, "--helper"],
        cwd=BASE_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    out, _ = proc.communicate()
    return out











def handle_optimizer_finished(
    workers: List[OptimizerWorker], finished_worker: OptimizerWorker
):
    idx = finished_worker.idx
    neighbor_idx = get_unfinished_neighbor(workers, idx)
    if neighbor_idx is None:
        print(f"[Manager] optimizer{idx} has no unfinished neighbor to help.")
        return
    print(f"[Manager] optimizer{idx} will assist optimizer{neighbor_idx}.")
    target_worker = workers[neighbor_idx - 1]
    print(
        f"[Manager] Waiting for optimizer{neighbor_idx} to output at least one [Iter ...] line ..."
    )
    while target_worker.last_iter_line is None and target_worker.status != "finished":
        time.sleep(0.1)
    target_line = target_worker.last_iter_line
    target_iter = target_worker.last_iter_index
    if target_line is None:
        print(
            f"[Manager] optimizer{neighbor_idx} finished but produced no [Iter ...] lines; "
            f"skip cooperative optimization."
        )
        return
    helper_output = run_helper(
        helper_worker=finished_worker,
        target_idx=neighbor_idx,
        prev_line=target_line,
        target_iter=target_iter,
    )
    combined = aggregate_messages(target_line, helper_output)
    enhanced = improve_combined_message(combined)
    coop_msg_file = os.path.join(BASE_DIR, f"coop_msg_for_optimizer{neighbor_idx}.txt")
    with open(coop_msg_file, "w", encoding="utf-8") as f:
        f.write(enhanced + "\n")
    print("\n========= Combined & Enhanced Output =========")
    print("Raw primary line:")
    print(target_line)
    print("\nRaw helper output:")
    print(helper_output.strip())
    print("\nAggregated message:")
    print(combined)
    print("\nEnhanced message to be used by optimizer:")
    print(enhanced)
    print("=============================================\n")
def monitor(workers: List[OptimizerWorker]):
    target_worker = workers[0]
    helper_worker = workers[1]
    last_helper_iter = -1
    while True:
        finished_count = 0
        for w in workers:
            if w.status == "running" and w.process is not None:
                if w.process.poll() is not None:
                    w.status = "finished"
                    print(f"[Manager] optimizer{w.idx} terminated.")
            if w.status == "finished":
                finished_count += 1
        if target_worker.last_iter_index > last_helper_iter:
            last_helper_iter = target_worker.last_iter_index
            iter_idx = target_worker.last_iter_index
            prev_line = target_worker.last_iter_line
            print(
                f"\n[Manager] Cooperative call: optimizer2 assists optimizer1 at Iter {iter_idx}...\n"
            )
            helper_output = run_helper(
                helper_worker=helper_worker,
                target_idx=1,
                prev_line=prev_line,
                target_iter=iter_idx,
            )
            combined_msg = aggregate_messages(prev_line, helper_output)
            enhanced_msg = improve_combined_message(combined_msg)
            coop_msg_file = os.path.join(
                BASE_DIR, "coop_msg_for_optimizer1.txt"
            )
            try:
                with open(coop_msg_file, "w", encoding="utf-8") as f:
                    f.write(enhanced_msg + "\n")
            except Exception as e:
                print(f"[Manager] Failed to write coop message file: {e}")
            print("========= Raw helper output =========")
            print(helper_output.strip())
            print("========= Aggregated messages =========")
            print(combined_msg)
            print("========= Enhanced suggestion for optimizer1 =========")
            print(enhanced_msg)
            print("=========================================\n")
        if finished_count == len(workers):
            break
        time.sleep(0.1)
def main():
    workers: List[OptimizerWorker] = []
    for idx, script_name in OPTIMIZER_SCRIPTS.items():
        workers.append(OptimizerWorker(idx, script_name))
    for w in workers:
        w.launch()
    print("[Manager] All optimizers launched.")
    monitor(workers)
    print("[Manager] All optimizers finished.")
    print("[Manager] Final statuses:")
    for w in workers:
        print(f"  optimizer{w.idx}: status={w.status}")
if __name__ == "__main__":
    main()
