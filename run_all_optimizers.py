import os
import time
import threading
import subprocess
from typing import Optional, List, Dict

BASE_DIR = r"D:\Pycharm\Bobsleigh\Code"

OPTIMIZER_SCRIPTS: Dict[int, str] = {
    idx: f"optimizer{idx}.py" for idx in range(1, 34)
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
        self.last_iter_index: int = -1
    def launch(self):
        with self.lock:
            if not os.path.exists(self.path):
                print(f"[Manager] optimizer{self.idx} script not found: {self.path}")
                self.status = "finished"
                return
            print(f"[Manager] Launching optimizer{self.idx} ({self.filename}) ...")
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
def get_finished_neighbor_for_target(
    workers: List[OptimizerWorker],
    target_idx: int
) -> Optional[int]:
    n = len(workers)
    prev_idx = target_idx - 1 if target_idx > 1 else None
    next_idx = target_idx + 1 if target_idx < n else None
    if prev_idx is not None:
        w_prev = workers[prev_idx - 1]
        if w_prev.status == "finished":
            return prev_idx
    if next_idx is not None:
        w_next = workers[next_idx - 1]
        if w_next.status == "finished":
            return next_idx
    return None





















def run_helper(
    helper_worker: OptimizerWorker,
    target_idx: int,
    prev_line: str,
    target_iter: int
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
def aggregate_state(
    target_idx: int,
    helper_idx: int,
    iter_idx: int,
    target_line: str,
    helper_output: str,
) -> Dict[str, str]:
    helper_lines = [l.strip() for l in helper_output.splitlines() if l.strip()]
    helper_input_lines = [l for l in helper_lines if "input_from_optimizer" in l]
    helper_coop_lines = [l for l in helper_lines if "[Coop" in l]
    helper_input = helper_input_lines[-1] if helper_input_lines else ""
    helper_summary = helper_coop_lines[-1] if helper_coop_lines else (
        helper_lines[-1] if helper_lines else ""
    )
    state = {
        "target_idx": target_idx,
        "helper_idx": helper_idx,
        "iter_idx": iter_idx,
        "target_line": target_line.strip(),
        "helper_input": helper_input,
        "helper_summary": helper_summary,
    }
    return state
def improve_state_to_message(state: Dict[str, str]) -> str:
    target_idx = state["target_idx"]
    helper_idx = state["helper_idx"]
    iter_idx = state["iter_idx"]
    target_line = state["target_line"]
    helper_input = state["helper_input"]
    helper_summary = state["helper_summary"]
    lines = [
        f"[CoopCombined {helper_idx}->{target_idx} | Iter {iter_idx}]",
        f"target_iter: {target_line}",
    ]
    if helper_summary:
        lines.append(f"helper_suggestion: {helper_summary}")
    elif helper_input:
        lines.append(f"helper_view_target: {helper_input}")
    return "\n".join(lines)
def monitor(workers: List[OptimizerWorker]):
    last_helper_iter: Dict[int, int] = {w.idx: -1 for w in workers}
    while True:
        finished_count = 0
        for w in workers:
            if w.status == "running" and w.process is not None:
                if w.process.poll() is not None:
                    w.status = "finished"
                    print(f"[Manager] optimizer{w.idx} terminated.")
            if w.status == "finished":
                finished_count += 1
        for target in workers:
            if target.status != "running":
                continue
            if target.last_iter_line is None:
                continue
            iter_idx = target.last_iter_index
            if iter_idx <= last_helper_iter[target.idx]:
                continue
            neighbor_idx = get_finished_neighbor_for_target(workers, target.idx)
            if neighbor_idx is None:
                continue
            helper = workers[neighbor_idx - 1]
            prev_line = target.last_iter_line
            print(
                f"\n[Manager] Cooperative call: optimizer{neighbor_idx} "
                f"assists optimizer{target.idx} at Iter {iter_idx}...\n"
            )
            helper_output = run_helper(
                helper_worker=helper,
                target_idx=target.idx,
                prev_line=prev_line,
                target_iter=iter_idx,
            )
            state = aggregate_state(
                target_idx=target.idx,
                helper_idx=helper.idx,
                iter_idx=iter_idx,
                target_line=prev_line,
                helper_output=helper_output,
            )
            combined_msg = improve_state_to_message(state)
            coop_msg_file = os.path.join(
                BASE_DIR,
                f"coop_msg_for_optimizer{target.idx}.txt"
            )
            with open(coop_msg_file, "w", encoding="utf-8") as f:
                f.write(combined_msg)
            print("========= Cooperative Suggestion =========")
            print(combined_msg)
            print("=========================================\n")
            last_helper_iter[target.idx] = iter_idx
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
