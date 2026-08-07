"""
resume_smoke_test.py
====================
A throwaway smoke test for ChRBayesOpt's save/resume feature.

WHAT IT DOES
------------
  Runs a RESTORABLE BayesOpt experiment and interrupts + resumes it MULTIPLE
  times (default: 3 interrupts) before letting a final run finish. This proves
  resume works repeatedly, not just once -- the realistic case for a job that
  needs several SLURM wall-time windows to complete.

  Timeline (budget = TOTAL_TRIALS):
     cycle 0 : START    -> run a few trials -> INTERRUPT   (like a wall-time kill)
     cycle 1 : RESUME    -> run a few more  -> INTERRUPT
     cycle 2 : RESUME    -> run a few more  -> INTERRUPT
     final   : RESUME    -> run to completion
     analyze : verdict

MONITORING
----------
  The driver tails the trial log live, so you'll see each evaluation as it
  happens, plus banners for every interrupt/resume and a note if any config
  gets re-run (expected at most once per interrupt, for a trial caught
  in-flight at the moment of the kill).

HOW TO RUN
----------
    python resume_smoke_test.py          # local is fine -- no cluster/GPU needed

  Takes ~40-50s. Most of that is Ray restarting for each resume (unavoidable);
  the trials themselves are trivial. It re-invokes itself as
  `python resume_smoke_test.py worker` for each phase -- you don't call that.

TWO KNOBS YOU MAY NEED TO TOUCH
-------------------------------
  1. IMPORT PATH in run_worker() -- adjust if your package layout differs.
  2. KILL_SIGNAL -- SIGTERM (SLURM's first signal, graceful) by default; flip
     to signal.SIGKILL for the brutal no-grace test.

  Windows note: SIGTERM handling is flaky on Windows; if you're on Windows and
  it misbehaves, say so and I'll swap in a Windows-safe kill.
"""

import os
import sys
import json
import time
import signal
import shutil
import subprocess


# ============================== CONFIG ======================================
TOTAL_TRIALS    = 9                     # full budget for the experiment
SLEEP_PER_TRIAL = 1.0                   # seconds; slow enough to catch interrupts
NUM_INTERRUPTS  = 3                     # how many times to kill+resume before finishing
KILL_SIGNAL     = signal.SIGTERM        # SLURM's first signal; SIGKILL for brutal test
EXP_NAME        = "chr_resume_smoke_test"
STORAGE_DIR     = "chr_resume_smoke_storage"
LOG_PATH_ENV    = "CHR_TEST_LOG"        # absolute log path (env -> survives Ray's cwd change)
PHASE_ENV       = "CHR_TEST_PHASE"      # per-cycle label, written into the log
MODE_ENV        = "CHR_TEST_MODE"       # "start" or "resume" -> which restore flag to set
# ============================================================================


# ---------------------------------------------------------------------------
# Objective: a simple bowl (optimum at x=3, y=-1). simulate_fn logs every
# evaluation to an ABSOLUTE path from env -- absolute because Ray changes a
# trial's working directory mid-run.
# ---------------------------------------------------------------------------
def simulate_fn(config):
    x = config["x"]
    y = config["y"]
    time.sleep(SLEEP_PER_TRIAL)
    loss = (x - 3.0) ** 2 + (y + 1.0) ** 2

    log_path = os.environ[LOG_PATH_ENV]
    phase    = os.environ.get(PHASE_ENV, "?")
    with open(log_path, "a") as f:
        f.write(json.dumps({"phase": phase, "x": x, "y": y, "loss": loss}) + "\n")
        f.flush()
        os.fsync(f.fileno())

    return {"loss": loss}


def objective_fn(output):
    return output["loss"]


# ---------------------------------------------------------------------------
# WORKER: runs one phase (start or resume) in its own process.
# ---------------------------------------------------------------------------
def run_worker():
    # ADJUST THIS IMPORT if your package layout differs:
    from ChronoRay import ChRBayesOpt

    mode      = os.environ[MODE_ENV]          # "start" or "resume"
    is_start  = (mode == "start")
    is_resume = (mode == "resume")

    ChRBayesOpt(
        simulate_fn=simulate_fn,
        objective_fn=objective_fn,
        param_sample_space={
            "x": ChRBayesOpt.ChR_Distr.uniform(-10, 10),
            "y": ChRBayesOpt.ChR_Distr.uniform(-10, 10),
        },
        mode="min",
        total_trials=TOTAL_TRIALS,
        resources_per_trial={"cpu": 1, "gpu": 0},
        max_concurrent_trials=1,                 # sequential -> clean, deterministic test
        FLAG_auto_run=True,
        FLAG_start_restore_session=is_start,
        FLAG_restore_from_previous_session=is_resume,
        restore_experiment_name=EXP_NAME,
        restore_experiment_storage_path=STORAGE_DIR,
    )


# ---------------------------------------------------------------------------
# Log helpers
# ---------------------------------------------------------------------------
def _read_log(log_path):
    entries = []
    if not os.path.exists(log_path):
        return entries
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def _key(entry, nd=4):
    return (round(entry["x"], nd), round(entry["y"], nd))


# live tail printer: prints new evaluations since last call, flags re-runs
_seen_keys = set()
def _tail(log_path, already):
    entries = _read_log(log_path)
    for i, e in enumerate(entries[already:], start=already + 1):
        k = _key(e)
        tag = "   <-- re-run (trial was in-flight at the kill)" if k in _seen_keys else ""
        _seen_keys.add(k)
        print(f"     eval {i:02d} | {e['phase']:<10} | x={e['x']:+07.3f} y={e['y']:+07.3f} | loss={e['loss']:8.4f}{tag}")
    return len(entries)


def _unique_count(log_path):
    return len({_key(e) for e in _read_log(log_path)})


# ---------------------------------------------------------------------------
# Launch one phase and watch it. If kill_at is None, wait for it to finish.
# Otherwise, interrupt once the total evaluation count reaches kill_at.
# Returns: (seen_lines, exited_on_its_own, was_killed)
# ---------------------------------------------------------------------------
def launch_and_watch(mode, label, kill_at, env, log_path, seen):
    env = dict(env)
    env[MODE_ENV]  = mode
    env[PHASE_ENV] = label

    p = subprocess.Popen([sys.executable, __file__, "worker"], env=env)

    killed = False
    while True:
        seen = _tail(log_path, seen)

        if p.poll() is not None:                 # exited on its own
            seen = _tail(log_path, seen)
            return seen, True, False

        if kill_at is not None and len(_read_log(log_path)) >= kill_at:
            print(f"\n  >> INTERRUPT: {len(_read_log(log_path))} evals logged "
                  f"-- sending {KILL_SIGNAL.name} (simulating wall-time kill)\n")
            p.send_signal(KILL_SIGNAL)
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print("  >> graceful stop timed out -> SIGKILL")
                p.kill()
                p.wait()
            killed = True
            seen = _tail(log_path, seen)
            return seen, False, True

        time.sleep(0.4)


# ---------------------------------------------------------------------------
# ANALYZE: the verdict.
# ---------------------------------------------------------------------------
def analyze(log_path, interrupts_done):
    entries = _read_log(log_path)
    by_phase = {}
    for e in entries:
        by_phase.setdefault(e["phase"], []).append(e)

    unique     = {_key(e) for e in entries}
    duplicates = len(entries) - len(unique)

    print("\n================ RESULTS ================")
    print("Per-cycle evaluations:")
    for phase in sorted(by_phase):
        print(f"    {phase:<12}: {len(by_phase[phase])}")
    print(f"Total evaluations logged : {len(entries)}")
    print(f"Unique configs           : {len(unique)}  (budget = {TOTAL_TRIALS})")
    print(f"Duplicate (re-run) evals : {duplicates}")
    print(f"Interrupt+resume cycles  : {interrupts_done}")

    print("\n--- CHECKS ---")
    ok = True

    # (1) we actually interrupted+resumed more than once
    if interrupts_done >= 2:
        print(f"[PASS] Survived {interrupts_done} interrupt/resume cycles (multi-resume works).")
    else:
        ok = False
        print(f"[INCONCLUSIVE] Only {interrupts_done} interrupt(s) landed -- trials may be "
              "finishing too fast to catch. Raise SLEEP_PER_TRIAL or TOTAL_TRIALS.")

    # (2) reached full budget despite repeated interruptions
    if len(unique) >= TOTAL_TRIALS:
        print(f"[PASS] Reached the full budget after all resumes ({len(unique)} >= {TOTAL_TRIALS}).")
    else:
        ok = False
        print(f"[FAIL] Fell short of budget ({len(unique)} < {TOTAL_TRIALS}). "
              "State not restored properly across cycles.")

    # (3) completed trials preserved: allow up to one re-run per interrupt
    if duplicates <= interrupts_done:
        print(f"[PASS] Completed trials preserved across resumes "
              f"(duplicates={duplicates} <= interrupts={interrupts_done}).")
    else:
        print(f"[WARN] {duplicates} duplicate evals vs {interrupts_done} interrupts -- "
              "more re-running than expected; completed state may not be fully preserved.")

    # (4) soft: is the search converging as cycles progress?
    if entries:
        running_best = []
        best = float("inf")
        for e in entries:
            best = min(best, e["loss"])
            running_best.append(best)
        print(f"\n[INFO] best loss over time (optimum = 0): "
              f"{running_best[0]:.3f} -> {running_best[-1]:.3f}")
        print("       Should trend DOWN and not reset after a resume. If it looks like it")
        print("       restarts cold each cycle, the GP state isn't carrying over.")

    print("\n" + ("OVERALL: looks good." if ok else "OVERALL: needs a look."))
    print("=========================================")


# ---------------------------------------------------------------------------
# DRIVER
# ---------------------------------------------------------------------------
def drive():
    log_path    = os.path.abspath("chr_resume_smoke_log.jsonl")
    restore_dir = os.path.join(os.getcwd(), STORAGE_DIR, EXP_NAME)

    # clean slate
    if os.path.exists(log_path):
        os.remove(log_path)
    if os.path.isdir(STORAGE_DIR):
        shutil.rmtree(STORAGE_DIR)

    env = dict(os.environ)
    env[LOG_PATH_ENV] = log_path

    # cumulative kill points spread across the budget
    kill_points = [max(1, round(TOTAL_TRIALS * (i + 1) / (NUM_INTERRUPTS + 1)))
                   for i in range(NUM_INTERRUPTS)]

    # build the launch plan: NUM_INTERRUPTS killable phases + 1 final phase
    plan = [("start", "c0_start", kill_points[0])]
    for i in range(1, NUM_INTERRUPTS):
        plan.append(("resume", f"c{i}_resume", kill_points[i]))
    plan.append(("resume", "cF_final", None))

    print("=" * 55)
    print(f"MULTI-CYCLE RESUME TEST  (budget={TOTAL_TRIALS}, interrupts={NUM_INTERRUPTS})")
    print(f"Interrupt after these cumulative eval counts: {kill_points}")
    print("=" * 55)

    seen = 0
    interrupts_done = 0

    for idx, (mode, label, kill_at) in enumerate(plan):
        is_final = (kill_at is None)
        if mode == "start":
            print(f"\n### CYCLE {idx}: START restorable run "
                  f"(will interrupt at {kill_at} evals) ###")
        elif is_final:
            print(f"\n### FINAL: RESUME and run to completion ###")
            print(f"    (progress so far: {_unique_count(log_path)}/{TOTAL_TRIALS} unique trials)")
            print(f"    restoring from: {restore_dir}")
            print(f"    checkpoint present: {os.path.isdir(restore_dir)}")
        else:
            print(f"\n### CYCLE {idx}: RESUME "
                  f"(will interrupt at {kill_at} evals) ###")
            print(f"    (progress so far: {_unique_count(log_path)}/{TOTAL_TRIALS} unique trials)")
            print(f"    restoring from: {restore_dir}")
            print(f"    checkpoint present: {os.path.isdir(restore_dir)}")

        seen, exited, killed = launch_and_watch(mode, label, kill_at, env, log_path, seen)

        if killed:
            interrupts_done += 1
            time.sleep(1.5)      # let the filesystem settle after the kill
            continue

        if exited:
            # finished on its own
            if _unique_count(log_path) >= TOTAL_TRIALS:
                print(f"\n  (experiment completed during this cycle -- "
                      f"{_unique_count(log_path)}/{TOTAL_TRIALS})")
            else:
                print(f"\n  [WARN] phase exited early without reaching budget "
                      f"({_unique_count(log_path)}/{TOTAL_TRIALS}). Stopping.")
            break

    analyze(log_path, interrupts_done)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "worker":
        run_worker()
    else:
        drive()