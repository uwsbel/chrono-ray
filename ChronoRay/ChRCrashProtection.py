"""
crash_safe.py

A small, framework-agnostic wrapper that isolates a simulation function in a
child process so that a HARD failure (SIGSEGV from native C++ code, an OOM-kill,
or a hang) cannot take down the worker that's running it.

Why a child process?
    A segfault or an OOM-kill terminates the *process* from the outside -- there
    is nothing for try/except to catch, because the interpreter is already gone.
    The only way to survive it is to put the risky code in a separate process and
    watch that process from a parent that stays alive.

What it does on failure:
    Instead of letting the crash propagate, the protected simulate_fn returns a
    crash sentinel: the "worst possible" score for the optimization direction --
    +inf under mode="min" (largest misfit = worst) or -inf under mode="max"
    (smallest score = worst). Either way it is infinite, a value a real misfit can
    never take, so downstream code can recognise a failed run with is_crash() and
    handle it honestly:
      - the MCMC sampler treats it as a plain REJECT (stay put, re-record the
        current state) -- a crash is the ABSENCE of information about a point,
        not evidence that the point fits badly.
      - an optimizer can still treat it as a "worst possible / avoid this" score.

What it wraps:
    - simulate_fn (REQUIRED) is ALWAYS run in the isolated child process. This is
      the piece that can hard-crash, so it is the piece that must be protected --
      even in workflows that have no objective function, the Ray worker is kept
      alive.
    - objective_fn (OPTIONAL) is wrapped with a crash-aware guard: if the sim
      failed (its output is the crash sentinel), the guard passes the sentinel
      straight through instead of feeding it into your scoring arithmetic. So your
      objective_fn stays completely crash-unaware -- it only ever sees genuine
      simulation output. If you don't pass an objective_fn, this half is simply
      skipped and .objective_fn is None.

Design goals:
    - Wrap ANY callable `simulate_fn(config) -> result`. It doesn't care what
      `result` is.
    - Zero coupling to Ray / Tune / your optimizer. Callables in, callables out.
    - The wrapped result is passed straight through unchanged on success.

Usage -- protecting BOTH functions (e.g. Bayesian calibration):

    from crash_safe import ChRCrashProtection

    cp = ChRCrashProtection(simulate_fn, objective_fn, fn_timeout=900)

    cal = ChRBayesCali(
        cp,                 # <-- the object IS the crash-protected simulate_fn
        cp.objective_fn,    # <-- crash-aware objective (sentinel passes through)
        param_prior_space,
        sigma=0.05,
    )

Usage -- simulate only, no objective (workflows that don't score):

    cp = ChRCrashProtection(simulate_fn, fn_timeout=900)
    # cp is the protected sim; cp.objective_fn is None. The worker is still
    # protected from a sim crash exactly as before.

Notes:
    - objective_fn stays crash-unaware: the guard here means a crashed run (the
      crash sentinel) never reaches your scoring arithmetic. You do NOT need a
      penalty branch (e.g. `if output == 1`) anymore.
    - With this in place, trials no longer ERROR on a sim crash -- they complete
      with the sentinel -- so the run can't hang the way it did before, and
      `max_failures` becomes irrelevant for sim-level failures.
"""

#the disposable child 
import multiprocessing as mp
#force killing a child that reached timeout 
import os
import signal
#for the inf value (to indicate a crash) 
import math
#the communication between parent and child 
import queue
#timekeeping for timeout 
import time
#tool to send information to child 
import cloudpickle  


#0. shared entities between files 

#internal marker meaning "the mailbox was never filled" (a hard crash posted
#nothing). object() gives a one-of-a-kind value that can't collide with any real
#message, including None.
_EMPTY_QUEUE_SENTINEL = object()


def _crash_sentinel(mode):
    """Return the crash sentinel for `mode`: +inf under 'min', -inf under 'max'.

    The sentinel is the "worst possible" score for the optimization direction, so a
    crashed run is naturally rejected/avoided. It is infinite (never a real, finite
    misfit), survives pickling across the Ray boundary, and stays numeric so Ray
    never chokes on a non-number metric.
    """
    return float("inf") if mode == "min" else float("-inf")


def is_crash(value, mode="min"):
    """True iff `value` is the crash sentinel for the given optimization mode.

    The crash sentinel is the "worst possible" score for the mode: +inf under 'min'
    (largest misfit = worst) and -inf under 'max' (smallest score = worst). The sign
    guard matters -- the OPPOSITE infinity is an impossibly-GOOD score, not a crash,
    so it must never be flagged.
    """
    if not (isinstance(value, (int, float)) and math.isinf(value)):
        return False
    return value > 0 if mode == "min" else value < 0


def _child_entry(fnserialized_fn, extra_sys_path, config, result_q):
    """Runs in the CHILD process. Native crashes here only kill the child."""
    import sys
    # Make sure the child can import whatever the wrapped fn references
    # (e.g. LanderBaby, ChronoRay) even though it's a fresh interpreter.
    for p in extra_sys_path:
        if p and p not in sys.path:
            sys.path.insert(0, p)

    fn = cloudpickle.loads(fnserialized_fn)
    try:
        result = fn(config)
        result_q.put(("ok", result))
    except BaseException:
        # Catchable Python-level error (bad input, NaN you raised, etc.).
        import traceback
        result_q.put(("exc", traceback.format_exc()))


class ChRCrashProtection:
    """
    A callable object that runs a `simulate_fn(config)` inside an isolated child
    process, so a hard failure in the simulation (native crash, OOM-kill, or hang)
    kills only that disposable child -- never the Ray worker or the driver.

    On any failure it returns a crash sentinel (+inf under mode="min", -inf under
    mode="max") instead of propagating, so every call yields a usable value and the
    run keeps going.

    Two things get protected:
      - simulate_fn (REQUIRED): always run behind the child-process shield.
      - objective_fn (OPTIONAL): wrapped so a crashed run's sentinel passes
        straight through instead of hitting your scoring arithmetic.

    Both are exposed as attributes so the caller can hand them to Ray:
    `cp.simulate_fn` and `cp.objective_fn` (the latter is None if no objective
    was given).


    Parameters
    ----------
    simulate_fn : callable
        The simulation function, `simulate_fn(config) -> result`. ALWAYS run in
        the isolated child process.
    objective_fn : callable | None
        Optional scoring function, `objective_fn(output) -> misfit`. If given, a
        crash-aware version is exposed as `self.objective_fn`; a failed run's
        sentinel is passed through untouched. If None (workflows that don't
        score), `self.objective_fn` is None and only the sim is protected.
    fn_timeout : float | None
        Wall-clock seconds before a hung run is killed and treated as a failure.
        None disables the fn_timeout. (Your good runs were ~180-380s; pick a value
        above your worst legitimate run.)
    mode : str
        Optimization direction the downstream search uses, "min" (default) or
        "max". It sets which infinity means "crash": +inf when minimizing (largest
        = worst) or -inf when maximizing (smallest = worst). is_crash() then flags
        only that worst-direction infinity, so the opposite (an impossibly-good
        score) is never mistaken for a failure.
    process_build_type : str
        Multiprocessing start method. "spawn" (default) gives a clean interpreter
        with no inherited native/CUDA state, which is the safe choice for
        PyChrono/FSI. Use "fork" only if you have a specific reason to.
    verbose : bool
        Print a one-line reason whenever a failure is caught.
    """

    #===== CONSTRUCTOR =====#

    def __init__(self, simulate_fn, objective_fn=None, *, fn_timeout=None, mode="min", process_build_type="spawn", verbose=True):
        """Configure the wrapper: serialize the sim, capture settings, and build the
        crash-aware objective if one was supplied.
        """

        #0. sanity checks

        #0a. objective_fn is optional, but if supplied it must be callable
        if objective_fn is not None and not callable(objective_fn):
            raise ValueError("objective_fn must be callable or None")

        #0b. mode decides the sign of the crash sentinel, so it must be min or max
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")

        #1. set up child process config based on user args 

        self.serialized_fn = cloudpickle.dumps(simulate_fn)
        self.fn_timeout = fn_timeout
        self.mode = mode
        #crash return value = the worst-possible score for this mode (+inf min / -inf max)
        self.fn_crash_retval = _crash_sentinel(mode)
        self.process_build_type = process_build_type
        self.verbose = verbose

        #2. set up function name and docstring for log messages

        #internal: function's name, for our own log messages (falls back to a text repr)
        self._fn_name = getattr(simulate_fn, "__name__", repr(simulate_fn))
        #public: copy the name onto the wrapper so outside code reading .__name__ sees the real one
        self.__name__ = getattr(simulate_fn, "__name__", "simulate_fn")
        #copy the fuller "qualified" name too, defaulting to the plain name we just set
        self.__qualname__ = getattr(simulate_fn, "__qualname__", self.__name__)
        #copy the function's docstring (description), or None if it has none
        self.__doc__ = getattr(simulate_fn, "__doc__", None)

        #3. set up sys.path to make sure relevant modules are available in the child process
        paths = [os.getcwd()]
        try:
            import inspect
            paths.append(os.path.dirname(os.path.abspath(inspect.getfile(simulate_fn))))
        except Exception:
            pass
        self._extra_sys_path = list(dict.fromkeys(paths))  # de-duped, ordered

        #4. build the crash-aware objective if one was supplied, else leave it None.
        self.objective_fn = self._make_safe_objective(objective_fn) if objective_fn is not None else None

        #5. expose the simulation function as an attribute for API consistency
        self.simulate_fn = self 

    #===== PUBLIC METHODS =====#


    def __call__(self, config):
        """Run simulate_fn(config) in an isolated child process and return its result.

        On any failure -- hard crash, hang past fn_timeout, or a raised exception --
        returns the crash sentinel (self.fn_crash_retval) instead of propagating, so
        the caller always gets a usable value back.
        """

        #0. configure the child process     
        ctx = mp.get_context(self.process_build_type)
        result_q = ctx.Queue()
        proc = ctx.Process(
            target=_child_entry,
            args=(self.serialized_fn, self._extra_sys_path, config, result_q),
            daemon=False,
        )

        #1. launch the child process and set the deadline and the payload (container for ret val)
        proc.start()

        deadline = (time.monotonic() + self.fn_timeout) if self.fn_timeout else None
        payload = _EMPTY_QUEUE_SENTINEL

    
        #2. let the process run while monitoring: 
            #a. if the process has died 
            #b. if the execution has exceeded the deadline (curr time + fn_timeout)
        while True:
            try:
                payload = result_q.get(timeout=0.5) #constantly query the queue for the result 
                break #if we get a result, all is well and break the loop 
            except queue.Empty:
                if not proc.is_alive(): #(a) process died 
                    #one last chance in case it put a result just before exit.
                    try:
                        payload = result_q.get_nowait()
                    except queue.Empty:
                        payload = _EMPTY_QUEUE_SENTINEL
                    break
                if deadline and time.monotonic() >= deadline: #(b)timeout 
                    self._kill(proc)
                    return self._fail(f"timed out after {self.fn_timeout}s")

        #3. wait here for child process to fully finish and clean up 
        proc.join()

        #4. check if the process ended without sending anything --> hard crash 
        if payload is _EMPTY_QUEUE_SENTINEL:
            return self._fail(
                f"died without a result (exit={proc.exitcode}, "
                f"{self._signal_name(proc.exitcode)})"
            )

        #5. if the process sent a result, verify it and send it back to the parent process 
        status, data = payload
        if status == "ok":
            return data
        else:  # "exc"
            return self._fail(f"raised an exception:\n{data}")


    #===== (PRIVATE) HELPER METHODS =====#


    def _make_safe_objective(self, objective_fn):
        """Wrap objective_fn so a crashed sim's sentinel passes straight through
        instead of being fed into the user's scoring arithmetic.
        """

        #0. capture the crash return value and mode in the closure so the guard stays
        #   consistent with what this instance returns / how it detects a crash
        sentinel = self.fn_crash_retval
        mode = self.mode

        def safe_objective(output):
            #1a. crashed run -> pass the sentinel through untouched (no scoring).
                #detect it with the mode-aware is_crash (worst-direction infinity).
            if is_crash(output, mode):
                return sentinel

            #1b. otherwise score the genuine simulation output as normal
            return objective_fn(output)

        #2. preserve the original name / docstring so config reports & logs stay readable
        safe_objective.__name__ = getattr(objective_fn, "__name__", "objective_fn")
        safe_objective.__qualname__ = getattr(objective_fn, "__qualname__", safe_objective.__name__)
        safe_objective.__doc__ = getattr(objective_fn, "__doc__", None)
        
        return safe_objective

    def _fail(self, reason):
        """Log the failure reason (if verbose) and return the crash sentinel."""
        if self.verbose:
            print(f"[CrashSafe] {self._fn_name}: {reason}", flush=True)
        return self.fn_crash_retval

    @staticmethod
    def _kill(proc):
        """Force-terminate a child: ask politely (terminate), then SIGKILL if it
        refuses to die within the grace period.
        """
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            try:
                os.kill(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.join()

    @staticmethod
    def _signal_name(exitcode):
        """Turn a process exit code into a human-readable reason (e.g. 'killed by
        SIGSEGV') for logging.
        """
        #processes killed by a signal report a negative exit code.
            #SIGSEGV -> -11  (native crash) ; SIGKILL -> -9 (OOM-killer)
        if exitcode is not None and exitcode < 0:
            try:
                return f"killed by {signal.Signals(-exitcode).name}"
            except ValueError:
                return f"killed by signal {-exitcode}"
        return f"exit code {exitcode}"