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
    Instead of letting the crash propagate, it returns a sentinel value (default
    `1`). Your objective function then maps that sentinel to a penalty score, so
    the failed point becomes a normal (bad) observation that the search algorithm
    can learn from -- it will steer away from that region of parameter space.

Design goals:
    - Wrap ANY callable `fn(config) -> result`. It doesn't care what `result` is.
    - Zero coupling to Ray / Tune / your optimizer. It's just a callable in,
      callable out.
    - The wrapped result is passed straight through unchanged on success.

Usage (drop-in for the lander experiment):

    from crash_safe import CrashSafe

    # fn_crash_retval=1 matches: `if output == 1: return CRASH_PENALTY`
    safe_simulate = CrashSafe(simulate_fn, fn_timeout=900, fn_crash_retval=1)

    ChRBayesOpt = ChRBayesOpt(
        safe_simulate,          # <-- the only change
        objective_fn,
        param_sample_space,
        mode="min",
        total_trials=50,
        max_concurrent_trials=2,
        FLAG_log_to_file=True,
        FLAG_auto_run=True,
    )

Notes:
    - `objective_fn` needs no changes: it already returns CRASH_PENALTY for the
      sentinel.
    - With this in place, trials no longer ERROR on a sim crash -- they complete
      with a penalty -- so the run can't hang the way it did before, and
      `max_failures` becomes irrelevant for sim-level failures.
"""

import os
import queue
import signal
import time
import multiprocessing as mp
import cloudpickle  


_NOTHING = object()


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


class CrashSafe:
    """Wrap a `fn(config)` so crashes/hangs become a sentinel instead of a kill.

    Parameters
    ----------
    fn : callable
        The simulation function, `fn(config) -> result`.
    fn_timeout : float | None
        Wall-clock seconds before a hung run is killed and treated as a failure.
        None disables the fn_timeout. (Your good runs were ~180-380s; pick a value
        above your worst legitimate run.)
    fn_crash_retval : Any
        Value returned on ANY failure. Default `1` to match an objective_fn that
        does `if output == 1: return PENALTY`.
    process_build_type : str
        Multiprocessing start method. "spawn" (default) gives a clean interpreter
        with no inherited native/CUDA state, which is the safe choice for
        PyChrono/FSI. Use "fork" only if you have a specific reason to.
    verbose : bool
        Print a one-line reason whenever a failure is caught.
    """

    def __init__(self, fn, *, fn_timeout=None, fn_crash_retval=1, process_build_type="spawn", verbose=True):
        #1. set up child process config based on user args 

        self.serialized_fn = cloudpickle.dumps(fn)
        self.fn_timeout = fn_timeout
        self.fn_crash_retval = fn_crash_retval
        self.process_build_type = process_build_type
        self.verbose = verbose

        #2. set up function name and docstring for log messages

        #internal: function's name, for our own log messages (falls back to a text repr)
        self._fn_name = getattr(fn, "__name__", repr(fn))
        #public: copy the name onto the wrapper so outside code reading .__name__ sees the real one
        self.__name__ = getattr(fn, "__name__", "simulate_fn")
        #copy the fuller "qualified" name too, defaulting to the plain name we just set
        self.__qualname__ = getattr(fn, "__qualname__", self.__name__)
        #copy the function's docstring (description), or None if it has none
        self.__doc__ = getattr(fn, "__doc__", None)

        #3. set up sys.path to make sure relevant modules are available in the child process
        paths = [os.getcwd()]
        try:
            import inspect
            paths.append(os.path.dirname(os.path.abspath(inspect.getfile(fn))))
        except Exception:
            pass
        self._extra_sys_path = list(dict.fromkeys(paths))  # de-duped, ordered



    def __call__(self, config):
        #1. launch the child process 
        
        ctx = mp.get_context(self.process_build_type)
        result_q = ctx.Queue()
        proc = ctx.Process(
            target=_child_entry,
            args=(self.serialized_fn, self._extra_sys_path, config, result_q),
            daemon=False,
        )
        proc.start()

        deadline = (time.monotonic() + self.fn_timeout) if self.fn_timeout else None
        payload = _NOTHING

        # Wait for a result, the process dying, or the fn_timeout -- whichever first.
        # Draining the queue via get() (rather than join()-then-get()) avoids the
        # classic deadlock when the returned object is large.
        while True:
            try:
                payload = result_q.get(timeout=0.5)
                break
            except queue.Empty:
                if not proc.is_alive():
                    # One last chance in case it put a result just before exit.
                    try:
                        payload = result_q.get_nowait()
                    except queue.Empty:
                        payload = _NOTHING
                    break
                if deadline and time.monotonic() >= deadline:
                    self._kill(proc)
                    return self._fail(f"timed out after {self.fn_timeout}s")

        proc.join()

        if payload is _NOTHING:
            # Process ended without sending anything -> hard crash.
            return self._fail(
                f"died without a result (exit={proc.exitcode}, "
                f"{self._signal_name(proc.exitcode)})"
            )

        status, data = payload
        if status == "ok":
            return data
        else:  # "exc"
            return self._fail(f"raised an exception:\n{data}")

    # -- helpers ------------------------------------------------------------

    def _fail(self, reason):
        if self.verbose:
            print(f"[CrashSafe] {self._fn_name}: {reason}", flush=True)
        return self.fn_crash_retval

    @staticmethod
    def _kill(proc):
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
        # Processes killed by a signal report a negative exit code.
        #   SIGSEGV -> -11  (native crash) ; SIGKILL -> -9 (OOM-killer)
        if exitcode is not None and exitcode < 0:
            try:
                return f"killed by {signal.Signals(-exitcode).name}"
            except ValueError:
                return f"killed by signal {-exitcode}"
        return f"exit code {exitcode}"