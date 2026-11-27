"""Strict argument check and write-once outputs for experiment scripts.

check_argv: stop at once on an unknown flag (a mistyped or merged flag used to fall back to another model silently).
guard_outputs: refuse to start if any OOF file the run would write already exists, unless --force is given, so
saved results are never overwritten by accident (project rule: write new files, never modify saved ones).
"""
import sys


def check_argv(flags=(), options=(), n_positional=0):
    """flags: names of --flag switches; options: names of --name=value options. --force is always allowed."""
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(pos) != n_positional:
        raise SystemExit(f"expected {n_positional} positional argument(s), got {pos}")
    for a in sys.argv[1:]:
        if not a.startswith("--"):
            continue
        key, eq, _ = a[2:].partition("=")
        if not ((eq and key in options) or (not eq and key in set(flags) | {"force"})):
            raise SystemExit(f"unknown argument {a!r}; flags {sorted(flags)}, options {sorted(options)}")


def guard_outputs(out_dir, names, seeds):
    """Raise before any work if out_dir already holds oof_<name>_<seed>.npy for a planned (name, seed)."""
    if "--force" in sys.argv:
        return
    existing = [f"oof_{n}_{s}.npy" for n in names for s in seeds if (out_dir / f"oof_{n}_{s}.npy").exists()]
    if existing:
        raise SystemExit(f"{out_dir} already has {existing}; refusing to overwrite (pass --force to replace them, "
                         f"or set FRAUD_OUTPUTS to write elsewhere)")
