"""
Script: verify_env.py
Post-installation health check for the `neuralprophet_env` environment.

Run this once on every new machine, immediately after creating the Conda
environment and before opening any notebook. It verifies, in order:

1. Interpreter identity (Python version, platform, CPU architecture).
2. Presence and version of every package the pipeline imports.
3. The numpy-below-2.0 constraint required by neuralprophet 0.8.0.
4. That torch and pytorch-lightning can actually train a NeuralProphet model
   end to end, which is the one link in the dependency chain that a Conda
   solve cannot prove on its own.
5. That the `jupytext` command-line entry point is on PATH, which the
   auto-watcher depends on.
6. That the local, git-ignored `data/` and `outputs/` directory trees exist.

Inputs: none (reads the active environment and the repository it sits in).
Outputs: a printed report on stdout; exit code 0 if every mandatory check
passed, 1 otherwise. Missing data directories are reported as warnings, not
failures, because a fresh clone is expected to lack them.

Usage:
    conda activate neuralprophet_env
    python verify_env.py
"""

import importlib
import os
import platform
import shutil
import sys
import tempfile
import warnings
from importlib import metadata

# Packages that must import cleanly for the pipeline to run. The second element
# is the attribute holding the version string, when it is not `__version__`.
REQUIRED_PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "seaborn",
    "sklearn",
    "statsmodels",
    "xgboost",
    "torch",
    "pytorch_lightning",
    "neuralprophet",
    "jupytext",
    "watchdog",
    "tqdm",
]

# Import names whose installed distribution is published under a different name,
# needed when the module exposes no `__version__` attribute of its own.
DISTRIBUTION_NAMES = {
    "sklearn": "scikit-learn",
    "pytorch_lightning": "pytorch-lightning",
}

# Directory tree that the notebooks read from and write to. These are ignored by
# Git, so they never arrive with a clone and must be created by hand.
EXPECTED_DIRS = [
    "data/raw/sensor",
    "data/raw/proxies",
    "data/interim/sensor",
    "data/interim/aligned",
    "data/processed",
    "outputs/figures",
    "outputs/tables",
    "outputs/models",
]

OK = "PASS"
BAD = "FAIL"
WARN = "WARN"


def _version_of(module, import_name):
    """
    Returns a best-effort version string for an imported module.

    Falls back to the installed distribution metadata for packages that do not
    expose a `__version__` attribute, such as watchdog.

    Parameters
    ----------
    module : module
        An already-imported Python module.
    import_name : str
        The name used to import the module, used to look up the distribution.

    Returns
    -------
    version : str
        The module version, or "unknown" if no version could be determined.
    """
    for attr in ("__version__", "version", "VERSION"):
        value = getattr(module, attr, None)
        if isinstance(value, str):
            return value

    distribution = DISTRIBUTION_NAMES.get(import_name, import_name)
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"


def check_interpreter():
    """
    Prints interpreter identity and verifies the Python minor version.

    The pipeline is pinned to Python 3.10 because neuralprophet 0.8.0 caps
    several of its dependencies below versions that dropped 3.10 support.

    Returns
    -------
    ok : bool
        True if the running interpreter is Python 3.10.
    """
    print("Interpreter")
    print(f"  executable   : {sys.executable}")
    print(f"  python       : {platform.python_version()}")
    print(f"  platform     : {platform.system()} {platform.release()}")
    print(f"  architecture : {platform.machine()}")

    ok = sys.version_info[:2] == (3, 10)
    status = OK if ok else BAD
    print(f"  [{status}] expected Python 3.10, found {platform.python_version()}")

    if platform.system() == "Darwin" and platform.machine() != "arm64":
        print(
            "  [WARN] Running an x86_64 interpreter on macOS. On Apple Silicon this "
            "means Conda is emulated through Rosetta and will be markedly slower; "
            "reinstall a native arm64 distribution."
        )
    print()
    return ok


def check_packages():
    """
    Imports every required package and reports its version.

    Returns
    -------
    ok : bool
        True if every package imported successfully.
    versions : dict
        Mapping of package name to version string for those that imported.
    """
    print("Packages")
    versions = {}
    failures = []

    for name in REQUIRED_PACKAGES:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                module = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - we want the reason, whatever it is
            failures.append(name)
            print(f"  [{BAD}] {name:<20} import failed: {exc}")
            continue

        version = _version_of(module, name)
        versions[name] = version
        print(f"  [{OK}] {name:<20} {version}")

    print()
    return not failures, versions


def check_numpy_pin(versions):
    """
    Verifies the numpy major version required by neuralprophet 0.8.0.

    A numpy 2.x installation alongside Conda-built scipy, xgboost, statsmodels
    and pandas is the single most common breakage on a fresh machine: pip
    downgrades numpy after Conda has already compiled the stack against the
    numpy 2 ABI.

    Parameters
    ----------
    versions : dict
        Package version mapping produced by `check_packages`.

    Returns
    -------
    ok : bool
        True if numpy is present and below version 2.0.
    """
    print("Dependency constraints")
    numpy_version = versions.get("numpy")
    if numpy_version is None:
        print(f"  [{BAD}] numpy not importable, cannot check the version pin")
        print()
        return False

    major = int(numpy_version.split(".")[0])
    ok = major < 2
    status = OK if ok else BAD
    print(f"  [{status}] numpy {numpy_version} (neuralprophet 0.8.0 requires <2.0)")
    if not ok:
        print(
            "         Fix: recreate the environment from environment.yml, which "
            "pins numpy>=1.25,<2."
        )
    print()
    return ok


def check_training():
    """
    Trains a one-epoch NeuralProphet model on synthetic data.

    This is the decisive check for the torch / pytorch-lightning pairing. The
    Conda solve cannot validate it, because both come from pip and neuralprophet
    caps pytorch-lightning below 2.0 while allowing any torch 2.x.

    Returns
    -------
    ok : bool
        True if the model fitted without raising.
    """
    print("Training smoke test")
    original_cwd = os.getcwd()
    try:
        import numpy as np
        import pandas as pd
        from neuralprophet import NeuralProphet, set_log_level

        set_log_level("ERROR")

        periods = 96
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=periods, freq="h"),
                "y": np.sin(np.linspace(0, 8, periods)),
            }
        )

        # Lightning writes `lightning_logs/` and a learning-rate-finder checkpoint
        # into the working directory. Run the fit inside a temporary directory so
        # the repository is left untouched.
        with tempfile.TemporaryDirectory() as scratch:
            try:
                os.chdir(scratch)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = NeuralProphet(epochs=1, n_lags=4, daily_seasonality=True)
                    model.fit(df, freq="h", progress=None)
            finally:
                # The working directory must be restored before the context
                # manager removes the scratch directory. Windows refuses to
                # remove a directory that is a process's working directory, and
                # TemporaryDirectory's cleanup responds by retrying through a
                # recursive call, which exhausts the stack and kills the
                # interpreter with an access violation rather than an exception.
                os.chdir(original_cwd)

        print(f"  [{OK}] NeuralProphet fitted a 1-epoch model successfully")
        print()
        return True
    except Exception as exc:  # noqa: BLE001 - report whatever the stack raised
        print(f"  [{BAD}] NeuralProphet training failed: {type(exc).__name__}: {exc}")
        print(
            "         This usually means the installed torch is too new for "
            "pytorch-lightning 1.9.x. See the troubleshooting table in README.md."
        )
        print()
        return False
    finally:
        os.chdir(original_cwd)


def check_jupytext_cli():
    """
    Verifies that the `jupytext` executable is on PATH.

    `auto_watcher.py` shells out to this command, so an importable jupytext
    package is not sufficient on its own.

    Returns
    -------
    ok : bool
        True if the executable was found.
    """
    print("Command-line tools")
    path = shutil.which("jupytext")
    ok = path is not None
    status = OK if ok else BAD
    print(f"  [{status}] jupytext executable: {path or 'not found on PATH'}")
    if not ok:
        print(
            "         auto_watcher.py cannot sync without it. Confirm the "
            "neuralprophet_env environment is active."
        )
    print()
    return ok


def check_data_dirs():
    """
    Reports which of the git-ignored data and output directories are missing.

    Missing directories are a warning rather than a failure: a fresh clone is
    expected to lack them until the user creates them and copies the datasets in.

    Returns
    -------
    ok : bool
        Always True; this check never blocks.
    """
    print("Local directory tree (git-ignored, must be created by hand)")
    repo_root = os.path.dirname(os.path.abspath(__file__))
    missing = []

    for relative in EXPECTED_DIRS:
        absolute = os.path.join(repo_root, *relative.split("/"))
        if os.path.isdir(absolute):
            print(f"  [{OK}] {relative}")
        else:
            missing.append(relative)
            print(f"  [{WARN}] {relative} is missing")

    if missing:
        print(
            "         Create the missing directories and copy your datasets in "
            "before running notebook 00. See 'Supplying the Data' in README.md."
        )
    print()
    return True


def main():
    """
    Runs every check and returns a process exit code.

    Returns
    -------
    exit_code : int
        0 if all mandatory checks passed, 1 otherwise.
    """
    print("=" * 72)
    print("neuralprophet_env verification")
    print("=" * 72)
    print()

    results = []
    results.append(("interpreter", check_interpreter()))

    packages_ok, versions = check_packages()
    results.append(("packages", packages_ok))
    results.append(("numpy pin", check_numpy_pin(versions)))

    # Only attempt training if the imports that it needs actually succeeded.
    if packages_ok:
        results.append(("training", check_training()))
    else:
        print("Training smoke test")
        print(f"  [{WARN}] skipped because some packages failed to import")
        print()
        results.append(("training", False))

    results.append(("jupytext cli", check_jupytext_cli()))
    check_data_dirs()

    failed = [name for name, ok in results if not ok]

    print("=" * 72)
    if failed:
        print(f"RESULT: FAILED - {', '.join(failed)}")
        print("Consult the troubleshooting table in README.md.")
        print("=" * 72)
        return 1

    print("RESULT: PASSED - the environment is ready to run the pipeline.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
