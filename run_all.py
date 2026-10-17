# =============================================================================
#  run_all.py -- the whole evaluation of the paper in ONE notebook cell
#  (or: python notebook/run_all.py, from the repository root): both datasets,
#  ten seeds, every stage, on deduplicated data. It runs the program
#  mlnids_robustness_evaluation.py, which must be in the current folder.
#
#  What this cell does, in order:
#   0. checks the folder: the program with the expected MD5, and no data,
#      models or results of an earlier run;
#   1. DATA STEP (minutes): builds data_final/ from the Kaggle files with the
#      deduplication and prints the counts, the residual overlap, the new
#      top-30 feature sets and the PREFLIGHT check of the stage settings.
#      If the preflight fails it STOPS before anything is trained: the
#      stage settings must then be updated to the printed feature sets;
#   2. THE RUN: all 60 parts in parallel, every stage, then the tables
#      (rerun_summary.txt). This is long. If it is interrupted, run this cell
#      again: finished work is skipped;
#   3. rerun_results_dedup.zip (results, figures, tables, data counts) with a
#      download link, and backup_models_data.zip (models_final/, data_final/,
#      per-flow results). COPY THE BACKUP OFF THIS MACHINE.
#
#  Run it in a folder that holds only the program (no data, models or results
#  of an earlier run), from a RESTARTED kernel.
# =============================================================================
import os, sys, glob, time, base64, hashlib, zipfile, subprocess

PROGRAM = "mlnids_robustness_evaluation.py"
PROGRAM_MD5 = "531032fd1a70cd592c49248c32718463"
ZIP_RESULTS = "rerun_results_dedup.zip"
ZIP_BACKUP = "backup_models_data.zip"
DATA_LOG = "run_all_data_dedup.out"


def _md5_text(path):
    with open(path, encoding="utf-8") as fh:
        t = fh.read().replace("\r\n", "\n").rstrip("\n") + "\n"
    return hashlib.md5(t.encode("utf-8")).hexdigest()


def _folder_ok():
    if not os.path.isfile(PROGRAM):
        print(f"STOPPED, nothing was run: {PROGRAM} is not in this folder "
              f"({os.getcwd()}). Copy it here first.")
        return False
    h = _md5_text(PROGRAM)
    if h != PROGRAM_MD5:
        print(f"STOPPED, nothing was run: {PROGRAM} has MD5 {h}, expected "
              f"{PROGRAM_MD5}. Copy the file again (unchanged).")
        return False
    old_data = [p for p in glob.glob(os.path.join("data_final", "*.npz"))
                if not p.endswith(("_meta.npz", "_scaler.npz"))
                and not os.path.isfile(p[:-4] + "_meta.npz")]
    if old_data:
        print("STOPPED, nothing was run: data_final/ holds data built WITHOUT "
              f"deduplication ({', '.join(old_data)}). Use a new, empty folder.")
        return False
    has_meta = bool(glob.glob(os.path.join("data_final", "*_meta.npz")))
    if not has_meta and (os.path.isdir("models_final") or os.path.isdir("results_final")):
        print("STOPPED, nothing was run: this folder holds models or results of "
              "an earlier run. Use a new, empty folder.")
        return False
    return True


def _run(env_extra, log=None):
    env = dict(os.environ, PYTHONUNBUFFERED="1", **env_extra)
    env.pop("RERUN_LAUNCHER_PID", None)
    fh = open(log, "w", encoding="utf-8") if log else None
    p = subprocess.Popen([sys.executable, PROGRAM], env=env, text=True,
                         encoding="utf-8", errors="replace",
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        for line in p.stdout:
            print(line, end="", flush=True)
            if fh:
                fh.write(line)
        return p.wait()
    finally:
        if p.poll() is None:
            p.terminate()
        if fh:
            fh.close()


def _zip(path, files):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for f in files:
            z.write(f)
    return os.path.getsize(path)


def run_dedup():
    now = lambda: time.strftime("%H:%M")
    if not _folder_ok():
        return
    print(f"{PROGRAM}: MD5 {PROGRAM_MD5} (as expected)")

    # ---- 1. data step: deduplication, new feature sets, preflight -------------
    print(f"[{now()}] DATA STEP: building data_final/ with deduplication "
          f"(output also in {DATA_LOG})", flush=True)
    rc = _run({"RERUN_PART": "data", "RERUN_STAGES": ""}, DATA_LOG)
    if rc != 0:
        print(f"\n[{now()}] STOPPED after the data step, before anything was "
              "trained.\nSee the output above (or the file "
              f"{DATA_LOG}): the stage settings must be updated to the new "
              "feature sets.")
        return
    print(f"[{now()}] data step passed (deduplication counts, feature sets and "
          "preflight are printed above)\n", flush=True)

    # ---- 2. the whole run: 60 parts, every stage, then the tables -------------
    print(f"[{now()}] THE RUN: all parts and stages. Long; progress below. If "
          "it is interrupted, run this cell again.", flush=True)
    rc = _run({"RERUN_PART": "", "RERUN_STAGES": ""})
    if rc != 0 or not os.path.isfile("rerun_summary.txt"):
        print(f"\n[{now()}] STOPPED: the run did not finish (see above and the "
              "run_all_<part>.out files). Finished work is saved; run this "
              "cell again after the cause is fixed.")
        return

    # ---- 3. results zip (with link) and the backup of models and data ---------
    res = (["rerun_summary.txt", DATA_LOG]
           + sorted(glob.glob(os.path.join("results_final", "*_final.csv")))
           + sorted(glob.glob(os.path.join("figures_final", "*")))
           + sorted(glob.glob(os.path.join("data_final", "*_meta.npz"))))
    n = _zip(ZIP_RESULTS, [f for f in res if os.path.isfile(f)])
    print(f"[{now()}] {ZIP_RESULTS}: {n:,} bytes", flush=True)
    bak = [f for d in ("models_final", "data_final", os.path.join("results_final", "flows"))
           for f in sorted(glob.glob(os.path.join(d, "**", "*"), recursive=True))
           if os.path.isfile(f) and ".tmp" not in os.path.basename(f)]
    n = _zip(ZIP_BACKUP, bak)
    print(f"[{now()}] {ZIP_BACKUP}: {len(bak):,} files, {n:,} bytes in "
          f"{os.path.abspath(ZIP_BACKUP)}\n   COPY THIS FILE OFF THIS MACHINE "
          "(Google Drive, a USB disk): it is the only copy of the trained "
          "models.", flush=True)
    try:
        from IPython.display import HTML, display
        with open(ZIP_RESULTS, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        display(HTML(f'<a download="{ZIP_RESULTS}" href="data:application/zip;'
                     f'base64,{b64}" style="font-size:22px;font-weight:bold">'
                     f'CLICK HERE to download {ZIP_RESULTS}</a>'))
    except ImportError:
        pass
    print(f"[{now()}] FINISHED")


run_dedup()
