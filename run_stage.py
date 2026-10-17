# run_stage.py -- ONE notebook cell (or: python notebook/run_stage.py STAGE).
# Runs ONE stage of mlnids_robustness_evaluation.py on the SAVED models of a
# finished run (models_final/, data_final/, results_final/flows/ present), for
# all 60 parts, then writes the tables. Nothing is trained that is already
# saved; finished cells are skipped, so re-running after an interruption
# continues. The full rerun_summary.txt is kept; the new one is saved as
# rerun_summary_<STAGE>.txt and zipped with the stage's result rows.
import os, sys, glob, shutil, base64, hashlib, zipfile, subprocess, time

PROGRAM = "mlnids_robustness_evaluation.py"
PROGRAM_MD5 = "531032fd1a70cd592c49248c32718463"
STAGE = "atfgsm"   # the stage to run (names: see README); "python run_stage.py NAME" overrides it
if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
    STAGE = sys.argv[1]
ZIP = f"rerun_results_{STAGE}.zip"

def _md5_text(path):
    with open(path, encoding="utf-8") as fh:
        t = fh.read().replace("\r\n", "\n").rstrip("\n") + "\n"
    return hashlib.md5(t.encode("utf-8")).hexdigest()

def run_stage():
    for need in (PROGRAM, "models_final", "data_final", os.path.join("results_final", "flows")):
        if not os.path.exists(need):
            print(f"STOPPED, nothing was run: {need} is not in {os.getcwd()}."); return
    h = _md5_text(PROGRAM)
    if h != PROGRAM_MD5:
        print(f"STOPPED, nothing was run: {PROGRAM} has MD5 {h}, expected {PROGRAM_MD5}."); return
    if os.path.isfile("rerun_summary.txt") and not os.path.isfile("rerun_summary_full.txt"):
        shutil.copy("rerun_summary.txt", "rerun_summary_full.txt")
    env = dict(os.environ, PYTHONUNBUFFERED="1", RERUN_PART="", RERUN_STAGES=STAGE)
    env.pop("RERUN_LAUNCHER_PID", None)
    t0 = time.time()
    p = subprocess.Popen([sys.executable, PROGRAM], env=env, text=True, encoding="utf-8",
                         errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout:
        print(line, end="", flush=True)
    rc = p.wait()
    if os.path.isfile("rerun_summary.txt"):
        shutil.copy("rerun_summary.txt", f"rerun_summary_{STAGE}.txt")
    if os.path.isfile("rerun_summary_full.txt"):
        shutil.copy("rerun_summary_full.txt", "rerun_summary.txt")
    if rc != 0:
        print(f"\nSTOPPED (see above). Finished cells are saved; run this cell again."); return
    files = [f"rerun_summary_{STAGE}.txt"] + sorted(glob.glob(os.path.join("results_final", f"{STAGE}_*.csv")))
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            if os.path.isfile(f):
                z.write(f)
    print(f"\nFINISHED in {(time.time() - t0) / 60:.0f} min. {ZIP}: "
          f"{os.path.getsize(ZIP):,} bytes in {os.path.abspath(ZIP)}")
    try:
        from IPython.display import HTML, display
        with open(ZIP, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        display(HTML(f'<a download="{ZIP}" href="data:application/zip;base64,{b64}" '
                     f'style="font-size:22px;font-weight:bold">CLICK HERE to download {ZIP}</a>'))
    except ImportError:
        pass

run_stage()
