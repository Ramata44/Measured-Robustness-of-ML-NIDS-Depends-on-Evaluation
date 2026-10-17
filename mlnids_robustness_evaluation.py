# =============================================================================
#  mlnids_robustness_evaluation.py
#  Every model-dependent result of "Measured Robustness of ML-NIDS Depends on
#  Evaluation" from one configuration and one saved set of models.
#
#  - One configuration: the SETTINGS block below defines every setting.
#  - One model set: each model is trained once, saved in models_final/, and
#    only loaded afterwards.
#  - One value per measurement: each (model, capability level, eps, attack) is
#    computed once, with a seed derived from those keys.
#  - One data artefact: the preprocessed, deduplicated data are frozen in
#    data_final/ and every part loads them.
#
#  Run:  python mlnids_robustness_evaluation.py          (all 60 parts in parallel)
#        RERUN_PART=summary python mlnids_robustness_evaluation.py > rerun_summary.txt
#  See README.md for the environment variables and the stages.
# =============================================================================


import os
# RUN ON CPU.
FORCE_CPU = True
if FORCE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
# ONE THREAD PER PROCESS.
N_THREADS = int(os.environ.get("HSJ_THREADS", "1"))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = str(N_THREADS)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "0")

import sys
if os.environ.get("RERUN_LAUNCHER_PID"):
    # This process is one part started by the parallel launcher.
    try:
        import ctypes, signal
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)   # PR_SET_PDEATHSIG
        if os.getppid() != int(os.environ["RERUN_LAUNCHER_PID"]):
            os._exit(1)                         # launcher already gone
    except (OSError, AttributeError, ValueError):
        pass                                    # not Linux: no such safeguard
if FORCE_CPU and "torch" in sys.modules:
    print("!! torch is already imported in this kernel, so the CPU switch cannot "
          "take effect -- RESTART THE KERNEL and run this cell first.\n",
          flush=True)
if "torch" in sys.modules:
    print("!! torch already imported -- RESTART THE KERNEL and run this first.\n",
          flush=True)

# ---- REQUIRED LIBRARIES: install the missing ones before pandas is imported ----
import subprocess
from importlib import metadata as _md
_NEED = ["pyarrow",                          # reads the Kaggle parquet files
         "imbalanced-learn",                 # RandomUnderSampler
         "adversarial-robustness-toolbox",   # PGD, BIM, HopSkipJump
         "kagglehub",                        # downloads the two datasets
         "matplotlib"]                       # the figures (summary step)


def _have(dist):
    try:
        _md.version(dist)
        return True
    except _md.PackageNotFoundError:
        return False


_missing = [p for p in _NEED if not _have(p)]
if _missing:
    print(f"installing missing libraries: {', '.join(_missing)} ...", flush=True)
    _r = subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + _missing,
                        capture_output=True, text=True)
    import importlib
    importlib.invalidate_caches()
    _still = [p for p in _missing if not _have(p)]
    if _r.returncode != 0 or _still:
        _msg = ("could not install " + ", ".join(_still or _missing)
                + ". Install them yourself (pip install "
                + " ".join(_still or _missing) + "), then run this again.")
        print((_r.stdout or "")[-2000:] + (_r.stderr or "")[-2000:], flush=True)
        print("\n" + "!" * 74 + f"\n{_msg}\n" + "!" * 74, flush=True)
        raise RuntimeError(_msg)
    _loaded = [m for m in ("numpy", "pandas", "sklearn", "torch")
               if m in sys.modules]
    if _loaded:
        _msg = ("installed " + ", ".join(_missing) + ". This kernel had already "
                "loaded " + ", ".join(_loaded) + " before they existed, so "
                "RESTART THE KERNEL and run this cell again (nothing is lost).")
        print("\n" + "!" * 74 + f"\n{_msg}\n" + "!" * 74, flush=True)
        raise RuntimeError(_msg)
    print("  installed: " + ", ".join(f"{p} {_md.version(p)}" for p in _missing),
          flush=True)

import glob, json, re, time, warnings, hashlib
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

torch.set_num_threads(N_THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass            # already set in this kernel; harmless
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# The ONE configuration.
SEED          = 42                      # data split, undersampling, pool
SEEDS         = [1, 2, 3, 4, 5, 6, 7, 8, 9, 42]   # model seeds: 1, 7, 42 of the
DATASETS      = ["unsw", "cic"]
ACTIVATIONS   = ["relu", "elu", "tanh"]
ARCHS         = {1: [64], 2: [64, 128], 3: [64, 128, 128],
                 4: [64, 128, 512, 128], 5: [64, 128, 512, 128, 64]}
ACTS          = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}
DEPTHS        = [1, 2, 3, 4, 5]
FEATS         = 30                      # every experiment except the k sweep
K_SWEEP       = {"cic": [10, 20, 50, 78], "unsw": [10, 20, 50, 100, 188]}
MAX_EPOCHS, CHUNK, PATIENCE, TOL = 120, 10, 2, 1e-3
BATCH_SIZE, LR = 16384, 1e-3
VAL_FRACTION  = 0.1                     # split with random_state=SEED
SAMPLE_PER_FILE = 300000                # CIC rows read per file
SLUG = {"cic": "dhoogla/distrinetcsecicids2018", "unsw": "dhoogla/unswnb15"}
N_EVAL_BY_DS  = {"cic": 2000, "unsw": 500}   # fixed malicious pool
EPS_LIST      = [0.1, 0.3, 0.5]
LEVELS        = [0, 1, 2]
# Free-feature counts at L1/L2 of the top-30 set.
EXPECTED_FREE = {}
DEDUP = True
_DATA_META = {}           # filled by load(), saved by get_data()
_PREFLIGHT_DONE = set()
FGSM_EPS_STEP_DIV, PGD_BIM_MAX_ITER, PGD_RANDOM_INIT = 10, 40, 1
SPSA_STEPS, SPSA_NDIRS, SPSA_C = 20, 8, 0.01
HSJ_MAX_ITER, HSJ_MAX_EVAL, HSJ_INIT_EVAL = 50, 1000, 100
HSJ_VERBOSE   = False
SUBSTITUTE    = dict(depth=2, act="tanh", seed_offset=1000)
ATTACK_TIMEOUT = 24 * 3600              # a timeout STOPS the part
SD_DDOF       = 1                       # sample SD over the seeds
DEVICE        = "cpu"
MATCH_PUBLISHED_SPLIT = True
CW_MAX_SAMPLES = 500                    # (unused here)
ATTACK_SEED_BASE = 50000
MODEL_DIR     = "models_final"
DATA_DIR      = "data_final"
RES_DIR       = "results_final"
CACHE_DIR, DEPTH = MODEL_DIR, 4         # names the shared helpers expect
OLD_TABLE5_F1 = {}
RERUN_STAGES = os.environ.get("RERUN_STAGES", "").strip()
# ---- settings of the later stages (each between its own markers) ----
# ======== stage wbbb: settings ========
WB_SPSA_M_STEPS  = 100                  # ascent steps of the matched run
WB_SPSA_EPS      = [0.3]
WB_SPSA_LOSSES   = ["prob", "logit"]    # "prob" = the base SPSA's loss
# (2) Unbounded-budget check.
WB_UNB_EPS       = [1.0, 3.0, 10.0, 100.0]
WB_UNB_STEP_MULT = 2.5
# (3) Convergence of PGD in iterations.
WB_ITER_LIST     = [1, 5, 10, 20, 40, 100, 400]
WB_ITER_LEVELS   = [0, 2]
WB_ITER_EPS      = [0.3]
WB_FIG_EPS       = 0.3
WB_FIG_DIR       = "figures_final"
WB_FIG_BARS      = ["pgd", "bim", "hsj_inf", "spsa_matched"]
WB_FIG_MATCHED_CELL = ("matched", "prob")   # (budget, loss) of the wbspsa cell
WB_FIG_STYLE     = {   # bar -> (legend label, colour, hatch); colours as published
    "pgd":          ("PGD (WB)",           (0.231, 0.435, 0.690), ""),
    "bim":          ("BIM (WB)",           (0.435, 0.639, 0.847), ""),
    "hsj_inf":      ("HSJ (BB)",           (0.753, 0.314, 0.302), ""),
    "spsa":         ("SPSA, 320 queries (BB)", (0.886, 0.569, 0.373), "////"),
    "spsa_matched": ("SPSA (BB)",          (0.886, 0.569, 0.373), "")}
WB_FIG_SIZE      = (10.5, 4.2)          # inches, as the published figure
WB_FIG_GROUP_W   = 0.8                  # width of one activation group of bars
WB_FIG_DPI       = 1600
# ======== stage restarts: settings ========
RS_EPS_LIST    = [0.1, 0.3, 0.5]
RS_RESTARTS    = [1, 5, 20, 50, 100]    # restart budgets; the table prints the
RS_APGD_MAX_ITER, RS_APGD_STEP_DIV, RS_APGD_INIT = 100, 4, 1   # APGD-CE, one
RS_PROBE_ITERS    = 400                 # "iterations do not substitute for
RS_PROBE_RESTARTS = 20                  #  restarts": 400 x 1 against 40 x 20
# ======== stage capa: settings ========
CA_ATTACKS       = ["pgd", "bim", "spsa", "transfer"]   # as the base L1/L2 rows
CA_FIVE_DROP     = ["Fwd RST Flags"]   # deduplicated data: Fwd PSH Flags left
CA_FIVE_FREE     = 5                    # the FIVE set must have exactly this many
# Exact L2 worst case by enumeration.
CA_ENUM_GRID     = 9
CA_ENUM_MAX_FREE = 4
CA_ENUM_CHUNK    = 64                   # flows per forward batch (speed only)
CA_RUN_SWEEP     = True
CA_SWEEP_EPS     = [1.0, 2.0, 3.0, 5.0, 10.0]
CA_UNCLIPPED_EPS = [1.0, 3.0, 10.0, 30.0, 100.0]        # no input clip
CA_UNCLIPPED_ATTACKS = ["pgd", "bim"]
CA_SAT_LEVEL     = 0.9                  # "L2 worst case reaches ..." report line
CA_RAWUNITS_EPS  = 0.3
CA_LOCK_POLL_S   = 5                    # wait while another part builds the scaler
CA_LOCK_STALE_S  = 60                   # an empty lock older than this is dropped
CA_UNITS = {"Init Fwd Win Bytes": "bytes (TCP window)",
            "Fwd Seg Size Min": "bytes", "Fwd Packet Length Min": "bytes",
            "Fwd IAT Total": "$\\mu$s", "Fwd IAT Max": "$\\mu$s",
            "Fwd PSH Flags": "count", "Fwd RST Flags": "count",
            "swin": "TCP window value", "stcpb": "TCP sequence number",
            "sinpkt": "ms", "smean": "bytes (mean packet size)",
            "trans_depth": "count (HTTP transactions)"}
# ======== stage capb: settings ========
CB_ATTACKS    = ["pgd", "bim", "spsa", "transfer"]  # the four natively-
CB_TABLE_EPS  = 0.3
CB_ACT_LEVELS = [0, 2]
CB_SE_ASR     = 0.5       # ASR at which the pool's binomial standard error is
CB_TIE_TOL    = 1e-9      # floating-point guard only: two means closer than
# ======== stage cnn: settings ========
CNN_GRID        = (5, 6)        # the FEATS = 30 features, ANOVA-rank order, row-major
CNN_FILTERS     = (8, 16)       # filters of the two convolutions
CNN_KERNEL      = 2             # 2 x 2 convolutions, stride 1, "same" padding
CNN_POOL_KERNEL, CNN_POOL_STRIDE = 2, 1      # max-pool after each convolution
CNN_DENSE       = 64            # width of the dense layer
CNN_P_DROP      = 0.3           # dropout after the second pool and after the dense layer
CNN_LEVELS      = LEVELS        # attacked at every capability level, both datasets
CNN_ATTACKS     = ["pgd", "bim", "spsa", "transfer"]
CNN_TABLE_LEVELS = (0, 2)
CNN_ROLE        = "cnn"         # role name of the saved models in models_final/
# ======== stage advtrain: settings ========
AT_ROLE        = "pgd_at"   # models_final/<ds>_d4_f30_<act>_s<seed>_pgd_at.pt
AT_EPS         = 0.3
AT_STEPS       = 7
AT_RATIO       = 0.5
AT_STEP_FACTOR = 2.5        # inner step size = AT_EPS / AT_STEPS * 2.5
AT_CLAMP       = 1e6        # the inner PGD clamps to +-1e6 only (no data box)
AT_LEVELS      = [0, 2]
AT_ATTACKS     = ["pgd", "bim", "spsa", "transfer"]
AT_HSJ         = ("inf", np.inf)   # decision-based check of the defended models
AT_TPR_TARGETS = [0.90, 0.95, 0.99]   # FPR at these fixed TPRs is recorded
AT_TPR_TABLE   = 0.95
AT_THRESHOLD   = 0.5        # default decision threshold on P(malicious)
AT_FIG_DIR     = "figures_final"
AT_FIG_FILE    = "fig2.pdf"
AT_FIG_DPI, AT_FIG_SIZE = 1600, (10.0, 4.0)
AT_FIG_SERIES  = {"standard": ("Standard", (0.231, 0.435, 0.690)),
                  AT_ROLE:    ("PGD-AT",   (0.753, 0.314, 0.302))}
AT_FIG_LEVELS  = [(2, "-", "L2 source-side"), (0, "--", "L0 unconstrained")]
# ======== stage xai: settings ========
XAI_N_BASELINES  = 64
XAI_N_ALPHA      = 32
XAI_TARGET_CLASS = 1
XAI_ABS_PER_SAMPLE = False
XAI_LEVELS       = [0, 2]
XAI_TAB_EPS      = [0.1, 0.3]
XAI_TOPK_FIG     = 8
XAI_TOPK_TEXT    = 5
XAI_DIFF_NOTED   = 0.01
XAI_UNSW_PAIR    = ["rate", "sload"]
XAI_FIG_DIR      = "figures_final"
XAI_FIG_SIZE     = (11.0, 4.2)      # one row of fig3.pdf: two panels side by side
XAI_FIG_FILE     = "fig3.pdf"       # Figure 3: ONE file, a row per ranking activation
XAI_FIG_DPI      = 1600
XAI_FIG_COLORS   = {"relu": "#3b6fb0", "elu": "#c0504d", "tanh": "#e2915f"}
XAI_FIG_ABBREV   = [(" Packet Length ", " PktLen "), ("Seg Size", "Seg Sz"),
                    ("Flag Count", "Flag Cnt"), ("Win Bytes", "Win B"),
                    ("_ltm", "")]   # tick labels only, as in the published figure
XAI_DS_LABEL     = {"cic": "CSE-CIC-IDS2018", "unsw": "UNSW-NB15"}
# ======== stage caat: settings ========
CAAT_ROLE       = "ca_at"   # models_final/<ds>_d4_f30_<act>_s<seed>_ca_at.pt
CAAT_MASK_LEVEL = 2         # inner PGD confined to the L2 (source-side) mask
# ======== stage capu: settings ========
CU_DATASET  = "unsw"
CU_TCP_ONLY = ["swin", "stcpb", "trans_depth"]   # source-side fields that
CU_TCP_FLAG = "proto_tcp"         # one-hot protocol indicator among the 30
# ======== stage fgsm: settings ========
FG_EPS_STEP_FACTOR = 1.0              # step = FG_EPS_STEP_FACTOR x eps
FG_REF_CFG   = ("cic", 1, 30, "relu") # undefended configuration of its Fig. 16
FG_REF_LEVEL = 0.3                    # level it reports there at eps = 0.5
# ======== stage capd: settings ========
CD_ADD_ONLY = {"cic": ["Fwd IAT Total", "Fwd IAT Max", "Fwd Packet Length Min",
                       "Fwd Seg Size Min"],
               "unsw": ["smean"]}   # deduplicated data: sinpkt left the top-30;
CD_VARIANTS = ["sym", "addonly", "up", "down"]
CD_TOL      = 1e-5      # a displacement beyond the box by more than this is counted
# ======== stage rvstd1: settings ========
RV1_ROLE = "std_1adam"   # models_final/<ds>_d4_f30_<act>_s<seed>_std_1adam.pt
# ======== stage rvspsam: settings ========
RV2_LOSS = "margin"
# ======== stage rvthr: settings ========
RV3_DEFENCES = ["standard", AT_ROLE, CAAT_ROLE]
RV3_FPR      = {"cic": [0.001, 0.01], "unsw": [0.05, 0.10]}
RV3_TUNE     = ["val", "test"]
# ======== stage rvstrong: settings ========
RV4_DEFENCES = ["standard", AT_ROLE, CAAT_ROLE]
RV4_RESTARTS = 100
RV4_REPORT   = [1, 10, 100]    # unions over the first R starts that are written
# ======== stage rvhsjc2: settings ========
RV5_DEFENCES = ["standard", AT_ROLE, CAAT_ROLE]
RV5_LEVEL    = 2
# ---- later-stage settings go above this line ----
# ---- EDIT HERE WHEN RUNNING IN A NOTEBOOK CELL ----
NOTEBOOK_PART = ""   # "data" | "unsw-relu-1" ... | "summary" | "" = all parts
# SPEED ONLY -- changes no number.
PARALLEL = int(os.environ.get("RERUN_PARALLEL", "0"))
HEARTBEAT = 900                         # seconds between progress reports
RERUN_PART = (os.environ.get("RERUN_PART", "") or NOTEBOOK_PART).strip()
PARTS = [(d, a, s) for d in DATASETS for a in ACTIVATIONS for s in SEEDS]
if RERUN_PART not in ("", "data", "summary"):
    _d, _a, _s = RERUN_PART.split("-")
    PARTS = [(_d, _a, int(_s))]
    assert _d in DATASETS and _a in ACTIVATIONS and int(_s) in SEEDS, RERUN_PART
OUT = os.path.join(RES_DIR, f"attacks_{RERUN_PART or 'all'}.csv")
OUT_FINAL = os.path.join(RES_DIR, "attacks_final.csv")

def _stop(msg):
    """Abort loudly from inside a notebook cell."""
    print("\n" + "!" * 74 + f"\n{msg}\n" + "!" * 74, flush=True)
    raise RuntimeError(msg)


UNSW_LABELS = ["label", "Label", "attack_cat"]
UNSW_DROP = {"id", "attack_cat", "label", "srcip", "dstip", "sport", "dsport",
             "stime", "ltime", "src ip", "dst ip"}

UNSW_CATEGORICAL = (r"^(proto|service|state)_", r"^is_sm_ips_ports$",
                    r"^is_ftp_login$")
UNSW_DEST_SIDE   = (r"^d(bytes|pkts|load|loss|mean|ttl|tcpb|win|jit|inpkt)$",
                    r"^res_bdy_len$", r"^dur$")
UNSW_CT_AGG      = (r"^ct_",)
UNSW_DERIVED     = (
    r"^rate$", r"^sload$", r"^tcprtt$", r"^synack$", r"^ackdat$",
)
CIC_CATEGORICAL  = (r"Flags$", r"^Protocol$")
CIC_SENDER_SET   = (r"^Fwd PSH Flags$", r"^Fwd URG Flags$",
                    r"^Fwd RST Flags$")
CIC_DEST_SIDE    = (r"^Bwd ", r"^Init Bwd Win Bytes$")
CIC_DERIVED      = (r"Flag Count$",
                    r"^Flow ", r"^Idle ", r"^Active ", r"^Packet Length ",
                    r"^Average Packet Size$", r"^Down/Up Ratio$")


def make_deterministic(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def env_report():
    return {"torch": torch.__version__, "cuda": (torch.version.cuda or "cpu"),
            "device": DEVICE,
            "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available()
                    else "cpu"),
            "numpy": np.__version__,
            "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "<unset>"),
            "python": sys.version.split()[0], "pandas": pd.__version__,
            "sklearn": _md.version("scikit-learn"),
            "art": _md.version("adversarial-robustness-toolbox"),
            "imblearn": _md.version("imbalanced-learn"),
            "pyarrow": _md.version("pyarrow")}


def _any(p, n):
    return any(re.search(x, n) for x in p)


def build_mask(names, ds, level):
    if level == 0:
        return np.ones(len(names), dtype=np.float32)
    cat = UNSW_CATEGORICAL if ds == "unsw" else CIC_CATEGORICAL
    dst = UNSW_DEST_SIDE   if ds == "unsw" else CIC_DEST_SIDE
    der = (UNSW_CT_AGG + UNSW_DERIVED) if ds == "unsw" else CIC_DERIVED
    snd = () if ds == "unsw" else CIC_SENDER_SET
    out = []
    for n in names:
        n = str(n)
        if _any(snd, n):
            out.append(1.0); continue
        is_dst = _any(dst, n)
        is_cat = _any(cat, n) and not is_dst
        frozen = (is_cat if level == 1
                  else (is_cat or is_dst or _any(der, n)))
        out.append(0.0 if frozen else 1.0)
    return np.array(out, dtype=np.float32)

def _kaggle(ds):
    # Only the Kaggle dataset itself.
    slug = SLUG[ds].split("/")[1].lower()
    bases = [os.path.expanduser("~/.cache/kagglehub")]
    if os.environ.get("NIDS_DATA_DIR"):
        bases.append(os.environ["NIDS_DATA_DIR"])
    for base in bases:
        for ext in ("*.parquet", "*.csv"):
            hits = [h for h in glob.glob(os.path.join(base, "**", ext),
                                         recursive=True)
                    if slug in h.lower().replace("-", "").replace("_", "")]
            if hits:
                print(f"  data: {len(hits)} file(s) under {base}", flush=True)
                return sorted(hits)
    print(f"  downloading {SLUG[ds]} ...", flush=True)
    import kagglehub
    p = kagglehub.dataset_download(SLUG[ds])
    hits = (glob.glob(os.path.join(p, "**", "*.parquet"), recursive=True)
            or glob.glob(os.path.join(p, "**", "*.csv"), recursive=True))
    if not hits:
        _stop(f"no data under {p}")
    return sorted(hits)


def _read(paths, sample=None):
    fr = []
    for p in paths:
        df = (pd.read_parquet(p) if p.endswith(".parquet")
              else pd.read_csv(p, low_memory=False))
        df.columns = [str(c).strip() for c in df.columns]
        if sample and len(df) > sample:
            df = df.sample(sample, random_state=SEED)
        fr.append(df)
    common = set(fr[0].columns)
    for f in fr[1:]:
        common &= set(f.columns)
    common = [c for c in fr[0].columns if c in common]
    return pd.concat([f[common] for f in fr], ignore_index=True)


def _proto_cic(v):
    try:
        v = int(round(float(v)))
    except (TypeError, ValueError):
        return "other"
    return {6: "TCP", 17: "UDP"}.get(v, f"other({v})")


def _dedup(X, y, fam, proto, ds):
    A = np.ascontiguousarray(X.to_numpy(np.float32) + np.float32(0.0))
    v = A.view(np.dtype((np.void, A.dtype.itemsize * A.shape[1]))).ravel()
    _, first, inv = np.unique(v, return_index=True, return_inverse=True)
    inv = np.asarray(inv).ravel()
    yb = np.asarray(y) != 0
    ng = len(first)
    has0, has1 = np.zeros(ng, bool), np.zeros(ng, bool)
    has0[inv[~yb]] = True
    has1[inv[yb]] = True
    conflict = has0 & has1
    is_first = np.zeros(len(inv), bool)
    is_first[first] = True
    keep = is_first & ~conflict[inv]
    nfam = (pd.DataFrame({"g": inv, "f": np.asarray(fam)}).groupby("g").f
            .nunique().reindex(range(ng), fill_value=1).to_numpy())
    counts = dict(
        n_before=len(inv), n_after=int(keep.sum()),
        n_mal_before=int(yb.sum()), n_mal_after=int(yb[keep].sum()),
        n_dup_removed=int((~is_first & ~conflict[inv]).sum()),
        n_conflict_rows=int(conflict[inv].sum()),
        n_conflict_vectors=int(conflict.sum()),
        n_family_ambiguous=int(((nfam > 1) & ~conflict).sum()))
    print(f"  {ds}: deduplication: {counts['n_before']:,} rows -> "
          f"{counts['n_after']:,} ({counts['n_dup_removed']:,} exact duplicates"
          f" removed; {counts['n_conflict_rows']:,} rows of "
          f"{counts['n_conflict_vectors']:,} feature vectors that occur with "
          f"both labels removed); malicious {counts['n_mal_before']:,} -> "
          f"{counts['n_mal_after']:,}", flush=True)
    idx = np.flatnonzero(keep)
    return (X.iloc[idx].reset_index(drop=True), np.asarray(y)[idx],
            np.asarray(fam)[idx], np.asarray(proto)[idx], counts)


def _flag_identity(X, ds):
    names, eq, ge = [], [], []
    cols = set(X.columns)
    for fl in ("FIN", "SYN", "RST", "PSH", "ACK", "URG", "CWE", "ECE"):
        f, b, t = f"Fwd {fl} Flags", f"Bwd {fl} Flags", f"{fl} Flag Count"
        if f in cols and t in cols:
            fw = X[f].to_numpy(np.float64)
            tt = X[t].to_numpy(np.float64)
            bw = X[b].to_numpy(np.float64) if b in cols else np.zeros(len(X))
            names.append(fl)
            eq.append(float(np.mean(tt == fw + bw)))
            ge.append(float(np.mean(tt >= fw)))
    if names:
        print(f"  {ds}: flag-count identity in the raw data: " + ", ".join(
            f"{n} total = fwd + bwd on {e:.2%} of rows (total >= fwd {g:.2%})"
            for n, e, g in zip(names, eq, ge)), flush=True)
    return names, eq, ge


def load(ds):
    print(f"  loading {ds} ...", flush=True)
    paths = _kaggle(ds)
    if ds == "cic":
        df = _read(paths, sample=SAMPLE_PER_FILE)
        lab = next((c for c in df.columns if c.strip().lower() == "label"),
                   df.columns[-1])
        yr = df[lab].astype(str).str.strip()
        X = (df.drop(columns=[lab]).apply(pd.to_numeric, errors="coerce")
             .astype("float32"))
        X = X.replace([np.inf, -np.inf], np.nan)
        keep = X.notna().all(axis=1) & (yr.str.lower() != "label")
        X, yr = X[keep], yr[keep]
        y = (~yr.str.lower().isin({"benign", "normal"})).astype(np.int64).values
        fam = yr.to_numpy(dtype=str)
        pcol = next((c for c in X.columns if c.strip().lower() == "protocol"),
                    None)
        proto = (np.array([_proto_cic(v) for v in X[pcol].to_numpy()])
                 if pcol is not None else np.full(len(X), "unknown"))
    else:
        df = _read(paths)
        lab = next((c for c in UNSW_LABELS if c in df.columns), None)
        if lab is None:
            _stop(f"no label column among {UNSW_LABELS}")
        if lab == "label" or df[lab].dropna().isin([0, 1]).all():
            y = df[lab].astype(np.int64).values
        else:
            y = (~df[lab].astype(str).str.strip().str.lower()
                 .isin({"normal", "benign"})).astype(np.int64).values
        fcol = next((c for c in df.columns if c.strip().lower() == "attack_cat"),
                    None)
        if fcol is not None:
            fam_all = (df[fcol].astype(str).str.strip()
                       .replace({"": "Normal", "nan": "Normal", "None": "Normal"}))
        else:
            fam_all = pd.Series(np.where(y != 0, "malicious", "Normal"),
                                index=df.index)
        pcol = next((c for c in df.columns if c.strip().lower() == "proto"), None)
        proto_all = (df[pcol].astype(str).str.strip().str.lower()
                     if pcol is not None else pd.Series("unknown", index=df.index))
        X = df.drop(columns=[c for c in df.columns
                             if c.strip().lower() in UNSW_DROP])
        cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat:
            X = pd.get_dummies(X, columns=cat, dtype=np.float32)
        X = X.apply(pd.to_numeric, errors="coerce").astype("float32")
        X = X.replace([np.inf, -np.inf], np.nan)
        keep = X.notna().all(axis=1)
        X, y = X[keep], y[keep]
        fam = fam_all[keep].to_numpy(dtype=str)
        proto = proto_all[keep].to_numpy(dtype=str)
    if DEDUP:
        X, y, fam, proto, dd = _dedup(X, y, fam, proto, ds)
    else:
        dd = {}
    fid = _flag_identity(X, ds)
    names = X.columns.tolist()
    X = X.to_numpy(np.float32)
    Xtr, Xte, ytr, yte, ftr, fte, ptr, pte = train_test_split(
        X, y, fam, proto, test_size=0.2, random_state=SEED, stratify=y)
    Xtr_raw, Xte_raw = Xtr, Xte
    sc = StandardScaler().fit(Xtr)
    Xtr = sc.transform(Xtr).astype(np.float32)
    Xte = sc.transform(Xte).astype(np.float32)
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler(random_state=SEED)
    Xtr, ytr = rus.fit_resample(Xtr, ytr)
    scores = np.nan_to_num(SelectKBest(f_classif, k="all").fit(Xtr, ytr).scores_)
    order = np.argsort(scores)[::-1]
    tr_raw = Xtr_raw[rus.sample_indices_]
    ov = {}
    for tag, cols in (("all", slice(None)), ("top30", order[:FEATS])):
        seen = {r.tobytes() for r in
                np.ascontiguousarray(tr_raw[:, cols] + np.float32(0.0))}
        ov[tag] = int(sum(r.tobytes() in seen for r in
                          np.ascontiguousarray(Xte_raw[:, cols] + np.float32(0.0))))
    print(f"  {ds}: test flows identical to a training flow: {ov['all']:,} "
          f"on all features, {ov['top30']:,} on the top-{FEATS} "
          f"(of {len(yte):,})", flush=True)
    _DATA_META[ds] = dict(fam_te=np.asarray(fte, dtype=str),
                          proto_te=np.asarray(pte, dtype=str),
                          overlap_all=ov["all"], overlap_top30=ov["top30"],
                          n_test=len(yte), n_test_mal=int((yte != 0).sum()),
                          n_train=len(ytr),
                          flagid_names=np.asarray(fid[0], dtype=str),
                          flagid_eq=np.asarray(fid[1], dtype=np.float64),
                          flagid_ge=np.asarray(fid[2], dtype=np.float64), **dd)
    print(f"  {ds}: train={Xtr.shape} test={Xte.shape}", flush=True)
    return Xtr, ytr, Xte, yte, order, names

class _F32(nn.Module):
    def __init__(self, m):
        super().__init__(); self.model = m
    def forward(self, x):
        return self.model(x.float())


def build_model(depth, n_features, activation):
    layers, prev = [], n_features
    for h in ARCHS[depth]:
        layers += [nn.Linear(prev, h), ACTS[activation]()]
        prev = h
    layers += [nn.Linear(prev, 2)]
    return nn.Sequential(*layers)


def train_clean(model, Xtr, ytr, device, epochs):
    """Global-RNG randperm, fresh Adam per call."""
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    n = len(Xt)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            F.cross_entropy(model(Xt[idx].to(device)),
                            yt[idx].to(device)).backward()
            opt.step()


@torch.no_grad()
def _probs(model, X, device, bs=16384):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i + bs], dtype=torch.float32, device=device)
        out.append(F.softmax(model(xb), 1).cpu().numpy())
    return np.concatenate(out, 0)


@torch.no_grad()
def macro_f1(model, X, y, device):
    return f1_score(y, _probs(model, X, device).argmax(1),
                    average="macro", zero_division=0)


def train_to_convergence(model, Xtr, ytr, Xval, yval, device):
    """Blocks of CHUNK epochs, patience on validation macro-F1, LAST model kept."""
    best, stale, done = -1.0, 0, 0
    while done < MAX_EPOCHS:
        train_clean(model, Xtr, ytr, device, CHUNK)
        done += CHUNK
        f1 = macro_f1(model, Xval, yval, device)
        if f1 > best + TOL:
            best, stale = f1, 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    return best, done


def _art_clf(model, X, device):
    from art.estimators.classification import PyTorchClassifier
    return PyTorchClassifier(
        model=_F32(model).to(device), loss=nn.CrossEntropyLoss(),
        input_shape=(X.shape[1],), nb_classes=2,
        clip_values=(float(X.min()), float(X.max())),
        device_type=("gpu" if str(device).startswith("cuda") else "cpu"))


def _ce(model, X, y, device):
    p = np.clip(_probs(model, X, device), 1e-8, 1.0)
    return -np.log(p[np.arange(len(y)), y])


def _asr(model, adv, X, y, device, eps=None):
    if eps is not None:
        adv = (X + np.clip(adv - X, -eps, eps)).astype(np.float32)
    correct = _probs(model, X, device).argmax(1) == y
    flip = correct & (_probs(model, adv, device).argmax(1) != y)
    return float(flip.sum() / max(correct.sum(), 1))

def hits_of(model, adv, X, y, device, eps, mask=None):
    """Per-flow success SET, not a rate -- this is what makes the union possible."""
    delta = adv.astype(np.float32) - X.astype(np.float32)
    viol = 0
    if mask is not None:
        frozen = (np.asarray(mask).reshape(-1) == 0)
        if frozen.any():
            viol = int((np.abs(delta[:, frozen]) > 1e-6).sum())
    advp = (X + np.clip(delta, -eps, eps)).astype(np.float32)
    correct = _probs(model, X, device).argmax(1) == y
    flipped = correct & (_probs(model, advp, device).argmax(1) != y)
    return correct, flipped, viol

def _apply(X, adv, mask):
    """Confine a finished perturbation to the perturbable features."""
    if mask is None:
        return adv
    return (X + (adv - X) * mask).astype(np.float32)


def _kw(cls, **kw):
    """Keep only the kwargs this ART version's constructor accepts -- ART's signatures drift between releases."""
    import inspect
    ok = set(inspect.signature(cls.__init__).parameters)
    return {k: v for k, v in kw.items() if k in ok}


def _ref(clf, X, eps, mask, which):
    from art.attacks.evasion import BasicIterativeMethod, ProjectedGradientDescent
    kw = dict(eps=eps, eps_step=eps / FGSM_EPS_STEP_DIV,
              max_iter=PGD_BIM_MAX_ITER, batch_size=BATCH_SIZE,
              verbose=False)      # no progress bars: they flood notebook output
    atk = (ProjectedGradientDescent(clf, num_random_init=PGD_RANDOM_INIT, **kw)
           if which == "pgd" else BasicIterativeMethod(clf, **kw))
    gk = {} if mask is None else {"mask": np.broadcast_to(
        mask, X.shape).astype(np.float32).copy()}
    return atk.generate(x=X.astype(np.float32), **gk)


def a_spsa(model, X, y, eps, seed, mask):
    rng = np.random.default_rng(seed)
    x0 = np.ascontiguousarray(X, np.float32); x = x0.copy()
    m = 1.0 if mask is None else mask
    alpha = eps / SPSA_STEPS * 2.5
    for _ in range(SPSA_STEPS):
        g = np.zeros_like(x)
        for _ in range(SPSA_NDIRS):
            d = rng.choice([-1.0, 1.0], size=x.shape).astype(np.float32) * m
            g += ((_ce(model, x + SPSA_C * d, y, DEVICE) -
                   _ce(model, x - SPSA_C * d, y, DEVICE))
                  / (2 * SPSA_C))[:, None] * d
        x = x + alpha * np.sign(g) * m
        x = np.clip(x, x0 - eps, x0 + eps)
    return x.astype(np.float32)


def a_apgd(clf, X, eps, mask):
    from art.attacks.evasion import AutoProjectedGradientDescent
    atk = AutoProjectedGradientDescent(**_kw(
        AutoProjectedGradientDescent, estimator=clf, norm=np.inf, eps=eps,
        eps_step=eps / 4.0, max_iter=100, nb_random_init=5,
        loss_type="cross_entropy", targeted=False, batch_size=512, verbose=False))
    return _apply(X, atk.generate(x=X.astype(np.float32)), mask)


def a_deepfool(clf, X, eps, mask):
    """DeepFool, minimum-norm, projected into the eps-ball -> LOWER BOUND."""
    from art.attacks.evasion import DeepFool
    atk = DeepFool(**_kw(DeepFool, classifier=clf, estimator=clf, max_iter=100,
                         nb_grads=2, batch_size=512, verbose=False))
    return _apply(X, atk.generate(x=X.astype(np.float32)), mask)


class _Timeout(Exception):
    pass


def _run_with_timeout(fn, seconds):
    import signal
    def _h(signum, frame):
        raise _Timeout()
    old = signal.signal(signal.SIGALRM, _h); signal.alarm(seconds)
    try:
        return fn()
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old)


def a_cw(clf, X, eps, mask):
    """Carlini-Wagner L_inf, capped pool."""
    from art.attacks.evasion import CarliniLInfMethod
    n = min(CW_MAX_SAMPLES, len(X))
    Xs = X[:n].astype(np.float32)
    adv = CarliniLInfMethod(**_kw(
        CarliniLInfMethod, classifier=clf, estimator=clf, eps=eps, max_iter=10,
        batch_size=512, verbose=False)).generate(x=Xs)
    return _apply(Xs, adv, mask), np.arange(n)


def a_hsj_bounded(clf, X, eps, mask):
    """HopSkipJump seeded with an eps-bounded PGD example."""
    from art.attacks.evasion import HopSkipJump, ProjectedGradientDescent
    pgd = ProjectedGradientDescent(clf, eps=eps, eps_step=eps / FGSM_EPS_STEP_DIV,
                                   max_iter=PGD_BIM_MAX_ITER,
                                   num_random_init=1, batch_size=BATCH_SIZE)
    kw = {} if mask is None else {"mask": np.broadcast_to(
        mask, X.shape).astype(np.float32).copy()}
    init = pgd.generate(x=X.astype(np.float32), **kw)
    atk = HopSkipJump(classifier=clf, targeted=False, norm=np.inf,
                      max_iter=HSJ_MAX_ITER, max_eval=HSJ_MAX_EVAL,
                      init_eval=HSJ_INIT_EVAL, batch_size=128,
                      verbose=HSJ_VERBOSE)
    assert atk.norm == np.inf, f"HopSkipJump norm is {atk.norm}, not inf"
    adv = atk.generate(x=X.astype(np.float32),
                       x_adv_init=init.astype(np.float32), **kw)
    adv = np.where(np.isfinite(adv).all(1, keepdims=True), adv, init)
    adv = _apply(X, adv, mask)
    adv = (X + np.clip(adv - X, -eps, eps)).astype(np.float32)
    # Per-flow best of {projected HSJ, its own PGD seed}.
    init_b = (X + np.clip(_apply(X, init, mask) - X, -eps, eps)).astype(np.float32)
    try:
        ev_adv = clf.predict(adv).argmax(1) == 0
        ev_ini = clf.predict(init_b).argmax(1) == 0
        take = (~ev_adv) & ev_ini
        if take.any():
            adv = adv.copy()
            adv[take] = init_b[take]
    except Exception:
        pass
    return adv

_SUB_CACHE = {}


def a_transfer(Xtrn, ytrn, Xval, yval, k, X, eps, seed, mask):
    """Mismatched surrogate (depth 2, tanh, seed+1000)."""
    if seed not in _SUB_CACHE:
        s = seed + SUBSTITUTE["seed_offset"]
        torch.manual_seed(s); np.random.seed(s); torch.cuda.manual_seed_all(s)
        sub = build_model(SUBSTITUTE["depth"], k, SUBSTITUTE["act"]).to(DEVICE)
        train_to_convergence(sub, Xtrn, ytrn, Xval, yval, DEVICE)
        sub.eval()
        _SUB_CACHE[seed] = (sub, _art_clf(sub, Xtrn, DEVICE))
    sub, sub_clf = _SUB_CACHE[seed]
    from art.attacks.evasion import ProjectedGradientDescent
    atk = ProjectedGradientDescent(sub_clf, eps=eps,
                                   eps_step=eps / FGSM_EPS_STEP_DIV,
                                   max_iter=PGD_BIM_MAX_ITER, num_random_init=1,
                                   batch_size=BATCH_SIZE)
    kw = {} if mask is None else {"mask": np.broadcast_to(
        mask, X.shape).astype(np.float32).copy()}
    return atk.generate(x=X.astype(np.float32), **kw)


# ---- model cache ----
def _cache_path(ds, depth, feats, act, seed, defence):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR,
                        f"{ds}_d{depth}_f{feats}_{act}_s{seed}_{defence}.pt")


def train_cached(ds, act, seed, k, Xtrn, ytrn, Xval, yval, Xte, yte,
                 defence="standard", device=None):
    """Train once, reuse thereafter."""
    device = device or DEVICE
    p = _cache_path(ds, DEPTH, k, act, seed, defence)
    if os.path.exists(p):
        try:
            blob = torch.load(p, map_location=device, weights_only=False)
            model = build_model(DEPTH, k, act).to(device)
            model.load_state_dict(blob["state_dict"])
            model.eval()
            f1 = macro_f1(model, Xte, yte, device)
            if abs(f1 - blob["clean_f1"]) <= 1e-6:
                return model, f1, int(blob.get("epochs", -1)), "cache"
            print(f"      !! cache at {p} re-measures {f1:.6f} against stored "
                  f"{blob['clean_f1']:.6f}; retraining", flush=True)
        except Exception as e:
            print(f"      !! cache at {p} unreadable ({type(e).__name__}); "
                  f"retraining", flush=True)
    torch.manual_seed(seed); np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = build_model(DEPTH, k, act).to(device)
    _, epochs = train_to_convergence(model, Xtrn, ytrn, Xval, yval, device)
    model.eval()
    f1 = macro_f1(model, Xte, yte, device)
    torch.save({"state_dict": model.state_dict(), "clean_f1": float(f1),
                "epochs": int(epochs), "ds": ds, "depth": DEPTH, "features": k,
                "activation": act, "seed": seed, "defence": defence}, p)
    return model, f1, epochs, "trained"


def mask_self_check_cic(free_names):
    pats = {"categorical": CIC_CATEGORICAL, "dest_side": CIC_DEST_SIDE,
            "derived": CIC_DERIVED}
    leaks = []
    for n in free_names:
        if _any(CIC_SENDER_SET, n):
            continue
        for g, ps in pats.items():
            if _any(ps, n):
                leaks.append((n, g))
    if leaks:
        _stop(f"CIC MASK SELF-CHECK FAILED: {leaks} free at L2 but "
              f"matching a frozen pattern group.")
    return ", ".join(pats) + " (sender-set exempt)"


def append_row(path, row):
    pd.DataFrame([{c: row.get(c, "") for c in COLS}], columns=COLS).to_csv(
        path, mode="a", index=False, header=not os.path.exists(path))


def done_keys(path):
    if not os.path.exists(path):
        return set()
    d = pd.read_csv(path)
    missing = [c for c in COLS if c not in d.columns]
    if missing:
        n = 1
        while os.path.exists(f"{path}.v{n}"):
            n += 1
        os.rename(path, f"{path}.v{n}")
        print(f"  {path} has a different schema (missing {missing}); moved to "
              f"{path}.v{n} and starting fresh", flush=True)
        return set()
    return set(zip(d.dataset.astype(str), d.activation.astype(str),
                   d.seed.astype(int), d.level.astype(int),
                   d.eps.astype(float).round(3)))


COLS = ["dataset", "depth", "features", "activation", "seed", "level", "eps",
        "attack", "asr", "n_correct", "n_hit", "clean_f1", "n_free",
        "required_linf", "attack_seed", "elapsed_s"]
_KEY = ["dataset", "depth", "features", "activation", "seed", "level", "eps",
        "attack"]


def append_row(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame([{c: row.get(c, "") for c in COLS}], columns=COLS).to_csv(
        path, mode="a", index=False, header=not os.path.exists(path))


def done_keys(path):
    if not os.path.exists(path):
        return set()
    d = pd.read_csv(path)
    if [c for c in COLS if c not in d.columns]:
        _stop(f"{path} has a different column layout; it was not written by "
              f"this script. Move it away and re-run.")
    return set(zip(d.dataset.astype(str), d.depth.astype(int),
                   d.features.astype(int), d.activation.astype(str),
                   d.seed.astype(int), d.level.astype(int),
                   d.eps.astype(float).round(3), d.attack.astype(str)))


def _seed_of(*parts):
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return ATTACK_SEED_BASE + int(h[:8], 16) % 1_000_000


def _reseed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def _sha(*arrs):
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def get_meta(ds):
    p = os.path.join(DATA_DIR, f"{ds}_meta.npz")
    if not os.path.exists(p):
        _stop(f"{p} not found: build the data first (RERUN_PART=data).")
    return dict(np.load(p, allow_pickle=False))


def preflight(ds, names, order):
    sel = [str(names[i]) for i in np.asarray(order)[:FEATS]]
    key = (ds, tuple(sel))
    if key in _PREFLIGHT_DONE:
        return
    f1 = [n for n, v in zip(sel, build_mask(sel, ds, 1)) if v > 0]
    f2 = [n for n, v in zip(sel, build_mask(sel, ds, 2)) if v > 0]
    print(f"  {ds}: top-{FEATS} features (ANOVA order): {sel}", flush=True)
    old = {"cic": (29, 7), "unsw": (14, 3)}.get(ds, ("?", "?"))
    print(f"  {ds}: free at L1: {len(f1)}; free at L2 ({len(f2)}): {f2}  "
          f"[paper, before deduplication: {old[0]} and {old[1]}]", flush=True)
    bad, warn = [], []
    if max(K_SWEEP[ds]) > len(names):
        bad.append(f"K_SWEEP[{ds!r}] asks for {max(K_SWEEP[ds])} features, the "
                   f"data have {len(names)}")
    for n in CD_ADD_ONLY.get(ds, []):
        if n not in f2:
            bad.append(f"CD_ADD_ONLY[{ds!r}]: {n!r} is not free at L2")
    if ds == "cic":
        for n in CA_FIVE_DROP:
            if n not in f2:
                bad.append(f"CA_FIVE_DROP: {n!r} is not free at L2")
        n5 = len([n for n in f2 if n not in CA_FIVE_DROP])
        if n5 != CA_FIVE_FREE:
            bad.append(f"CA_FIVE_FREE = {CA_FIVE_FREE}, but L2 minus "
                       f"CA_FIVE_DROP leaves {n5}")
        mask_self_check_cic(f2)
    if ds == CU_DATASET:
        for n in CU_TCP_ONLY:
            if n not in f2:
                bad.append(f"CU_TCP_ONLY: {n!r} is not free at L2")
        if CU_TCP_FLAG not in sel:
            bad.append(f"CU_TCP_FLAG {CU_TCP_FLAG!r} is not among the top-{FEATS}")
    miss = [n for n in f2 if n not in CA_UNITS]
    if miss:
        warn.append(f"CA_UNITS has no unit for {miss} (tab:rawunits prints it "
                    f"blank)")
    if ds == "unsw":
        if [n for n in XAI_UNSW_PAIR if n not in sel]:
            warn.append(f"XAI_UNSW_PAIR {XAI_UNSW_PAIR}: not all in the top-{FEATS}")
        if len(f2) > CA_ENUM_MAX_FREE:
            warn.append(f"{len(f2)} free features at L2 > CA_ENUM_MAX_FREE: the "
                        f"lattice evaluation will not run")
    tcp_free = [n for n in CX_TCP_ONLY.get(ds, []) if n in f2]
    warn.append(f"cxcomb: TCP-only fields free at L2: {tcp_free or 'none'}")
    for fwd, tot in CX_FLAG_TOTALS.get(ds, {}).items():
        if fwd in f2:
            warn.append(f"cxcomb: {fwd} is free; its total {tot} is "
                        + ("a model input and is recomputed" if tot in sel
                           else "not a model input (nothing to recompute)"))
    for w in warn:
        print(f"  PREFLIGHT note ({ds}): {w}", flush=True)
    if bad:
        _stop(f"PREFLIGHT FAILED for {ds}; NOTHING HAS BEEN TRAINED.\n  - "
              + "\n  - ".join(bad) + "\nThe deduplicated data changed the top-"
              f"{FEATS} set. Send this output: the stage settings named above "
              f"must be updated to the new feature set before the run.")
    _PREFLIGHT_DONE.add(key)
    print(f"  {ds}: preflight passed", flush=True)


def get_data(ds, check=True):
    p = os.path.join(DATA_DIR, f"{ds}.npz")
    pm = os.path.join(DATA_DIR, f"{ds}_meta.npz")
    if not os.path.exists(p):
        # Built automatically the first time it is needed.
        print(f"  {p} not found: building it once from the Kaggle data ...",
              flush=True)
        make_deterministic(SEED)
        Xtr, ytr, Xte, yte, order, names = load(ds)
        os.makedirs(DATA_DIR, exist_ok=True)
        sel = [str(names[i]) for i in np.asarray(order)[:FEATS]]
        meta = dict(_DATA_META[ds],
                    free_l1=int(build_mask(sel, ds, 1).sum()),
                    free_l2=int(build_mask(sel, ds, 2).sum()),
                    data_hash=_sha(Xtr, Xte, ytr, yte), dedup=int(DEDUP))
        tmpm = pm[:-4] + f".tmp{os.getpid()}.npz"
        np.savez(tmpm, **{k: np.asarray(v) for k, v in meta.items()})
        os.replace(tmpm, pm)
        tmp = p[:-4] + f".tmp{os.getpid()}.npz"
        np.savez(tmp, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte,
                 order=np.asarray(order), names=np.asarray(names, dtype=str))
        os.replace(tmp, p)
    if not os.path.exists(pm):
        _stop(f"{p} exists but {pm} does not: {p} was built by a program "
              f"without deduplication. Run in an EMPTY folder.")
    z = np.load(p, allow_pickle=False)
    out = (z["Xtr"], z["ytr"], z["Xte"], z["yte"], z["order"],
           [str(x) for x in z["names"]])
    zm = np.load(pm, allow_pickle=False)
    h = _sha(out[0], out[2], out[1], out[3])
    if str(zm["data_hash"]) != h or int(zm["dedup"]) != int(DEDUP):
        _stop(f"{p} and {pm} do not belong together (data hash {h}, meta "
              f"{zm['data_hash']}, dedup {int(zm['dedup'])}). Run in an EMPTY "
              f"folder.")
    EXPECTED_FREE[(ds, 1)] = int(zm["free_l1"])
    EXPECTED_FREE[(ds, 2)] = int(zm["free_l2"])
    print(f"  {ds}: train={out[0].shape} test={out[2].shape} "
          f"data hash {h}", flush=True)
    if check:
        preflight(ds, out[5], out[4])
    return out


def make_pool(ds, yte):
    rng = np.random.default_rng(SEED)
    idx = np.where(yte != 0)[0]
    return rng.choice(idx, size=min(N_EVAL_BY_DS[ds], len(idx)), replace=False)


def model_path(ds, depth, k, act, seed, role="standard"):
    return os.path.join(MODEL_DIR, f"{ds}_d{depth}_f{k}_{act}_s{seed}_{role}.pt")


def get_model(ds, depth, k, act, seed, Xtr_k, ytr, Xte_k, yte, role="standard",
              load_only=False):
    """Train once and save."""
    p = model_path(ds, depth, k, act, seed, role)
    if load_only and not os.path.exists(p):
        _stop(f"{p} is missing: the base run of this part has not saved it. "
              f"Run the part without RERUN_STAGES first.")
    if os.path.exists(p):
        blob = torch.load(p, map_location=DEVICE, weights_only=False)
        model = build_model(depth, k, act).to(DEVICE)
        model.load_state_dict(blob["state_dict"]); model.eval()
        f1 = macro_f1(model, Xte_k, yte, DEVICE)
        if abs(f1 - blob["clean_f1"]) > 1e-9:
            _stop(f"{p} re-measures clean F1 {f1:.6f}, saved {blob['clean_f1']:.6f}"
                  f": the data or code changed since it was trained.")
        return model, f1, "saved"
    Xtrn, Xval, ytrn, yval = train_test_split(
        Xtr_k, ytr, test_size=VAL_FRACTION, random_state=SEED, stratify=ytr)
    _reseed(seed)
    model = build_model(depth, k, act).to(DEVICE)
    _, epochs = train_to_convergence(model, Xtrn, ytrn, Xval, yval, DEVICE)
    model.eval()
    f1 = macro_f1(model, Xte_k, yte, DEVICE)
    os.makedirs(MODEL_DIR, exist_ok=True)
    tmp = p + f".tmp{os.getpid()}"
    torch.save({"state_dict": model.state_dict(), "clean_f1": float(f1),
                "epochs": int(epochs), "ds": ds, "depth": depth, "features": k,
                "activation": act, "seed": seed, "role": role,
                "script": "rerun_all.py"}, tmp)
    os.replace(tmp, p)                    # atomic: parallel parts may race
    return model, f1, "trained"


def get_substitute(ds, seed, Xtr_k, ytr, Xte_k, yte):
    s = seed + SUBSTITUTE["seed_offset"]
    m, f1, src = get_model(ds, SUBSTITUTE["depth"], Xtr_k.shape[1],
                           SUBSTITUTE["act"], s, Xtr_k, ytr, Xte_k, yte,
                           role="substitute")
    print(f"    transfer substitute {ds} s{s}: {src} (F1 {f1:.4f})", flush=True)
    return _art_clf(m, Xtr_k, DEVICE)


def a_transfer_saved(sub_clf, X, eps, mask):
    from art.attacks.evasion import ProjectedGradientDescent
    atk = ProjectedGradientDescent(sub_clf, eps=eps,
                                   eps_step=eps / FGSM_EPS_STEP_DIV,
                                   max_iter=PGD_BIM_MAX_ITER,
                                   num_random_init=PGD_RANDOM_INIT,
                                   batch_size=BATCH_SIZE, verbose=False)
    kw = {} if mask is None else {"mask": np.broadcast_to(
        mask, X.shape).astype(np.float32).copy()}
    return atk.generate(x=X.astype(np.float32), **kw)


def _ref_q(clf, X, eps, mask, which):
    """PGD / BIM exactly as _ref, without progress bars."""
    from art.attacks.evasion import BasicIterativeMethod, ProjectedGradientDescent
    kw = dict(eps=eps, eps_step=eps / FGSM_EPS_STEP_DIV,
              max_iter=PGD_BIM_MAX_ITER, batch_size=BATCH_SIZE, verbose=False)
    atk = (ProjectedGradientDescent(clf, num_random_init=PGD_RANDOM_INIT, **kw)
           if which == "pgd" else BasicIterativeMethod(clf, **kw))
    gk = {} if mask is None else {"mask": np.broadcast_to(
        mask, X.shape).astype(np.float32).copy()}
    return atk.generate(x=X.astype(np.float32), **gk)


def hsj_unseeded(clf, X, norm, seed):
    """HopSkipJump, no starting point, returned UNPROJECTED."""
    from art.attacks.evasion import HopSkipJump
    atk = HopSkipJump(classifier=clf, targeted=False, norm=norm,
                      max_iter=HSJ_MAX_ITER, max_eval=HSJ_MAX_EVAL,
                      init_eval=HSJ_INIT_EVAL, batch_size=128,
                      verbose=HSJ_VERBOSE)
    assert atk.norm == norm, f"HopSkipJump norm is {atk.norm}, not {norm}"
    _orig, _n = np.random.RandomState, [0]

    def _seeded(*a, **k):
        if a or k:
            return _orig(*a, **k)
        _n[0] += 1
        return _orig((seed + _n[0]) % (2 ** 32))
    np.random.RandomState = _seeded
    try:
        adv = atk.generate(x=X.astype(np.float32))
    finally:
        np.random.RandomState = _orig
    ok = np.isfinite(adv).all(1)
    return np.where(ok[:, None], adv, X).astype(np.float32)


def _timed(fn):
    try:
        return _run_with_timeout(fn, ATTACK_TIMEOUT)
    except _Timeout:
        _stop(f"attack exceeded {ATTACK_TIMEOUT}s; nothing was written for "
              f"this cell. Raise ATTACK_TIMEOUT before ANY part runs.")


# ---- one part ----
def run_part(ds, act, seed, have, data):
    Xtr_f, ytr, Xte_f, yte, order, names = data
    pool = make_pool(ds, yte)
    sel30 = [names[i] for i in np.asarray(order)[:FEATS]]
    masks = {0: None}
    for lv in (1, 2):
        masks[lv] = build_mask(sel30, ds, lv)
        nfree = int(masks[lv].sum())
        if nfree != EXPECTED_FREE[(ds, lv)]:
            _stop(f"{ds} L{lv}: {nfree} free features, expected "
                  f"{EXPECTED_FREE[(ds, lv)]}.")
    if ds == "cic":
        mask_self_check_cic([n for n, v in zip(sel30, masks[2]) if v > 0])

    cfgs = [(4, FEATS)] + [(d, FEATS) for d in DEPTHS if d != 4]
    if act == "relu":
        cfgs += [(1, k) for k in K_SWEEP[ds] if k != FEATS]
    sub_clf = None
    for depth, k in cfgs:
        cols = np.asarray(order)[:k]
        Xtr, Xte = Xtr_f[:, cols], Xte_f[:, cols]
        Xe, ye = Xte_f[pool][:, cols].astype(np.float32), yte[pool]
        levels = LEVELS if (depth == 4 and k == FEATS) else [0]
        attacks = ["pgd", "bim", "spsa"] + (["transfer"] if k == FEATS else [])
        hsj_norms = [("inf", np.inf)] + ([("l2", 2)] if depth == 4 else [])
        want = {(ds, depth, k, act, seed, lv, round(e, 3), a)
                for lv in levels for e in EPS_LIST for a in attacks + ["union"]}
        want |= {(ds, depth, k, act, seed, 0, round(e, 3), f"hsj_{n}{x}")
                 for n, _ in hsj_norms for e in EPS_LIST for x in ("", "_proj")}
        if want <= have:
            print(f"  skip {ds} d{depth} k{k} {act} s{seed}: done", flush=True)
            continue
        t0 = time.time()
        model, f1, src = get_model(ds, depth, k, act, seed, Xtr, ytr, Xte, yte)
        old = (OLD_TABLE5_F1.get((ds, act, seed)) if (depth, k) == (4, FEATS)
               else None)
        print(f"  {ds} d{depth} k{k} {act} s{seed}: {src} cleanF1={f1:.4f}"
              + ("" if old is None else f"  (earlier Table 5 model: {old:.4f})")
              + f" [{time.time() - t0:.0f}s]", flush=True)
        clf = _art_clf(model, Xtr, DEVICE)
        if k == FEATS and sub_clf is None:
            sub_clf = get_substitute(ds, seed, Xtr, ytr, Xte, yte)
        npz = os.path.join(RES_DIR, "flows",
                           f"{ds}_d{depth}_f{k}_{act}_s{seed}.npz")
        saved = dict(np.load(npz)) if os.path.exists(npz) else {}
        saved["pool"] = pool
        correct = _probs(model, Xe, DEVICE).argmax(1) == ye
        saved["correct"] = correct
        n_ok = max(int(correct.sum()), 1)
        base = dict(dataset=ds, depth=depth, features=k, activation=act,
                    seed=seed, clean_f1=round(float(f1), 4))

        def save_npz():
            os.makedirs(os.path.dirname(npz), exist_ok=True)
            tmp = npz[:-4] + f".tmp{os.getpid()}.npz"
            np.savez_compressed(tmp, **saved); os.replace(tmp, npz)

        # -- gradient / score / transfer attacks, every level and budget.
        for lv in levels:
            mask = masks[lv]
            nfree = k if mask is None else int(mask.sum())
            for eps in EPS_LIST:
                if (ds, depth, k, act, seed, lv, round(eps, 3), "union") in have:
                    continue
                union, rates = None, {}
                for a in attacks:
                    s = _seed_of(ds, depth, k, act, seed, lv, eps, a)
                    _reseed(s); t2 = time.time()
                    fn = {"pgd": lambda: _ref_q(clf, Xe, eps, mask, "pgd"),
                          "bim": lambda: _ref_q(clf, Xe, eps, mask, "bim"),
                          "spsa": lambda: a_spsa(model, Xe, ye, eps, s, mask),
                          "transfer": lambda: a_transfer_saved(sub_clf, Xe,
                                                               eps, mask)}[a]
                    adv = _timed(fn)
                    o, h, v = hits_of(model, adv, Xe, ye, DEVICE, eps, mask)
                    if v:
                        _stop(f"capability violation: {v} frozen coordinates "
                              f"moved ({ds} d{depth} {act} s{seed} L{lv} "
                              f"eps={eps} {a})")
                    saved[f"L{lv}_e{eps}_{a}"] = h
                    union = h if union is None else (union | h)
                    rates[a] = float(h.sum() / n_ok)
                    append_row(OUT, dict(base, level=lv, eps=round(eps, 3),
                                         attack=a, asr=round(rates[a], 4),
                                         n_correct=n_ok, n_hit=int(h.sum()),
                                         n_free=nfree, attack_seed=s,
                                         elapsed_s=round(time.time() - t2, 1)))
                saved[f"L{lv}_e{eps}_union"] = union
                save_npz()
                append_row(OUT, dict(base, level=lv, eps=round(eps, 3),
                                     attack="union",
                                     asr=round(float(union.sum() / n_ok), 4),
                                     n_correct=n_ok, n_hit=int(union.sum()),
                                     n_free=nfree, attack_seed=0, elapsed_s=0))
                have.add((ds, depth, k, act, seed, lv, round(eps, 3), "union"))
                print(f"    L{lv} eps={eps}: " + " ".join(
                    f"{a.upper()} {rates[a]:.3f}" for a in attacks)
                    + f" | union {union.sum() / n_ok:.3f}", flush=True)

        for nm, norm in hsj_norms:
            if all((ds, depth, k, act, seed, 0, round(e, 3), f"hsj_{nm}") in have
                   for e in EPS_LIST):
                continue
            s = _seed_of(ds, depth, k, act, seed, "hsj", nm)
            _reseed(s); t2 = time.time()
            raw = _timed(lambda: hsj_unseeded(clf, Xe, norm, s))
            saved[f"hsj_{nm}_raw"] = raw
            dist = np.abs(raw - Xe).max(1)
            wrong = _probs(model, raw, DEVICE).argmax(1) != ye
            evaded = correct & wrong
            req = float(np.median(dist[evaded])) if evaded.any() else float("nan")
            el = round((time.time() - t2) / len(EPS_LIST), 1)
            msg = []
            for eps in EPS_LIST:
                within = evaded & (dist <= eps + 1e-6)
                proj = (Xe + np.clip(raw - Xe, -eps, eps)).astype(np.float32)
                _, hp, _ = hits_of(model, proj, Xe, ye, DEVICE, eps, None)
                for a, h in ((f"hsj_{nm}", within), (f"hsj_{nm}_proj", hp)):
                    saved[f"L0_e{eps}_{a}"] = h
                    append_row(OUT, dict(base, level=0, eps=round(eps, 3),
                                         attack=a,
                                         asr=round(float(h.sum() / n_ok), 4),
                                         n_correct=n_ok, n_hit=int(h.sum()),
                                         n_free=k, required_linf=round(req, 4),
                                         attack_seed=s, elapsed_s=el))
                msg.append(f"eps={eps} {within.sum() / n_ok:.3f}"
                           f"/{hp.sum() / n_ok:.3f}")
            save_npz()
            print(f"    HopSkipJump {nm} (within/projected): " + "  ".join(msg)
                  + f"  required l_inf {req:.3f}  [{time.time() - t2:.0f}s]",
                  flush=True)


# ---- LATER STAGES ----
# Everything below the base run.
class Ctx:
    """What a stage is given for one part."""

    def split(self):
        return train_test_split(self.Xtr, self.ytr, test_size=VAL_FRACTION,
                                random_state=SEED, stratify=self.ytr)

    def substitute(self):
        if getattr(self, "_sub", None) is None:
            self._sub = get_substitute(self.ds, self.seed, self.Xtr, self.ytr,
                                       self.Xte, self.yte)
        return self._sub


def part_ctx(ds, act, seed, data):
    Xtr_f, ytr, Xte_f, yte, order, names = data
    c = Ctx()
    c.ds, c.act, c.seed, c.part, c.data = ds, act, seed, f"{ds}-{act}-{seed}", data
    c.cols = np.asarray(order)[:FEATS]
    c.names = [names[i] for i in c.cols]
    c.Xtr, c.ytr, c.Xte, c.yte = Xtr_f[:, c.cols], ytr, Xte_f[:, c.cols], yte
    c.pool = make_pool(ds, yte)
    c.Xe, c.ye = Xte_f[c.pool][:, c.cols].astype(np.float32), yte[c.pool]
    c.masks = {0: None}
    for lv in (1, 2):
        c.masks[lv] = build_mask(c.names, ds, lv)
        if int(c.masks[lv].sum()) != EXPECTED_FREE[(ds, lv)]:
            _stop(f"{ds} L{lv}: {int(c.masks[lv].sum())} free features, "
                  f"expected {EXPECTED_FREE[(ds, lv)]}.")
    c.model, c.f1, _ = get_model(ds, 4, FEATS, act, seed, c.Xtr, ytr, c.Xte, yte,
                                 load_only=True)
    c.clf = _art_clf(c.model, c.Xtr, DEVICE)
    c.correct = _probs(c.model, c.Xe, DEVICE).argmax(1) == c.ye
    c.n_ok = max(int(c.correct.sum()), 1)
    return c


def stage_file(stage):
    """File THIS process appends to."""
    if not stage.isalnum() or stage == "attacks":
        _stop(f"bad stage table name {stage!r}: letters and digits only.")
    return os.path.join(RES_DIR, f"{stage}_{RERUN_PART or 'all'}.csv")


def _files_to_consult(prefix):
    """Result files whose finished rows this process must not recompute."""
    own = os.path.join(RES_DIR, f"{prefix}_{RERUN_PART or 'all'}.csv")
    allf = os.path.join(RES_DIR, f"{prefix}_all.csv")
    final = os.path.join(RES_DIR, f"{prefix}_final.csv")
    fs = [own, allf]
    if not RERUN_PART:
        fs += sorted(glob.glob(os.path.join(RES_DIR, f"{prefix}_*.csv")))
    out = []
    for f in fs:
        if f not in out and f != final and os.path.exists(f):
            out.append(f)
    return out


def _k(*vals):
    return tuple(str(v) for v in vals)


def stage_have(stage, key_cols):
    """Keys of this stage table that are already written (see _files_to_consult)."""
    stage_file(stage)                       # validates the name
    out = set()
    for p in _files_to_consult(stage):
        d = pd.read_csv(p)
        if [c for c in key_cols if c not in d.columns]:
            _stop(f"{p} has a different column layout. Move it away and re-run.")
        out |= set(_k(*r) for r in d[list(key_cols)].itertuples(index=False))
    return out


def stage_append(stage, cols, row):
    p = stage_file(stage)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    missing = [c for c in row if c not in cols]
    if missing:
        _stop(f"stage {stage}: row has columns {missing} that are not in its "
              f"column list.")
    pd.DataFrame([{c: row.get(c, "") for c in cols}], columns=cols).to_csv(
        p, mode="a", index=False, header=not os.path.exists(p))


def stage_read(stage, key_cols, ignore=("elapsed_s",)):
    """All rows of a stage from every part, one per key."""
    final = os.path.join(RES_DIR, f"{stage}_final.csv")
    files = sorted(f for f in glob.glob(os.path.join(RES_DIR, f"{stage}_*.csv"))
                   if f != final)
    if not files:
        return None
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    key_cols = list(key_cols)
    val = [c for c in d.columns if c not in key_cols and c not in ignore]
    dup = d[d.duplicated(key_cols, keep=False)]
    if len(dup):
        txt = dup[key_cols + val].astype(str)
        bad = txt.groupby(key_cols).nunique().max(axis=1)
        if (bad > 1).any():
            _stop(f"stage {stage}: {int((bad > 1).sum())} measurements were "
                  f"repeated with DIFFERENT values:\n{bad[bad > 1].head(20)}")
        d = d.drop_duplicates(key_cols, keep="last").reset_index(drop=True)
    d.to_csv(final, index=False)
    return d


def stage_npz(stage, c):
    return os.path.join(RES_DIR, "flows",
                        f"{stage}_{c.ds}_{c.act}_s{c.seed}.npz")


def npz_load(path):
    return dict(np.load(path)) if os.path.exists(path) else {}


def npz_save(path, arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[:-4] + f".tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


STAGES = []     # (name, run_fn(ctx), summarize_fn()) in the order they run


# ======== stage wbbb: code ========
WB_SPSA_COLS = ["dataset", "activation", "seed", "level", "eps", "steps",
                "ndirs", "loss", "budget", "spsa_queries", "hsj_queries",
                "asr", "n_correct", "n_hit", "n_unmoved", "n_zero_loss",
                "n_zero_logit", "attack_seed", "elapsed_s"]
WB_SPSA_KEY = WB_SPSA_COLS[:8]
WB_UNB_COLS = ["dataset", "activation", "seed", "level", "eps", "max_iter",
               "eps_step", "asr", "n_correct", "n_hit", "n_nonfinite",
               "attack_seed", "elapsed_s"]
WB_UNB_KEY = WB_UNB_COLS[:5]
WB_ITER_COLS = ["dataset", "activation", "seed", "level", "eps", "max_iter",
                "asr", "n_correct", "n_hit", "n_free", "attack_seed",
                "elapsed_s"]
WB_ITER_KEY = WB_ITER_COLS[:6]


def wb_hsj_queries():
    """Queries per flow of HopSkipJump's schedule, from the base settings."""
    import math
    return int(sum(min(HSJ_MAX_EVAL, HSJ_INIT_EVAL * math.sqrt(t))
                   for t in range(1, HSJ_MAX_ITER + 1)))


def wb_spsa_matched_ndirs():
    return max(SPSA_NDIRS,
               int(round(wb_hsj_queries() / (2 * WB_SPSA_M_STEPS))))


def wb_spsa_budgets():
    """[(name, steps, directions)]."""
    out = [("base", SPSA_STEPS, SPSA_NDIRS)]
    if (WB_SPSA_M_STEPS, wb_spsa_matched_ndirs()) != (SPSA_STEPS, SPSA_NDIRS):
        out.append(("matched", WB_SPSA_M_STEPS, wb_spsa_matched_ndirs()))
    return out


def wb_ce_logit(model, X, y, device):
    model.eval()
    yt = torch.tensor(y, dtype=torch.long, device=device)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            t = torch.tensor(X[i:i + BATCH_SIZE], dtype=torch.float32,
                             device=device)
            out.append(F.cross_entropy(model(t), yt[i:i + BATCH_SIZE],
                                       reduction="none").cpu().numpy())
    return np.concatenate(out)


def wb_spsa(model, X, y, eps, seed, mask, steps, ndirs, loss):
    global SPSA_STEPS, SPSA_NDIRS, _ce
    if loss not in ("prob", "logit"):
        _stop(f"unknown SPSA loss {loss!r}")
    keep = (SPSA_STEPS, SPSA_NDIRS, _ce)
    fn, calls = (keep[2] if loss == "prob" else wb_ce_logit), [0]

    def counted(m, Z, yy, dev):
        calls[0] += 1                   # one call = one query per flow
        return fn(m, Z, yy, dev)
    SPSA_STEPS, SPSA_NDIRS, _ce = steps, ndirs, counted
    try:
        adv = a_spsa(model, X, y, eps, seed, mask)
    finally:
        SPSA_STEPS, SPSA_NDIRS, _ce = keep
    if calls[0] != 2 * steps * ndirs:
        _stop(f"SPSA issued {calls[0]} loss queries per flow, expected "
              f"{2 * steps * ndirs} ({steps} steps x {ndirs} directions x 2): "
              f"the requested loss or budget was not the one used.")
    return adv


def wb_base_row(c, lv, eps, attack):
    """(n_hit, n_correct) of this part's base row for Model 4, read from the base result files."""
    for f in _files_to_consult("attacks"):
        d = pd.read_csv(f)
        r = d[(d.dataset == c.ds) & (d.depth == DEPTH) & (d.features == FEATS)
              & (d.activation == c.act) & (d.seed == c.seed) & (d.level == lv)
              & np.isclose(d.eps, eps) & (d.attack == attack)]
        if len(r):
            return int(r.n_hit.iloc[-1]), int(r.n_correct.iloc[-1])
    return None


def wb_clf_unclipped(model, k):
    """_art_clf without the clip to the training range."""
    from art.estimators.classification import PyTorchClassifier
    return PyTorchClassifier(
        model=_F32(model).to(DEVICE), loss=nn.CrossEntropyLoss(),
        input_shape=(k,), nb_classes=2, clip_values=None,
        device_type=("gpu" if str(DEVICE).startswith("cuda") else "cpu"))


def wb_pgd(clf, X, eps, mask, max_iter, eps_step):
    """PGD exactly as _ref_q (one random start, same batch size), with the iteration count and the step as arguments."""
    from art.attacks.evasion import ProjectedGradientDescent
    atk = ProjectedGradientDescent(clf, eps=eps, eps_step=eps_step,
                                   max_iter=max_iter,
                                   num_random_init=PGD_RANDOM_INIT,
                                   batch_size=BATCH_SIZE, verbose=False)
    gk = {} if mask is None else {"mask": np.broadcast_to(
        mask, X.shape).astype(np.float32).copy()}
    return atk.generate(x=X.astype(np.float32), **gk)


def wb_score(c, adv, eps, mask, what):
    """Per-flow success as hits_of."""
    adv = np.asarray(adv, np.float32)
    bad = ~np.isfinite(adv).all(1)
    adv = np.where(bad[:, None], c.Xe, adv).astype(np.float32)
    proj = (c.Xe + np.clip(adv - c.Xe, -eps, eps)).astype(np.float32)
    bad2 = ~np.isfinite(_probs(c.model, proj, DEVICE)).all(1)
    if bad2.any():
        adv = np.where(bad2[:, None], c.Xe, adv).astype(np.float32)
    _, h, viol = hits_of(c.model, adv, c.Xe, c.ye, DEVICE, eps, mask)
    if viol:
        _stop(f"capability violation: {viol} frozen coordinates moved "
              f"({c.part} {what})")
    return h, int((bad | bad2).sum())


def wb_stage(c):
    npz = stage_npz("wbbb", c)
    saved = npz_load(npz)
    ident = dict(dataset=c.ds, activation=c.act, seed=c.seed)

    have = stage_have("wbspsa", WB_SPSA_KEY)
    hq = wb_hsj_queries()
    # flows on which the loss is exactly 0.0 at the clean point, for the base ("prob") loss and for the logit loss.
    zero = int((c.correct & (_ce(c.model, c.Xe, c.ye, DEVICE) == 0)).sum())
    zero_lg = int((c.correct
                   & (wb_ce_logit(c.model, c.Xe, c.ye, DEVICE) == 0)).sum())
    for eps in WB_SPSA_EPS:
        for bname, steps, ndirs in wb_spsa_budgets():
            # The base budget runs with the BASE row's seed, so that (base, prob) is the base measurement itself.
            s = (_seed_of(c.ds, DEPTH, FEATS, c.act, c.seed, 0, eps, "spsa")
                 if bname == "base" else
                 _seed_of("wbbb", "spsa", c.ds, c.act, c.seed, 0, eps, steps,
                          ndirs))
            for loss in WB_SPSA_LOSSES:
                if _k(c.ds, c.act, c.seed, 0, round(eps, 3), steps, ndirs,
                      loss) in have:
                    continue
                _reseed(s); t0 = time.time()
                adv = _timed(lambda: wb_spsa(c.model, c.Xe, c.ye, eps, s, None,
                                             steps, ndirs, loss))
                h, bad = wb_score(c, adv, eps, None,
                                  f"spsa {bname} {loss} eps={eps}")
                if bad:
                    _stop(f"{c.part}: SPSA ({bname} budget, {loss} loss) "
                          f"produced {bad} non-finite flows at eps={eps}.")
                if (bname, loss) == ("base", "prob"):
                    ref = wb_base_row(c, 0, eps, "spsa")
                    if ref != (int(h.sum()), c.n_ok):
                        _stop(f"{c.part} eps={eps}: SPSA with the base budget, "
                              f"loss and seed gives {int(h.sum())}/{c.n_ok}, "
                              f"the base row {ref}. They must be equal; "
                              f"nothing was written for this cell.")
                # flows SPSA never moved.
                still = c.correct & (np.abs(adv - c.Xe).max(1) == 0)
                saved[f"spsa_{bname}_{loss}_e{eps}_adv"] = adv
                saved[f"spsa_{bname}_{loss}_e{eps}_hit"] = h
                npz_save(npz, saved)
                stage_append("wbspsa", WB_SPSA_COLS, dict(
                    ident, level=0, eps=round(eps, 3), steps=steps,
                    ndirs=ndirs, loss=loss, budget=bname,
                    spsa_queries=2 * steps * ndirs, hsj_queries=hq,
                    asr=round(float(h.sum() / c.n_ok), 4), n_correct=c.n_ok,
                    n_hit=int(h.sum()), n_unmoved=int(still.sum()),
                    n_zero_loss=zero, n_zero_logit=zero_lg, attack_seed=s,
                    elapsed_s=round(time.time() - t0, 1)))
                print(f"    SPSA {bname} budget ({2 * steps * ndirs} queries/"
                      f"flow; HopSkipJump {hq}), {loss} loss, eps={eps}: "
                      f"{h.sum() / c.n_ok:.4f}  never moved: "
                      f"{int(still.sum())} of {c.n_ok} (zero clean loss: "
                      f"prob {zero}, logit {zero_lg})  "
                      f"[{time.time() - t0:.0f}s]", flush=True)

    have = stage_have("wbunb", WB_UNB_KEY)
    clf_nc, msg = None, []
    for eps in WB_UNB_EPS:
        if _k(c.ds, c.act, c.seed, 0, round(eps, 3)) in have:
            continue
        if clf_nc is None:
            clf_nc = wb_clf_unclipped(c.model, c.Xe.shape[1])
        step = eps * WB_UNB_STEP_MULT / PGD_BIM_MAX_ITER
        s = _seed_of("wbbb", "unbounded", c.ds, c.act, c.seed, eps)
        _reseed(s); t0 = time.time()
        adv = _timed(lambda: wb_pgd(clf_nc, c.Xe, eps, None, PGD_BIM_MAX_ITER,
                                    step))
        h, bad = wb_score(c, adv, eps, None, f"unbounded eps={eps}")
        saved[f"unb_e{eps}_hit"] = h
        npz_save(npz, saved)
        stage_append("wbunb", WB_UNB_COLS, dict(
            ident, level=0, eps=round(eps, 3), max_iter=PGD_BIM_MAX_ITER,
            eps_step=round(step, 6), asr=round(float(h.sum() / c.n_ok), 4),
            n_correct=c.n_ok, n_hit=int(h.sum()), n_nonfinite=bad,
            attack_seed=s, elapsed_s=round(time.time() - t0, 1)))
        msg.append(f"eps={eps:g} {h.sum() / c.n_ok:.4f}")
    if msg:
        print("    unbounded PGD (unclipped): " + "  ".join(msg), flush=True)

    have = stage_have("wbiter", WB_ITER_KEY)
    for lv in WB_ITER_LEVELS:
        mask = c.masks[lv]
        nfree = FEATS if mask is None else int(mask.sum())
        for eps in WB_ITER_EPS:
            # the seed of the BASE run's PGD row for this model, level and budget.
            s = _seed_of(c.ds, DEPTH, FEATS, c.act, c.seed, lv, eps, "pgd")
            msg = []
            for it in WB_ITER_LIST:
                if _k(c.ds, c.act, c.seed, lv, round(eps, 3), it) in have:
                    continue
                _reseed(s); t0 = time.time()
                adv = _timed(lambda: wb_pgd(c.clf, c.Xe, eps, mask, it,
                                            eps / FGSM_EPS_STEP_DIV))
                h, bad = wb_score(c, adv, eps, mask,
                                  f"PGD {it} iterations L{lv} eps={eps}")
                if bad:
                    _stop(f"{c.part}: PGD with {it} iterations produced {bad} "
                          f"non-finite flows (L{lv}, eps={eps}).")
                saved[f"iter_L{lv}_e{eps}_i{it}_hit"] = h
                npz_save(npz, saved)
                stage_append("wbiter", WB_ITER_COLS, dict(
                    ident, level=lv, eps=round(eps, 3), max_iter=it,
                    asr=round(float(h.sum() / c.n_ok), 4), n_correct=c.n_ok,
                    n_hit=int(h.sum()), n_free=nfree, attack_seed=s,
                    elapsed_s=round(time.time() - t0, 1)))
                msg.append(f"{it}: {h.sum() / c.n_ok:.4f}")
            if msg:
                print(f"    PGD iterations L{lv} eps={eps}: " + "  ".join(msg),
                      flush=True)


def wb_figure_canvas():
    import importlib.util
    if importlib.util.find_spec("matplotlib") is None:
        print("installing missing library: matplotlib ...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "matplotlib"], capture_output=True, text=True)
        importlib.invalidate_caches()
        if importlib.util.find_spec("matplotlib") is None:
            return None
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=WB_FIG_SIZE)
    FigureCanvasAgg(fig)
    return fig


def wb_fig1(val, acts, lab):
    rows, missing = [], []
    for ds, L in lab:
        for bar in WB_FIG_BARS:
            for (a, _), v in zip(acts, val[(ds, bar)]):
                if not np.isfinite(v).all():
                    missing.append(f"{ds}-{a} {bar}")
                rows.append(dict(
                    dataset=L, activation=a, bar=bar,
                    label=WB_FIG_STYLE[bar][0],
                    mean=round(float(np.mean(v)), 4),
                    sd=round(float(np.std(v, ddof=SD_DDOF)), 4),
                    bar_low=round(max(0.0, float(np.mean(v))
                                      - float(np.std(v, ddof=SD_DDOF))), 4),
                    bar_high=round(min(1.0, float(np.mean(v))
                                       + float(np.std(v, ddof=SD_DDOF))), 4),
                    n_seeds=int(np.isfinite(v).sum()),
                    source=("wbspsa_final.csv:budget={},loss={}".format(
                                *WB_FIG_MATCHED_CELL) if bar == "spsa_matched"
                            else f"attacks_final.csv:{bar}")))
    if missing:
        print(f"%% Figure 1 NOT written: {len(missing)} bar(s) lack a seed "
              f"({', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''}).")
        return
    fig = wb_figure_canvas()
    if fig is None:
        print("\n" + "!" * 74 + "\nFigure 1 NOT written: matplotlib is missing "
              "and could not be installed. Run  pip install matplotlib  and "
              "then RERUN_PART=summary again.\n" + "!" * 74)
        return
    os.makedirs(WB_FIG_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        os.path.join(WB_FIG_DIR, "fig1_plotted_values.csv"), index=False)
    axes = fig.subplots(1, 2, sharey=True)
    n = len(WB_FIG_BARS)
    width = WB_FIG_GROUP_W / n
    x = np.arange(len(acts))
    for ax, (ds, L) in zip(axes, lab):
        for i, bar in enumerate(WB_FIG_BARS):
            label, colour, hatch = WB_FIG_STYLE[bar]
            m = [float(np.mean(v)) for v in val[(ds, bar)]]
            sd = [float(np.std(v, ddof=SD_DDOF)) for v in val[(ds, bar)]]
            # A success rate lies in [0, 1].
            err = [[min(s_, m_) for m_, s_ in zip(m, sd)],
                   [min(s_, 1.0 - m_) for m_, s_ in zip(m, sd)]]
            ax.bar(x + (i - (n - 1) / 2) * width, m, width, yerr=err, capsize=3,
                   color=colour, label=label, hatch=hatch,
                   edgecolor=("white" if hatch else colour), linewidth=0,
                   error_kw=dict(lw=1))
        ax.set_xticks(x)
        ax.set_xticklabels([a.upper() for a, _ in acts])
        ax.set_title(L)
        ax.set_xlabel("Activation")
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel(f"Mean ASR at $\\epsilon = {WB_FIG_EPS}$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=n, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(WB_FIG_DIR, "fig1.pdf")
    fig.savefig(out, dpi=WB_FIG_DPI)
    print(f"%% Figure 1 written: {out} (values: {WB_FIG_DIR}/"
          f"fig1_plotted_values.csv). Bars, left to right in each group: "
          + ", ".join(WB_FIG_STYLE[k][0] for k in WB_FIG_BARS)
          + ". A bar named 'spsa_matched' in WB_FIG_BARS is the wbspsa cell "
          f"(budget, loss) = {WB_FIG_MATCHED_CELL}; a bar named 'spsa' is the "
          f"base row of Table 4. Plotted: {WB_FIG_BARS}. Error "
          f"bars: mean +- one SD over the seeds (ddof = {SD_DDOF}), truncated "
          f"to the range [0, 1] of a rate.")


def wb_summarize():
    if not os.path.exists(OUT_FINAL):
        print("%% stage wbbb: the base table (attacks_final.csv) is missing; "
              "RERUN_PART=summary writes it first.")
        return
    b = pd.read_csv(OUT_FINAL)
    b = b[(b.depth == DEPTH) & (b.features == FEATS)]
    spsa = stage_read("wbspsa", WB_SPSA_KEY)
    unb = stage_read("wbunb", WB_UNB_KEY)
    itr = stage_read("wbiter", WB_ITER_KEY)
    acts = (("relu", "ReLU"), ("elu", "ELU "), ("tanh", "Tanh"))
    lab = (("cic", "CSE-CIC-IDS2018"), ("unsw", "UNSW-NB15"))
    hq = wb_hsj_queries()
    budgets = {n: (st, nd) for n, st, nd in wb_spsa_budgets()}
    f4 = lambda v: " ".join(f"{x:.4f}" for x in v)
    mean = lambda v: float(np.mean(v))

    def base(ds, act, eps, atk, lv=0, col="asr"):
        """Per-seed values of one base cell, in the order of SEEDS (NaN where a part has not run)."""
        r = b[(b.dataset == ds) & (b.activation == act) & (b.level == lv)
              & np.isclose(b.eps, eps) & (b.attack == atk)]
        return r.set_index("seed")[col].reindex(SEEDS).to_numpy(float)

    def cell(ds, act, eps, budget, loss, col="asr"):
        """Per-seed values of one cell of the SPSA 2 x 2 (NaN where missing)."""
        if spsa is None or budget not in budgets:
            return np.full(len(SEEDS), np.nan)
        r = spsa[(spsa.dataset == ds) & (spsa.activation == act)
                 & (spsa.level == 0) & np.isclose(spsa.eps, eps)
                 & (spsa.steps == budgets[budget][0])
                 & (spsa.ndirs == budgets[budget][1]) & (spsa.loss == loss)]
        return r.set_index("seed")[col].reindex(SEEDS).to_numpy(float)

    def matched(ds, act, eps, col="asr"):
        """The cell Figure 1 plots as its matched-budget SPSA bar."""
        return cell(ds, act, eps, *WB_FIG_MATCHED_CELL, col=col)

    mname = "SPSA " + "/".join(WB_FIG_MATCHED_CELL)

    # ---- A. in-text numbers that are base rows ----
    A5 = (("pgd", "PGD"), ("bim", "BIM"), ("hsj_inf", "HSJ"),
          ("hsj_inf_proj", "HSJ-proj"), ("spsa", "SPSA"))
    print("\n%% ---- sec:wbbb, in-text numbers that are BASE rows (Model 4, "
          f"{FEATS} features, L0); nothing here is measured by this stage")
    print("%% mean+-SD over seeds: PGD | BIM | HopSkipJump l_inf within eps | "
          f"the same output projected | SPSA ({2 * SPSA_STEPS * SPSA_NDIRS} "
          "queries)")
    e_small = min(EPS_LIST)
    for eps in (e_small, WB_FIG_EPS):
        print(f"%% eps = {eps}")
        for ds, L in lab:
            for a, A in acts:
                print(f"  {ds:4s} {A} & " + " & ".join(
                    _ms(base(ds, a, eps, k)) for k, _ in A5) + " \\\\")
    print(f"%% per seed (seeds {SEEDS}), 4 decimals")
    for eps in (e_small, WB_FIG_EPS):
        for ds, L in lab:
            for a, A in acts:
                print(f"  eps={eps} {ds:4s} {A}: " + " | ".join(
                    f"{K} {f4(base(ds, a, eps, k))}" for k, K in A5))
    e = WB_FIG_EPS
    print(f"%% eps = {e}: differences of the seed means (white-box minus "
          "black-box)")
    for ds, L in lab:
        wb = [mean(base(ds, a, e, k)) for a, _ in acts for k in ("pgd", "bim")]
        hs = [mean(base(ds, a, e, "hsj_inf")) for a, _ in acts]
        pb = max(abs(mean(base(ds, a, e, "pgd")) - mean(base(ds, a, e, "bim")))
                 for a, _ in acts)
        print(f"  {ds:4s}: lowest white-box mean (PGD, BIM; three activations) "
              f"{min(wb):.3f}; lowest decision-based mean {min(hs):.3f}; "
              f"largest |PGD - BIM| {pb:.3f}")
        for a, A in acts:
            p, h = mean(base(ds, a, e, "pgd")), mean(base(ds, a, e, "hsj_inf"))
            s, m = mean(base(ds, a, e, "spsa")), mean(matched(ds, a, e))
            print(f"  {ds:4s} {A}: PGD - HSJ {p - h:+.3f} | PGD - SPSA "
                  f"{p - s:+.3f} | PGD - stronger of (HSJ, SPSA) "
                  f"{p - max(h, s):+.3f} | PGD - {mname} {p - m:+.3f} | "
                  f"PGD - stronger of (HSJ, {mname}) "
                  f"{p - (max(h, m) if m == m else m):+.3f}")
    print(f"%% eps = {e}: models on which a gradient-free attack exceeds PGD "
          "(per model, not per mean)")
    for ds, L in lab:
        for k, K in (("hsj_inf", "HopSkipJump within eps"),
                     ("hsj_inf_proj", "HopSkipJump projected"),
                     ("spsa", "SPSA base/prob (Table 4)"),
                     (("base", "logit"), "SPSA base/logit"),
                     (("matched", "prob"), "SPSA matched/prob"),
                     (("matched", "logit"), "SPSA matched/logit")):
            cells = []
            for a, _ in acts:
                p = base(ds, a, e, "pgd")
                g = (base(ds, a, e, k) if isinstance(k, str)
                     else cell(ds, a, e, *k))
                cells += [f"{a} s{sd} ({gv:.4f} vs {pv:.4f})"
                          for sd, gv, pv in zip(SEEDS, g, p) if gv > pv]
            print(f"  {ds:4s} {K}: {len(cells)} of {len(acts) * len(SEEDS)}"
                  + (": " + "; ".join(cells) if cells else ""))
    print(f"%% eps = {e_small}: PGD minus HopSkipJump ({HSJ_MAX_ITER} "
          "iterations, full pool -- the main protocol, not a reduced probe)")
    for ds, L in lab:
        for a, A in acts:
            p = base(ds, a, e_small, "pgd")
            h = base(ds, a, e_small, "hsj_inf")
            hp = base(ds, a, e_small, "hsj_inf_proj")
            print(f"  {ds:4s} {A}: PGD {mean(p):.4f}  within eps {mean(h):.4f} "
                  f"(gap {mean(p) - mean(h):+.4f}; per seed {f4(p - h)})  "
                  f"projected {mean(hp):.4f} (gap {mean(p) - mean(hp):+.4f}; "
                  f"per seed {f4(p - hp)})")
    print(f"%% eps = {e}: effect of the search norm on HopSkipJump (l_inf "
          "minus l_2; Table 5)")
    for ds, L in lab:
        for a, A in acts:
            dw = mean(base(ds, a, e, "hsj_inf")) - mean(base(ds, a, e, "hsj_l2"))
            dp = (mean(base(ds, a, e, "hsj_inf_proj"))
                  - mean(base(ds, a, e, "hsj_l2_proj")))
            print(f"  {ds:4s} {A}: within eps {dw:+.3f} | projected {dp:+.3f}")

    # ---- B. unbounded-budget check ----
    print("\n%% ---- unbounded-budget check: PGD on the UNCLIPPED classifier, "
          f"eps in {WB_UNB_EPS}, {PGD_BIM_MAX_ITER} iterations, step eps * "
          f"{WB_UNB_STEP_MULT} / {PGD_BIM_MAX_ITER}, one random start")
    if unb is None:
        print("%% not run yet")
    else:
        print("%% per model: ASR at each eps | best over the sweep | smallest "
              "eps at which EVERY correctly classified flow is evaded")
        n_models = n_full = n_drop = 0
        first, runmax = [], {}
        for ds, L in lab:
            for a, A in acts:
                for sd in SEEDS:
                    r = unb[(unb.dataset == ds) & (unb.activation == a)
                            & (unb.seed == sd)].sort_values("eps")
                    if len(r) != len(WB_UNB_EPS):
                        print(f"  {ds:4s} {a:4s} s{sd:<2d}: {len(r)} of "
                              f"{len(WB_UNB_EPS)} budgets run")
                        continue
                    n_models += 1
                    full = r[r.n_hit == r.n_correct]
                    n_full += int(len(full) > 0)
                    n_drop += int((np.diff(r.asr.to_numpy()) < 0).any())
                    if len(full):
                        first.append(float(full.eps.min()))
                    runmax.setdefault(ds, []).append(
                        np.maximum.accumulate(r.asr.to_numpy()))
                    bad = int(r.n_nonfinite.sum())
                    print(f"  {ds:4s} {a:4s} s{sd:<2d}: {f4(r.asr)} | best "
                          f"{r.asr.max():.4f} | "
                          + (f"all evaded at eps = {full.eps.min():g}"
                             if len(full) else "never all evaded")
                          + (f" | {bad} non-finite flows counted as not evaded"
                             if bad else ""))
        print(f"  {n_full} of {n_models} models are fully evaded (n_hit = "
              "n_correct) at some budget"
              + (f", every one of them by eps = {max(first):g}; "
                 + ", ".join(f"{first.count(x)} at eps = {x:g}"
                             for x in sorted(set(first))) if first else "")
              + f". The raw ASR decreases somewhere along the sweep for "
              f"{n_drop} of {n_models} models.")
        print("%% mean over the models of a dataset of the running maximum "
              "over the sweep (the balls are nested), after the base PGD "
              f"(clipped) at eps = {max(EPS_LIST)}")
        for ds, L in lab:
            if ds in runmax:
                bp = [mean(base(ds, a, max(EPS_LIST), "pgd")) for a, _ in acts]
                rm = np.mean(runmax[ds], axis=0)
                print(f"  {ds:4s}: base eps={max(EPS_LIST)} {np.mean(bp):.4f} | "
                      + "  ".join(f"eps={x:g} {v:.4f}"
                                  for x, v in zip(WB_UNB_EPS, rm))
                      + f"  ({len(runmax[ds])} models)")

    # ---- C. SPSA: {budget} x {loss} ----
    C4 = [(bn, ls) for bn in budgets for ls in WB_SPSA_LOSSES]
    print("\n%% ---- SPSA, {budget} x {loss}, L0. Budgets (steps x directions "
          "x 2 = queries per flow): " + "; ".join(
              f"{n} {st} x {nd} x 2 = {2 * st * nd}"
              for n, (st, nd) in budgets.items())
          + f". HopSkipJump's schedule: {hq} queries per flow (HopSkipJump / "
          f"base SPSA = {hq / (2 * SPSA_STEPS * SPSA_NDIRS):.0f}). Loss "
          "\"prob\" = the base SPSA (-log of the clipped softmax probability); "
          "\"logit\" = cross-entropy from the logits (log-softmax). Same "
          "directions for both losses of a budget.")
    if spsa is None:
        print("%% not run yet")
    else:
        n_chk = 0
        for r in spsa[(spsa.budget == "base") & (spsa.loss == "prob")].itertuples():
            q = b[(b.dataset == r.dataset) & (b.activation == r.activation)
                  & (b.seed == r.seed) & (b.level == r.level)
                  & np.isclose(b.eps, r.eps) & (b.attack == "spsa")]
            if len(q) == 1:
                n_chk += 1
                if (int(q.n_hit.iloc[0]), int(q.n_correct.iloc[0])) != \
                        (int(r.n_hit), int(r.n_correct)):
                    _stop(f"wbspsa {r.dataset}-{r.activation}-{r.seed} "
                          f"eps={r.eps}: base budget with the prob loss gives "
                          f"{r.n_hit}/{r.n_correct}, the base SPSA row "
                          f"{int(q.n_hit.iloc[0])}/{int(q.n_correct.iloc[0])}."
                          f" They must be equal.")
        print("  check: the (base budget, prob loss) rows equal the base SPSA "
              f"rows of Table 4 (flows evaded and flows correct): yes, {n_chk} "
              "rows")
        hdr = " | ".join(f"{bn}/{ls}" for bn, ls in C4)
        for eps in WB_SPSA_EPS:
            print(f"%% eps = {eps}: mean+-SD over seeds: {hdr} | PGD")
            for ds, L in lab:
                for a, A in acts:
                    for bn, ls in C4:
                        nc = cell(ds, a, eps, bn, ls, "n_correct")
                        nb = base(ds, a, eps, "pgd", col="n_correct")
                        if np.isfinite(nc).all() and np.isfinite(nb).all() \
                                and not np.array_equal(nc, nb):
                            _stop(f"wbspsa {ds} {a} {bn}/{ls}: n_correct {nc} "
                                  f"differs from the base rows {nb}: not the "
                                  f"same models/pool.")
                    print(f"  {ds:4s} {A} & " + " & ".join(
                        _ms(cell(ds, a, eps, bn, ls)) for bn, ls in C4)
                        + f" & {_ms(base(ds, a, eps, 'pgd'))} \\\\")
            print(f"%% eps = {eps}: per seed (seeds {SEEDS}), 4 decimals")
            for ds, L in lab:
                for a, A in acts:
                    print(f"  {ds:4s} {A}: " + " | ".join(
                        f"{bn}/{ls} {f4(cell(ds, a, eps, bn, ls))}"
                        for bn, ls in C4)
                        + f" | PGD {f4(base(ds, a, eps, 'pgd'))}")
            print(f"%% eps = {eps}: gap to PGD (PGD mean minus cell mean)")
            for ds, L in lab:
                worst = {k: [] for k in C4}
                for a, A in acts:
                    p = mean(base(ds, a, eps, "pgd"))
                    for k in C4:
                        worst[k].append(p - mean(cell(ds, a, eps, *k)))
                    print(f"  {ds:4s} {A}: " + " | ".join(
                        f"{k[0]}/{k[1]} {worst[k][-1]:+.3f}" for k in C4))
                print(f"  {ds:4s} largest over the activations: " + " | ".join(
                    f"{k[0]}/{k[1]} {max(worst[k]):+.3f}" for k in C4))
            if "matched" in budgets:
                print(f"%% eps = {eps}: what closes the base SPSA's shortfall "
                      "to PGD (differences of the seed means). shortfall = PGD "
                      "- base/prob; loss alone = base/logit - base/prob; "
                      "budget alone = matched/prob - base/prob; both = "
                      "matched/logit - base/prob; left = PGD - matched/logit. "
                      "Also: budget under the logit loss = matched/logit - "
                      "base/logit; loss at the matched budget = matched/logit "
                      "- matched/prob")
                for ds, L in lab:
                    for a, A in acts:
                        p = mean(base(ds, a, eps, "pgd"))
                        v = {k: mean(cell(ds, a, eps, *k)) for k in C4}
                        bp, bl = v[("base", "prob")], v[("base", "logit")]
                        mp, ml = v[("matched", "prob")], v[("matched", "logit")]
                        print(f"  {ds:4s} {A}: shortfall {p - bp:+.3f} | loss "
                              f"alone {bl - bp:+.3f} | budget alone "
                              f"{mp - bp:+.3f} | both {ml - bp:+.3f} | left "
                              f"{p - ml:+.3f} | budget under logit "
                              f"{ml - bl:+.3f} | loss at matched "
                              f"{ml - mp:+.3f}")
            print(f"%% eps = {eps}: per seed (seeds {SEEDS}): flows correctly "
                  "classified | of those, flows whose clean loss is exactly "
                  "0.0 in float32, prob loss and logit loss | flows never "
                  "moved by SPSA (perturbation exactly zero after the "
                  "attack), by cell")
            cnt = lambda v: " ".join("-" if x != x else f"{x:.0f}" for x in v)
            for ds, L in lab:
                for a, A in acts:
                    k0 = C4[0]
                    print(f"  {ds:4s} {A}: correct "
                          f"{cnt(cell(ds, a, eps, *k0, col='n_correct'))} | "
                          "zero clean loss: prob "
                          f"{cnt(cell(ds, a, eps, *k0, col='n_zero_loss'))}, "
                          "logit "
                          f"{cnt(cell(ds, a, eps, *k0, col='n_zero_logit'))} | "
                          "never moved: " + " | ".join(
                              f"{bn}/{ls} "
                              f"{cnt(cell(ds, a, eps, bn, ls, 'n_unmoved'))}"
                              for bn, ls in C4))
                tot = {k: np.nansum([cell(ds, a, eps, *k, col="n_unmoved")
                                     for a, _ in acts]) for k in C4}
                ok = np.nansum([cell(ds, a, eps, *C4[0], col="n_correct")
                                for a, _ in acts])
                zl = np.nansum([cell(ds, a, eps, *C4[0], col="n_zero_loss")
                                for a, _ in acts])
                zg = np.nansum([cell(ds, a, eps, *C4[0], col="n_zero_logit")
                                for a, _ in acts])
                print(f"  {ds:4s} pooled over the models: correct {ok:.0f} | "
                      f"zero clean loss: prob {zl:.0f}, logit {zg:.0f} | "
                      "never moved: " + " | ".join(
                          f"{k[0]}/{k[1]} {tot[k]:.0f}" for k in C4))

    # ---- D. PGD convergence in iterations ----
    print("\n%% ---- PGD convergence in iterations (step eps / "
          f"{FGSM_EPS_STEP_DIV}, one random start = the base row's start; "
          f"the base setting is {PGD_BIM_MAX_ITER} iterations)")
    if itr is None:
        print("%% not run yet")
    else:
        n_chk = 0
        for r in itr[itr.max_iter == PGD_BIM_MAX_ITER].itertuples():
            q = b[(b.dataset == r.dataset) & (b.activation == r.activation)
                  & (b.seed == r.seed) & (b.level == r.level)
                  & np.isclose(b.eps, r.eps) & (b.attack == "pgd")]
            if len(q) == 1:
                n_chk += 1
                if (int(q.n_hit.iloc[0]), int(q.n_correct.iloc[0])) != \
                        (int(r.n_hit), int(r.n_correct)):
                    _stop(f"wbiter {r.dataset}-{r.activation}-{r.seed} "
                          f"L{r.level} eps={r.eps}: {PGD_BIM_MAX_ITER} "
                          f"iterations give {r.n_hit}/{r.n_correct}, the base "
                          f"PGD row {int(q.n_hit.iloc[0])}/"
                          f"{int(q.n_correct.iloc[0])}. They must be equal.")
        print(f"  check: the {PGD_BIM_MAX_ITER}-iteration rows equal the base "
              f"PGD rows (flows evaded and flows correct): yes, {n_chk} rows")
        print("%% pooled over the models of a dataset: flows evaded / flows "
              "correctly classified (rate), by number of iterations")
        for ds, L in lab:
            for lv in WB_ITER_LEVELS:
                for eps in WB_ITER_EPS:
                    r = itr[(itr.dataset == ds) & (itr.level == lv)
                            & np.isclose(itr.eps, eps)]
                    g = r.groupby("max_iter").agg(
                        hit=("n_hit", "sum"), ok=("n_correct", "sum"),
                        n=("seed", "size"), asr=("asr", "mean"))
                    print(f"  {ds:4s} L{lv} eps={eps}: " + "  ".join(
                        f"{it}: {int(x.hit)}/{int(x.ok)} ({x.asr:.4f})"
                        for it, x in g.iterrows())
                        + f"   [mean of the per-model rates in brackets; "
                        f"{int(g.n.min())} models]")
        print("%% by activation: mean over seeds of the ASR, by number of "
              "iterations")
        for ds, L in lab:
            for lv in WB_ITER_LEVELS:
                for eps in WB_ITER_EPS:
                    for a, A in acts:
                        r = itr[(itr.dataset == ds) & (itr.level == lv)
                                & np.isclose(itr.eps, eps)
                                & (itr.activation == a)]
                        g = r.groupby("max_iter").asr.mean()
                        print(f"  {ds:4s} L{lv} eps={eps} {A}: " + "  ".join(
                            f"{it}: {x:.4f}" for it, x in g.items()))

    print("\n%% ---- Figure 1 (fig:fig1): Model 4, L0, eps = "
          f"{WB_FIG_EPS}; mean+-SD over seeds of the plotted bars")
    val = {}
    for ds, L in lab:
        for bar in WB_FIG_BARS:
            val[(ds, bar)] = [
                matched(ds, a, WB_FIG_EPS) if bar == "spsa_matched"
                else base(ds, a, WB_FIG_EPS, bar) for a, _ in acts]
            # "spsa" is the base row = the (base, prob) cell (verified above).
        for i, (a, A) in enumerate(acts):
            print(f"  {ds:4s} {A} & " + " & ".join(
                _ms(val[(ds, bar)][i]) for bar in WB_FIG_BARS) + " \\\\")
    wb_fig1(val, acts, lab)


STAGES.append(("wbbb", wb_stage, wb_summarize))


# ======== stage wbunbm: code ========
WBM_COLS = ["dataset", "activation", "seed", "level", "eps", "loss", "ret",
            "max_iter", "eps_step", "asr", "n_correct", "n_hit", "n_nonfinite",
            "n_zero_grad_start", "n_zero_grad_end", "n_lost_at_end",
            "n_diff_wbunb", "attack_seed", "elapsed_s"]
WBM_KEY = WBM_COLS[:7]
WBM_LOSSES = ["ce", "margin"]
WBM_RETURNS = ["last", "best"]


def wbm_loss(logits, y, loss):
    """The quantity the attack ASCENDS, reduced by the mean over the batch as ART reduces the cross-entropy."""
    if loss == "ce":
        return F.cross_entropy(logits, y, reduction="mean")
    if loss == "margin":
        fy = logits.gather(1, y[:, None]).squeeze(1)
        fo = logits.gather(1, (1 - y)[:, None]).squeeze(1)
        return (fo - fy).mean()
    _stop(f"wbunbm: unknown loss {loss!r}")


def wbm_pgd(model, X, y, eps, step, n_iter, loss, seed):
    """L_inf PGD on the UNCLIPPED model with the loss as an argument."""
    model.eval()
    _reseed(seed)
    X0 = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y, dtype=torch.long, device=DEVICE)
    noise = np.random.uniform(-eps, eps, size=X.shape).astype(np.float32)
    x = X0 + torch.tensor(noise, device=DEVICE)

    def grad_and_adv(xc):
        xc = xc.detach().requires_grad_(True)
        logits = model(xc)
        g, = torch.autograd.grad(wbm_loss(logits, yt, loss), xc)
        g = torch.nan_to_num(g, nan=0.0)
        return g.detach(), (logits.detach().argmax(1) != yt)

    g, adv_now = grad_and_adv(x)
    zero_start = int((g == 0).all(1).sum())
    found = adv_now.clone()
    best = x.clone()
    for _ in range(n_iter):
        x = (x + step * g.sign()).detach()
        x = X0 + torch.clamp(x - X0, -eps, eps)
        g, adv_now = grad_and_adv(x)
        new = adv_now & ~found
        best[new] = x[new]
        found |= adv_now
    zero_end = int((g == 0).all(1).sum())
    lost = int((found & ~adv_now).sum())
    best = torch.where(found[:, None], best, x)
    return (x.cpu().numpy(), best.cpu().numpy(), zero_start, zero_end, lost)


def stage_wbunbm(c):
    have = stage_have("wbunbm", WBM_KEY)
    ident = dict(dataset=c.ds, activation=c.act, seed=c.seed, level=0)
    model = _F32(c.model).to(DEVICE)
    base_hits = npz_load(stage_npz("wbbb", c))     # wbunb's per-flow hits
    msg = []
    for eps in WB_UNB_EPS:
        step = eps * WB_UNB_STEP_MULT / PGD_BIM_MAX_ITER
        s = _seed_of("wbbb", "unbounded", c.ds, c.act, c.seed, eps)  # wbunb's seed
        for loss in WBM_LOSSES:
            todo = [r for r in WBM_RETURNS
                    if _k(c.ds, c.act, c.seed, 0, round(eps, 3), loss, r)
                    not in have]
            if not todo:
                continue
            t0 = time.time()
            last, best, z0, z1, lost = _timed(lambda: wbm_pgd(
                model, c.Xe, c.ye, eps, step, PGD_BIM_MAX_ITER, loss, s))
            el = round(time.time() - t0, 1)
            for ret, adv in (("last", last), ("best", best)):
                if ret not in todo:
                    continue
                h, bad = wb_score(c, adv, eps, None,
                                  f"wbunbm eps={eps} {loss}/{ret}")
                bk = f"unb_e{eps}_hit"
                ndiff = (int((np.asarray(base_hits[bk], bool) != h).sum())
                         if (loss, ret) == ("ce", "last") and bk in base_hits
                         else -1)
                stage_append("wbunbm", WBM_COLS, dict(
                    ident, eps=round(eps, 3), loss=loss, ret=ret,
                    max_iter=PGD_BIM_MAX_ITER, eps_step=round(step, 6),
                    asr=round(float(h.sum() / c.n_ok), 4), n_correct=c.n_ok,
                    n_hit=int(h.sum()), n_nonfinite=bad,
                    n_zero_grad_start=z0, n_zero_grad_end=z1,
                    n_lost_at_end=lost, n_diff_wbunb=ndiff, attack_seed=s,
                    elapsed_s=el))
                msg.append(f"e={eps:g} {loss}/{ret} {h.sum() / c.n_ok:.4f}")
    if msg:
        print("    wbunbm: " + "  ".join(msg), flush=True)


def summarize_wbunbm():
    d = stage_read("wbunbm", WBM_KEY)
    if d is None:
        print("%% stage wbunbm: not run yet"); return
    bp = os.path.join(RES_DIR, "wbunb_final.csv")   # read only: stage_read
    base = pd.read_csv(bp) if os.path.isfile(bp) else None   # would rewrite it
    sd = lambda v: float(np.std(v, ddof=SD_DDOF)) if len(v) > 1 else 0.0
    cells = [(l, r) for l in WBM_LOSSES for r in WBM_RETURNS]
    print("%% mean ASR over seeds (SD). 'ART' = table wbunb; (ce,last) is the "
          "control and must agree with it.")
    print("%% ds   act   eps    | ART           | " + " | ".join(
        f"{l}/{r:4s}       " for l, r in cells))
    for ds in DATASETS:
        for a in ACTIVATIONS:
            for eps in WB_UNB_EPS:
                q = d[(d.dataset == ds) & (d.activation == a)
                      & (d.eps.round(3) == round(eps, 3))]
                if q.empty:
                    continue
                parts = []
                if base is not None:
                    b = base[(base.dataset == ds) & (base.activation == a)
                             & (base.eps.round(3) == round(eps, 3))].asr
                    parts.append(f"{b.mean():.3f}({sd(b):.3f})")
                else:
                    parts.append("    n/a     ")
                for l, r in cells:
                    v = q[(q.loss == l) & (q.ret == r)].asr
                    parts.append(f"{v.mean():.3f}({sd(v):.3f}) n{len(v)}")
                print(f"  {ds:4s} {a:4s} {eps:6g} | " + " | ".join(parts))
    print("\n%% diagnostics, mean over seeds of the per-flow counts "
          "(pool flows): zero gradient at start / at end, lost at the end")
    for ds in DATASETS:
        for a in ACTIVATIONS:
            for eps in WB_UNB_EPS:
                q = d[(d.dataset == ds) & (d.activation == a)
                      & (d.eps.round(3) == round(eps, 3)) & (d.ret == "last")]
                if q.empty:
                    continue
                txt = []
                for l in WBM_LOSSES:
                    r = q[q.loss == l]
                    txt.append(f"{l}: {r.n_zero_grad_start.mean():.1f} / "
                               f"{r.n_zero_grad_end.mean():.1f} / "
                               f"{r.n_lost_at_end.mean():.1f}")
                print(f"  {ds:4s} {a:4s} {eps:6g} | " + " | ".join(txt))
    ctl = d[(d.loss == "ce") & (d.ret == "last")]
    if len(ctl):
        print(f"\n%% CONTROL (ce, last) against wbunb, flow by flow: "
              f"{int((ctl.n_diff_wbunb > 0).sum())} of {len(ctl)} model-budget "
              f"cells differ, {int(ctl.n_diff_wbunb.clip(lower=0).sum())} "
              f"flows in all; {int((ctl.n_diff_wbunb < 0).sum())} cells had no "
              f"saved wbunb hits to compare")
    print("\n%% models whose ASR falls somewhere along the sweep, per cell")
    for l, r in cells:
        q = d[(d.loss == l) & (d.ret == r)]
        n = drop = 0
        for _, g in q.groupby(["dataset", "activation", "seed"]):
            if len(g) != len(WB_UNB_EPS):
                continue
            v = g.sort_values("eps").asr.to_numpy()
            n += 1; drop += int((np.diff(v) < 0).any())
        print(f"  {l}/{r}: {drop} of {n} models")


STAGES.append(("wbunbm", stage_wbunbm, summarize_wbunbm))


# ======== stage restarts: code ========
RS_TABLE = "rsrestarts"
RS_COLS = ["dataset", "activation", "seed", "level", "eps", "attack", "iters",
           "restarts", "asr", "n_correct", "n_hit", "n_free", "mask_native",
           "frozen_moved", "clean_f1", "attack_seed", "elapsed_s"]
RS_KEY = RS_COLS[:8]
RS_ACT_LABEL = (("relu", "ReLU"), ("elu", "ELU "), ("tanh", "Tanh"))
RS_DS_LABEL = (("unsw", "UNSW-NB15"), ("cic", "CSE-CIC-IDS2018"))


def rs_pgd_iters(clf, X, eps, mask, iters):
    from art.attacks.evasion import ProjectedGradientDescent
    atk = ProjectedGradientDescent(clf, eps=eps, eps_step=eps / FGSM_EPS_STEP_DIV,
                                   max_iter=iters,
                                   num_random_init=PGD_RANDOM_INIT,
                                   batch_size=BATCH_SIZE, verbose=False)
    gk = {} if mask is None else {"mask": np.broadcast_to(
        mask, X.shape).astype(np.float32).copy()}
    return atk.generate(x=X.astype(np.float32), **gk)


def rs_apgd(clf, X, eps, mask=None):
    from art.attacks.evasion import AutoProjectedGradientDescent
    import art.attacks.evasion.auto_projected_gradient_descent as _apgd_mod
    atk = AutoProjectedGradientDescent(**_kw(
        AutoProjectedGradientDescent, estimator=clf, norm=np.inf, eps=eps,
        eps_step=eps / RS_APGD_STEP_DIV, max_iter=RS_APGD_MAX_ITER,
        nb_random_init=RS_APGD_INIT, loss_type="cross_entropy", targeted=False,
        batch_size=BATCH_SIZE, verbose=False))
    x = X.astype(np.float32)
    if mask is None:
        adv = atk.generate(x=x)
    else:
        m = np.asarray(mask, np.float32).reshape(-1)
        orig = _apgd_mod.random_sphere

        def _masked_sphere(nb_points, nb_dims, radius, norm):
            return orig(nb_points, nb_dims, radius, norm) * m[None, :]

        _apgd_mod.random_sphere = _masked_sphere
        try:
            adv = atk.generate(x=x, mask=m)
        finally:
            _apgd_mod.random_sphere = orig
    return np.where(np.isfinite(adv).all(1, keepdims=True), adv, x).astype(
        np.float32)


def rs_score(c, adv, eps, mask, what):
    """Per-flow success of one finished attack."""
    _, h, viol = hits_of(c.model, adv, c.Xe, c.ye, DEVICE, eps, mask)
    if viol:
        _stop(f"capability violation: {viol} frozen coordinates moved "
              f"(stage restarts, {c.part}, eps={eps}, {what})")
    return h


def rs_sweep(c, lv, eps, attack, seed0):
    """max(RS_RESTARTS) single-start runs of one attack."""
    mask, n = c.masks[lv], max(RS_RESTARTS)
    each = np.zeros((n, len(c.ye)), bool)
    seeds, moved = np.zeros(n, np.int64), np.zeros(n, np.int64)
    secs, t0 = np.zeros(n), time.time()
    for i in range(n):
        s = (seed0 if (i == 0 and seed0 is not None) else
             _seed_of("restarts", c.ds, c.act, c.seed, lv, eps, attack, i))
        _reseed(s)
        if attack == "pgd":
            adv = _timed(lambda: _ref_q(c.clf, c.Xe, eps, mask, "pgd"))
        else:
            raw = _timed(lambda: rs_apgd(c.clf, c.Xe, eps, mask))
            if mask is not None:
                moved[i] = int((np.abs(raw - c.Xe)[:, mask == 0] > 1e-6).sum())
                if moved[i]:
                    _stop(f"capability violation: APGD moved {moved[i]} frozen "
                          f"coordinates (stage restarts, {c.part}, L{lv}, "
                          f"eps={eps}, restart {i})")
            adv = _apply(c.Xe, raw, mask)
        each[i] = rs_score(c, adv, eps, mask, f"L{lv} {attack} restart {i}")
        seeds[i], secs[i] = s, time.time() - t0
    return each, seeds, moved, secs


def rs_check_settings():
    if 1 not in RS_RESTARTS or RS_PROBE_RESTARTS not in RS_RESTARTS:
        _stop("stage restarts: RS_RESTARTS must contain 1 and "
              "RS_PROBE_RESTARTS.")
    if [e for e in RS_EPS_LIST if e not in EPS_LIST]:
        _stop("stage restarts: RS_EPS_LIST must be budgets of EPS_LIST (the "
              "one-restart PGD is the base measurement).")


def rs_stage(c):
    rs_check_settings()
    have = stage_have(RS_TABLE, RS_KEY)
    it, ia = PGD_BIM_MAX_ITER, RS_APGD_MAX_ITER
    path = stage_npz(RS_TABLE, c)
    saved = npz_load(path)
    if saved and not (np.array_equal(saved.get("pool"), c.pool) and
                      np.array_equal(saved.get("correct"), c.correct)):
        _stop(f"{path} was written for a different pool or model. Move it "
              f"away and re-run.")
    saved["pool"], saved["correct"] = c.pool, c.correct
    saved["restart_budgets"] = np.asarray(RS_RESTARTS)
    bpath = os.path.join(RES_DIR, "flows",
                         f"{c.ds}_d{DEPTH}_f{FEATS}_{c.act}_s{c.seed}.npz")
    basef = npz_load(bpath)

    def same_as_base(key, h):
        # ONE value per measurement.
        if key not in basef:
            _stop(f"{bpath} has no '{key}': the base run of {c.part} did not "
                  f"save its per-flow results. Run the part without "
                  f"RERUN_STAGES first.")
        if not np.array_equal(np.asarray(basef[key]).astype(bool), h):
            _stop(f"stage restarts, {c.part}: '{key}' differs from the base "
                  f"run flow by flow ({int(h.sum())} against "
                  f"{int(np.asarray(basef[key]).sum())} evaded). The same "
                  f"seed must give the same result.")

    for eps in RS_EPS_LIST:
        e3 = round(eps, 3)
        for lv in LEVELS:
            mask = c.masks[lv]
            tag = f"L{lv}_e{eps}"
            row0 = dict(dataset=c.ds, activation=c.act, seed=c.seed, level=lv,
                        eps=e3, n_correct=c.n_ok,
                        n_free=FEATS if mask is None else int(mask.sum()),
                        clean_f1=round(float(c.f1), 4))

            def missing(attack, iters, budgets):
                return [R for R in budgets if _k(c.ds, c.act, c.seed, lv, e3,
                                                 attack, iters, R) not in have]

            def put(attack, iters, R, h, **kw):
                stage_append(RS_TABLE, RS_COLS, dict(
                    row0, attack=attack, iters=iters, restarts=R,
                    asr=round(float(h.sum() / c.n_ok), 4), n_hit=int(h.sum()),
                    **kw))
                have.add(_k(c.ds, c.act, c.seed, lv, e3, attack, iters, R))

            if not (missing("bim", it, [1]) or missing("pgd", it, RS_RESTARTS)
                    or missing("pgd", RS_PROBE_ITERS, [1])
                    or missing("apgd", ia, RS_RESTARTS)
                    or missing("union3", 0, RS_RESTARTS)):
                print(f"    L{lv} eps={eps}: done", flush=True)
                continue

            if missing("bim", it, [1]) or f"{tag}_bim" not in saved:
                s = _seed_of(c.ds, DEPTH, FEATS, c.act, c.seed, lv, eps, "bim")
                _reseed(s); t0 = time.time()
                hb = rs_score(c, _timed(lambda: _ref_q(c.clf, c.Xe, eps, mask,
                                                       "bim")), eps, mask,
                              f"L{lv} bim")
                same_as_base(f"{tag}_bim", hb)
                saved[f"{tag}_bim"] = hb
                npz_save(path, saved)
                for R in missing("bim", it, [1]):
                    put("bim", it, R, hb, mask_native=1, frozen_moved=0,
                        attack_seed=s, elapsed_s=round(time.time() - t0, 1))
            hb = np.asarray(saved[f"{tag}_bim"]).astype(bool)

            # -- PGD and APGD, max(RS_RESTARTS) single-start runs each.
            for a, iters in (("pgd", it), ("apgd", ia)):
                if missing(a, iters, RS_RESTARTS) or f"{tag}_{a}_each" not in saved:
                    s0 = (_seed_of(c.ds, DEPTH, FEATS, c.act, c.seed, lv, eps,
                                   "pgd") if a == "pgd" else None)
                    each, seeds, moved, secs = rs_sweep(c, lv, eps, a, s0)
                    if a == "pgd":
                        same_as_base(f"{tag}_pgd", each[0])
                    saved[f"{tag}_{a}_each"] = each
                    saved[f"{tag}_{a}_seeds"] = seeds
                    saved[f"{tag}_{a}_moved"] = moved
                    npz_save(path, saved)
                    for R in missing(a, iters, RS_RESTARTS):
                        put(a, iters, R, each[:R].any(0),
                            mask_native=1,
                            frozen_moved=int(moved[:R].sum()),
                            attack_seed=int(seeds[0]),
                            elapsed_s=round(float(secs[R - 1]), 1))
                    print(f"    L{lv} eps={eps} {a.upper():4s} restarts "
                          + " ".join(f"{R}:{each[:R].any(0).sum() / c.n_ok:.4f}"
                                     for R in RS_RESTARTS), flush=True)
            pe = np.asarray(saved[f"{tag}_pgd_each"]).astype(bool)
            ae = np.asarray(saved[f"{tag}_apgd_each"]).astype(bool)

            if missing("pgd", RS_PROBE_ITERS, [1]):
                s = _seed_of(c.ds, DEPTH, FEATS, c.act, c.seed, lv, eps, "pgd")
                _reseed(s); t0 = time.time()
                hp = rs_score(c, _timed(lambda: rs_pgd_iters(
                    c.clf, c.Xe, eps, mask, RS_PROBE_ITERS)), eps, mask,
                    f"L{lv} pgd {RS_PROBE_ITERS} iterations")
                saved[f"{tag}_pgd{RS_PROBE_ITERS}"] = hp
                npz_save(path, saved)
                put("pgd", RS_PROBE_ITERS, 1, hp, mask_native=1, frozen_moved=0,
                    attack_seed=s, elapsed_s=round(time.time() - t0, 1))
                print(f"    L{lv} eps={eps} PGD {RS_PROBE_ITERS} iterations x 1: "
                      f"{hp.sum() / c.n_ok:.4f}", flush=True)

            # -- per-flow union of the three attacks at each budget.
            for R in missing("union3", 0, RS_RESTARTS):
                put("union3", 0, R, pe[:R].any(0) | ae[:R].any(0) | hb,
                    mask_native=1, frozen_moved=0,
                    attack_seed=0, elapsed_s=0)


def rs_ms4(v):
    v = np.asarray(v, float)
    return f"{v.mean():.4f}$\\pm${v.std(ddof=SD_DDOF):.4f}"


def rs_summarize():
    rs_check_settings()
    d = stage_read(RS_TABLE, RS_KEY)
    if d is None:
        print("%% stage restarts: not run yet"); return
    T = {}
    for x in d.itertuples(index=False):
        T[(x.dataset, x.activation, int(x.seed), int(x.level),
           round(float(x.eps), 3), x.attack, int(x.iters), int(x.restarts))] = x
    print(f"%% {len(d)} rows from "
          f"{d.groupby(['dataset', 'activation', 'seed']).ngroups} of "
          f"{len(DATASETS) * len(ACTIVATIONS) * len(SEEDS)} parts. Model 4, "
          f"{FEATS} features, the SAVED networks of the base run.")
    print(f"%% PGD and BIM: {PGD_BIM_MAX_ITER} iterations, step eps/"
          f"{FGSM_EPS_STEP_DIV}, capability mask applied inside the attack. "
          f"APGD-CE: {RS_APGD_MAX_ITER} iterations, step eps/{RS_APGD_STEP_DIV};"
          f" capability mask applied inside the search, random start "
          f"included (ART 1.20.1 masks only the gradient step; rs_apgd "
          f"confines the start as well).")
    print("%% A restart is one single-start run; restart i is seeded from its "
          "index; the value at R restarts is the per-flow union of restarts "
          "1..R. Restart 1 of PGD is the base PGD measurement itself.")
    for eps in RS_EPS_LIST:
        rs_summarize_eps(T, round(eps, 3))
    rs_check_base(d)


def rs_check_base(d):
    """The one-restart PGD and the BIM rows must be the base rows."""
    if not os.path.exists(OUT_FINAL):
        print("%% base table not found: the comparison with it was skipped")
        return
    b = pd.read_csv(OUT_FINAL)
    b = b[(b.depth == DEPTH) & (b.features == FEATS)]
    bk = {(x.dataset, x.activation, int(x.seed), int(x.level),
           round(float(x.eps), 3), x.attack): (int(x.n_hit), int(x.n_correct))
          for x in b.itertuples(index=False)}
    m = d[(d.restarts == 1) & (d.iters == PGD_BIM_MAX_ITER)
          & d.attack.isin(["pgd", "bim"])]
    n, bad = 0, []
    for x in m.itertuples(index=False):
        k = (x.dataset, x.activation, int(x.seed), int(x.level),
             round(float(x.eps), 3), x.attack)
        if k in bk:
            n += 1
            if bk[k] != (int(x.n_hit), int(x.n_correct)):
                bad.append(k)
    if bad:
        _stop(f"stage restarts: {len(bad)} one-restart rows differ from the "
              f"base table, e.g. {bad[:5]}. The files come from different runs.")
    print(f"\n%% ---- consistency: the one-restart PGD and the BIM of this "
          f"stage are the base rows (Table 4 / levels): {n} of {len(m)} rows "
          f"compared, all identical")


def rs_summarize_eps(T, e3):
    it, ia = PGD_BIM_MAX_ITER, RS_APGD_MAX_ITER
    r1, rN = min(RS_RESTARTS), max(RS_RESTARTS)

    def g(ds, a, s, lv, atk, iters, R, f="asr"):
        x = T.get((ds, a, s, lv, e3, atk, iters, R))
        return None if x is None else float(getattr(x, f))

    def worst(ds, a, s, lv, R):
        v = [g(ds, a, s, lv, "pgd", it, R), g(ds, a, s, lv, "apgd", ia, R),
             g(ds, a, s, lv, "bim", it, 1)]
        return None if None in v else max(v)

    def union3(ds, a, s, lv, R):
        return g(ds, a, s, lv, "union3", 0, R)

    def pgd(ds, a, s, lv, R):
        return g(ds, a, s, lv, "pgd", it, R)

    def apgd(ds, a, s, lv, R):
        return g(ds, a, s, lv, "apgd", ia, R)

    def col(fn, ds, a, lv, R):
        v = [fn(ds, a, s, lv, R) for s in SEEDS]
        return None if None in v else np.array(v, float)

    def table(fn):
        for ds, L in RS_DS_LABEL:
            print(f"\\multicolumn{{5}}{{l}}{{\\textit{{{L}}}}} \\\\")
            for lv in LEVELS:
                for i, (a, A) in enumerate(RS_ACT_LABEL):
                    head = f"L{lv}" if i == 0 else "  "
                    v1, vN = col(fn, ds, a, lv, r1), col(fn, ds, a, lv, rN)
                    if v1 is None or vN is None:
                        print(f"{head} & {A} & missing & missing & missing \\\\")
                        continue
                    dl = round(vN.mean(), 4) - round(v1.mean(), 4) + 0.0
                    print(f"{head} & {A} & {rs_ms4(v1)} & {rs_ms4(vN)} & "
                          f"${dl:+.4f}$ \\\\")
            print("\\hline")

    print(f"\n%% ---- tab:restarts (eps = {e3}): Level & Activation & "
          f"{r1} restart & {rN} restarts & Delta. Worst case = per-seed "
          f"maximum of the three rates (PGD at R restarts, APGD-CE at R "
          f"restarts, BIM); mean, sample SD over seeds {SEEDS}; Delta = "
          f"difference of the two tabulated means.")
    table(worst)
    print("\n%% ---- the same table with the per-flow UNION of the three "
          "attacks in place of the maximum of their rates (not the published "
          "convention; an adversary who picks the best attack per flow)")
    table(union3)

    ok = {ds: all(col(fn, ds, a, lv, R) is not None
                  for fn in (worst, union3) for a, _ in RS_ACT_LABEL
                  for lv in LEVELS for R in RS_RESTARTS)
          and all(g(ds, a, s, lv, "pgd", RS_PROBE_ITERS, 1) is not None
                  for a, _ in RS_ACT_LABEL for s in SEEDS for lv in LEVELS)
          for ds, _ in RS_DS_LABEL}
    for ds, _ in RS_DS_LABEL:
        if not ok[ds]:
            print(f"%% {ds}: parts are missing; the numbers below are printed "
                  f"for complete datasets only")
    dss = [ds for ds, _ in RS_DS_LABEL if ok[ds]]
    cells = [(ds, lv, a) for ds in dss for lv in LEVELS for a, _ in RS_ACT_LABEL]
    if not cells:
        return

    print(f"\n%% ---- every restart budget {RS_RESTARTS}: mean over seeds "
          f"(worst case | PGD alone | APGD-CE alone | BIM)")
    for ds, lv, a in cells:
        f4 = lambda fn: " ".join(f"{col(fn, ds, a, lv, R).mean():.4f}"
                                 for R in RS_RESTARTS)
        bim = np.mean([g(ds, a, s, lv, "bim", it, 1) for s in SEEDS])
        print(f"  {ds:4s} L{lv} {a:4s}: {f4(worst)} | {f4(pgd)} | {f4(apgd)} "
              f"| {bim:.4f}")

    print(f"\n%% ---- per seed {SEEDS}: worst case at {r1} restart -> at {rN} "
          f"restarts")
    for ds, lv, a in cells:
        v1, vN = col(worst, ds, a, lv, r1), col(worst, ds, a, lv, rN)
        print(f"  {ds:4s} L{lv} {a:4s}: " + "  ".join(
            f"s{s} {x:.4f}->{y:.4f}" for s, x, y in zip(SEEDS, v1, vN)))

    print(f"\n%% ---- text (sec:wbbb): change of the worst case from {r1} to "
          f"{rN} restarts. Delta of the tabulated means [difference of the "
          f"unrounded means] largest single-seed change")
    dl = {}
    for ds, lv, a in cells:
        v1, vN = col(worst, ds, a, lv, r1), col(worst, ds, a, lv, rN)
        dl[(ds, lv, a)] = (round(vN.mean(), 4) - round(v1.mean(), 4) + 0.0,
                           vN.mean() - v1.mean(), float((vN - v1).max()),
                           float((vN - v1).min()))
        print(f"  {ds:4s} L{lv} {a:4s}: {dl[(ds, lv, a)][0]:+.4f} "
              f"[{dl[(ds, lv, a)][1]:+.5f}] {dl[(ds, lv, a)][2]:+.4f}")
    neg = [k for k in dl if dl[k][3] < 0]
    print("  every single-seed change is >= 0, as the union over restarts "
          "requires" if not neg else
          f"  !! NEGATIVE change in {neg}: the union over restarts cannot "
          f"decrease; do not use these rows")
    for ds in dss:
        print(f"  {ds}: ReLU rises by " + ", ".join(
            f"{dl[(ds, lv, 'relu')][0]:+.4f} at L{lv}" for lv in LEVELS)
            + "; largest change for ELU and Tanh at any level: "
            + f"{max(dl[(ds, lv, a)][0] for lv in LEVELS for a in ('elu', 'tanh')):+.4f}")
        k = max((k for k in dl if k[0] == ds), key=lambda k: dl[k][1])
        print(f"  {ds}: largest movement anywhere: {dl[k][0]:+.4f} "
              f"[{dl[k][1]:+.5f}] at L{k[1]} {k[2]}")
        for lv in LEVELS:
            v1 = col(worst, ds, "relu", lv, r1)
            vN = col(worst, ds, "relu", lv, rN)
            print(f"  {ds}: ReLU seed SD at L{lv}: {v1.std(ddof=SD_DDOF):.4f} "
                  f"at {r1} restart -> {vN.std(ddof=SD_DDOF):.4f} at {rN}")

    pr = RS_PROBE_RESTARTS
    print(f"\n%% ---- text (sec:wbbb): iterations against restarts, PGD alone. "
          f"{it} iterations x 1 restart | {RS_PROBE_ITERS} x 1 (same random "
          f"start) | {it} x {pr} restarts | {it} x {rN} restarts")
    print("%% mean over seeds:")
    for ds, lv, a in cells:
        pv = np.mean([g(ds, a, s, lv, "pgd", RS_PROBE_ITERS, 1) for s in SEEDS])
        print(f"  {ds:4s} L{lv} {a:4s}: {col(pgd, ds, a, lv, 1).mean():.4f} | "
              f"{pv:.4f} | {col(pgd, ds, a, lv, pr).mean():.4f} | "
              f"{col(pgd, ds, a, lv, rN).mean():.4f}")
    print("%% each model:")
    for ds, lv, a in cells:
        for s in SEEDS:
            print(f"  {ds:4s} L{lv} {a:4s} s{s:<2d}: {pgd(ds, a, s, lv, 1):.4f} | "
                  f"{g(ds, a, s, lv, 'pgd', RS_PROBE_ITERS, 1):.4f} | "
                  f"{pgd(ds, a, s, lv, pr):.4f} | {pgd(ds, a, s, lv, rN):.4f}")

    # ---- flow counts pooled over the models of a dataset ----
    print("\n%% ---- text (sec:wbbb, the convergence checks): PGD alone, "
          "flows evaded of the correctly classified flows, pooled over the "
          "models of a dataset")
    for ds in dss:
        for lv in LEVELS:
            ms = [(a, s) for a, _ in RS_ACT_LABEL for s in SEEDS]
            tot = int(sum(g(ds, a, s, lv, "pgd", it, 1, "n_correct")
                          for a, s in ms))
            cnt = {R: int(sum(g(ds, a, s, lv, "pgd", it, R, "n_hit")
                              for a, s in ms)) for R in RS_RESTARTS}
            p4 = int(sum(g(ds, a, s, lv, "pgd", RS_PROBE_ITERS, 1, "n_hit")
                         for a, s in ms))
            print(f"  {ds:4s} L{lv}: of {tot} flows, " + ", ".join(
                f"{cnt[R]} at {R}" for R in RS_RESTARTS)
                + f" restart(s) (gain {cnt[rN] - cnt[r1]} flows, "
                + f"{cnt[r1] / tot:.4f} -> {cnt[rN] / tot:.4f}); "
                + f"{RS_PROBE_ITERS} iterations x 1 restart: {p4} "
                + f"({p4 - cnt[r1]:+d} flows against {it} x 1); one flow = "
                + f"{1 / tot:.5f}")

    # ---- CIC L1 against L0, and the L2 block ----
    print("\n%% ---- text (sec:wbbb): L1 against L0, worst case, seed by seed")
    for ds in dss:
        for a, _ in RS_ACT_LABEL:
            same = [abs(worst(ds, a, s, 1, R) - worst(ds, a, s, 0, R)) < 5e-5
                    for s in SEEDS for R in (r1, rN)]
            m0 = [col(worst, ds, a, 0, R).mean() for R in (r1, rN)]
            m1 = [col(worst, ds, a, 1, R).mean() for R in (r1, rN)]
            print(f"  {ds:4s} {a:4s}: L1 equals L0 in {sum(same)} of "
                  f"{len(same)} seed-by-budget values; means L0 "
                  f"{m0[0]:.4f}/{m0[1]:.4f}, L1 {m1[0]:.4f}/{m1[1]:.4f}")
    print(f"%% L2, worst case at {rN} restarts, per seed "
          f"(and its change from {r1} restart):")
    for ds in dss:
        for a, _ in RS_ACT_LABEL:
            v1, vN = col(worst, ds, a, 2, r1), col(worst, ds, a, 2, rN)
            print(f"  {ds:4s} L2 {a:4s}: mean {vN.mean():.4f}; " + ", ".join(
                f"seed {s}: {y:.4f} ({y - x:+.4f})"
                for s, x, y in zip(SEEDS, v1, vN)))

    # ---- PGD minus APGD, both masked inside ----
    print(f"\n%% ---- text (sec:wbbb): natively masked PGD minus APGD-CE "
          f"(also masked inside its search), both at {rN} restarts. Largest single "
          f"model | smallest single model | largest activation mean | mean of "
          f"the models | frozen coordinates APGD had moved before the "
          f"restriction")
    for ds in dss:
        for lv in LEVELS:
            gap = {(a, s): pgd(ds, a, s, lv, rN) - apgd(ds, a, s, lv, rN)
                   for a, _ in RS_ACT_LABEL for s in SEEDS}
            am = {a: np.mean([gap[(a, s)] for s in SEEDS])
                  for a, _ in RS_ACT_LABEL}
            kmax = max(gap, key=gap.get)
            amax = max(am, key=am.get)
            mv = int(sum(g(ds, a, s, lv, "apgd", ia, rN, "frozen_moved")
                         for a, s in gap))
            print(f"  {ds:4s} L{lv}: {gap[kmax]:+.4f} ({kmax[0]} s{kmax[1]}) | "
                  f"{min(gap.values()):+.4f} | {am[amax]:+.4f} ({amax}) | "
                  f"{np.mean(list(gap.values())):+.4f} | {mv}")


STAGES.append(("restarts", rs_stage, rs_summarize))

# ======== stage capa: code ========
CA_QCOLS = ["dataset", "activation", "seed", "variant", "eps", "attack", "asr",
            "n_correct", "n_hit", "n_free", "n_int_free", "attack_seed",
            "elapsed_s"]
CA_QKEY = CA_QCOLS[:6]
CA_VARIANTS = ["five", "seven", "whole"]
CA_ECOLS = ["dataset", "activation", "seed", "level", "eps", "n_free", "grid",
            "n_points", "n_distinct", "n_correct", "n_hit_exact",
            "n_hit_corner", "asr_exact", "asr_corner", "n_hit_union",
            "n_union_not_exact", "n_clipped", "n_hit_box", "n_hit_corner_box",
            "n_union_not_box", "elapsed_s"]
CA_EKEY = CA_ECOLS[:5]
CA_SCOLS = ["dataset", "activation", "seed", "level", "eps", "clipped",
            "attack", "asr", "n_correct", "n_hit", "n_free", "attack_seed",
            "elapsed_s"]
CA_SKEY = CA_SCOLS[:7]
CA_UCOLS = ["dataset", "feature", "rank", "free_l1", "free_l2", "integer_all",
            "integer_mal", "scale", "mean", "mal_median", "mal_p01", "mal_p99",
            "mal_distinct", "n_mal", "data_hash", "source"]
CA_UKEY = CA_UCOLS[:2]
CA_LAB = (("cic", "CSE-CIC-IDS2018"), ("unsw", "UNSW-NB15"))
_CA_SC = {}


# ---- the scaler of the frozen data (isolated) ----
def ca_scaler_path(ds):
    return os.path.join(DATA_DIR, f"{ds}_scaler.npz")


def ca_scaler_build(ds, data, path):
    """THE ONLY STEP OF THIS STAGE THAT READS THE KAGGLE FILES."""
    Xtr_f, ytr_f, Xte_f, yte_f, order_f, names_f = data
    n_te = len(Xte_f)
    cap = {"sc": None, "int": None, "raw_te": None}

    class _CaScaler(StandardScaler):
        def fit(self, X, y=None, sample_weight=None):
            cap["sc"] = self
            return super().fit(X, y, sample_weight)

        def transform(self, X, copy=None):
            A = np.asarray(X)
            isint = np.array([bool(np.all(np.mod(A[:, j], 1.0) == 0.0))
                              for j in range(A.shape[1])])
            cap["int"] = isint if cap["int"] is None else (cap["int"] & isint)
            if len(A) == n_te:
                cap["raw_te"] = A
            return super().transform(X, copy=copy)

    print(f"  {path} not found: recomputing the scaler of the frozen {ds} "
          f"data from the Kaggle files (once) ...", flush=True)
    g = globals()
    orig = g["StandardScaler"]
    g["StandardScaler"] = _CaScaler
    try:
        make_deterministic(SEED)
        Xtr, ytr, Xte, yte, order, names = load(ds)
    finally:
        g["StandardScaler"] = orig
    sc, raw_te = cap["sc"], cap["raw_te"]
    if sc is None or raw_te is None or cap["int"] is None:
        _stop(f"{ds}: load() did not fit and apply a StandardScaler as "
              f"expected; the scaler could not be captured.")
    h_new, h_old = _sha(Xtr, Xte, ytr, yte), _sha(Xtr_f, Xte_f, ytr_f, yte_f)
    if ([str(n) for n in names] != [str(n) for n in names_f]
            or not np.array_equal(np.asarray(order), np.asarray(order_f))
            or h_new != h_old):
        _stop(f"{ds}: replaying load() on the Kaggle files does not reproduce "
              f"{DATA_DIR}/{ds}.npz (hash {h_new}, frozen {h_old}). The scaler "
              f"of the frozen data therefore cannot be established and "
              f"nothing was written. The Kaggle files or the library "
              f"versions differ from those of the data step.")
    mal = raw_te[np.asarray(yte) != 0]          # malicious test flows, raw units
    nf = mal.shape[1]
    med = np.array([float(np.median(mal[:, j])) for j in range(nf)])
    p01 = np.array([float(np.percentile(mal[:, j], 1)) for j in range(nf)])
    p99 = np.array([float(np.percentile(mal[:, j], 99)) for j in range(nf)])
    nun = np.array([len(np.unique(mal[:, j])) for j in range(nf)], np.int64)
    imal = np.array([bool(np.all(np.isclose(mal[:, j], np.round(mal[:, j]))))
                     for j in range(nf)])
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = path[:-4] + f".tmp{os.getpid()}.npz"
    np.savez(tmp, names=np.asarray([str(n) for n in names], dtype=str),
             mean=np.asarray(sc.mean_, np.float64),
             scale=np.asarray(sc.scale_, np.float64),
             is_int_all=cap["int"], mal_median=med, mal_p01=p01, mal_p99=p99,
             mal_distinct=nun, mal_is_int=imal, n_mal=np.asarray(len(mal)),
             data_hash=np.asarray(h_old), source=np.asarray("kaggle"))
    os.replace(tmp, path)
    print(f"  scaler of {ds} saved in {path} (replay hash {h_new} = frozen "
          f"data hash)", flush=True)


def ca_pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def ca_scaler(ds, data):
    if ds in _CA_SC:
        return _CA_SC[ds]
    p = ca_scaler_path(ds)
    lock, told = p + ".lock", False
    while not os.path.exists(p):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                with open(lock) as fh:
                    owner = int(fh.read().strip() or 0)
                age = time.time() - os.path.getmtime(lock)
            except (OSError, ValueError):
                continue                          # lock just went away
            if ((owner and not ca_pid_alive(owner))
                    or (not owner and age > CA_LOCK_STALE_S)):
                try:
                    os.remove(lock)               # its owner is gone
                except OSError:
                    pass
                continue
            if not told:
                print(f"  waiting: another part is building {p} ...",
                      flush=True)
                told = True
            time.sleep(CA_LOCK_POLL_S)
            continue
        try:
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            if not os.path.exists(p):
                ca_scaler_build(ds, data, p)
        finally:
            try:
                os.remove(lock)
            except OSError:
                pass
    z = np.load(p, allow_pickle=False)
    h = _sha(data[0], data[2], data[1], data[3])
    if ([str(x) for x in z["names"]] != [str(x) for x in data[5]]
            or str(z["data_hash"]) != h):
        _stop(f"{p} does not belong to the frozen data in {DATA_DIR}/{ds}.npz "
              f"(data hash {h}, file says {str(z['data_hash'])}). Move it "
              f"away and re-run: it is rebuilt from the Kaggle files.")
    sc = {k: z[k] for k in ("mean", "scale", "mal_median", "mal_p01",
                            "mal_p99", "mal_distinct")}
    sc["is_int_all"] = z["is_int_all"].astype(bool)
    sc["mal_is_int"] = z["mal_is_int"].astype(bool)
    sc["n_mal"], sc["data_hash"] = int(z["n_mal"]), h
    sc["source"] = str(z["source"])
    if sc["source"] != "kaggle":
        print(f"  !! {p} is marked '{sc['source']}', not recomputed from the "
              f"Kaggle files: raw-unit results are NOT valid.", flush=True)
    _CA_SC[ds] = sc
    return sc


# ---- shared helpers ----
def ca_base_flows(c):
    """Per-flow success sets the BASE run saved for this part's Model 4."""
    p = os.path.join(RES_DIR, "flows",
                     f"{c.ds}_d{DEPTH}_f{FEATS}_{c.act}_s{c.seed}.npz")
    if not os.path.exists(p):
        _stop(f"{p} is missing: the base run of this part has not saved its "
              f"per-flow outputs. Run the part without RERUN_STAGES first.")
    return dict(np.load(p))


def ca_attack(c, a, eps, mask, s, clf=None):
    """One of the four attacks exactly as the base run calls it, seeded s."""
    clf = c.clf if clf is None else clf
    sub = c.substitute() if a == "transfer" else None    # loaded BEFORE seeding
    fn = {"pgd": lambda: _ref_q(clf, c.Xe, eps, mask, "pgd"),
          "bim": lambda: _ref_q(clf, c.Xe, eps, mask, "bim"),
          "spsa": lambda: a_spsa(c.model, c.Xe, c.ye, eps, s, mask),
          "transfer": lambda: a_transfer_saved(sub, c.Xe, eps, mask)}[a]
    _reseed(s)
    adv = _timed(fn)
    if not np.isfinite(adv).all():
        _stop(f"{a} returned non-finite values ({c.part} eps={eps}); nothing "
              f"was written for this cell.")
    return adv


def ca_hits(c, adv, eps, mask, what):
    _, h, viol = hits_of(c.model, adv, c.Xe, c.ye, DEVICE, eps, mask)
    if viol:
        _stop(f"capability violation: {viol} frozen coordinates moved "
              f"({c.part} eps={eps} {what})")
    return h


def ca_quantise(X, adv, scale, quant_idx):
    d = (adv.astype(np.float64) - X.astype(np.float64))
    if len(quant_idx):
        qi = np.asarray(quant_idx, dtype=int)
        raw = d[:, qi] * scale[None, qi]
        d[:, qi] = np.trunc(raw) / scale[None, qi]
    return (X.astype(np.float32) + d.astype(np.float32))


# ---- part 1: whole-unit quantisation (CIC) ----
def ca_part_quant(c):
    sc = ca_scaler(c.ds, c.data)
    scale_k, int_k = sc["scale"][c.cols], sc["is_int_all"][c.cols]
    free7 = c.masks[2]
    free5 = free7.copy()
    for nm in CA_FIVE_DROP:
        if nm not in c.names:
            _stop(f"{nm} is not among the {FEATS} selected features.")
        free5[c.names.index(nm)] = 0.0
    if int(free5.sum()) != CA_FIVE_FREE:
        _stop(f"the FIVE set has {int(free5.sum())} features, expected "
              f"{CA_FIVE_FREE}.")
    q_idx = [i for i in range(len(c.names)) if free7[i] > 0 and int_k[i]]
    n7 = int(free7.sum())
    have = stage_have("caquant", CA_QKEY)
    base = ca_base_flows(c)
    npz = stage_npz("caquant", c)
    saved = npz_load(npz)
    kb = dict(dataset=c.ds, activation=c.act, seed=c.seed, n_correct=c.n_ok)
    nfree = {"five": CA_FIVE_FREE, "seven": n7, "whole": n7}
    nint = {"five": 0, "seven": 0, "whole": len(q_idx)}

    def put(v, eps, a, h, s, el):
        key = _k(c.ds, c.act, c.seed, v, round(eps, 3), a)
        if key not in have:
            stage_append("caquant", CA_QCOLS, dict(
                kb, variant=v, eps=round(eps, 3), attack=a,
                asr=round(float(h.sum() / c.n_ok), 4), n_hit=int(h.sum()),
                n_free=nfree[v], n_int_free=nint[v], attack_seed=s,
                elapsed_s=round(el, 1)))
            have.add(key)

    for eps in EPS_LIST:
        if all(_k(c.ds, c.act, c.seed, v, round(eps, 3), "union") in have
               for v in CA_VARIANTS):
            continue
        union = {v: np.zeros(len(c.Xe), bool) for v in CA_VARIANTS}
        for a in CA_ATTACKS:
            t0 = time.time()
            s7 = _seed_of(c.ds, DEPTH, FEATS, c.act, c.seed, 2, eps, a)
            adv7 = ca_attack(c, a, eps, free7, s7)
            h7 = ca_hits(c, adv7, eps, free7, f"seven {a}")
            if not np.array_equal(h7, base[f"L2_e{eps}_{a}"]):
                _stop(f"{c.part} eps={eps} {a}: the regenerated L2 attack "
                      f"evades {int(h7.sum())} flows, the base run recorded "
                      f"{int(base[f'L2_e{eps}_{a}'].sum())} (or other flows). "
                      f"The base attack is not being reproduced; nothing was "
                      f"written for this cell.")
            t1 = time.time()
            advw = ca_quantise(c.Xe, adv7, scale_k, q_idx)
            hw = ca_hits(c, advw, eps, free7, f"whole {a}")
            t2 = time.time()
            s5 = _seed_of("capa", "five", c.ds, c.act, c.seed, eps, a)
            adv5 = ca_attack(c, a, eps, free5, s5)
            h5 = ca_hits(c, adv5, eps, free5, f"five {a}")
            t3 = time.time()
            for v, h, s, el in (("seven", h7, s7, t1 - t0),
                                ("whole", hw, s7, t2 - t1),
                                ("five", h5, s5, t3 - t2)):
                saved[f"{v}_e{eps}_{a}"] = h
                union[v] |= h
                put(v, eps, a, h, s, el)
        if not np.array_equal(union["seven"], base[f"L2_e{eps}_union"]):
            _stop(f"{c.part} eps={eps}: regenerated L2 union differs from the "
                  f"base run's.")
        for v in CA_VARIANTS:
            saved[f"{v}_e{eps}_union"] = union[v]
        npz_save(npz, saved)
        for v in CA_VARIANTS:
            put(v, eps, "union", union[v], 0, 0.0)
        print(f"    quant eps={eps}: " + "  ".join(
            f"{v} {union[v].sum() / c.n_ok:.4f} ({int(union[v].sum())})"
            for v in CA_VARIANTS) + f"  of {c.n_ok} flows", flush=True)


# ---- part 2: large-budget extension ----
def ca_clf_unclipped(c):
    from art.estimators.classification import PyTorchClassifier
    return PyTorchClassifier(
        model=_F32(c.model).to(DEVICE), loss=nn.CrossEntropyLoss(),
        input_shape=(c.Xtr.shape[1],), nb_classes=2, clip_values=None,
        device_type=("gpu" if str(DEVICE).startswith("cuda") else "cpu"))


def ca_part_sweep(c):
    have = stage_have("casweep", CA_SKEY)
    npz = stage_npz("casweep", c)
    saved = npz_load(npz)
    clf_u = None
    kb = dict(dataset=c.ds, activation=c.act, seed=c.seed, n_correct=c.n_ok)
    for lv in LEVELS:
        mask = c.masks[lv]
        nfree = len(c.names) if mask is None else int(mask.sum())
        for clipped, eps_list, attacks in (
                (1, CA_SWEEP_EPS, CA_ATTACKS),
                (0, CA_UNCLIPPED_EPS, CA_UNCLIPPED_ATTACKS)):
            msg = []
            for eps in eps_list:
                e = round(eps, 3)
                if _k(c.ds, c.act, c.seed, lv, e, clipped, "union") in have:
                    continue
                if not clipped and clf_u is None:
                    clf_u = ca_clf_unclipped(c)
                tag = f"L{lv}_e{eps}_{'c' if clipped else 'u'}"
                union = np.zeros(len(c.Xe), bool)
                for a in attacks:
                    t0 = time.time()
                    s = _seed_of("capa", "sweep", c.ds, c.act, c.seed, lv, eps,
                                 clipped, a)
                    adv = ca_attack(c, a, eps, mask, s,
                                    clf=(c.clf if clipped else clf_u))
                    h = ca_hits(c, adv, eps, mask, f"sweep L{lv} {a}")
                    saved[f"{tag}_{a}"] = h
                    union |= h
                    key = _k(c.ds, c.act, c.seed, lv, e, clipped, a)
                    if key not in have:
                        stage_append("casweep", CA_SCOLS, dict(
                            kb, level=lv, eps=e, clipped=clipped, attack=a,
                            asr=round(float(h.sum() / c.n_ok), 4),
                            n_hit=int(h.sum()), n_free=nfree, attack_seed=s,
                            elapsed_s=round(time.time() - t0, 1)))
                        have.add(key)
                saved[f"{tag}_union"] = union
                npz_save(npz, saved)
                stage_append("casweep", CA_SCOLS, dict(
                    kb, level=lv, eps=e, clipped=clipped, attack="union",
                    asr=round(float(union.sum() / c.n_ok), 4),
                    n_hit=int(union.sum()), n_free=nfree, attack_seed=0,
                    elapsed_s=0))
                have.add(_k(c.ds, c.act, c.seed, lv, e, clipped, "union"))
                msg.append(f"eps={eps} {union.sum() / c.n_ok:.3f}")
            if msg:
                print(f"    sweep L{lv} {'clipped' if clipped else 'UNCLIPPED'}"
                      f" union: " + "  ".join(msg), flush=True)


# ---- part 3: the L2 worst case by enumeration (UNSW) ----
def ca_offsets(eps, k):
    """Offsets of the enumeration on the k free coordinates."""
    import itertools
    axes = np.meshgrid(*([np.linspace(-eps, eps, CA_ENUM_GRID)] * k),
                       indexing="ij")
    lat = np.stack([a.reshape(-1) for a in axes], 1).astype(np.float32)
    cor = np.array(list(itertools.product((-eps, eps), repeat=k)), np.float32)
    return lat, cor


def ca_enumerate(c, eps, free):
    """Evaluate every lattice point and every corner of each flow's box."""
    lat, cor = ca_offsets(eps, len(free))
    offs = np.concatenate([lat, cor], 0)
    lo, hi = float(c.Xtr.min()), float(c.Xtr.max())     # as _art_clf
    n, d = c.Xe.shape
    out = {k: np.zeros(n, bool) for k in ("exact", "corner", "box",
                                          "boxcorner")}
    n_clip = 0

    def wrong_at(p, y):
        return (_probs(c.model, p.reshape(-1, d), DEVICE).argmax(1)
                .reshape(p.shape[0], p.shape[1]) != y[:, None])

    for i in range(0, n, CA_ENUM_CHUNK):
        sl = slice(i, i + CA_ENUM_CHUNK)
        xb, yb, ok = c.Xe[sl], c.ye[sl], c.correct[sl]
        pts = np.repeat(xb[:, None, :], len(offs), 1).astype(np.float32)
        pts[:, :, free] += offs[None]
        cl = np.clip(pts, lo, hi)
        changed = (cl != pts).any(2)
        n_clip += int((changed & ok[:, None]).sum())
        w = wrong_at(cl, yb)
        wb = wrong_at(pts, yb) if changed.any() else w
        out["exact"][sl] = w.any(1) & ok
        out["corner"][sl] = w[:, len(lat):].any(1) & ok
        out["box"][sl] = wb.any(1) & ok
        out["boxcorner"][sl] = wb[:, len(lat):].any(1) & ok
    return out, n_clip, len(offs), len(np.unique(offs, axis=0))


def ca_part_enum(c):
    free = np.where(np.asarray(c.masks[2]) > 0)[0]
    if len(free) > CA_ENUM_MAX_FREE:
        return                      # too many free features to enumerate
    have = stage_have("caenum", CA_EKEY)
    base = ca_base_flows(c)
    sweep = npz_load(stage_npz("casweep", c)) if CA_RUN_SWEEP else {}
    npz = stage_npz("caenum", c)
    saved = npz_load(npz)
    for eps in EPS_LIST + (CA_SWEEP_EPS if CA_RUN_SWEEP else []):
        e = round(eps, 3)
        if _k(c.ds, c.act, c.seed, 2, e) in have:
            continue
        ukey = f"L2_e{eps}_union" if eps in EPS_LIST else f"L2_e{eps}_c_union"
        src = base if eps in EPS_LIST else sweep
        if ukey not in src:
            _stop(f"{c.part}: the attack union at L2 eps={eps} is not saved; "
                  f"the enumeration is compared with it.")
        union = src[ukey].astype(bool)
        t0 = time.time()
        en, n_clip, n_pts, n_dist = ca_enumerate(c, eps, free)
        exact, corner = en["exact"], en["corner"]
        for k_, v_ in en.items():
            saved[f"{k_}_e{eps}"] = v_
        npz_save(npz, saved)
        stage_append("caenum", CA_ECOLS, dict(
            dataset=c.ds, activation=c.act, seed=c.seed, level=2, eps=e,
            n_free=len(free), grid=CA_ENUM_GRID, n_points=n_pts,
            n_distinct=n_dist, n_correct=c.n_ok,
            n_hit_exact=int(exact.sum()), n_hit_corner=int(corner.sum()),
            asr_exact=round(float(exact.sum() / c.n_ok), 4),
            asr_corner=round(float(corner.sum() / c.n_ok), 4),
            n_hit_union=int(union.sum()),
            n_union_not_exact=int((union & ~exact).sum()),
            n_clipped=n_clip, n_hit_box=int(en["box"].sum()),
            n_hit_corner_box=int(en["boxcorner"].sum()),
            n_union_not_box=int((union & ~en["box"]).sum()),
            elapsed_s=round(time.time() - t0, 1)))
        have.add(_k(c.ds, c.act, c.seed, 2, e))
        print(f"    enumeration L2 eps={eps}: {int(exact.sum())} of {c.n_ok} "
              f"flows ({n_pts} points, {n_dist} distinct); corners only "
              f"{int(corner.sum())}; without the clip {int(en['box'].sum())}"
              f"; attack union {int(union.sum())}", flush=True)


# ---- part 4: raw units of the selected features (data only) ----
def ca_part_units(c):
    if (c.act, c.seed) != (ACTIVATIONS[0], SEEDS[0]):
        return                      # a property of the dataset: one part writes it
    have = stage_have("caunits", CA_UKEY)
    if all(_k(c.ds, n) in have for n in c.names):
        return
    sc = ca_scaler(c.ds, c.data)
    for r, (i, n) in enumerate(zip(c.cols, c.names)):
        if _k(c.ds, n) in have:
            continue
        stage_append("caunits", CA_UCOLS, dict(
            dataset=c.ds, feature=n, rank=r + 1,
            free_l1=int(c.masks[1][r] > 0), free_l2=int(c.masks[2][r] > 0),
            integer_all=int(sc["is_int_all"][i]),
            integer_mal=int(sc["mal_is_int"][i]),
            scale=float(sc["scale"][i]), mean=float(sc["mean"][i]),
            mal_median=float(sc["mal_median"][i]),
            mal_p01=float(sc["mal_p01"][i]), mal_p99=float(sc["mal_p99"][i]),
            mal_distinct=int(sc["mal_distinct"][i]), n_mal=sc["n_mal"],
            data_hash=sc["data_hash"], source=sc["source"]))


def ca_stage(c):
    if c.ds == "cic":
        ca_part_quant(c)
    if CA_RUN_SWEEP:
        ca_part_sweep(c)
    ca_part_enum(c)
    ca_part_units(c)


def ca_ms(v, nd=3):
    v = np.asarray(v, float)
    if not len(v):
        return "(not run)"
    sd = v.std(ddof=SD_DDOF) if len(v) > SD_DDOF else float("nan")
    return f"{v.mean():.{nd}f}$\\pm${sd:.{nd}f}"


def ca_span(s):
    if not len(s):
        return "(not run)"
    lo, hi = int(s.min()), int(s.max())
    return f"{lo}" if lo == hi else f"{lo} to {hi}"


def ca_pick(d, **kw):
    m = np.ones(len(d), bool)
    for col, v in kw.items():
        m &= (np.isclose(d[col].astype(float), v) if col == "eps"
              else (d[col] == v).values)
    return d[m]


def ca_seedavg(r, col="asr"):
    """Activation-averaged value per seed (the quantity whose mean and SD the capability tables report)."""
    return [r[r.seed == s][col].mean() for s in SEEDS if (r.seed == s).any()]


def ca_base_rows():
    """The base rows of Model 4 (30 features), one per measurement, with the unrounded rate n_hit / n_correct."""
    files = sorted(f for f in glob.glob(os.path.join(RES_DIR, "attacks_*.csv"))
                   if f != OUT_FINAL)
    if not files:
        return None
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d = d.drop_duplicates(_KEY, keep="last")
    d = d[(d.depth == DEPTH) & (d.features == FEATS)].reset_index(drop=True)
    d["rate"] = d.n_hit / d.n_correct
    return d


def ca_sum_levels(b):
    n_models = len(ACTIVATIONS) * len(SEEDS)
    print("\n%% ---- tab:levels: union of PGD, BIM, SPSA, "
          "transfer, activation-averaged; the SAME base rows the base summary "
          "prints")
    U = lambda ds, lv, e: ca_pick(b, dataset=ds, level=lv, eps=e,
                                  attack="union")
    for ds, L in CA_LAB:
        n = len(U(ds, 2, EPS_LIST[0]))
        if n != n_models:
            print(f"%% !! {ds}: {n} of {n_models} models present -- the "
                  f"numbers below are INCOMPLETE")
        print(f"\\multicolumn{{5}}{{l}}{{\\textit{{{L}}}}} \\\\")
        for e in EPS_LIST:
            v = {lv: ca_seedavg(U(ds, lv, e)) for lv in LEVELS}
            print(f"$\\epsilon={e}$ & " + " & ".join(ca_ms(v[lv])
                                                      for lv in LEVELS)
                  + f" & {100 * np.mean(v[2]) / np.mean(v[0]):.0f}\\% \\\\")
        print("\\hline")
    print("\n%% ---- in-text, lines 971-986 and 1097-1121 (from the same rows)")
    for ds, L in CA_LAB:
        for e in EPS_LIST:
            v = {lv: float(np.mean(ca_seedavg(U(ds, lv, e)))) for lv in LEVELS}
            print(f"  {ds:4s} eps={e}: L0 {v[0]:.3f} -> L1 {v[1]:.3f} -> L2 "
                  f"{v[2]:.3f}; L2/L0 = {100 * v[2] / v[0]:.1f}%; "
                  f"L1/L0 = {100 * v[1] / v[0]:.1f}%")
    print("  worst case by activation, mean$\\pm$SD over seeds "
          "(the entries of tab:levels_act):")
    for ds, L in CA_LAB:
        for lv in (0, 2):
            for e in EPS_LIST:
                r = U(ds, lv, e)
                per = {a: r[r.activation == a].sort_values("seed").asr
                       for a in ACTIVATIONS}
                mean = [per[a].mean() for a in ACTIVATIONS]
                print(f"  {ds:4s} L{lv} eps={e}: " + ", ".join(
                    f"{a} {ca_ms(per[a])}" for a in ACTIVATIONS)
                    + f"; span of the means {max(mean) - min(mean):.3f}; "
                    f"smallest mean {min(mean):.3f}")
    for ds, L in CA_LAB:
        print(f"  {ds}: unpaired binomial standard error of one rate on the "
              f"{N_EVAL_BY_DS[ds]}-flow pool, at most 0.5/sqrt(n) = "
              f"{0.5 / np.sqrt(N_EVAL_BY_DS[ds]):.3f} (arithmetic, no model)")
    print("  L1 against L0 (line 1115: 'identical to four decimals' on "
          "CSE-CIC-IDS2018), mean rate over the models:")
    for ds, L in CA_LAB:
        for e in EPS_LIST:
            r0 = U(ds, 0, e).sort_values(["activation", "seed"])
            r1 = U(ds, 1, e).sort_values(["activation", "seed"])
            same = int((r0.n_hit.values == r1.n_hit.values).sum()) \
                if len(r0) == len(r1) else -1
            print(f"  {ds:4s} eps={e}: L0 {r0.rate.mean():.4f}  L1 "
                  f"{r1.rate.mean():.4f}; same flow count in {same} of "
                  f"{len(r0)} models")
    print("\n%% ---- in-text, lines 993-997: evaded flows per model at L2 "
          "(union)")
    for ds, L in CA_LAB:
        for e in EPS_LIST:
            r = U(ds, 2, e)
            print(f"  {ds:4s} eps={e}: {ca_span(r.n_hit)} flows per model, "
                  f"of {ca_span(r.n_correct)} correctly-classified; all "
                  f"models {int(r.n_hit.sum())} of {int(r.n_correct.sum())}")


def ca_sum_unionmax(b):
    print("\n%% ---- in-text, lines 1051-1058: union against the maximum of "
          "the per-attack rates, Model 4 MLPs, cells = (level, budget, "
          "activation, seed)")
    for ds, L in CA_LAB:
        d = b[(b.dataset == ds) & b.level.isin(LEVELS)]
        d = d[np.isin(d.eps.round(3), [round(e, 3) for e in EPS_LIST])]
        key = ["level", "eps", "activation", "seed"]
        u = d[d.attack == "union"].set_index(key)
        m = (d[d.attack.isin(CA_ATTACKS)].groupby(key).n_hit.max()
             .reindex(u.index))
        diff = (u.n_hit - m) / u.n_correct
        if not len(diff):
            print(f"  {ds}: no rows")
            continue
        print(f"  {L}: {len(diff)} cells; union above the maximum of rates "
              f"in {int((diff > 0).sum())}, below it in "
              f"{int((diff < 0).sum())}; difference on average "
              f"{diff.mean():.4f}, at most {diff.max():.4f}")
    print("%% the CNN and adversarial-training cells "
          "figures of lines 1059-1063 come from the stages that own those "
          "models")


def ca_sum_units_text(un):
    print("\n%% ---- in-text, lines 988-992 and 1012-1018: what a budget buys "
          "in raw units on the CSE-CIC-IDS2018 source-side features "
          "(scaler of the frozen data; no model)")
    if un is None or not (un.dataset == "cic").any():
        print("%% caunits (cic): not run yet")
        return
    g = un[(un.dataset == "cic") & (un.free_l2 == 1)]
    if (g.source != "kaggle").any():
        print("%% !! the scaler file is a STAND-IN, not recomputed from the "
              "Kaggle files: these raw-unit numbers are NOT valid")
    for r in g.itertuples():
        print(f"  {r.feature}: SD {r.scale:.4g} raw units; "
              + "; ".join(f"eps={e} buys {e * r.scale:.2f}" for e in EPS_LIST)
              + f"; {int(r.mal_distinct)} distinct values among malicious "
              f"test flows; raw column integer-valued: "
              f"{'yes' if r.integer_all else 'no'}")
    for e in EPS_LIST:
        imm = [r.feature for r in g.itertuples()
               if r.integer_all and e * r.scale < 1.0]
        print(f"  eps={e}: {len(g) - len(imm)} of {len(g)} free coordinates "
              f"can move by a whole unit; cannot: {imm if imm else 'none'}")


def ca_sum_quant(b, un):
    print("\n%% ---- tab:quant: CSE-CIC-IDS2018, L2 "
          "worst case (union of the four attacks) under three capability "
          "sets; mean$\\pm$SD over seeds of the activation-averaged value")
    q = stage_read("caquant", CA_QKEY)
    if q is None:
        print("%% caquant: not run yet")
        return
    # the SEVEN column is the base L2 row.
    rep = q[q.variant == "seven"].copy()
    rep["eps"] = rep.eps.round(3)
    bb = b[(b.dataset == "cic") & (b.level == 2)].copy()
    bb["eps"] = bb.eps.round(3)
    m = rep.merge(bb, on=["activation", "seed", "eps", "attack"],
                  suffixes=("", "_base"))
    bad = int(((m.n_hit != m.n_hit_base)
               | (m.n_correct != m.n_correct_base)).sum())
    if bad or len(m) != len(rep):
        _stop(f"tab:quant: {bad} regenerated SEVEN rows differ from the base "
              f"L2 rows ({len(rep) - len(m)} have no base row).")
    print(f"%% SEVEN = the base L2 rows; the stage regenerated all "
          f"{len(rep)} of them identically (flow counts equal)")
    u = q[q.attack == "union"]
    n_models = len(ACTIVATIONS) * len(SEEDS)
    n = len(ca_pick(u, variant="whole", eps=EPS_LIST[0]))
    if n != n_models:
        print(f"%% !! {n} of {n_models} models present -- INCOMPLETE")
    T = {}
    for e in EPS_LIST:
        T[e] = {"five": ca_seedavg(ca_pick(u, variant="five", eps=e)),
                "seven": ca_seedavg(ca_pick(b, dataset="cic", level=2, eps=e,
                                            attack="union")),
                "whole": ca_seedavg(ca_pick(u, variant="whole", eps=e))}
        print(f"$\\epsilon={e}$ & " + " & ".join(
            ca_ms(T[e][v]) for v in CA_VARIANTS) + " \\\\")
    print("\\hline")
    print("%% in-text, lines 1006-1013 (and 1300-1303, 1327-1328)")
    for e in EPS_LIST:
        f, s, w = (float(np.mean(T[e][v])) for v in CA_VARIANTS)
        l0 = float(np.mean(ca_seedavg(ca_pick(b, dataset="cic", level=0, eps=e,
                                              attack="union"))))
        print(f"  eps={e}: freeing the two flag features raises the L2 worst "
              f"case by {s - f:.3f} ({f:.3f} -> {s:.3f}); whole raw units "
              f"lower it from {s:.3f} to {w:.3f} (by {s - w:.3f}, "
              f"{(100 * (s - w) / (s - f)) if s != f else float('nan'):.0f}% "
              f"of that increase); whole-unit L2 / L0 = {100 * w / l0:.1f}%")
    for e in EPS_LIST:
        r = ca_pick(u, variant="whole", eps=e)
        print(f"  eps={e}: the whole-unit adversary evades "
              f"{ca_span(r.n_hit)} flows per model, of {ca_span(r.n_correct)}"
              f"; all models {int(r.n_hit.sum())} of {int(r.n_correct.sum())}"
              f" ({ca_span(r.n_int_free)} of {ca_span(r.n_free)} free "
              f"coordinates integer-valued)")
    ca_sum_units_text(un)


def ca_sum_enum(b):
    print("\n%% ---- in-text, lines 1066-1095: the L2 worst case by "
          "enumeration (no attack), 4 decimals")
    en = stage_read("caenum", CA_EKEY)
    if en is None:
        print("%% caenum: not run yet")
        return
    n_models = len(ACTIVATIONS) * len(SEEDS)
    for ds in [d for d, _ in CA_LAB if (en.dataset == d).any()]:
        g_ds = en[en.dataset == ds]
        r0 = g_ds.iloc[0]
        print(f"  {ds}: {int(r0.n_free)} free features; {int(r0.grid)} values "
              f"per axis -> {int(r0.grid) ** int(r0.n_free)} lattice points + "
              f"{2 ** int(r0.n_free)} corners = {int(r0.n_points)} points "
              f"evaluated per flow, {int(r0.n_distinct)} of them distinct "
              f"(the lattice runs from -eps to +eps, so it contains the "
              f"corners)")
        for hx, hc, hu, title in (
                ("n_hit_exact", "n_hit_corner", "n_union_not_exact",
                 "points clipped to the range of the training data, as the "
                 "classifier of PGD, BIM and transfer clips"),
                ("n_hit_box", "n_hit_corner_box", "n_union_not_box",
                 "the plain box, no clip")):
            print(f"  -- {title}")
            worst, worst_lc = None, None
            for e in EPS_LIST:
                g = ca_pick(g_ds, eps=e)
                bu = ca_pick(b, dataset=ds, level=2, eps=e, attack="union")
                m = g.merge(bu[["activation", "seed", "n_hit", "n_correct"]],
                            on=["activation", "seed"], suffixes=("", "_base"))
                if (len(m) != len(g) or (m.n_hit != m.n_hit_union).any()
                        or (m.n_correct != m.n_correct_base).any()):
                    _stop(f"caenum {ds} eps={e}: the attack union stored with "
                          f"the enumeration is not the base L2 union row.")
                if len(m) != n_models:
                    print(f"%% !! eps={e}: {len(m)} of {n_models} models "
                          f"present -- INCOMPLETE")
                if not len(m):
                    continue
                ex, co = m[hx] / m.n_correct, m[hc] / m.n_correct
                un = m.n_hit / m.n_correct
                print(f"  eps={e}: exact worst case {ca_ms(ex, 4)} over the "
                      f"{len(m)} models; four-attack union {un.mean():.4f}; "
                      f"exact - union {ex.mean() - un.mean():.4f}; "
                      f"{int(m[hx].sum())} evaded flows of "
                      f"{int(m.n_correct.sum())} trials (union "
                      f"{int(m.n_hit.sum())}); flows the union evades and "
                      f"the enumeration does not: {int(m[hu].sum())}")
                print(f"           corners only {co.mean():.4f} "
                      f"({int(m[hc].sum())} flows); lattice - corners "
                      f"{ex.mean() - co.mean():.4f}")
                pa = {a: ex[(m.activation == a).values] for a in ACTIVATIONS}
                mean = [pa[a].mean() for a in ACTIVATIONS]
                sds = [pa[a].std(ddof=SD_DDOF) if len(pa[a]) > SD_DDOF
                       else float("nan") for a in ACTIVATIONS]
                print("           by activation: " + ", ".join(
                    f"{a} {mu:.4f}" for a, mu in zip(ACTIVATIONS, mean))
                    + f"; span {max(mean) - min(mean):.4f}; seed SDs "
                    + ", ".join(f"{s:.4f}" for s in sds))
                for dv, tag in (((ex - un), "u"), ((ex - co), "c")):
                    i = int(np.argmax(dv.values))
                    cur = (float(dv.values[i]), m.activation.values[i],
                           int(m.seed.values[i]), e)
                    if tag == "u" and (worst is None or cur[0] > worst[0]):
                        worst = cur
                    if tag == "c" and (worst_lc is None
                                       or cur[0] > worst_lc[0]):
                        worst_lc = cur
            if worst is not None:
                print(f"  worst single cell, exact - union: {worst[0]:.4f} "
                      f"({worst[1]} seed {worst[2]}, eps={worst[3]}); "
                      f"lattice - corners: {worst_lc[0]:.4f} ({worst_lc[1]} "
                      f"seed {worst_lc[2]}, eps={worst_lc[3]})")
        g3 = g_ds[np.isin(g_ds.eps.round(3), [round(e, 3) for e in EPS_LIST])]
        print("  (flow, point) pairs changed by the clip, per budget: "
              + ", ".join(f"eps={e}: {int(ca_pick(g3, eps=e).n_clipped.sum())}"
                          for e in EPS_LIST)
              + f" of {int(g3.n_points.iloc[0])} x "
              f"{int(ca_pick(g3, eps=EPS_LIST[0]).n_correct.sum())}")


def ca_sum_sweep(b):
    print("\n%% ---- large-budget extension of the capability sweep; "
          "eps <= " + str(max(EPS_LIST)) + " are the base rows.")
    sw = stage_read("casweep", CA_SKEY)
    if sw is None:
        print("%% casweep: not run yet")
        return
    u = sw[sw.attack == "union"]
    cl, uc = u[u.clipped == 1], u[u.clipped == 0]
    eps_all = list(EPS_LIST) + list(CA_SWEEP_EPS)
    print("%% clipped classifier (the protocol of the paper): worst case, "
          "activation-averaged, mean$\\pm$SD over seeds")
    print("%% dataset level | " + " | ".join(f"eps={e}" for e in eps_all))
    T = {}
    for ds, L in CA_LAB:
        for lv in LEVELS:
            cells = []
            for e in eps_all:
                r = (ca_pick(b, dataset=ds, level=lv, eps=e, attack="union")
                     if e in EPS_LIST else
                     ca_pick(cl, dataset=ds, level=lv, eps=e))
                v = ca_seedavg(r)
                T[(ds, lv, e)] = float(np.mean(v)) if len(v) else float("nan")
                cells.append(ca_ms(v))
            print(f"  {ds:4s} L{lv} & " + " & ".join(cells) + " \\\\")
        print(f"  {ds:4s} L2/L0 retained: " + "  ".join(
            f"eps={e}: {100 * T[(ds, 2, e)] / T[(ds, 0, e)]:.0f}%"
            for e in eps_all if T[(ds, 0, e)] > 0))
    print(f"%% smallest budget at which the L2 worst case (mean over seeds) "
          f"reaches {CA_SAT_LEVEL}:")
    for ds, L in CA_LAB:
        for a in ACTIVATIONS:
            val = {}
            for e in eps_all:
                r = (ca_pick(b, dataset=ds, level=2, eps=e, attack="union",
                             activation=a) if e in EPS_LIST else
                     ca_pick(cl, dataset=ds, level=2, eps=e, activation=a))
                if len(r):
                    val[e] = float(r.asr.mean())
            hit = [e for e in eps_all if val.get(e, -1.0) >= CA_SAT_LEVEL]
            if val:
                print(f"  {ds:4s} {a:4s}: "
                      + (f"eps={hit[0]}" if hit
                         else f"not reached by eps={max(val)}")
                      + f"  (largest value {max(val.values()):.3f})")
    print("%% classifier WITHOUT input clip, PGD and BIM (union): raw value "
          "per budget (mean over the models), then the running maximum over "
          "the budgets per model (the balls are nested) as mean / minimum "
          "over the models")
    for ds, L in CA_LAB:
        for lv in LEVELS:
            r = ca_pick(uc, dataset=ds, level=lv)
            if not len(r):
                continue
            raw = "  ".join(
                f"eps={e}: {ca_pick(r, eps=e).asr.mean():.3f}"
                for e in CA_UNCLIPPED_EPS if len(ca_pick(r, eps=e)))
            mx = r.groupby(["activation", "seed"]).asr.max()
            print(f"  {ds:4s} L{lv}: {raw}  | running max: mean "
                  f"{mx.mean():.3f}, min {mx.min():.3f} over {len(mx)} models")
    en = stage_read("caenum", CA_EKEY)
    if en is not None:
        print("%% L2 by enumeration at the large budgets against the attack "
              "union (mean over the models)")
        for ds in [d for d, _ in CA_LAB if (en.dataset == d).any()]:
            for e in CA_SWEEP_EPS:
                g = ca_pick(en[en.dataset == ds], eps=e)
                if len(g):
                    print(f"  {ds:4s} eps={e}: enumeration "
                          f"{(g.n_hit_exact / g.n_correct).mean():.4f} "
                          f"(no clip {(g.n_hit_box / g.n_correct).mean():.4f})"
                          f"  union "
                          f"{(g.n_hit_union / g.n_correct).mean():.4f}  "
                          f"corners only "
                          f"{(g.n_hit_corner / g.n_correct).mean():.4f}  "
                          f"({int(g.n_hit_exact.sum())} ("
                          f"{int(g.n_hit_box.sum())}) / "
                          f"{int(g.n_hit_union.sum())} / "
                          f"{int(g.n_hit_corner.sum())} flows of "
                          f"{int(g.n_correct.sum())}); union flows the "
                          f"enumeration misses: "
                          f"{int(g.n_union_not_exact.sum())} "
                          f"({int(g.n_union_not_box.sum())})")


def ca_sum_rawunits(un):
    print(f"\n%% ---- tab:rawunits (data only, no model): "
          f"Feature & Unit & Median & SD & eps={CA_RAWUNITS_EPS} & x median & "
          f"Distinct.  Median and distinct values over the malicious flows of "
          f"the test set; SD = the scaler's scale.")
    if un is None:
        print("%% caunits: not run yet")
        return
    if (un.source != "kaggle").any():
        print("%% !! a scaler file is a STAND-IN, not recomputed from the "
              "Kaggle files: this table is NOT valid")
    rank = {n: i for i, n in enumerate(CA_UNITS)}
    for ds, L in CA_LAB:
        g = un[(un.dataset == ds) & (un.free_l2 == 1)]
        if not len(g):
            print(f"%% caunits ({ds}): not run yet")
            continue
        g = g.assign(_o=[rank.get(n, len(rank) + r)
                         for n, r in zip(g.feature, g["rank"])]).sort_values("_o")
        print(f"\\multicolumn{{7}}{{l}}{{\\textit{{{L}}}}} \\\\")
        for r in g.itertuples():
            d = CA_RAWUNITS_EPS * r.scale
            xm = f"{d / abs(r.mal_median):.3g}" if r.mal_median else "---"
            print(f"\\texttt{{{r.feature}}} & {CA_UNITS.get(r.feature, '')} & "
                  f"{r.mal_median:.4g} & {r.scale:.4g} & {d:.4g} & {xm} & "
                  f"{int(r.mal_distinct)} \\\\")
        print("\\hline")
        print(f"%% {ds}: {int(g.integer_mal.sum())} of {len(g)} "
              f"adversary-controllable features are integer-valued among the "
              f"malicious test flows ({int(g.integer_all.sum())} over all "
              f"rows); 99th percentiles: " + ", ".join(
                  f"{r.feature} {r.mal_p99:.4g}" for r in g.itertuples()))


def ca_summarize():
    b = ca_base_rows()
    if b is None:
        print("%% stage capa: the base run has not written any rows yet")
        return
    un = stage_read("caunits", CA_UKEY)
    ca_sum_levels(b)
    ca_sum_quant(b, un)
    ca_sum_unionmax(b)
    ca_sum_enum(b)
    ca_sum_sweep(b)
    ca_sum_rawunits(un)


STAGES.append(("capa", ca_stage, ca_summarize))

# ======== stage capb: code ========
CB_ACT_LABEL = {"relu": "ReLU", "elu": "ELU", "tanh": "Tanh"}
CB_ATK_LABEL = {"pgd": "PGD", "bim": "BIM", "spsa": "SPSA",
                "transfer": "Transfer"}
CB_DS_LABEL = (("cic", "CSE-CIC-IDS2018"), ("unsw", "UNSW-NB15"))


def cb_stage(c):
    """Nothing to measure (see the note above)."""
    print(f"    capb: no measurement for {c.part}; this stage only summarizes "
          f"base rows (RERUN_PART=summary)", flush=True)


def cb_pct(num, den, nd=1):
    """100 * num / den as LaTeX text."""
    if den == 0:
        return "undefined (denominator is 0)"
    return f"{100 * num / den:.{nd}f}\\%"


def cb_ratio(num, den, nd=1):
    if den == 0:
        return "undefined (denominator is 0)"
    return f"{num / den:.{nd}f}"


def cb_rows():
    """The base rows this stage needs, as {key."""
    if not os.path.exists(OUT_FINAL):
        print(f"%% stage capb: {OUT_FINAL} is missing; the base summary writes "
              f"it. Nothing printed.")
        return None
    d = pd.read_csv(OUT_FINAL)
    d = d[(d.depth == DEPTH) & (d.features == FEATS)
          & d.attack.isin(CB_ATTACKS + ["union"])]
    rate, cnt = {}, {}
    for x in d.itertuples(index=False):
        key = (str(x.dataset), str(x.activation), int(x.seed), int(x.level),
               round(float(x.eps), 3), str(x.attack))
        if key in rate:
            _stop(f"stage capb: {OUT_FINAL} holds {key} twice.")
        rate[key] = float(x.asr)
        cnt[key] = (int(x.n_hit), int(x.n_correct))
    want = [(ds, a, s, lv, round(e, 3), t) for ds in DATASETS
            for a in ACTIVATIONS for s in SEEDS for lv in LEVELS
            for e in EPS_LIST for t in CB_ATTACKS + ["union"]]
    missing = [k for k in want if k not in rate]
    if missing:
        print(f"%% stage capb: {len(missing)} of the {len(want)} base rows it "
              f"reads are not in {OUT_FINAL} yet (first missing: {missing[0]})."
              f" Nothing printed; finish the base parts first.")
        return None
    return rate, cnt


def cb_summarize():
    got = cb_rows()
    if got is None:
        return
    rate, cnt = got
    n_models = len(ACTIVATIONS) * len(SEEDS)
    A = CB_ACT_LABEL

    def per_act(ds, lv, e, atk, act):
        return np.array([rate[(ds, act, s, lv, round(e, 3), atk)]
                         for s in SEEDS])

    def pooled(ds, lv, e, atk):
        return np.mean([per_act(ds, lv, e, atk, a) for a in ACTIVATIONS],
                       axis=0)

    def mean_sd(v):
        return float(np.mean(v)), float(np.std(v, ddof=SD_DDOF))

    def act_stats(ds, lv, e):
        """Union worst case by activation."""
        st = {a: mean_sd(per_act(ds, lv, e, "union", a)) for a in ACTIVATIONS}
        return {a: st[a][0] for a in ACTIVATIONS}, {a: st[a][1]
                                                    for a in ACTIVATIONS}

    def chain(mu, names, labels, nd=4):
        """names sorted from the highest mean to the lowest, as text."""
        o = sorted(names, key=lambda k: -mu[k])
        out = f"{labels[o[0]]} {mu[o[0]]:.{nd}f}"
        for a, b in zip(o[:-1], o[1:]):
            out += ((" = " if abs(mu[a] - mu[b]) < CB_TIE_TOL else " > ")
                    + f"{labels[b]} {mu[b]:.{nd}f}")
        return o, out

    def pool_se(ds):
        return float(np.sqrt(CB_SE_ASR * (1 - CB_SE_ASR) / N_EVAL_BY_DS[ds]))

    e_lo, e_mid, e_hi = EPS_LIST[0], EPS_LIST[len(EPS_LIST) // 2], EPS_LIST[-1]

    e = CB_TABLE_EPS
    print(f"\n%% ---- capb A. text on tab:levels_attack "
          f"(eps = {e}). The table rows themselves are printed "
          f"by the base summary above.")
    print("%% Each value: mean over the seeds of the activation-averaged "
          "rate, as in the table.")
    tag = {("unsw", "pgd"): "", ("unsw", "transfer"): "",
           ("cic", "pgd"): "", ("cic", "transfer"): ""}
    for ds, L in CB_DS_LABEL:
        print(f"%% {L}")
        mu = {(t, lv): float(np.mean(pooled(ds, lv, e, t)))
              for t in CB_ATTACKS for lv in LEVELS}
        for t in CB_ATTACKS:
            l0, l2 = mu[(t, 0)], mu[(t, 2)]
            nlow = sum(rate[(ds, a, s, 2, round(e, 3), t)]
                       < rate[(ds, a, s, 0, round(e, 3), t)]
                       for a in ACTIVATIONS for s in SEEDS)
            print(f"%%   {tag.get((ds, t), '')}{CB_ATK_LABEL[t]}: L0 {l0:.3f} "
                  f"-> L2 {l2:.3f}; reduction L0-L2 = {l0 - l2:.3f}; relative "
                  f"reduction {cb_pct(l0 - l2, l0)}; retained L2/L0 "
                  f"{cb_pct(l2, l0)}; L2 below L0 in {nlow} of {n_models} "
                  f"models")
        lose = [CB_ATK_LABEL[t] for t in CB_ATTACKS
                if not mu[(t, 2)] < mu[(t, 0)]]
        print("%%   every attack is lower at L2 than at L0: "
              + ("yes" if not lose else "NO -- not lower: " + ", ".join(lose)))
        if all(mu[(t, 0)] > 0 for t in CB_ATTACKS):
            rel = {t: (mu[(t, 0)] - mu[(t, 2)]) / mu[(t, 0)]
                   for t in CB_ATTACKS}
            top = max(rel.values())
            print("%%   " + ("" if ds == "unsw" else "(the text makes "
                             "this statement for UNSW-NB15 only) ")
                  + "largest relative reduction: "
                  + ", ".join(CB_ATK_LABEL[t] for t in CB_ATTACKS
                              if top - rel[t] < CB_TIE_TOL)
                  + f" ({100 * top:.1f}\\%)")
        else:
            print("%%   largest relative reduction: undefined (an L0 rate is 0)")
        for lv in LEVELS:
            _, txt = chain({t: mu[(t, lv)] for t in CB_ATTACKS}, CB_ATTACKS,
                           CB_ATK_LABEL)
            print(f"%%   order of the attacks at L{lv}: {txt}")
        for what, f in (("unrounded means", lambda v: v),
                        ("means rounded to 3 decimals as in the table",
                         lambda v: round(v, 3))):
            rev = []
            for i, t1 in enumerate(CB_ATTACKS):
                for t2 in CB_ATTACKS[i + 1:]:
                    d0 = f(mu[(t1, 0)]) - f(mu[(t2, 0)])
                    d2 = f(mu[(t1, 2)]) - f(mu[(t2, 2)])
                    if (abs(d0) >= CB_TIE_TOL and abs(d2) >= CB_TIE_TOL
                            and (d0 > 0) != (d2 > 0)):
                        rev.append(f"{CB_ATK_LABEL[t1]}/{CB_ATK_LABEL[t2]} "
                                   f"(L0 {d0:+.4f}, L2 {d2:+.4f})")
            print(f"%%   pairs of attacks ordered one way at L0 "
                  f"and the other way at L2, {what}: "
                  + ("none" if not rev else "; ".join(rev)))

    # ---- B. per seed, and separability ----
    print("\n%% ---- capb B. tab:levels_act, MLP columns PER SEED ("
          "; caption: \"the text states which comparisons "
          "are separable\").")
    print("%% Union worst case of each seed as rate (evaded/correctly "
          "classified flows), then mean and sample SD as in the table.")
    print("%% 'separated' = the gap between two neighbouring activations "
          "exceeds BOTH seed SDs (the rule the base summary uses for "
          "tab:depth_act).")
    for ds, L in CB_DS_LABEL:
        se = pool_se(ds)
        print(f"%% {L}: binomial standard error of the "
              f"{N_EVAL_BY_DS[ds]}-flow pool at ASR {CB_SE_ASR} = {se:.4f}")
        for e in EPS_LIST:
            for lv in CB_ACT_LEVELS:
                mu, sg = act_stats(ds, lv, e)
                for a in ACTIVATIONS:
                    v = per_act(ds, lv, e, "union", a)
                    cells = "  ".join(
                        f"s{s} {v[i]:.4f} ({cnt[(ds, a, s, lv, round(e, 3), 'union')][0]}"
                        f"/{cnt[(ds, a, s, lv, round(e, 3), 'union')][1]})"
                        for i, s in enumerate(SEEDS))
                    print(f"%%   eps={e} L{lv} {A[a]:4s}: {cells}  -> "
                          f"{mu[a]:.3f}$\\pm${sg[a]:.3f}")
                o, txt = chain(mu, ACTIVATIONS, A)
                span = mu[o[0]] - mu[o[-1]]
                gaps = "; ".join(
                    f"{A[a]}-{A[b]} {mu[a] - mu[b]:.3f} (SDs {sg[a]:.3f}, "
                    f"{sg[b]:.3f}: "
                    + ("separated" if (mu[a] - mu[b] > sg[a]
                                       and mu[a] - mu[b] > sg[b])
                       else "not separated") + ")"
                    for a, b in zip(o[:-1], o[1:]))
                print(f"%%     eps={e} L{lv}: {txt}; span {span:.3f}; largest "
                      f"seed SD {max(sg.values()):.3f}; span "
                      + ("exceeds" if span > se else "does not exceed")
                      + f" the pool standard error {se:.3f}; gaps: {gaps}")

    # ---- C. CSE-CIC-IDS2018 by activation ----
    ds = "cic"
    print("\n%% ---- capb C. CSE-CIC-IDS2018 by activation"
          "(union worst case, mean over seeds)")
    mu0, sg0 = act_stats(ds, 0, e_mid)
    mu2, sg2 = act_stats(ds, 2, e_mid)
    o2, txt2 = chain(mu2, ACTIVATIONS, A, 3)
    print(f"%% eps={e_mid} L0 values span {min(mu0.values()):.3f} to "
          f"{max(mu0.values()):.3f}")
    print(f"%% eps={e_mid} L0 largest seed SD: "
          f"{max(sg0.values()):.3f}")
    print(f"%% eps={e_mid} L2 values span "
          f"{min(mu2.values()):.3f} to {max(mu2.values()):.3f}; highest/lowest"
          f" = {cb_ratio(max(mu2.values()), min(mu2.values()))}")
    print(f"%% eps={e_mid} L2 order: {txt2}")
    mu2h, sg2h = act_stats(ds, 2, e_hi)
    o2h, txt2h = chain(mu2h, ACTIVATIONS, A, 3)
    print(f"%% eps={e_hi} L2 order: {txt2h}; same order as at "
          f"eps={e_mid}: " + ("yes" if o2h == o2 else "NO"))
    print(f"%% binomial standard error of the {N_EVAL_BY_DS[ds]}-flow"
          f" pool: {pool_se(ds):.3f}; L2 span at eps={e_mid}: "
          f"{mu2[o2[0]] - mu2[o2[-1]]:.3f}")
    print(f"%% eps={e_mid} L2 seed SDs: "
          + ", ".join(f"{A[a]} {sg2[a]:.3f}" for a in ACTIVATIONS))
    gap = mu2[o2[0]] - mu2[o2[-1]]
    print(f"%% gap between the most evaded ({A[o2[0]]}) and the "
          f"least evaded ({A[o2[-1]]}) at eps={e_mid} L2: {gap:.3f} = "
          f"{cb_ratio(gap, sg2[o2[0]])} seed SDs of {A[o2[0]]}, "
          f"{cb_ratio(gap, sg2[o2[-1]])} of {A[o2[-1]]}")
    print(f"%% the same for ELU against ReLU, the pair the text "
          f"names: {mu2['elu'] - mu2['relu']:.3f} = "
          f"{cb_ratio(mu2['elu'] - mu2['relu'], sg2['elu'])} seed SDs of ELU")
    print(f"%% eps={e_hi} L2 seed SDs: "
          + ", ".join(f"{A[a]} {sg2h[a]:.3f}" for a in ACTIVATIONS)
          + f"; largest {max(sg2h.values()):.3f}")

    # ---- D. UNSW-NB15 by activation ----
    ds = "unsw"
    print("\n%% ---- capb D. UNSW-NB15 by activation "
          "(union worst case, mean over seeds)")
    for e, ln in ((e_lo, ""), (e_mid, "")):
        mu, sg = act_stats(ds, 0, e)
        o, txt = chain(mu, ACTIVATIONS, A, 3)
        print(f"%% {ln}eps={e} L0: {txt}; ReLU minus ELU "
              f"{mu['relu'] - mu['elu']:+.3f}, ReLU minus Tanh "
              f"{mu['relu'] - mu['tanh']:+.3f}; seed SD of ReLU "
              f"{sg['relu']:.3f}, largest seed SD {max(sg.values()):.3f}")
    last, first = [], []
    for e, ln in zip(EPS_LIST, ("", "", "")):
        mu, sg = act_stats(ds, 2, e)
        o, txt = chain(mu, ACTIVATIONS, A, 3)
        last.append(o[-1])
        first.append(o[0])
        print(f"%% {ln}eps={e} L2: {txt}; span "
              f"{mu[o[0]] - mu[o[-1]]:.3f}")
    print(f"%% binomial standard error of the {N_EVAL_BY_DS[ds]}-flow"
          f" pool: {pool_se(ds):.3f}")
    print("%% least evaded activation at L2 by budget: "
          + ", ".join(f"eps={e} {A[a]}" for e, a in zip(EPS_LIST, last))
          + "; most evaded: "
          + ", ".join(f"eps={e} {A[a]}" for e, a in zip(EPS_LIST, first))
          + "; the least evaded changes with the budget: "
          + ("yes" if len(set(last)) > 1 else "NO")
          + "; the most evaded changes: "
          + ("yes" if len(set(first)) > 1 else "NO"))

    # ---- E. retained fractions ----
    print("\n%% ---- capb E. fraction of unconstrained success that survives, "
          "MLPs pooled over activations (union worst case; means over seeds "
          "of the activation-averaged rate, as in tab:levels)")
    keep, pm = {}, {}
    for ds, L in CB_DS_LABEL:
        print(f"%% {L}")
        for e in EPS_LIST:
            l0, l1, l2 = (float(np.mean(pooled(ds, lv, e, "union")))
                          for lv in LEVELS)
            pm[(ds, e)] = (l0, l1, l2)
            keep[(ds, e)] = None if l0 == 0 else 100 * l2 / l0
            print(f"%%   eps={e}: L0 {l0:.3f}  L1 {l1:.3f}  L2 {l2:.3f}; "
                  f"retained L2/L0 = {cb_pct(l2, l0)} ({cb_pct(l2, l0, 0)}); "
                  f"removed = {cb_pct(l0 - l2, l0)}; share of the L0-to-L2 "
                  f"reduction already reached at L1 = "
                  f"{cb_pct(l0 - l1, l0 - l2)}")
    if any(v is None for v in keep.values()):
        print("%% an L0 worst case is 0, so the lines below cannot be formed.")
    else:
        kc = [keep[("cic", e)] for e in EPS_LIST]
        ku = [keep[("unsw", e)] for e in EPS_LIST]
        f1 = lambda v: ", ".join(f"{x:.1f}\\%" for x in v)
        f0 = lambda v: ", ".join(f"{x:.0f}\\%" for x in v)
        print(f"%% UNSW-NB15 MLPs retained at the three budgets:"
              f" {f0(ku)}")
        print(f"%% CSE-CIC-IDS2018 MLPs retained at eps={e_lo}: "
              f"{kc[0]:.1f}\\%")
        print(f"%% CSE-CIC-IDS2018 MLPs retained at the two larger "
              f"budgets: {f0(kc[1:])}")
        print(f"%% range over both datasets and all budgets: "
              f"{min(kc + ku):.0f}\\% to {max(kc + ku):.0f}\\%")
        print(f"%% CSE-CIC-IDS2018 surviving fraction: {f1(kc)}")
        print(f"%% CSE-CIC-IDS2018 removed at eps={e_hi}: "
              f"{100 - kc[-1]:.1f}\\%")
        print(f"%% UNSW-NB15 surviving fraction: {f1(ku)}; "
              f"between {min(ku):.1f}\\% and {max(ku):.1f}\\%")
    print("%% \"as for the MLPs, freezing the categorical "
          "features alone accounts for most of the reduction\": UNSW-NB15 MLP "
          "share of the L0-to-L2 reduction reached at L1 = "
          + ", ".join(f"eps={e} "
                      + cb_pct(pm[("unsw", e)][0] - pm[("unsw", e)][1],
                               pm[("unsw", e)][0] - pm[("unsw", e)][2])
                      for e in EPS_LIST))
    print("%% CSE-CIC-IDS2018 L2 worst case, seven features, "
          "continuous (the value each whole-unit figure is compared with): "
          + ", ".join(f"eps={e} {pm[('cic', e)][2]:.3f}" for e in EPS_LIST))
    print("%% denominators for the whole-unit adversary's "
          "retained fraction (CSE-CIC-IDS2018 L0 worst case): "
          + ", ".join(f"eps={e} {pm[('cic', e)][0]:.4f}" for e in EPS_LIST))

    # ---- F. what this stage does not print ----
    print("\n%% ---- capb F. numbers printed elsewhere")
    print("%% tab:levels_attack rows and the MLP columns of "
          "tab:levels_act: base summary.")
    print("%% CNN column of tab:levels_act and every CNN number "
          " the CNN stage.")
    print("%% Whole-unit L2 values (the second number of each "
          "pair; the three percentages): the stage that "
          "regenerates tab:quant. The "
          "percentages are those whole-unit values divided by "
          "the L0 values printed in the last line of E.")


STAGES.append(("capb", cb_stage, cb_summarize))

# ======== stage cnn: code ========
# The convolutional detector.
CNN_ACOLS = ["dataset", "model", "features", "seed", "level", "eps", "attack",
             "asr", "n_correct", "n_hit", "clean_f1", "n_free", "attack_seed",
             "elapsed_s"]
CNN_AKEY = ["dataset", "seed", "level", "eps", "attack"]
CNN_MCOLS = ["dataset", "model", "features", "seed", "clean_f1", "roc_auc",
             "epochs", "n_params", "n_pool", "n_correct", "elapsed_s"]
CNN_MKEY = ["dataset", "seed"]


def cnn_pooled(n):
    """Side length left by one max-pool (no padding)."""
    return (n - CNN_POOL_KERNEL) // CNN_POOL_STRIDE + 1


class CNN_FlowNet(nn.Module):
    """Conv-BatchNorm-ReLU, max-pool, Conv-ReLU, max-pool, dropout, dense-ReLU, dropout, dense."""

    def __init__(self):
        super().__init__()
        self.h, self.w = CNN_GRID
        f1, f2 = CNN_FILTERS
        self.conv1 = nn.Conv2d(1, f1, kernel_size=CNN_KERNEL, stride=1,
                               padding="same")
        self.bn1   = nn.BatchNorm2d(f1)
        self.pool1 = nn.MaxPool2d(kernel_size=CNN_POOL_KERNEL,
                                  stride=CNN_POOL_STRIDE)
        self.conv2 = nn.Conv2d(f1, f2, kernel_size=CNN_KERNEL, stride=1,
                               padding="same")
        self.pool2 = nn.MaxPool2d(kernel_size=CNN_POOL_KERNEL,
                                  stride=CNN_POOL_STRIDE)
        self.drop2 = nn.Dropout(CNN_P_DROP)
        self.fc1   = nn.Linear(f2 * cnn_pooled(cnn_pooled(self.h))
                               * cnn_pooled(cnn_pooled(self.w)), CNN_DENSE)
        self.drop3 = nn.Dropout(CNN_P_DROP)
        self.fc2   = nn.Linear(CNN_DENSE, 2)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 1, self.h, self.w)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.drop2(self.pool2(x))
        x = self.drop3(self.relu(self.fc1(x.flatten(1))))
        return self.fc2(x)


def cnn_layer_rows():
    m = CNN_FlowNet().eval()
    seen = []
    hooks = [mod.register_forward_hook(
        lambda _m, _i, out, _n=name: seen.append((_n, tuple(out.shape[1:]))))
        for name, mod in m.named_children()]
    with torch.no_grad():
        m(torch.zeros(1, CNN_GRID[0] * CNN_GRID[1]))
    for h in hooks:
        h.remove()
    order = [n for n, _ in seen]
    expect = ["conv1", "bn1", "relu", "pool1", "conv2", "relu", "pool2",
              "drop2", "fc1", "relu", "drop3", "fc2"]
    if order != expect:
        _stop(f"stage cnn: layers are applied as {order}, not as "
              f"Table tab:arch_cnn lists them ({expect}).")
    shp = [s for _, s in seen]
    par = {n: sum(p.numel() for p in mod.parameters())
           for n, mod in m.named_children()}
    k, pk, ps = CNN_KERNEL, CNN_POOL_KERNEL, CNN_POOL_STRIDE
    pool = f"Max-pool ${pk} \\times {pk}$, str.\\ ${ps}$"
    rows = [("Input grid", (1,) + tuple(CNN_GRID), 0),
            (f"Conv ${k} \\times {k}$, ${CNN_FILTERS[0]}$ filters", shp[0],
             par["conv1"]),
            ("BatchNorm, ReLU", shp[2], par["bn1"]),
            (pool, shp[3], 0),
            (f"Conv ${k} \\times {k}$, ${CNN_FILTERS[1]}$ filters", shp[4],
             par["conv2"]),
            ("ReLU", shp[5], 0),
            (pool, shp[6], 0),
            (f"Dropout ${CNN_P_DROP}$, flatten", (int(np.prod(shp[7])),), 0),
            (f"Dense ${CNN_DENSE}$, ReLU, dropout ${CNN_P_DROP}$", shp[10],
             par["fc1"]),
            (f"Dense ${m.fc2.out_features}$ (output)", shp[11], par["fc2"])]
    total = sum(p.numel() for p in m.parameters())
    if sum(r[2] for r in rows) != total or m.fc1.in_features != rows[7][1][0]:
        _stop("stage cnn: the rows of Table tab:arch_cnn do not add up to the "
              "network's parameter count.")
    return rows, total


def cnn_model_path(ds, seed):
    return os.path.join(MODEL_DIR, f"{ds}_cnn_g{CNN_GRID[0]}x{CNN_GRID[1]}"
                                   f"_f{FEATS}_s{seed}_{CNN_ROLE}.pt")


def cnn_get_model(c):
    """The CNN of this dataset and seed."""
    from sklearn.metrics import roc_auc_score
    p = cnn_model_path(c.ds, c.seed)
    if os.path.exists(p):
        blob = torch.load(p, map_location=DEVICE, weights_only=False)
        model = CNN_FlowNet().to(DEVICE)
        model.load_state_dict(blob["state_dict"]); model.eval()
        f1 = macro_f1(model, c.Xte, c.yte, DEVICE)
        if abs(f1 - blob["clean_f1"]) > 1e-9:
            _stop(f"{p} re-measures clean F1 {f1:.6f}, saved "
                  f"{blob['clean_f1']:.6f}: the data or code changed since it "
                  f"was trained.")
        return model, f1, float(blob["roc_auc"]), int(blob["epochs"]), "saved"
    Xtrn, Xval, ytrn, yval = c.split()
    _reseed(c.seed)
    model = CNN_FlowNet().to(DEVICE)
    _, epochs = train_to_convergence(model, Xtrn, ytrn, Xval, yval, DEVICE)
    model.eval()
    f1 = macro_f1(model, c.Xte, c.yte, DEVICE)
    auc = float(roc_auc_score((c.yte != 0).astype(int),
                              _probs(model, c.Xte, DEVICE)[:, 1]))
    os.makedirs(MODEL_DIR, exist_ok=True)
    tmp = p + f".tmp{os.getpid()}"
    torch.save({"state_dict": model.state_dict(), "clean_f1": float(f1),
                "roc_auc": auc, "epochs": int(epochs), "ds": c.ds,
                "features": FEATS, "grid": tuple(CNN_GRID), "seed": c.seed,
                "role": CNN_ROLE, "script": "rerun_all.py"}, tmp)
    os.replace(tmp, p)                    # atomic, as in get_model
    return model, f1, auc, int(epochs), "trained"


def cnn_attack(a, model, clf, sub_clf, Xe, ye, eps, mask, s):
    """One attack of the union, by the base functions, exactly as run_part calls them for the MLPs."""
    if a == "pgd":
        return _ref_q(clf, Xe, eps, mask, "pgd")
    if a == "bim":
        return _ref_q(clf, Xe, eps, mask, "bim")
    if a == "spsa":
        return a_spsa(model, Xe, ye, eps, s, mask)
    if a == "transfer":
        return a_transfer_saved(sub_clf, Xe, eps, mask)
    _stop(f"stage cnn: unknown attack {a!r}")


def cnn_stage(c):
    if c.act != ACTIVATIONS[0]:
        print(f"    the CNN has no activation: it is run in the "
              f"{ACTIVATIONS[0]} part of each dataset and seed", flush=True)
        return
    if CNN_GRID[0] * CNN_GRID[1] != FEATS:
        _stop(f"stage cnn: {FEATS} features do not fill a {CNN_GRID} grid.")
    have = stage_have("cnnattacks", CNN_AKEY)
    have_m = stage_have("cnnmodels", CNN_MKEY)
    want = {_k(c.ds, c.seed, lv, round(e, 3), a) for lv in CNN_LEVELS
            for e in EPS_LIST for a in CNN_ATTACKS + ["union"]}
    npz = stage_npz("cnn", c)
    saved = npz_load(npz)

    def sets_saved(lv, eps):
        """The per-flow success sets of this cell are all in the per-flow file."""
        return all(f"L{lv}_e{eps}_{a}" in saved
                   for a in CNN_ATTACKS + ["union"])

    if (want <= have and _k(c.ds, c.seed) in have_m
            and all(sets_saved(lv, e) for lv in CNN_LEVELS for e in EPS_LIST)):
        print(f"    skip cnn {c.ds} s{c.seed}: done", flush=True)
        return
    t0 = time.time()
    model, f1, auc, epochs, src = cnn_get_model(c)
    n_par = sum(p.numel() for p in model.parameters())
    correct = _probs(model, c.Xe, DEVICE).argmax(1) == c.ye
    n_ok = max(int(correct.sum()), 1)
    print(f"    cnn {c.ds} s{c.seed}: {src} {epochs}ep cleanF1={f1:.4f} "
          f"ROC-AUC={auc:.5f} params={n_par} pool {int(correct.sum())}/"
          f"{len(c.ye)} correct [{time.time() - t0:.0f}s]", flush=True)
    if _k(c.ds, c.seed) not in have_m:
        stage_append("cnnmodels", CNN_MCOLS, dict(
            dataset=c.ds, model="cnn", features=FEATS, seed=c.seed,
            clean_f1=round(float(f1), 4), roc_auc=round(auc, 6),
            epochs=epochs, n_params=n_par, n_pool=len(c.ye),
            n_correct=int(correct.sum()),
            elapsed_s=round(time.time() - t0, 1)))
    clf = _art_clf(model, c.Xtr, DEVICE)      # clip range as for the MLPs
    sub_clf = c.substitute()                  # the SAME saved MLP substitute
    for name, now in (("pool", c.pool), ("correct", correct)):
        if name in saved and not np.array_equal(saved[name], now):
            _stop(f"{npz} was written for a different {name}: the model or "
                  f"the data changed. Move it away and re-run.")
        saved[name] = now
    base = dict(dataset=c.ds, model="cnn", features=FEATS, seed=c.seed,
                clean_f1=round(float(f1), 4), n_correct=n_ok)
    for lv in CNN_LEVELS:
        mask = c.masks[lv]
        nfree = FEATS if mask is None else int(mask.sum())
        for eps in EPS_LIST:
            ukey = _k(c.ds, c.seed, lv, round(eps, 3), "union")
            if ukey in have and sets_saved(lv, eps):
                continue
            union, rates = None, {}
            for a in CNN_ATTACKS:
                key = _k(c.ds, c.seed, lv, round(eps, 3), a)
                name = f"L{lv}_e{eps}_{a}"
                if key in have and name in saved:
                    h = saved[name].astype(bool)   # finished before a restart
                else:
                    s = _seed_of("cnn", c.ds, c.seed, lv, eps, a)
                    _reseed(s); t2 = time.time()
                    adv = _timed(lambda: cnn_attack(a, model, clf, sub_clf,
                                                    c.Xe, c.ye, eps, mask, s))
                    _, h, v = hits_of(model, adv, c.Xe, c.ye, DEVICE, eps, mask)
                    if v:
                        _stop(f"capability violation: {v} frozen coordinates "
                              f"moved (cnn {c.ds} s{c.seed} L{lv} eps={eps} "
                              f"{a})")
                    saved[name] = h
                    npz_save(npz, saved)      # the set first, then its row
                    if key not in have:
                        stage_append("cnnattacks", CNN_ACOLS, dict(
                            base, level=lv, eps=round(eps, 3), attack=a,
                            asr=round(float(h.sum() / n_ok), 4),
                            n_hit=int(h.sum()), n_free=nfree, attack_seed=s,
                            elapsed_s=round(time.time() - t2, 1)))
                        have.add(key)
                union = h if union is None else (union | h)
                rates[a] = float(h.sum() / n_ok)
            saved[f"L{lv}_e{eps}_union"] = union
            npz_save(npz, saved)
            if ukey not in have:
                stage_append("cnnattacks", CNN_ACOLS, dict(
                    base, level=lv, eps=round(eps, 3), attack="union",
                    asr=round(float(union.sum() / n_ok), 4),
                    n_hit=int(union.sum()), n_free=nfree, attack_seed=0,
                    elapsed_s=0))
                have.add(ukey)
            print(f"    cnn L{lv} eps={eps}: " + " ".join(
                f"{a.upper()} {rates[a]:.3f}" for a in CNN_ATTACKS)
                + f" | union {union.sum() / n_ok:.3f}", flush=True)


def cnn_summarize():
    A = stage_read("cnnattacks", CNN_AKEY)
    M = stage_read("cnnmodels", CNN_MKEY)
    lab = (("cic", "CSE-CIC-IDS2018"), ("unsw", "UNSW-NB15"))
    acts = ["relu", "elu", "tanh"]
    TL = list(CNN_TABLE_LEVELS)
    num = lambda n: f"{n:,}".replace(",", "{,}")

    rows, total = cnn_layer_rows()
    print("%% ---- tab:arch_cnn, read off the network "
          "this stage builds")
    times = " \\times "
    for name, shape, n in rows:
        print(f"{name} & ${times.join(str(x) for x in shape)}$ & "
              f"${num(n)}$ \\\\")
    print(f"\\hline\nTotal & & ${num(total)}$ \\\\")
    m4 = sum(p.numel() for p in
             build_model(DEPTH, FEATS, ACTIVATIONS[0]).parameters())
    print(f"%% grid {CNN_GRID[0]} x {CNN_GRID[1]}, "
          f"{num(total)} parameters; Model {DEPTH} MLP {num(m4)} "
          f"(ratio {m4 / total:.1f})")

    if A is None or M is None:
        print("%% stage cnn: not run yet")
        return
    if (M.n_params != total).any():
        _stop("stage cnn: a saved CNN has a different parameter count from "
              "the network this script builds.")
    keys = set(_k(*r) for r in A[CNN_AKEY].itertuples(index=False))
    mkeys = set(_k(*r) for r in M[CNN_MKEY].itertuples(index=False))
    ok = {}
    for ds, _ in lab:
        miss = sorted({s for s in SEEDS if _k(ds, s) not in mkeys or any(
            _k(ds, s, lv, round(e, 3), a) not in keys for lv in CNN_LEVELS
            for e in EPS_LIST for a in CNN_ATTACKS + ["union"])})
        ok[ds] = not miss
        if miss:
            print(f"%% stage cnn: {ds} INCOMPLETE -- seed(s) {miss} not "
                  f"finished (run the parts {ds}-{ACTIVATIONS[0]}-<seed>); "
                  f"nothing is printed for {ds}")
    done = [(ds, L) for ds, L in lab if ok[ds]]

    def per_seed(ds, lv, e, a="union", col="asr"):
        r = A[(A.dataset == ds) & (A.level == lv) & np.isclose(A.eps, e)
              & (A.attack == a)].set_index("seed")
        return [float(r.loc[s, col]) for s in SEEDS]

    def mcol(ds, col):
        r = M[M.dataset == ds].set_index("seed")
        return [float(r.loc[s, col]) for s in SEEDS]

    def pct(a, b):
        return f"{100 * a / b:.1f}\\%" if b > 0 else "n/a (L0 rate is 0)"

    B = pd.read_csv(OUT_FINAL) if os.path.exists(OUT_FINAL) else None

    def mlp(ds, lv, e, act):
        if B is None:
            return None
        r = B[(B.dataset == ds) & (B.depth == DEPTH) & (B.features == FEATS)
              & (B.activation == act) & (B.level == lv) & np.isclose(B.eps, e)
              & (B.attack == "union")].set_index("seed")
        if any(s not in r.index for s in SEEDS):
            return None
        return [float(r.loc[s, "asr"]) for s in SEEDS]

    def mlp_pooled(ds, lv, e):
        v = [mlp(ds, lv, e, a) for a in acts]
        return None if any(x is None for x in v) else list(np.mean(v, axis=0))

    def mlp_kept(ds, e):
        v0, v2 = mlp_pooled(ds, 0, e), mlp_pooled(ds, 2, e)
        if v0 is None or v2 is None:
            return "(base rows not available)"
        return pct(np.mean(v2), np.mean(v0))

    print("\n%% ---- CNN clean detection, full test set ("
          "'same clean macro-F1 to within')")
    for ds, L in done:
        f, u, ep = mcol(ds, "clean_f1"), mcol(ds, "roc_auc"), mcol(ds, "epochs")
        print(f"  {L}: macro-F1 {_ms(f)}  (per seed {SEEDS}: "
              f"{[round(x, 4) for x in f]}, max-min {max(f) - min(f):.4f})   "
              f"ROC-AUC mean {np.mean(u):.4f} (per seed "
              f"{[round(x, 5) for x in u]})   epochs trained "
              f"{[int(x) for x in ep]}")

    def head(e, i, lv):
        return (f"$\\epsilon = {e}$ & L{lv}" if i == 0
                else f"                 & L{lv}")

    print("\n%% ---- tab:levels_act, CNN column (same "
          "rows and order as the base block)")
    for ds, L in done:
        print(f"\\multicolumn{{6}}{{l}}{{\\textit{{{L}}}}} \\\\")
        for e in EPS_LIST:
            for i, lv in enumerate(TL):
                print(head(e, i, lv) + " & " + _ms(per_seed(ds, lv, e))
                      + " \\\\")
        print("\\hline")

    print("\n%% ---- tab:levels_act, complete rows (ReLU, ELU, Tanh from the "
          "base rows in " + OUT_FINAL + "; CNN from this stage)")
    for ds, L in done:
        print(f"\\multicolumn{{6}}{{l}}{{\\textit{{{L}}}}} \\\\")
        for e in EPS_LIST:
            for i, lv in enumerate(TL):
                cells = [mlp(ds, lv, e, a) for a in acts]
                print(head(e, i, lv) + " & " + " & ".join(
                    "(no base row)" if v is None else _ms(v) for v in cells)
                    + " & " + _ms(per_seed(ds, lv, e)) + " \\\\")
        print("\\hline")

    print("\n%% ---- CNN, union worst case at every level, per seed "
          f"{SEEDS} and mean+-SD")
    for ds, L in done:
        print(f"  {L}")
        for lv in CNN_LEVELS:
            for e in EPS_LIST:
                v = per_seed(ds, lv, e)
                print(f"    L{lv} eps={e}: {[round(x, 4) for x in v]}  "
                      f"{_ms(v)}")

    print("\n%% ---- Section 4.5 text on the CNN, the "
          "same quantities for both datasets")
    lo, hi = TL[0], TL[-1]
    for ds, L in done:
        print(f"  {L}")
        print(f"    fraction of L{lo} success kept at L{hi} (mean L{hi} / mean "
              f"L{lo}):")
        for e in EPS_LIST:
            v0, v2 = per_seed(ds, lo, e), per_seed(ds, hi, e)
            ps = ", ".join(f"s{s} {pct(b, a)}" for s, a, b in zip(SEEDS, v0, v2))
            print(f"      eps={e}: L{lo} {np.mean(v0):.3f}  L{hi} "
                  f"{np.mean(v2):.3f}  kept {pct(np.mean(v2), np.mean(v0))}"
                  f"   MLPs pooled over activations: {mlp_kept(ds, e)}"
                  f"   per seed: {ps}")
        sds = [np.std(per_seed(ds, hi, e), ddof=SD_DDOF) for e in EPS_LIST]
        print(f"    largest seed SD at L{hi} over the budgets: "
              f"{max(sds):.3f}")
        print("    mean by level (quotes L0 and L1 on UNSW at the "
              "largest budget): " + ";  ".join(
                  f"eps={e}: " + " ".join(
                      f"L{lv} {np.mean(per_seed(ds, lv, e)):.3f}"
                      for lv in CNN_LEVELS) for e in EPS_LIST))
        nc = per_seed(ds, hi, EPS_LIST[0], col="n_correct")
        print(f"    successful flows at L{hi}, summed over the seeds, of "
              f"{num(int(sum(nc)))} trials: " + ", ".join(
                  f"eps={e} {num(int(sum(per_seed(ds, hi, e, col='n_hit'))))}"
                  for e in EPS_LIST)
              + f"   (correctly classified pool flows per seed "
                f"{[int(x) for x in nc]})")
        for e in EPS_LIST:
            v0, v2 = per_seed(ds, lo, e), per_seed(ds, hi, e)
            ratio = (f"{max(v2) / min(v2):.1f}" if min(v2) > 0
                     else "n/a (a seed has rate 0)")
            print(f"    eps={e}, per seed {SEEDS}: L{hi} "
                  f"{[round(x, 3) for x in v2]}, largest / smallest {ratio}; "
                  f"L{lo} between {min(v0):.3f} and {max(v0):.3f}")

    print("\n%% ---- union minus maximum of the per-attack rates, CNN "
          "")
    for ds, L in done:
        for lvs in (list(CNN_LEVELS), TL):
            gap = []                # from the flow counts, not rounded rates
            for lv in lvs:
                for e in EPS_LIST:
                    u = per_seed(ds, lv, e, col="n_hit")
                    n = per_seed(ds, lv, e, col="n_correct")
                    r = [per_seed(ds, lv, e, a, col="n_hit")
                         for a in CNN_ATTACKS]
                    gap += [(u[i] - max(x[i] for x in r)) / n[i]
                            for i in range(len(SEEDS))]
            print(f"  {L}, levels {lvs}: {len(gap)} cells, mean "
                  f"{np.mean(gap):.4f}, at most {max(gap):.4f}, smallest "
                  f"{min(gap):.4f}, positive in {sum(g > 1e-9 for g in gap)}")
            if min(gap) < -1e-9:
                _stop("stage cnn: a union is below one of its own attacks.")


STAGES.append(("cnn", cnn_stage, cnn_summarize))

# ======== stage advtrain: code ========
AT_COLS = ["dataset", "depth", "features", "activation", "seed", "defence",
           "level", "eps", "attack", "asr", "n_correct", "n_hit", "clean_f1",
           "n_free", "required_linf", "attack_seed", "elapsed_s"]
AT_KEY = ["dataset", "activation", "seed", "defence", "level", "eps", "attack"]


def at_tpr_col(t):
    return f"fpr_at_tpr{int(round(t * 100))}"


AT_FCOLS = (["dataset", "activation", "seed", "defence", "depth", "features",
             "epochs", "clean_f1", "roc_auc"]
            + [at_tpr_col(t) for t in AT_TPR_TARGETS]
            + ["threshold", "fp_at_thr", "tp_at_thr", "n_benign", "n_malicious",
               "fpr_at_thr", "tpr_at_thr", "n_pool", "n_pool_correct",
               "elapsed_s"])
AT_FKEY = ["dataset", "activation", "seed", "defence"]


def at_train_seed(c):
    """The bare model seed, as in the published defence and in get_model."""
    return int(c.seed)


def at_attack_seed(c, lv, eps, a):
    return _seed_of("advtrain", c.ds, DEPTH, c.Xtr.shape[1], c.act, c.seed,
                    AT_ROLE, lv, eps, a)


def at_train_settings(c):
    """Everything that determines the defended weights."""
    return dict(at_eps=AT_EPS, at_steps=AT_STEPS, at_ratio=AT_RATIO,
                at_step_factor=AT_STEP_FACTOR, at_clamp=AT_CLAMP, lr=LR,
                batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, chunk=CHUNK,
                patience=PATIENCE, tol=TOL, val_fraction=VAL_FRACTION,
                split_seed=SEED, train_seed=at_train_seed(c))


def at_pgd_batch(model, X, y):
    model.eval()
    x0 = X.clone().detach()
    x = x0 + torch.empty_like(x0).uniform_(-AT_EPS, AT_EPS)
    step = AT_EPS / AT_STEPS * AT_STEP_FACTOR
    for _ in range(AT_STEPS):
        x = x.clone().detach().requires_grad_(True)
        loss = F.cross_entropy(model(x), y)
        g, = torch.autograd.grad(loss, x)
        upd = step * g.sign()
        x = torch.clamp(x0 + torch.clamp(x + upd - x0, -AT_EPS, AT_EPS),
                        -AT_CLAMP, AT_CLAMP)
    model.train()
    return x.detach()


def at_train(model, Xtrn, ytrn, Xval, yval, tag):
    """The published train_adversarial."""
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    Xt = torch.tensor(Xtrn, dtype=torch.float32)
    yt = torch.tensor(ytrn, dtype=torch.long)
    n, best, stale, done, t0 = len(Xt), -1.0, 0, 0, time.time()
    while done < MAX_EPOCHS:
        model.train()
        for _ in range(CHUNK):
            perm = torch.randperm(n)
            for i in range(0, n, BATCH_SIZE):
                idx = perm[i:i + BATCH_SIZE]
                xb = Xt[idx].to(DEVICE); yb = yt[idx].to(DEVICE)
                nadv = int(len(xb) * AT_RATIO)
                if nadv > 0:
                    xb = torch.cat([at_pgd_batch(model, xb[:nadv], yb[:nadv]),
                                    xb[nadv:]], 0)
                opt.zero_grad()
                F.cross_entropy(model(xb), yb).backward()
                opt.step()
        done += CHUNK
        f1 = macro_f1(model, Xval, yval, DEVICE)
        print(f"      {tag}: {done} epochs, validation F1 {f1:.4f} "
              f"[{time.time() - t0:.0f}s]", flush=True)
        if f1 > best + TOL:
            best, stale = f1, 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.eval()
    return best, done


def at_get_model(c):
    """The defended Model 4 of this part."""
    k = c.Xtr.shape[1]
    p = model_path(c.ds, DEPTH, k, c.act, c.seed, AT_ROLE)
    cfg = at_train_settings(c)
    if os.path.exists(p):
        blob = torch.load(p, map_location=DEVICE, weights_only=False)
        if blob.get("train") != cfg:
            _stop(f"{p} was trained with other settings than this file's:\n"
                  f"  saved   {blob.get('train')}\n  current {cfg}\n"
                  f"Move it away and re-run this part.")
        model = build_model(DEPTH, k, c.act).to(DEVICE)
        model.load_state_dict(blob["state_dict"]); model.eval()
        f1 = macro_f1(model, c.Xte, c.yte, DEVICE)
        if abs(f1 - blob["clean_f1"]) > 1e-9:
            _stop(f"{p} re-measures clean F1 {f1:.6f}, saved "
                  f"{blob['clean_f1']:.6f}: the data or code changed since it "
                  f"was trained.")
        return model, f1, int(blob["epochs"]), "saved"
    Xtrn, Xval, ytrn, yval = c.split()
    _reseed(cfg["train_seed"]); t0 = time.time()
    model = build_model(DEPTH, k, c.act).to(DEVICE)
    best, epochs = at_train(model, Xtrn, ytrn, Xval, yval,
                            f"{AT_ROLE} {c.part}")
    model.eval()
    f1 = macro_f1(model, c.Xte, c.yte, DEVICE)
    os.makedirs(MODEL_DIR, exist_ok=True)
    tmp = p + f".tmp{os.getpid()}"
    torch.save({"state_dict": model.state_dict(), "clean_f1": float(f1),
                "epochs": int(epochs), "val_best": float(best), "ds": c.ds,
                "depth": DEPTH, "features": k, "activation": c.act,
                "seed": c.seed, "role": AT_ROLE, "train": cfg,
                "train_s": round(time.time() - t0, 1),
                "script": "rerun_all.py"}, tmp)
    os.replace(tmp, p)
    return model, f1, int(epochs), "trained"


def at_attack(c, model, clf, lv, eps, a, s):
    """ONE attack on the pool, exactly as run_part runs it on the standard model."""
    mask = c.masks[lv]
    _reseed(s)
    fn = {"pgd": lambda: _ref_q(clf, c.Xe, eps, mask, "pgd"),
          "bim": lambda: _ref_q(clf, c.Xe, eps, mask, "bim"),
          "spsa": lambda: a_spsa(model, c.Xe, c.ye, eps, s, mask),
          "transfer": lambda: a_transfer_saved(c.substitute(), c.Xe, eps,
                                               mask)}[a]
    adv = _timed(fn)
    _, h, v = hits_of(model, adv, c.Xe, c.ye, DEVICE, eps, mask)
    if v:
        _stop(f"capability violation: {v} frozen coordinates moved "
              f"({c.part} {AT_ROLE} L{lv} eps={eps} {a})")
    return h


def at_hsj(c, model, clf, correct, s):
    """The base HopSkipJump of run_part on the pool, L0."""
    _reseed(s)
    raw = _timed(lambda: hsj_unseeded(clf, c.Xe, AT_HSJ[1], s))
    dist = np.abs(raw - c.Xe).max(1)
    wrong = _probs(model, raw, DEVICE).argmax(1) != c.ye
    evaded = correct & wrong
    req = float(np.median(dist[evaded])) if evaded.any() else float("nan")
    sets = {}
    for eps in EPS_LIST:
        within = evaded & (dist <= eps + 1e-6)
        proj = (c.Xe + np.clip(raw - c.Xe, -eps, eps)).astype(np.float32)
        _, hp, _ = hits_of(model, proj, c.Xe, c.ye, DEVICE, eps, None)
        sets[eps] = (within, hp)
    return raw, req, sets


def at_operating_row(c, model, defence, f1, epochs, correct):
    from sklearn.metrics import roc_auc_score, roc_curve
    t0 = time.time()
    p = _probs(model, c.Xte, DEVICE)[:, 1]
    y = (c.yte != 0).astype(int)
    nb, nm = int((y == 0).sum()), int((y == 1).sum())
    fpr, tpr, _ = roc_curve(y, p)
    row = dict(dataset=c.ds, activation=c.act, seed=c.seed, defence=defence,
               depth=DEPTH, features=c.Xte.shape[1], epochs=int(epochs),
               clean_f1=float(f1), roc_auc=float(roc_auc_score(y, p)),
               threshold=AT_THRESHOLD)
    for t in AT_TPR_TARGETS:
        row[at_tpr_col(t)] = (float(np.interp(t, tpr, fpr)) if tpr[-1] >= t
                              else float("nan"))
    pred = (p >= AT_THRESHOLD).astype(int)
    row["fp_at_thr"] = int(((pred == 1) & (y == 0)).sum())
    row["tp_at_thr"] = int(((pred == 1) & (y == 1)).sum())
    row["n_benign"], row["n_malicious"] = nb, nm
    row["fpr_at_thr"] = row["fp_at_thr"] / max(nb, 1)
    row["tpr_at_thr"] = row["tp_at_thr"] / max(nm, 1)
    row["n_pool"], row["n_pool_correct"] = len(c.pool), int(correct.sum())
    row["elapsed_s"] = round(time.time() - t0, 1)
    return row


def stage_advtrain(c):
    have = stage_have("atattacks", AT_KEY)
    havef = stage_have("atfpr", AT_FKEY)
    hsj = f"hsj_{AT_HSJ[0]}"
    key = lambda lv, eps, a: _k(c.ds, c.act, c.seed, AT_ROLE, lv,
                                round(eps, 3), a)
    want = {key(lv, e, a) for lv in AT_LEVELS for e in EPS_LIST
            for a in AT_ATTACKS + ["union"]}
    want |= {key(0, e, hsj + x) for e in EPS_LIST for x in ("", "_proj")}
    wantf = {_k(c.ds, c.act, c.seed, dfc) for dfc in ("standard", AT_ROLE)}
    if want <= have and wantf <= havef:
        print(f"    skip {AT_ROLE} {c.part}: done", flush=True)
        return
    k = c.Xtr.shape[1]
    t0 = time.time()
    model, f1, epochs, src = at_get_model(c)
    print(f"    {AT_ROLE} {c.part}: {src}, {epochs} epochs, cleanF1={f1:.4f} "
          f"(standard {c.f1:.4f}) [{time.time() - t0:.0f}s]", flush=True)
    clf = _art_clf(model, c.Xtr, DEVICE)
    correct = _probs(model, c.Xe, DEVICE).argmax(1) == c.ye
    n_ok = max(int(correct.sum()), 1)

    std_epochs = int(torch.load(
        model_path(c.ds, DEPTH, k, c.act, c.seed), map_location=DEVICE,
        weights_only=False).get("epochs", -1))
    for dfc, m, mf1, mep, mcorr in (("standard", c.model, c.f1, std_epochs,
                                     c.correct),
                                    (AT_ROLE, model, f1, epochs, correct)):
        if _k(c.ds, c.act, c.seed, dfc) in havef:
            continue
        row = at_operating_row(c, m, dfc, mf1, mep, mcorr)
        stage_append("atfpr", AT_FCOLS, row)
        print(f"    {dfc:8s} AUC {row['roc_auc']:.5f}  FPR@TPR{AT_TPR_TABLE} "
              f"{row[at_tpr_col(AT_TPR_TABLE)]:.5f}  at {AT_THRESHOLD}: FPR "
              f"{row['fpr_at_thr']:.5f} TPR {row['tpr_at_thr']:.5f}  pool "
              f"{row['n_pool_correct']}/{row['n_pool']}", flush=True)

    npz = stage_npz("advtrain", c)
    saved = npz_load(npz)
    saved["pool"], saved["correct"] = c.pool, correct
    base = dict(dataset=c.ds, depth=DEPTH, features=k, activation=c.act,
                seed=c.seed, defence=AT_ROLE, clean_f1=round(float(f1), 4))

    # -- PGD, BIM, SPSA, transfer and their per-flow union, L0 and L2.
    for lv in AT_LEVELS:
        nfree = k if c.masks[lv] is None else int(c.masks[lv].sum())
        for eps in EPS_LIST:
            if key(lv, eps, "union") in have:
                continue
            union, rates = None, {}
            for a in AT_ATTACKS:
                s = at_attack_seed(c, lv, eps, a)
                t2 = time.time()
                h = at_attack(c, model, clf, lv, eps, a, s)
                saved[f"L{lv}_e{eps}_{a}"] = h
                union = h if union is None else (union | h)
                rates[a] = float(h.sum() / n_ok)
                stage_append("atattacks", AT_COLS, dict(
                    base, level=lv, eps=round(eps, 3), attack=a,
                    asr=round(rates[a], 4), n_correct=n_ok, n_hit=int(h.sum()),
                    n_free=nfree, attack_seed=s,
                    elapsed_s=round(time.time() - t2, 1)))
            saved[f"L{lv}_e{eps}_union"] = union
            npz_save(npz, saved)
            stage_append("atattacks", AT_COLS, dict(
                base, level=lv, eps=round(eps, 3), attack="union",
                asr=round(float(union.sum() / n_ok), 4), n_correct=n_ok,
                n_hit=int(union.sum()), n_free=nfree, attack_seed=0,
                elapsed_s=0))
            have.add(key(lv, eps, "union"))
            print(f"    {AT_ROLE} L{lv} eps={eps}: " + " ".join(
                f"{a.upper()} {rates[a]:.3f}" for a in AT_ATTACKS)
                + f" | union {union.sum() / n_ok:.3f}", flush=True)

    # -- decision-based check.
    if not all(key(0, e, hsj + x) in have for e in EPS_LIST
               for x in ("", "_proj")):
        s = _seed_of("advtrain", c.ds, DEPTH, k, c.act, c.seed, AT_ROLE, "hsj",
                     AT_HSJ[0])
        t2 = time.time()
        raw, req, sets = at_hsj(c, model, clf, correct, s)
        saved[f"{hsj}_raw"] = raw
        el = round((time.time() - t2) / len(EPS_LIST), 1)
        msg, rows = [], []
        for eps in EPS_LIST:
            for a, h in zip((hsj, hsj + "_proj"), sets[eps]):
                saved[f"L0_e{eps}_{a}"] = h
                rows.append(dict(
                    base, level=0, eps=round(eps, 3), attack=a,
                    asr=round(float(h.sum() / n_ok), 4), n_correct=n_ok,
                    n_hit=int(h.sum()), n_free=k, required_linf=round(req, 4),
                    attack_seed=s, elapsed_s=el))
            msg.append(f"eps={eps} {sets[eps][0].sum() / n_ok:.3f}"
                       f"/{sets[eps][1].sum() / n_ok:.3f}")
        npz_save(npz, saved)
        for row in rows:
            stage_append("atattacks", AT_COLS, row)
        print(f"    {AT_ROLE} HopSkipJump {AT_HSJ[0]} (within/projected): "
              + "  ".join(msg) + f"  required l_inf {req:.3f}  "
              f"[{time.time() - t2:.0f}s]", flush=True)


def at_ms(v, nd=3):
    v = np.asarray(v, float)
    return f"{v.mean():.{nd}f}$\\pm${v.std(ddof=SD_DDOF):.{nd}f}"


def at_figure(u, path):
    try:
        import matplotlib
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "matplotlib"], capture_output=True, text=True)
        import importlib
        importlib.invalidate_caches()
        try:
            import matplotlib
        except ImportError:
            print("!! matplotlib is missing and could not be installed: "
                  f"{path} was NOT written. pip install matplotlib, then run "
                  "RERUN_PART=summary again.", flush=True)
            return False
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=AT_FIG_SIZE)
    for ax, (ds, title) in zip(axes, (("cic", "CSE-CIC-IDS2018"),
                                      ("unsw", "UNSW-NB15"))):
        x = u[u.dataset == ds]
        per_seed = x.groupby(["level", "eps", "defence", "seed"]).urd.mean()
        g = per_seed.groupby(["level", "eps", "defence"])
        mean, sd = g.mean(), g.std(ddof=SD_DDOF)
        for lv, ls, _ in AT_FIG_LEVELS:
            for dfc, (_, colour) in AT_FIG_SERIES.items():
                mm = [float(mean.loc[(lv, e, dfc)]) for e in EPS_LIST]
                ss = [float(sd.loc[(lv, e, dfc)]) for e in EPS_LIST]
                ax.errorbar(EPS_LIST, mm,
                            yerr=[[min(s_, m_) for m_, s_ in zip(mm, ss)],
                                  [min(s_, 1.0 - m_) for m_, s_ in zip(mm, ss)]],
                            color=colour, linestyle=ls, marker="o",
                            markersize=5, linewidth=2, capsize=3, zorder=3)
        ax.set_xlabel(r"Attack budget $\epsilon$")
        ax.set_title(title)
        ax.set_xticks(EPS_LIST)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(axis="y", alpha=0.3, which="major", zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(r"Robust detection  TPR $\times$ (1 $-$ ASR)")
    handles, labels = [], []
    for lv, ls, lvlab in AT_FIG_LEVELS:
        for dfc, (label, colour) in AT_FIG_SERIES.items():
            handles.append(Line2D([], [], color=colour, linestyle=ls,
                                  marker="o", markersize=5, linewidth=2))
            labels.append(f"{label}, {lvlab.split(' ', 1)[1]}")
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.6, handlelength=2.6)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[:-4] + f".tmp{os.getpid()}.pdf"
    fig.savefig(tmp, dpi=AT_FIG_DPI)
    plt.close(fig)
    os.replace(tmp, path)
    return True


def summarize_advtrain():
    a = stage_read("atattacks", AT_KEY)
    f = stage_read("atfpr", AT_FKEY)
    if a is None or f is None:
        print("%% stage advtrain: not run yet")
        return
    if not os.path.exists(OUT_FINAL):
        _stop(f"{OUT_FINAL} is missing: the standard-model columns are read "
              f"from it. Run RERUN_PART=summary (it writes that file first).")
    b = pd.read_csv(OUT_FINAL)
    b = b[(b.depth == DEPTH) & (b.features == FEATS)
          & b.level.isin(AT_LEVELS)].copy()
    b["defence"] = "standard"
    a = a[(a.depth == DEPTH) & (a.features == FEATS)]
    cols = ["dataset", "activation", "seed", "defence", "level", "eps",
            "attack", "asr", "n_correct", "n_hit", "clean_f1"]
    d = pd.concat([b[cols], a[cols]], ignore_index=True)
    d["eps"] = d.eps.astype(float).round(3)
    lab = (("cic", "CSE-CIC-IDS2018"), ("unsw", "UNSW-NB15"))
    acts = (("relu", "ReLU"), ("elu", "ELU "), ("tanh", "Tanh"))
    DEF = ("standard", AT_ROLE)
    hsj = f"hsj_{AT_HSJ[0]}"
    e_mid, e_max = EPS_LIST[len(EPS_LIST) // 2], EPS_LIST[-1]
    ref = lambda lines, txt: print(f"  [l.{lines}] {txt}" if lines else
                                   f"  [not quoted] {txt}")

    def q(df, **kw):
        m = np.ones(len(df), bool)
        for col, v in kw.items():
            m &= (df[col] == v).values
        return df[m]

    # ---- which datasets are complete (nothing is printed from a partial one) ----
    done = []
    for ds, L in lab:
        miss = 0
        for ac in ACTIVATIONS:
            for s in SEEDS:
                for dfc in DEF:
                    r = q(d, dataset=ds, activation=ac, seed=s, defence=dfc)
                    need = {(lv, round(e, 3), x) for lv in AT_LEVELS
                            for e in EPS_LIST for x in AT_ATTACKS + ["union"]}
                    if dfc == AT_ROLE:
                        need |= {(0, round(e, 3), hsj) for e in EPS_LIST}
                    got = set(zip(r.level.astype(int), r.eps, r.attack))
                    miss += len(need - got)
                    miss += int(len(q(f, dataset=ds, activation=ac, seed=s,
                                      defence=dfc)) != 1)
        if miss:
            print(f"%% stage advtrain: {L} is INCOMPLETE ({miss} measurements "
                  f"missing); nothing is printed for it")
        else:
            done.append((ds, L))
    if not done:
        return

    def sv(ds, dfc, lv, e, atk="union", col="asr"):
        r = q(d, dataset=ds, defence=dfc, level=lv, eps=round(e, 3), attack=atk)
        return np.array([float(r[r.seed == s][col].mean()) for s in SEEDS])

    def f1s(ds, dfc, ac):
        """clean macro-F1 per seed, as stored (4 dp) with the attack rows."""
        r = q(d, dataset=ds, defence=dfc, activation=ac)
        g = r.groupby("seed").clean_f1
        if (g.nunique() != 1).any():
            _stop(f"{ds} {dfc} {ac}: clean F1 differs between rows of one model")
        return np.array([float(g.first()[s]) for s in SEEDS])

    def fs(ds, dfc, col):
        r = q(f, dataset=ds, defence=dfc)
        return np.array([float(r[r.seed == s][col].mean()) for s in SEEDS])

    # per-model unconditional robust detection, exact from the integer counts.
    u = d[d.attack == "union"].copy()
    pools = {}
    for ds, _ in done:
        n = q(f, dataset=ds).n_pool.unique()
        if len(n) != 1:
            _stop(f"{ds}: the pool size differs between parts: {n}")
        pools[ds] = int(n[0])
        r = q(f, dataset=ds)
        for x in r.itertuples():
            nc = q(u, dataset=ds, activation=x.activation, seed=x.seed,
                   defence=x.defence).n_correct
            if set(nc) != {max(int(x.n_pool_correct), 1)}:
                _stop(f"{ds} {x.activation} s{x.seed} {x.defence}: n_correct "
                      f"of the attack rows {sorted(set(nc))} is not the "
                      f"model's pool count {x.n_pool_correct}")
    u = u[u.dataset.isin(pools)].copy()
    u["N"] = u.dataset.map(pools)
    u["tpr"] = u.n_correct / u.N
    u["urd"] = (u.n_correct - u.n_hit) / u.N

    def us(ds, dfc, lv, e, col="urd"):
        r = q(u, dataset=ds, defence=dfc, level=lv, eps=round(e, 3))
        return np.array([float(r[r.seed == s][col].mean()) for s in SEEDS])

    print("%% Standard = the saved base Model 4 (rows of attacks_final.csv, the "
          "rows tab:levels prints); PGD-AT = this stage.")
    print(f"%% PGD-AT: eps={AT_EPS}, {AT_STEPS} inner steps, {AT_RATIO} of each "
          f"batch, unconstrained inner adversary; worst case = per-flow union of "
          f"{', '.join(AT_ATTACKS)}.")

    print("\n%% ---- tab:advtrain: L0 Standard | L0 PGD-AT | "
          "L2 Standard | L2 PGD-AT")
    for ds, L in done:
        print(f"\\multicolumn{{5}}{{l}}{{\\textit{{{L}}}}} \\\\")
        for e in EPS_LIST:
            print(f"$\\epsilon={e}$ & " + " & ".join(
                at_ms(sv(ds, dfc, lv, e)) for lv in AT_LEVELS for dfc in DEF)
                + " \\\\")
        print("\\hline")
    print("%% caption: the Standard columns ARE the L0 and L2 "
          "columns of tab:levels (same rows of attacks_final.csv, same "
          "aggregation), so they agree exactly; no re-evaluation is involved.")

    print("\n%% ---- tab:advtrain_cost: Standard | PGD-AT | Cost "
          "(difference of the unrounded means)")
    cost = {}
    for ds, L in done:
        print(f"\\multicolumn{{4}}{{l}}{{\\textit{{{L}}}}} \\\\")
        for ac, A in acts:
            s0, s1 = f1s(ds, "standard", ac), f1s(ds, AT_ROLE, ac)
            cost[(ds, ac)] = s0.mean() - s1.mean()
            print(f"{A} & {at_ms(s0)} & {at_ms(s1)} & {cost[(ds, ac)]:.3f} \\\\")
        print("\\hline")
    for ds, L in done:
        r = q(f, dataset=ds)
        gap = max(abs(round(x.clean_f1, 4) - float(
            q(d, dataset=ds, activation=x.activation, seed=x.seed,
              defence=x.defence).clean_f1.iloc[0])) for x in r.itertuples())
        ep = r.groupby("defence").epochs
        print(f"%% {L}: training length, epochs min/mean/max -- " + "; ".join(
            f"{dfc} {ep.min()[dfc]}/{ep.mean()[dfc]:.1f}/{ep.max()[dfc]}"
            for dfc in DEF) + f". Clean F1 of the operating-characteristic "
            f"rows vs the attack rows: max difference {gap:.4f} (same models).")

    print("\n%% ---- in-text numbers, Section 4.6 first part")
    for ds, L in done:
        m = lambda dfc, lv, e: sv(ds, dfc, lv, e).mean()
        ref("1423-1424", f"{L}: L0 worst case at eps={e_mid}: "
            f"{m('standard', 0, e_mid):.3f} -> {m(AT_ROLE, 0, e_mid):.3f}")
        l2 = [m(AT_ROLE, 2, e) for e in EPS_LIST]
        ref("1425-1426, 1540-1541, 2088-2089", f"{L}: L2 PGD-AT worst case, "
            f"largest over the budgets: {max(l2):.4f} = {100 * max(l2):.2f}% ("
            + ", ".join(f"eps={e} {v:.4f}" for e, v in zip(EPS_LIST, l2)) + ")")
        r = q(u, dataset=ds, defence=AT_ROLE, level=2)
        tot = r.groupby("eps")[["n_hit", "n_correct"]].sum()
        ref("1426-1428", f"{L}: L2 PGD-AT evasions summed over the "
            f"{r.groupby(['activation', 'seed']).ngroups} models: " + "; ".join(
                f"eps={e} {int(tot.loc[round(e, 3), 'n_hit'])} of "
                f"{int(tot.loc[round(e, 3), 'n_correct']):,}" for e in EPS_LIST))
        per = r.groupby(["activation", "seed"]).n_hit.max()
        ref("1428-1430", f"{L}: L2 PGD-AT, per model: largest count at any "
            f"budget {int(r.n_hit.max())}; n_correct {int(r.n_correct.min()):,}"
            f"..{int(r.n_correct.max()):,}; models at zero or one at every "
            f"budget: {int((per <= 1).sum())} of {len(per)}")
        print("      evaded flows per model (rows eps; columns activation, seed):")
        print("      " + r.pivot_table(
            index="eps", columns=["activation", "seed"], values="n_hit",
            aggfunc="first").to_string().replace("\n", "\n      "))
        ref("1432-1435", f"{L}: L0 PGD-AT worst case at eps={e_max}: "
            f"{m(AT_ROLE, 0, e_max):.3f}")
        cs = [cost[(ds, ac)] for ac, _ in acts]
        ref("1437-1439, 1570", f"{L}: clean macro-F1 cost between "
            f"{min(cs):.3f} and {max(cs):.3f}")

    print("\n%% ---- tab:fpr: ROC-AUC | FPR at TPR = "
          f"{AT_TPR_TABLE} | FPR at threshold {AT_THRESHOLD} | TPR at "
          f"threshold {AT_THRESHOLD}")
    fc = ["roc_auc", at_tpr_col(AT_TPR_TABLE), "fpr_at_thr", "tpr_at_thr"]
    for ds, L in done:
        print(f"\\multicolumn{{5}}{{l}}{{\\textit{{{L}}}}} \\\\")
        for dfc, D in zip(DEF, ("Standard", "PGD-AT")):
            print(f"{D} & " + " & ".join(at_ms(fs(ds, dfc, x), 5) for x in fc)
                  + " \\\\")
        print("\\hline")
    print("%% in-text numbers on the operating characteristic ("
          "and the discussion)")
    for ds, L in done:
        nb = q(f, dataset=ds).n_benign.unique()
        if len(nb) != 1:
            _stop(f"{ds}: the benign test count differs between parts: {nb}")
        nb = int(nb[0])
        g = {dfc: {x: fs(ds, dfc, x) for x in fc + ["fp_at_thr"]}
             for dfc in DEF}
        s0, s1 = g["standard"], g[AT_ROLE]
        tcol = at_tpr_col(AT_TPR_TABLE)
        ref("1447, 1458-1459", f"{L}: ROC-AUC standard "
            f"{s0['roc_auc'].mean():.4f} ({s0['roc_auc'].mean():.3f}), PGD-AT "
            f"{s1['roc_auc'].mean():.4f} ({s1['roc_auc'].mean():.3f}); gap "
            f"{s0['roc_auc'].mean() - s1['roc_auc'].mean():+.5f} = "
            f"{(s0['roc_auc'].mean() - s1['roc_auc'].mean()) / s0['roc_auc'].std(ddof=SD_DDOF):.1f}"
            f" x the seed SD of the standard detectors, "
            f"{(s0['roc_auc'].mean() - s1['roc_auc'].mean()) / s1['roc_auc'].std(ddof=SD_DDOF):.1f}"
            f" x that of the defended ones")
        ref("1449-1451, 1460-1463", f"{L}: FPR at TPR = {AT_TPR_TABLE}: "
            f"standard {s0[tcol].mean():.5f}, PGD-AT {s1[tcol].mean():.5f}; on "
            f"the {nb:,} benign test flows that is "
            f"{s0[tcol].mean() * nb:,.0f} alarms (standard) against "
            f"{s1[tcol].mean() * nb:,.0f} (PGD-AT)")
        ref("1451-1453, 1463-1465, 2006", f"{L}: at threshold {AT_THRESHOLD}: "
            f"FPR standard {s0['fpr_at_thr'].mean():.3f}, PGD-AT "
            f"{s1['fpr_at_thr'].mean():.3f}; false alarms (mean over the "
            f"models) {s0['fp_at_thr'].mean():,.0f} against "
            f"{s1['fp_at_thr'].mean():,.0f}, PGD-AT / standard = "
            f"{s1['fp_at_thr'].mean() / max(s0['fp_at_thr'].mean(), 1e-12):.1f}"
            f"; TPR standard {s0['tpr_at_thr'].mean():.3f}, PGD-AT "
            f"{s1['tpr_at_thr'].mean():.3f}")
        others = [t for t in AT_TPR_TARGETS if t != AT_TPR_TABLE]
        print("      also recorded, not in the manuscript: " + "; ".join(
            f"FPR at TPR = {t}: standard {fs(ds, 'standard', at_tpr_col(t)).mean():.5f}"
            f", PGD-AT {fs(ds, AT_ROLE, at_tpr_col(t)).mean():.5f}"
            for t in others))
    print("  not applicable in the single run: tab:fpr is "
          "measured on the very models of tab:advtrain and tab:advtrain_cost "
          "(no separate re-training).")

    print("\n%% ---- cost and benefit")
    for ds, L in done:
        m = lambda dfc, lv, e: sv(ds, dfc, lv, e).mean()
        for e in EPS_LIST:
            b0 = m("standard", 0, e) - m(AT_ROLE, 0, e)
            b2 = m("standard", 2, e) - m(AT_ROLE, 2, e)
            ratio = (f"{b0 / b2:.2f}" if b2 > 0 else "undefined")
            l20 = (f"{m('standard', 0, e) / m('standard', 2, e):.2f}"
                   if m("standard", 2, e) > 0 else "undefined")
            ref("1534-1539" if e == e_mid else
                ("1547-1549" if e == e_max else ""),
                f"{L} eps={e}: L0 reduction {b0:.3f} ({m('standard', 0, e):.4f}"
                f" -> {m(AT_ROLE, 0, e):.4f}); L2 reduction {b2:.3f} ("
                f"{m('standard', 2, e):.4f} -> {m(AT_ROLE, 2, e):.4f}); ratio "
                f"of the two benefits {ratio} to 1; standard L0/L2 ratio of "
                f"tab:levels {l20}")
    allc = [v for (ds, _), v in cost.items() if ds in dict(done)]
    ref("80, 89, 1535-1536, 2002-2003", f"clean macro-F1 cost over "
        f"{' and '.join(L for _, L in done)}: {min(allc):.3f} to "
        f"{max(allc):.3f}")
    for ds, L in done:
        cs = {ac: cost[(ds, ac)] for ac, _ in acts}
        rest = [v for ac, v in cs.items() if ac != "relu"]
        ref("1623-1625", f"{L}: cost by activation: ReLU {cs['relu']:.3f}, "
            f"ELU {cs['elu']:.3f}, Tanh {cs['tanh']:.3f}; ELU and Tanh span "
            f"{min(rest):.3f}--{max(rest):.3f}; smallest: "
            f"{min(cs, key=cs.get)}")

    print("\n%% ---- unconditional robust detection, (n_c - n_h) / N "
          "")
    rec = {}
    for ds, L in done:
        t0 = us(ds, "standard", 0, e_mid, "tpr").mean()
        t1 = us(ds, AT_ROLE, 0, e_mid, "tpr").mean()
        rec[ds] = t0 - t1
        ref("1567-1570, 2011" if ds == "unsw" else "1586-1588",
            f"{L}: share of the {pools[ds]:,}-flow pool classified correctly: "
            f"standard {t0:.4f}, PGD-AT {t1:.4f}, loss {t0 - t1:.4f}")
        for lv, _, lvlab in AT_FIG_LEVELS:
            m0 = [us(ds, "standard", lv, e).mean() for e in EPS_LIST]
            m1 = [us(ds, AT_ROLE, lv, e).mean() for e in EPS_LIST]
            sd0 = [us(ds, "standard", lv, e).std(ddof=SD_DDOF) for e in EPS_LIST]
            sd1 = [us(ds, AT_ROLE, lv, e).std(ddof=SD_DDOF) for e in EPS_LIST]
            where = {("unsw", 2): "1576-1579", ("cic", 2): "1589-1591",
                     ("cic", 0): "1591-1593"}.get((ds, lv), "1507, figure")
            ref(where, f"{L} {lvlab}: standard " + ", ".join(
                f"{v:.3f}" for v in m0) + " against PGD-AT " + ", ".join(
                f"{v:.3f}" for v in m1) + f" at eps = {EPS_LIST}")
            print("      plotted mean+-SD: standard " + "  ".join(
                f"{v:.4f}+-{s:.4f}" for v, s in zip(m0, sd0)) + " | PGD-AT "
                + "  ".join(f"{v:.4f}+-{s:.4f}" for v, s in zip(m1, sd1)))
            sgn = np.sign(np.array(m0) - np.array(m1))
            cross = [i for i in range(len(EPS_LIST) - 1) if sgn[i] != sgn[i + 1]]
            worse = [e for e, x in zip(EPS_LIST, sgn) if x > 0]
            txt = ("; ".join(f"CROSSOVER between eps={EPS_LIST[i]} and eps="
                             f"{EPS_LIST[i + 1]}" for i in cross) if cross else
                   "no crossover: " + ("PGD-AT" if (sgn < 0).all() else
                                       "standard" if (sgn > 0).all() else
                                       "neither") + " higher at every budget")
            ref("1579-1582, 2013-2014" if (ds, lv) == ("unsw", 2) else
                "1586-1593" if ds == "cic" else "1507, figure",
                f"{L} {lvlab}: {txt}; budgets at which the defended model is "
                f"the WORSE one: {worse if worse else 'none'}")
    if len(rec) == 2:
        lo, hi = sorted(rec.values(), key=abs)
        ref("1593-1595", "recall cost of the defence: "
            + " against ".join(f"{rec[ds]:.4f} ({L})" for ds, L in done)
            + (f"; ratio {abs(hi) / abs(lo):.0f}" if lo != 0 else
               "; ratio undefined (one cost is zero)"))
    print("  NOT regenerated here: substituting the exact L2 "
          "worst case needs the enumeration of Section 4.5 (capability stage); "
          "with its per-model n_hit_exact the quantity is (n_c - n_hit_exact) "
          "/ N.")

    print(f"\n%% ---- decision-based check of the defended models"
          f": HopSkipJump l_{AT_HSJ[0]}, unseeded start, success within eps, L0")
    for ds, L in done:
        hv = {e: sv(ds, AT_ROLE, 0, e, hsj) for e in EPS_LIST}
        uv = {e: sv(ds, AT_ROLE, 0, e) for e in EPS_LIST}
        ref("1610-1612", f"{L}: HopSkipJump " + ", ".join(
            at_ms(hv[e]) for e in EPS_LIST) + f" at eps = {EPS_LIST}")
        ref("1613-1617", f"{L}: HopSkipJump minus the four-attack union of "
            f"tab:advtrain: " + ", ".join(
                f"eps={e} {hv[e].mean() - uv[e].mean():+.3f}" for e in EPS_LIST)
            + f"; at eps={e_max} union / HopSkipJump = "
            + (f"{uv[e_max].mean() / hv[e_max].mean():.1f}"
               if hv[e_max].mean() > 0 else "undefined (HopSkipJump 0)"))
        r = q(a, dataset=ds, defence=AT_ROLE, level=0)
        pv = r[r.attack.isin([hsj, hsj + "_proj"])].groupby(
            ["attack", "eps"]).asr.mean().unstack()
        print("      mean over the models, within eps / projected onto "
              "the eps-ball: " + ", ".join(
                  f"eps={e} {pv.loc[hsj, round(e, 3)]:.4f}/"
                  f"{pv.loc[hsj + '_proj', round(e, 3)]:.4f}" for e in EPS_LIST))
    print("  not applicable in the single run: the check is made "
          "on the very models of tab:advtrain (no separate re-training).")

    print("\n%% ---- union minus the maximum of the per-attack rates "
          "(Section 4.5; outside this section, same rows)")
    for ds, L in done:
        x = d[(d.dataset == ds) & d.attack.isin(AT_ATTACKS + ["union"])]
        idx = ["defence", "level", "eps", "activation", "seed"]
        mx = x[x.attack != "union"].groupby(idx).asr.max()
        un = x[x.attack == "union"].set_index(idx).asr
        gap = (un - mx)
        for name, g in (("PGD-AT", gap.xs(AT_ROLE)),
                        ("standard and PGD-AT, L0 and L2", gap)):
            print(f"  {L}, {name}: {len(g)} cells, mean {g.mean():+.4f}, max "
                  f"{g.max():+.4f}, min {g.min():+.4f}, positive in "
                  f"{int((g > 1e-9).sum())}")

    if len(done) == len(lab):
        path = os.path.join(AT_FIG_DIR, AT_FIG_FILE)
        if at_figure(u, path):
            print(f"\n%% fig:robust written: {path} "
                  f"({os.path.getsize(path):,} bytes)")
    else:
        print("\n%% fig:robust NOT written: it needs both datasets complete.")


STAGES.append(("advtrain", stage_advtrain, summarize_advtrain))

# ======== stage xai: code ========
XAI_ATTR_COLS = ["dataset", "activation", "seed", "feature_idx", "feature",
                 "free_l1", "free_l2", "attr_abs", "attr_abs_sample",
                 "attr_signed", "n_flows", "n_baselines", "n_alpha", "clean_f1",
                 "draw_seed", "elapsed_s"]
XAI_ATTR_KEY = XAI_ATTR_COLS[:4]
XAI_CHECK_COLS = ["dataset", "activation", "seed", "n_flows", "mean_out_pool",
                  "mean_out_baselines", "mean_abs_delta_out",
                  "mean_abs_completeness_gap", "draw_seed", "elapsed_s"]
XAI_CHECK_KEY = XAI_CHECK_COLS[:3]
XAI_MARGIN_COLS = ["dataset", "activation", "seed", "level", "eps", "n_correct",
                   "n_correct_base", "n_denominator_diff", "n_pred", "pred_asr",
                   "median_eps_hat", "median_eps_hat_finite", "mean_margin",
                   "mean_grad_l1", "frac_zero_grad", "n_free", "clean_f1",
                   "elapsed_s"]
XAI_MARGIN_KEY = XAI_MARGIN_COLS[:5]


def xai_draws(ds, n_train):
    """The 64 baseline rows and the 64 x 32 interpolation points."""
    s = _seed_of("xai", ds, "eg-draws")
    _reseed(s)
    rng = np.random.default_rng(s)
    base_idx = rng.integers(0, n_train, size=XAI_N_BASELINES)
    alphas = rng.random((XAI_N_BASELINES, XAI_N_ALPHA)).astype(np.float32)
    return s, base_idx, alphas


def xai_expected_gradients(model, Xe, B, alphas):
    """Expected gradients (GradientSHAP) of the malicious-class logit."""
    model.eval()
    X = torch.tensor(Xe, dtype=torch.float32, device=DEVICE)
    n, d = X.shape
    nb, na = alphas.shape
    per = max(1, BATCH_SIZE // n)           # interpolation points per batch
    signed = torch.zeros((n, d), dtype=torch.float64)
    absval = torch.zeros((n, d), dtype=torch.float64)
    for b in range(nb):
        base = torch.tensor(B[b], dtype=torch.float32, device=DEVICE)
        diff = X - base
        for j in range(0, na, per):
            a = torch.tensor(alphas[b, j:j + per], dtype=torch.float32,
                             device=DEVICE)
            x = (base + a[:, None, None] * diff[None, :, :]).reshape(-1, d)
            x = x.clone().detach().requires_grad_(True)
            g, = torch.autograd.grad(model(x)[:, XAI_TARGET_CLASS].sum(), x)
            t = (g.reshape(len(a), n, d) * diff[None, :, :]).double().cpu()
            signed += t.sum(0)
            absval += t.abs().sum(0)
    return (signed / (nb * na)).numpy(), (absval / (nb * na)).numpy()


@torch.no_grad()
def xai_logit(model, X):
    model.eval()
    out = []
    for i in range(0, len(X), BATCH_SIZE):
        xb = torch.tensor(X[i:i + BATCH_SIZE], dtype=torch.float32, device=DEVICE)
        out.append(model(xb)[:, XAI_TARGET_CLASS].double().cpu().numpy())
    return np.concatenate(out, 0)


def xai_margin_and_grad(model, X, y):
    model.eval()
    ms, gs = [], []
    for i in range(0, len(X), BATCH_SIZE):
        xb = torch.tensor(X[i:i + BATCH_SIZE], dtype=torch.float32,
                          device=DEVICE, requires_grad=True)
        yb = torch.tensor(y[i:i + BATCH_SIZE], dtype=torch.long, device=DEVICE)
        out = model(xb)
        m = (out.gather(1, yb[:, None]).squeeze(1)
             - out.gather(1, (1 - yb)[:, None]).squeeze(1))
        g, = torch.autograd.grad(m.sum(), xb)
        ms.append(m.detach().cpu().numpy())
        gs.append(g.detach().cpu().numpy())
    return np.concatenate(ms, 0), np.concatenate(gs, 0)


def xai_stage_attrib(c):
    have = stage_have("xaiattrib", XAI_ATTR_KEY)
    done = stage_have("xaicheck", XAI_CHECK_KEY)
    keys = [_k(c.ds, c.act, c.seed, j) for j in range(FEATS)]
    if all(k in have for k in keys) and _k(c.ds, c.act, c.seed) in done:
        print(f"    attribution {c.part}: done", flush=True)
        return
    t0 = time.time()
    Xtrn = c.split()[0]                     # the training split of the model
    s, base_idx, alphas = xai_draws(c.ds, len(Xtrn))
    B = Xtrn[base_idx].astype(np.float32)
    eg, eg_abs = xai_expected_gradients(c.model, c.Xe, B, alphas)
    f_pool, f_base = xai_logit(c.model, c.Xe), xai_logit(c.model, B)
    delta = f_pool - f_base.mean()
    gap = np.abs(eg.sum(1) - delta)
    a_abs, a_smp, a_sgn = np.abs(eg).mean(0), eg_abs.mean(0), eg.mean(0)
    path = stage_npz("xai", c)
    z = npz_load(path)
    z.update(pool=c.pool, names=np.asarray(c.names, dtype=str),
             free_l1=c.masks[1], free_l2=c.masks[2], eg=eg, eg_abs_sample=eg_abs,
             base_idx=base_idx, alphas=alphas, logit_pool=f_pool,
             logit_baselines=f_base)
    npz_save(path, z)
    el = round(time.time() - t0, 1)
    for j in range(FEATS):
        if keys[j] in have:
            continue
        stage_append("xaiattrib", XAI_ATTR_COLS, dict(
            dataset=c.ds, activation=c.act, seed=c.seed, feature_idx=j,
            feature=c.names[j], free_l1=int(c.masks[1][j]),
            free_l2=int(c.masks[2][j]), attr_abs=round(float(a_abs[j]), 10),
            attr_abs_sample=round(float(a_smp[j]), 10),
            attr_signed=round(float(a_sgn[j]), 10), n_flows=len(c.Xe),
            n_baselines=XAI_N_BASELINES, n_alpha=XAI_N_ALPHA,
            clean_f1=round(float(c.f1), 4), draw_seed=s, elapsed_s=el))
    if _k(c.ds, c.act, c.seed) not in done:
        stage_append("xaicheck", XAI_CHECK_COLS, dict(
            dataset=c.ds, activation=c.act, seed=c.seed, n_flows=len(c.Xe),
            mean_out_pool=round(float(f_pool.mean()), 6),
            mean_out_baselines=round(float(f_base.mean()), 6),
            mean_abs_delta_out=round(float(np.abs(delta).mean()), 6),
            mean_abs_completeness_gap=round(float(gap.mean()), 6),
            draw_seed=s, elapsed_s=el))
    print(f"    attribution {c.part}: n_eff {xai_neff(a_abs):.2f} "
          f"(per-term absolute values: {xai_neff(a_smp):.2f})  completeness gap "
          f"{gap.mean():.3f} of {np.abs(delta).mean():.3f}  [{el:.0f}s]",
          flush=True)


def xai_stage_margin(c):
    have = stage_have("xaimargin", XAI_MARGIN_KEY)
    if all(_k(c.ds, c.act, c.seed, lv, round(e, 3)) in have
           for lv in XAI_LEVELS for e in EPS_LIST):
        print(f"    margin test {c.part}: done", flush=True)
        return
    t0 = time.time()
    m, g = xai_margin_and_grad(c.model, c.Xe, c.ye)
    correct = m > 0                         # C of Eq. (predasr)
    n_corr = int(correct.sum())
    path = stage_npz("xai", c)
    z = npz_load(path)
    z.update(pool=c.pool, names=np.asarray(c.names, dtype=str),
             free_l1=c.masks[1], free_l2=c.masks[2], margin=m, grad=g,
             correct=correct)
    npz_save(path, z)
    for lv in XAI_LEVELS:
        mask = (np.ones(FEATS, dtype=np.float32) if c.masks[lv] is None
                else c.masks[lv])
        gl1 = np.abs(g[correct] * mask[None, :]).sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            eps_hat = np.where(gl1 > 0, m[correct] / gl1, np.inf)
        fin = np.isfinite(eps_hat)
        nan = float("nan")
        for eps in EPS_LIST:
            if _k(c.ds, c.act, c.seed, lv, round(eps, 3)) in have:
                continue
            n_pred = int((eps_hat <= eps).sum())
            stage_append("xaimargin", XAI_MARGIN_COLS, dict(
                dataset=c.ds, activation=c.act, seed=c.seed, level=lv,
                eps=round(eps, 3), n_correct=n_corr,
                n_correct_base=int(c.correct.sum()),
                n_denominator_diff=int((correct != c.correct).sum()),
                n_pred=n_pred, pred_asr=round(n_pred / max(n_corr, 1), 4),
                median_eps_hat=(round(float(np.median(eps_hat)), 6)
                                if n_corr else nan),
                median_eps_hat_finite=(round(float(np.median(eps_hat[fin])), 6)
                                       if fin.any() else nan),
                mean_margin=(round(float(m[correct].mean()), 6)
                             if n_corr else nan),
                mean_grad_l1=round(float(gl1.mean()), 6) if n_corr else nan,
                frac_zero_grad=(round(float((gl1 == 0).mean()), 6)
                                if n_corr else nan),
                n_free=int(mask.sum()), clean_f1=round(float(c.f1), 4),
                elapsed_s=round(time.time() - t0, 1)))
        print(f"    margin test {c.part} L{lv}: predicted ASR " + " ".join(
            f"eps={e} {(eps_hat <= e).sum() / max(n_corr, 1):.3f}"
            for e in EPS_LIST) + "  |g_S|_1 "
            + (f"{gl1.mean():.3f}" if n_corr else "nan"), flush=True)


def xai_stage(c):
    xai_stage_attrib(c)
    xai_stage_margin(c)


def xai_neff(v):
    v = np.asarray(v, float)
    tot = v.sum()
    if not tot > 0:
        return float("nan")
    return float(1.0 / np.square(v / tot).sum())


def xai_rank(v):
    return np.argsort(-np.asarray(v, float), kind="stable")


def xai_figures(mean_vec, names):
    def _mpl():
        # Figure + Agg canvas directly.
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        return Figure, FigureCanvasAgg
    try:
        Figure, FigureCanvasAgg = _mpl()
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "matplotlib"], capture_output=True, text=True)
        import importlib
        importlib.invalidate_caches()
        try:
            Figure, FigureCanvasAgg = _mpl()
        except ImportError:
            print("\n" + "!" * 74 + "\nfig:xai NOT written: matplotlib is not "
                  "installed (pip install matplotlib), then run the summary "
                  "again.\n" + "!" * 74, flush=True)
            return []
    os.makedirs(XAI_FIG_DIR, exist_ok=True)
    fig = Figure(figsize=(XAI_FIG_SIZE[0], XAI_FIG_SIZE[1] * len(ACTIVATIONS)))
    FigureCanvasAgg(fig)
    all_axes = fig.subplots(len(ACTIVATIONS), 2)
    for axes, ref in zip(all_axes, ACTIVATIONS):
        for ax, ds in zip(axes, ("cic", "unsw")):
            top = xai_rank(mean_vec[(ds, ref)])[:XAI_TOPK_FIG]
            x = np.arange(len(top))
            w = 0.8 / len(ACTIVATIONS)
            for j, a in enumerate(ACTIVATIONS):
                ax.bar(x + (j - (len(ACTIVATIONS) - 1) / 2) * w,
                       mean_vec[(ds, a)][top], w, label=a.upper(),
                       color=XAI_FIG_COLORS[a])
            labels = []
            for i in top:
                n = names[ds][i]
                for old, new in XAI_FIG_ABBREV:
                    n = n.replace(old, new)
                labels.append(n)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("Mean |attribution|")
            ax.set_title(f"{XAI_DS_LABEL[ds]} (ranked by {ref.upper()})")
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)
            ax.set_axisbelow(True)
    fig.tight_layout()
    path = os.path.join(XAI_FIG_DIR, XAI_FIG_FILE)
    tmp = path + f".tmp{os.getpid()}"
    fig.savefig(tmp, format="pdf", dpi=XAI_FIG_DPI, bbox_inches="tight",
                metadata={"CreationDate": None})
    os.replace(tmp, path)
    return [path]


def xai_summarize():
    A = stage_read("xaiattrib", XAI_ATTR_KEY)
    C = stage_read("xaicheck", XAI_CHECK_KEY)
    M = stage_read("xaimargin", XAI_MARGIN_KEY)
    if A is None or C is None or M is None:
        print("%% stage xai: not run yet")
        return
    if not os.path.exists(OUT_FINAL):
        print(f"%% stage xai: {OUT_FINAL} is missing (the base summary writes "
              f"it); nothing printed")
        return
    DSS = ("cic", "unsw")
    AL = (("relu", "ReLU"), ("elu", "ELU "), ("tanh", "Tanh"))
    ACT = [a for a, _ in AL]

    def part(d, ds, a, s):
        return d[(d.dataset == ds) & (d.activation == a) & (d.seed == s)]

    missing = [f"{ds}-{a}-{s}" for ds in DSS for a in ACT for s in SEEDS
               if len(part(A, ds, a, s)) != FEATS or len(part(C, ds, a, s)) != 1
               or len(part(M, ds, a, s)) != len(XAI_LEVELS) * len(EPS_LIST)]
    if missing:
        print(f"%% stage xai: INCOMPLETE, no table printed. Parts not finished: "
              f"{', '.join(missing)}")
        return
    base = pd.read_csv(OUT_FINAL)
    base = base[(base.depth == 4) & (base.features == FEATS)]

    def meas(ds, a, lv, e, atk):
        """Base rows (one per seed) of one Model 4 cell, in seed order."""
        r = base[(base.dataset == ds) & (base.activation == a)
                 & (base.level == lv) & np.isclose(base.eps, e)
                 & (base.attack == atk)].sort_values("seed")
        if sorted(r.seed.astype(int)) != sorted(SEEDS):
            _stop(f"stage xai: {OUT_FINAL} has no complete {atk} row set for "
                  f"{ds} {a} L{lv} eps={e}; run the base parts first.")
        return r

    # ---- the per-feature vectors: (seeds, features) per dataset and activation ----
    col = "attr_abs_sample" if XAI_ABS_PER_SAMPLE else "attr_abs"
    other = "attr_abs" if XAI_ABS_PER_SAMPLE else "attr_abs_sample"
    what = {"attr_abs": "mean over the pool of |attribution of the flow|",
            "attr_abs_sample": "mean over the pool AND over the baseline x "
                               "point terms of |term|"}
    names, free, V, W = {}, {}, {}, {}
    for ds in DSS:
        for a in ACT:
            rows = [part(A, ds, a, s).sort_values("feature_idx") for s in SEEDS]
            for r in rows:
                nm = [str(x) for x in r.feature]
                fr = {1: r.free_l1.to_numpy(int), 2: r.free_l2.to_numpy(int)}
                if ds not in names:
                    names[ds], free[ds] = nm, fr
                if nm != names[ds] or any((fr[k] != free[ds][k]).any()
                                          for k in (1, 2)):
                    _stop(f"stage xai: the parts of {ds} do not list the same "
                          f"features / capability sets.")
            V[(ds, a)] = np.vstack([r[col].to_numpy(float) for r in rows])
            W[(ds, a)] = np.vstack([r[other].to_numpy(float) for r in rows])
    mean_vec = {k: v.mean(0) for k, v in V.items()}     # averaged over seeds
    neff = {k: np.array([xai_neff(x) for x in v]) for k, v in V.items()}
    neff_o = {k: np.array([xai_neff(x) for x in v]) for k, v in W.items()}

    def share(vec, sel):
        return float(vec[sel].sum() / vec.sum())

    # saved per-flow arrays, when present, must give the same vectors.
    n_npz = 0
    for ds in DSS:
        for a in ACT:
            for i, s in enumerate(SEEDS):
                p = os.path.join(RES_DIR, "flows", f"xai_{ds}_{a}_s{s}.npz")
                if not os.path.exists(p):
                    continue
                z = np.load(p)
                v = (z["eg_abs_sample"].mean(0) if XAI_ABS_PER_SAMPLE
                     else np.abs(z["eg"]).mean(0))
                if not np.allclose(v, V[(ds, a)][i], rtol=0, atol=1e-9):
                    _stop(f"stage xai: {p} does not reproduce the rows of "
                          f"xaiattrib for {ds}-{a}-{s}.")
                n_npz += 1

    print(f"%% attribution: expected gradients, {XAI_N_BASELINES} baselines "
          f"from the training split x {XAI_N_ALPHA} interpolation points, "
          f"malicious-class logit, whole evaluation pool.")
    print(f"%% a-bar_j = {what[col]} (XAI_ABS_PER_SAMPLE = {XAI_ABS_PER_SAMPLE}).")
    print(f"%% {n_npz} of {len(DSS) * len(ACT) * len(SEEDS)} per-flow files "
          f"present; each reproduces its rows.")

    print("\n%% ---- tab:xai: effective number of features | "
          "PGD ASR eps=0.1 | PGD ASR eps=0.3 (Model 4, L0)")
    for ds in DSS:
        print(f"\\multicolumn{{4}}{{l}}{{\\textit{{{XAI_DS_LABEL[ds]}}}}} \\\\")
        for a, L in AL:
            v = neff[(ds, a)]
            print(f"{L} & {v.mean():.2f}$\\pm${v.std(ddof=SD_DDOF):.2f} & "
                  + " & ".join(_ms(meas(ds, a, 0, e, "pgd").asr)
                               for e in XAI_TAB_EPS) + " \\\\")
        print("\\hline")
    dm = max(abs(meas(ds, a, 0, e, "pgd").asr.mean()
                 - meas(ds, a, 0, e, "bim").asr.mean())
             for ds in DSS for a in ACT for e in XAI_TAB_EPS)
    ds_ = max(float(np.abs(meas(ds, a, 0, e, "pgd").asr.to_numpy()
                           - meas(ds, a, 0, e, "bim").asr.to_numpy()).max())
              for ds in DSS for a in ACT for e in XAI_TAB_EPS)
    print(f"%% caption: largest |PGD - BIM| over the 12 cells of the "
          f"table {dm:.4f} (means), {ds_:.4f} (single seed)")
    print("%% per seed, effective number of features:")
    for ds in DSS:
        for a in ACT:
            print(f"%%   {ds:4s} {a:4s} " + "  ".join(
                f"s{s}: {x:.3f}" for s, x in zip(SEEDS, neff[(ds, a)])))

    print("\n%% ---- breadth of attribution")
    for ds in DSS:
        mu = {a: neff[(ds, a)].mean() for a in ACT}
        sd = {a: neff[(ds, a)].std(ddof=SD_DDOF) for a in ACT}
        print(f"%%   {XAI_DS_LABEL[ds]}: n_eff " + ", ".join(
            f"{a} {mu[a]:.2f}+-{sd[a]:.2f}" for a in ACT)
            + f"; ReLU - ELU = {mu['relu'] - mu['elu']:.2f}, ReLU - Tanh = "
            f"{mu['relu'] - mu['tanh']:.2f}; largest seed SD {max(sd.values()):.2f}"
            f"; order " + " > ".join(sorted(ACT, key=lambda a: -mu[a])))
    for ds, e in (("cic", 0.1), ("unsw", 0.3), ("cic", 0.3), ("unsw", 0.1)):
        print(f"%%   {XAI_DS_LABEL[ds]} PGD ASR at eps={e}: " + ", ".join(
            f"{a} {meas(ds, a, 0, e, 'pgd').asr.mean():.3f}" for a in ACT)
            + "; least evaded: " + min(
                ACT, key=lambda a: meas(ds, a, 0, e, "pgd").asr.mean())
            + "; broadest: " + max(ACT, key=lambda a: neff[(ds, a)].mean()))

    print(f"\n%% ---- the most attributed features (attributions "
          f"averaged over the {len(SEEDS)} seeds)")
    for ds in DSS:
        nm = names[ds]
        rk = {a: xai_rank(mean_vec[(ds, a)]) for a in ACT}
        pos = {a: {int(j): r + 1 for r, j in enumerate(rk[a])} for a in ACT}
        print(f"%%   {XAI_DS_LABEL[ds]}")
        for a in ACT:
            v = mean_vec[(ds, a)]
            print(f"%%     {a:4s} top-{XAI_TOPK_TEXT}: " + ", ".join(
                f"{nm[j]} ({100 * v[j] / v.sum():.2f}%)"
                for j in rk[a][:XAI_TOPK_TEXT]))
        common = [nm[j] for j in rk[ACT[0]][:XAI_TOPK_TEXT]
                  if all(pos[a][int(j)] <= XAI_TOPK_TEXT for a in ACT)]
        print(f"%%     in the top-{XAI_TOPK_TEXT} of every activation: "
              + (", ".join(common) if common else "(none)"))
        for a in ACT:
            j = int(rk[a][0])
            print(f"%%     most attributed for {a}: {nm[j]}; its rank for "
                  + ", ".join(f"{b} {pos[b][j]}" for b in ACT if b != a))
        raw = np.mean([mean_vec[(ds, a)] for a in ACT], 0)
        shr = np.mean([mean_vec[(ds, a)] / mean_vec[(ds, a)].sum()
                       for a in ACT], 0)
        print("%%     averaged over the three activations, top-3: "
              + ", ".join(nm[j] for j in xai_rank(raw)[:3])
              + "  (averaging the normalised shares instead: "
              + ", ".join(nm[j] for j in xai_rank(shr)[:3]) + ")")

    print("\n%% ---- total attribution (sum over the 30 features of the "
          "mean |attribution|), mean+-SD over seeds")
    for ds in DSS:
        tot = {a: V[(ds, a)].sum(1) for a in ACT}
        print(f"%%   {XAI_DS_LABEL[ds]}: " + ", ".join(
            f"{a} {tot[a].mean():.3f}+-{tot[a].std(ddof=SD_DDOF):.3f}"
            for a in ACT) + "; largest: "
            + max(ACT, key=lambda a: tot[a].mean()))

    print("\n%% ---- clean macro-F1 of Model 4 (the base rows, "
          "tab:partd_f1)")
    for ds in DSS:
        print(f"%%   {XAI_DS_LABEL[ds]}: " + ", ".join(
            f"{a} {meas(ds, a, 0, EPS_LIST[0], 'pgd').clean_f1.mean():.3f}"
            for a in ACT))

    # ---- the first-order margin test ----
    def mrows(ds, a, lv, e=None):
        r = M[(M.dataset == ds) & (M.activation == a) & (M.level == lv)
              & np.isclose(M.eps, EPS_LIST[0] if e is None else e)]
        return r.sort_values("seed")

    bad = []
    for ds in DSS:
        for a in ACT:
            for lv in XAI_LEVELS:
                for e in EPS_LIST:
                    r, u = mrows(ds, a, lv, e), meas(ds, a, lv, e, "union")
                    if (list(r.n_correct) != list(u.n_correct)
                            or list(r.n_correct_base) != list(u.n_correct)
                            or (r.n_denominator_diff != 0).any()
                            or list(r.clean_f1.round(4)) != list(u.clean_f1.round(4))
                            or list(r.n_free) != list(u.n_free)):
                        bad.append(f"{ds}-{a} L{lv} eps={e}")
    print("\n%% ---- consistency with the base rows: originally-correct flows "
          "(m > 0), clean F1 and number of free features of every cell")
    print("%%   " + (f"identical in all {len(DSS) * len(ACT) * len(XAI_LEVELS) * len(EPS_LIST)} "
                     f"cells x {len(SEEDS)} seeds" if not bad else
                     "!! DIFFERENT in: " + ", ".join(bad)))

    cells = []
    for ds in DSS:
        for lv in XAI_LEVELS:
            for e in EPS_LIST:
                for a in ACT:
                    u = meas(ds, a, lv, e, "union").asr.to_numpy(float)
                    cells.append(dict(
                        dataset=ds, level=lv, eps=e, activation=a,
                        pred=float(mrows(ds, a, lv, e).pred_asr.mean()),
                        measured=float(u.mean()), ms=_ms(u)))
    E = pd.DataFrame(cells)
    E["error"] = E.pred - E.measured
    E["p3"] = E.pred.map(lambda x: f"{x:.3f}")
    E["m3"] = E.measured.map(lambda x: f"{x:.3f}")

    def cell(ds, lv, e, a):
        return E[(E.dataset == ds) & (E.level == lv) & np.isclose(E.eps, e)
                 & (E.activation == a)].iloc[0]

    def where(x):
        return (f"{x.dataset} L{int(x.level)} eps={x.eps} {x.activation}: "
                f"predicted {x.pred:.3f}, measured {x.measured:.3f}")

    nfree = {ds: int(free[ds][2].sum()) for ds in DSS}
    print("\n%% ---- tab:gradmargin: Pred. | Measured (union of "
          "PGD, BIM, SPSA, transfer; base rows) for ReLU, ELU, Tanh")
    print("%% S: all " + str(FEATS) + " features at L0; at L2 " + ", ".join(
        f"{nfree[ds]} on {XAI_DS_LABEL[ds]}" for ds in DSS))
    for ds in DSS:
        print(f"\\multicolumn{{8}}{{l}}{{\\textit{{{XAI_DS_LABEL[ds]}}}}} \\\\")
        for lv in XAI_LEVELS:
            for e in EPS_LIST:
                print(f"L{lv} & $\\epsilon={e}$ & " + " & ".join(
                    f"{cell(ds, lv, e, a).p3} & {cell(ds, lv, e, a).ms}"
                    for a in ACT) + " \\\\")
        print("\\hline")

    Q = {}
    for ds in DSS:
        for a in ACT:
            r0, r2 = mrows(ds, a, 0), mrows(ds, a, 2)
            Q[(ds, a)] = dict(
                margin=float(r0.mean_margin.mean()),
                g0=float(r0.mean_grad_l1.mean()), g2=float(r2.mean_grad_l1.mean()),
                e0=float(r0.median_eps_hat.mean()),
                e2=float(r2.median_eps_hat.mean()),
                zero=float(max(r0.frac_zero_grad.max(), r2.frac_zero_grad.max())),
                same=bool(np.allclose(
                    pd.concat([r0, r2]).median_eps_hat,
                    pd.concat([r0, r2]).median_eps_hat_finite, equal_nan=True)))
    print("\n%% ---- tab:gradquant: Margin | |g|_1 L0 | |g|_1 L2 "
          "| Retained | median eps-hat L0 | L2  (means over the seeds)")
    for ds in DSS:
        print(f"\\multicolumn{{7}}{{l}}{{\\textit{{{XAI_DS_LABEL[ds]}}}}} \\\\")
        for a, L in AL:
            q = Q[(ds, a)]
            print(f"{L.strip()} & {q['margin']:.2f} & {q['g0']:.1f} & "
                  f"{q['g2']:.1f} & {100 * q['g2'] / q['g0']:.1f}\\% & "
                  f"{q['e0']:.3f} & {q['e2']:.3f} \\\\")
        print("\\hline")
    zmax = max(q["zero"] for q in Q.values())
    print(f"%% largest fraction of originally-correct flows with |g_S|_1 = 0 "
          f"(eps-hat = infinity) in any model and level: {zmax:.4f}"
          + ("" if all(q["same"] for q in Q.values()) else
             "  -- the medians above INCLUDE those flows; the older scripts "
             "took the median over finite values only (column "
             "median_eps_hat_finite)"))

    def line(name, sub):
        r = float("nan")
        if len(sub) > 2 and sub.pred.std() > 0 and sub.measured.std() > 0:
            r = float(np.corrcoef(sub.pred, sub.measured)[0, 1])
        w = sub.loc[sub.error.abs().idxmax()]
        print(f"%%   {name:36s} n={len(sub):2d}  Pearson r {r:.3f}  mean "
              f"|pred - meas| {sub.error.abs().mean():.3f}  mean (pred - meas) "
              f"{sub.error.mean():+.3f}  worst cell {abs(w.error):.3f} "
              f"({where(w)})")

    print("\n%% ---- predicted against measured, seed means, the "
          f"{len(E)} cells of tab:gradmargin")
    noT, L2, L0 = E[E.activation != "tanh"], E[E.level == 2], E[E.level == 0]
    line("all cells", E)
    line("excluding Tanh", noT)
    line("constrained (L2)", L2)
    line("unconstrained (L0)", L0)
    for ds in DSS:
        line(f"L2, {XAI_DS_LABEL[ds]}", L2[L2.dataset == ds])
        sub = L2[L2.dataset == ds]
        print(f"%%     of which equal to three decimals: "
              f"{int((sub.p3 == sub.m3).sum())} of {len(sub)}")
    line("L2, ReLU and ELU", noT[noT.level == 2])
    for ds in DSS:
        line(f"L2, ReLU and ELU, {XAI_DS_LABEL[ds]}",
             noT[(noT.level == 2) & (noT.dataset == ds)])
    big = E[E.error.abs() > XAI_DIFF_NOTED]
    print(f"%%   cells differing by more than {XAI_DIFF_NOTED}: "
          f"{len(big)} of {len(E)}; prediction below measurement in "
          f"{int((big.error < 0).sum())}, above in {int((big.error > 0).sum())}")
    for x in big[big.error > 0].itertuples():
        print(f"%%     above: {where(x)} ({x.error:+.3f})")
    T = E[E.activation == "tanh"]
    wt = T.loc[T.error.abs().idxmax()]
    print(f"%%   Tanh, {len(T)} cells: prediction below measurement "
          f"in {int(((T.error < 0) & (T.p3 != T.m3)).sum())}, equal to three "
          f"decimals in {int((T.p3 == T.m3).sum())}, above in "
          f"{int(((T.error > 0) & (T.p3 != T.m3)).sum())}; mean |pred - meas| "
          f"{T.error.abs().mean():.3f}; largest {abs(wt.error):.3f} ({where(wt)})")
    print("%% every cell (prediction, measurement, difference):")
    print("\n".join("%%   " + s for s in E[
        ["dataset", "level", "eps", "activation", "pred", "measured", "error"]
    ].round(4).to_string(index=False).splitlines()))

    print("\n%% ---- margin against gradient norm")
    for ds in DSS:
        q = {a: Q[(ds, a)] for a in ACT}
        lo, hi = sorted((q["relu"]["margin"], q["elu"]["margin"]))
        print(f"%%   {XAI_DS_LABEL[ds]}: margins " + ", ".join(
            f"{a} {q[a]['margin']:.2f}" for a in ACT)
            + f"; ReLU and ELU differ by {100 * (hi - lo) / lo:.1f}% of the "
            f"smaller ({100 * (hi - lo) / hi:.1f}% of the larger); |g|_1 at L0 "
            + ", ".join(f"{a} {q[a]['g0']:.1f}" for a in ACT)
            + f"; ELU / ReLU = {q['elu']['g0'] / q['relu']['g0']:.2f}")
        print(f"%%     retained under the {nfree[ds]} source-side features "
              f"({FEATS - nfree[ds]} frozen): " + ", ".join(
                  f"{a} {100 * q[a]['g2'] / q[a]['g0']:.1f}%" for a in ACT)
              + f"; equal weighting would retain {nfree[ds]}/{FEATS} = "
              f"{100 * nfree[ds] / FEATS:.1f}%")
        print("%%     ReLU relative to the other two: margin "
              + ", ".join(f"ReLU/{a} {q['relu']['margin'] / q[a]['margin']:.2f}"
                          for a in ACT[1:]) + "; retained share "
              + ", ".join(
                  f"ReLU/{a} "
                  f"{(q['relu']['g2'] / q['relu']['g0']) / (q[a]['g2'] / q[a]['g0']):.2f}"
                  for a in ACT[1:]))
        e = EPS_LIST[0]
        print(f"%%     measured union at L0, eps={e}: " + ", ".join(
            f"{a} {cell(ds, 0, e, a).measured:.3f}" for a in ACT))
        p = [cell(ds, 2, e, a) for a in ACT]
        print(f"%%     predicted at L2, eps={e}: " + ", ".join(
            f"{a} {x.pred:.3f}" for a, x in zip(ACT, p))
            + f"; spread {max(x.pred for x in p) - min(x.pred for x in p):.3f}"
            f"; largest |pred - meas| {max(abs(x.error) for x in p):.3f}")
    print("\n%% ---- median predicted budget at L2 and the measured "
          "L2 rates")
    for ds in DSS:
        sub = L2[L2.dataset == ds]
        print(f"%%   {XAI_DS_LABEL[ds]}: median eps-hat at L2 " + ", ".join(
            f"{a} {Q[(ds, a)]['e2']:.3f}" for a in ACT)
            + f" (largest budget evaluated {max(EPS_LIST)}); measured L2 union "
            f"over the {len(sub)} cells {100 * sub.measured.min():.1f}%-"
            f"{100 * sub.measured.max():.1f}%")

    print("\n%% ---- share of attribution mass on the source-side "
          "(L2-free) features; attributions averaged over seeds")
    spread = {}
    for ds in DSS:
        sel = free[ds][2] > 0
        u = sel.sum() / FEATS
        sh = {a: share(mean_vec[(ds, a)], sel) for a in ACT}
        ps = {a: np.array([share(v, sel) for v in V[(ds, a)]]) for a in ACT}
        spread[ds] = max(sh.values()) - min(sh.values())
        print(f"%%   {XAI_DS_LABEL[ds]}: {int(sel.sum())} features ("
              + ", ".join(n for n, f in zip(names[ds], sel) if f)
              + f"); equal weighting {int(sel.sum())}/{FEATS} = {u:.4f}")
        print("%%     " + ", ".join(
            f"{a} {sh[a]:.4f} ({sh[a] / u:.2f}x)" for a in ACT)
            + f"; spread {spread[ds]:.4f}")
        print("%%     per seed, mean+-SD: " + ", ".join(
            f"{a} {ps[a].mean():.4f}+-{ps[a].std(ddof=SD_DDOF):.4f}"
            for a in ACT))
    if spread["unsw"] > 0:
        print(f"%%   spread on {XAI_DS_LABEL['cic']} / spread on "
              f"{XAI_DS_LABEL['unsw']} = {spread['cic'] / spread['unsw']:.1f}")
    e = EPS_LIST[1]
    print(f"%%   measured union at L2, eps={e}, "
          f"{XAI_DS_LABEL['cic']}: " + ", ".join(
              f"{a} {cell('cic', 2, e, a).ms}" for a in ACT))

    print("\n%% ---- share on " + " and ".join(XAI_UNSW_PAIR)
          + f" ({XAI_DS_LABEL['unsw']})")
    idx = [names["unsw"].index(n) for n in XAI_UNSW_PAIR if n in names["unsw"]]
    if len(idx) != len(XAI_UNSW_PAIR):
        print("%%   not all of them are among the " + str(FEATS)
              + " selected features of this run: "
              + ", ".join(n for n in XAI_UNSW_PAIR if n not in names["unsw"])
              + " missing")
    else:
        print("%%   " + ", ".join(
            f"{a} {share(mean_vec[('unsw', a)], idx):.4f}" for a in ACT)
            + "; free at L2: " + ", ".join(
                f"{names['unsw'][j]} {'yes' if free['unsw'][2][j] else 'no'}"
                for j in idx))
        both = (free["unsw"][2] > 0)
        both[idx] = True
        print(f"%%   share if they were left free ({int(both.sum())} features): "
              + ", ".join(f"{a} {share(mean_vec[('unsw', a)], both):.4f}"
                          for a in ACT)
              + f"; equal weighting {int(both.sum())}/{FEATS} = "
              f"{both.sum() / FEATS:.4f}")

    print("\n%% ---- share on the categorical features (frozen at "
          "L1 and L2)")
    for ds in DSS:
        sel = free[ds][1] == 0
        print(f"%%   {XAI_DS_LABEL[ds]}: {int(sel.sum())} feature(s) ("
              + ", ".join(n for n, f in zip(names[ds], sel) if f) + ")")
        print("%%     " + ", ".join(
            f"{a} {share(mean_vec[(ds, a)], sel):.4f}" for a in ACT))

    print("\n%% ---- orderings (largest first)")
    for ds in DSS:
        o = lambda f: " > ".join(sorted(ACT, key=lambda a: -f(a)))
        print(f"%%   {XAI_DS_LABEL[ds]}: n_eff " + ", ".join(
            f"{neff[(ds, a)].mean():.2f}" for a in ACT)
            + f" ({o(lambda a: neff[(ds, a)].mean())}); |g|_1 over all "
            f"features " + ", ".join(f"{Q[(ds, a)]['g0']:.1f}" for a in ACT)
            + f" ({o(lambda a: Q[(ds, a)]['g0'])}); |g_S|_1 over the "
            f"source-side features " + ", ".join(
                f"{Q[(ds, a)]['g2']:.1f}" for a in ACT)
            + f" ({o(lambda a: Q[(ds, a)]['g2'])})")
        print("%%     unrestricted and restricted gradient norm rank the "
              "activations "
              + ("the SAME way" if o(lambda a: Q[(ds, a)]["g0"])
                 == o(lambda a: Q[(ds, a)]["g2"]) else "differently"))

    print(f"\n%% ---- the other aggregation, same draws: a-bar_j = {what[other]}")
    print("%% (not used above; shown because Section 3.4 and the script behind "
          "the published tab:xai differ on this point)")
    for ds in DSS:
        sel = free[ds][2] > 0
        print(f"%%   {XAI_DS_LABEL[ds]}: n_eff " + ", ".join(
            f"{a} {neff_o[(ds, a)].mean():.2f}+-"
            f"{neff_o[(ds, a)].std(ddof=SD_DDOF):.2f}" for a in ACT)
            + "; source-side share " + ", ".join(
                f"{a} {share(W[(ds, a)].mean(0), sel):.4f}" for a in ACT))

    print("\n%% ---- estimator check: |sum of a flow's attributions - (f(x) - "
          "mean f(baseline))| / |f(x) - mean f(baseline)|, pool means")
    for ds in DSS:
        out = []
        for a in ACT:
            r = C[(C.dataset == ds) & (C.activation == a)]
            out.append(f"{a} {r.mean_abs_completeness_gap.mean():.3f} / "
                       f"{r.mean_abs_delta_out.mean():.3f}")
        print(f"%%   {XAI_DS_LABEL[ds]}: " + ", ".join(out))

    files = xai_figures(mean_vec, names)
    print(f"\n%% ---- fig:xai: top-{XAI_TOPK_FIG} features ranked "
          f"by each activation, {XAI_DS_LABEL['cic']} left, "
          f"{XAI_DS_LABEL['unsw']} right, averaged over {len(SEEDS)} seeds")
    for f in files:
        print(f"%%   wrote {f}")


STAGES.append(("xai", xai_stage, xai_summarize))

# ======== stage caat: code ========
# Capability-aware adversarial training.
def caat_mask(c):
    m = c.masks[CAAT_MASK_LEVEL]
    if m is None:
        _stop(f"CAAT_MASK_LEVEL={CAAT_MASK_LEVEL} has no mask: the inner attack "
              f"would be unconstrained, which is pgd_at, not ca_at.")
    return np.asarray(m, dtype=np.float32).reshape(-1)


def caat_attack_seed(c, lv, eps, a):
    return _seed_of("caat", c.ds, DEPTH, c.Xtr.shape[1], c.act, c.seed,
                    CAAT_ROLE, lv, eps, a)


def caat_train_settings(c):
    """The pgd_at settings plus the one thing that differs."""
    m = caat_mask(c)
    cfg = dict(at_train_settings(c))
    cfg["inner_mask_level"] = CAAT_MASK_LEVEL
    cfg["inner_mask"] = tuple(int(v) for v in m)
    cfg["inner_free"] = tuple(str(n) for n, v in zip(c.names, m) if v > 0)
    return cfg


def caat_pgd_batch(model, X, y, m):
    model.eval()
    x0 = X.clone().detach()
    x = x0 + torch.empty_like(x0).uniform_(-AT_EPS, AT_EPS) * m
    x = torch.where(m > 0, x, x0)
    step = AT_EPS / AT_STEPS * AT_STEP_FACTOR
    for _ in range(AT_STEPS):
        x = x.clone().detach().requires_grad_(True)
        loss = F.cross_entropy(model(x), y)
        g, = torch.autograd.grad(loss, x)
        upd = step * g.sign() * m
        x = torch.clamp(x0 + torch.clamp(x + upd - x0, -AT_EPS, AT_EPS),
                        -AT_CLAMP, AT_CLAMP)
        x = torch.where(m > 0, x, x0)
    model.train()
    return x.detach()


def caat_train(model, Xtrn, ytrn, Xval, yval, tag, mask):
    m = torch.as_tensor(np.asarray(mask, dtype=np.float32).reshape(1, -1),
                        device=DEVICE)
    frozen = (m.reshape(-1) == 0)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    Xt = torch.tensor(Xtrn, dtype=torch.float32)
    yt = torch.tensor(ytrn, dtype=torch.long)
    n, best, stale, done, t0 = len(Xt), -1.0, 0, 0, time.time()
    while done < MAX_EPOCHS:
        model.train()
        for _ in range(CHUNK):
            perm = torch.randperm(n)
            for i in range(0, n, BATCH_SIZE):
                idx = perm[i:i + BATCH_SIZE]
                xb = Xt[idx].to(DEVICE); yb = yt[idx].to(DEVICE)
                nadv = int(len(xb) * AT_RATIO)
                if nadv > 0:
                    adv = caat_pgd_batch(model, xb[:nadv], yb[:nadv], m)
                    if not torch.equal(adv[:, frozen], xb[:nadv][:, frozen]):
                        _stop(f"{tag}: the inner attack moved a frozen feature.")
                    xb = torch.cat([adv, xb[nadv:]], 0)
                opt.zero_grad()
                F.cross_entropy(model(xb), yb).backward()
                opt.step()
        done += CHUNK
        f1 = macro_f1(model, Xval, yval, DEVICE)
        print(f"      {tag}: {done} epochs, validation F1 {f1:.4f} "
              f"[{time.time() - t0:.0f}s]", flush=True)
        if f1 > best + TOL:
            best, stale = f1, 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.eval()
    return best, done


def caat_get_model(c):
    """The CA-AT Model 4 of this part."""
    k = c.Xtr.shape[1]
    p = model_path(c.ds, DEPTH, k, c.act, c.seed, CAAT_ROLE)
    cfg = caat_train_settings(c)
    if os.path.exists(p):
        blob = torch.load(p, map_location=DEVICE, weights_only=False)
        if blob.get("train") != cfg:
            _stop(f"{p} was trained with other settings than this file's:\n"
                  f"  saved   {blob.get('train')}\n  current {cfg}\n"
                  f"Move it away and re-run this part.")
        model = build_model(DEPTH, k, c.act).to(DEVICE)
        model.load_state_dict(blob["state_dict"]); model.eval()
        f1 = macro_f1(model, c.Xte, c.yte, DEVICE)
        if abs(f1 - blob["clean_f1"]) > 1e-9:
            _stop(f"{p} re-measures clean F1 {f1:.6f}, saved "
                  f"{blob['clean_f1']:.6f}: the data or code changed since it "
                  f"was trained.")
        return model, f1, int(blob["epochs"]), "saved"
    Xtrn, Xval, ytrn, yval = c.split()
    _reseed(cfg["train_seed"]); t0 = time.time()
    model = build_model(DEPTH, k, c.act).to(DEVICE)
    best, epochs = caat_train(model, Xtrn, ytrn, Xval, yval,
                              f"{CAAT_ROLE} {c.part}", caat_mask(c))
    model.eval()
    f1 = macro_f1(model, c.Xte, c.yte, DEVICE)
    os.makedirs(MODEL_DIR, exist_ok=True)
    tmp = p + f".tmp{os.getpid()}"
    torch.save({"state_dict": model.state_dict(), "clean_f1": float(f1),
                "epochs": int(epochs), "val_best": float(best), "ds": c.ds,
                "depth": DEPTH, "features": k, "activation": c.act,
                "seed": c.seed, "role": CAAT_ROLE, "train": cfg,
                "train_s": round(time.time() - t0, 1),
                "script": "rerun_all.py"}, tmp)
    os.replace(tmp, p)
    return model, f1, int(epochs), "trained"


def caat_attack(c, model, clf, lv, eps, a, s):
    try:
        return at_attack(c, model, clf, lv, eps, a, s)
    except RuntimeError as e:
        if "capability violation" in str(e):
            print(f"!! the violation above was on the {CAAT_ROLE} model of "
                  f"{c.part} (L{lv} eps={eps} {a}), not on {AT_ROLE}.",
                  flush=True)
        raise


def stage_caat(c):
    have = stage_have("caatattacks", AT_KEY)
    havef = stage_have("caatfpr", AT_FKEY)
    key = lambda lv, eps, a: _k(c.ds, c.act, c.seed, CAAT_ROLE, lv,
                                round(eps, 3), a)
    want = {key(lv, e, a) for lv in AT_LEVELS for e in EPS_LIST
            for a in AT_ATTACKS + ["union"]}
    fkey = _k(c.ds, c.act, c.seed, CAAT_ROLE)
    if want <= have and fkey in havef:
        print(f"    skip {CAAT_ROLE} {c.part}: done", flush=True)
        return
    k = c.Xtr.shape[1]
    t0 = time.time()
    model, f1, epochs, src = caat_get_model(c)
    free = [n for n, v in zip(c.names, caat_mask(c)) if v > 0]
    print(f"    {CAAT_ROLE} {c.part}: {src}, {epochs} epochs, cleanF1={f1:.4f} "
          f"(standard {c.f1:.4f}); inner attack on {len(free)} of {k} "
          f"features {free} [{time.time() - t0:.0f}s]", flush=True)
    clf = _art_clf(model, c.Xtr, DEVICE)
    correct = _probs(model, c.Xe, DEVICE).argmax(1) == c.ye
    n_ok = max(int(correct.sum()), 1)

    if fkey not in havef:
        row = at_operating_row(c, model, CAAT_ROLE, f1, epochs, correct)
        stage_append("caatfpr", AT_FCOLS, row)
        print(f"    {CAAT_ROLE:8s} AUC {row['roc_auc']:.5f}  FPR@TPR"
              f"{AT_TPR_TABLE} {row[at_tpr_col(AT_TPR_TABLE)]:.5f}  at "
              f"{AT_THRESHOLD}: FPR {row['fpr_at_thr']:.5f} TPR "
              f"{row['tpr_at_thr']:.5f}  pool {row['n_pool_correct']}/"
              f"{row['n_pool']}", flush=True)

    npz = stage_npz("caat", c)
    saved = npz_load(npz)
    saved["pool"], saved["correct"] = c.pool, correct
    base = dict(dataset=c.ds, depth=DEPTH, features=k, activation=c.act,
                seed=c.seed, defence=CAAT_ROLE, clean_f1=round(float(f1), 4))

    # -- PGD, BIM, SPSA, transfer and their per-flow union, L0 and L2.
    for lv in AT_LEVELS:
        nfree = k if c.masks[lv] is None else int(c.masks[lv].sum())
        for eps in EPS_LIST:
            if key(lv, eps, "union") in have:
                continue
            union, rates = None, {}
            for a in AT_ATTACKS:
                s = caat_attack_seed(c, lv, eps, a)
                t2 = time.time()
                h = caat_attack(c, model, clf, lv, eps, a, s)
                saved[f"L{lv}_e{eps}_{a}"] = h
                union = h if union is None else (union | h)
                rates[a] = float(h.sum() / n_ok)
                stage_append("caatattacks", AT_COLS, dict(
                    base, level=lv, eps=round(eps, 3), attack=a,
                    asr=round(rates[a], 4), n_correct=n_ok, n_hit=int(h.sum()),
                    n_free=nfree, attack_seed=s,
                    elapsed_s=round(time.time() - t2, 1)))
            saved[f"L{lv}_e{eps}_union"] = union
            npz_save(npz, saved)
            stage_append("caatattacks", AT_COLS, dict(
                base, level=lv, eps=round(eps, 3), attack="union",
                asr=round(float(union.sum() / n_ok), 4), n_correct=n_ok,
                n_hit=int(union.sum()), n_free=nfree, attack_seed=0,
                elapsed_s=0))
            have.add(key(lv, eps, "union"))
            print(f"    {CAAT_ROLE} L{lv} eps={eps}: " + " ".join(
                f"{a.upper()} {rates[a]:.3f}" for a in AT_ATTACKS)
                + f" | union {union.sum() / n_ok:.3f}", flush=True)


def caat_sign_p(k, n):
    """Two-sided exact sign test."""
    if n == 0:
        return 1.0
    from math import comb
    kk = max(k, n - k)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(kk, n + 1)) / 2 ** n)


def summarize_caat():
    a = stage_read("caatattacks", AT_KEY)
    f = stage_read("caatfpr", AT_FKEY)
    if a is None or f is None:
        print("%% stage caat: not run yet")
        return
    if not os.path.exists(OUT_FINAL):
        _stop(f"{OUT_FINAL} is missing: the standard-model columns are read "
              f"from it. Run RERUN_PART=summary (it writes that file first).")
    pa = stage_read("atattacks", AT_KEY)        # the advtrain stage, if run
    pf = stage_read("atfpr", AT_FKEY)
    b = pd.read_csv(OUT_FINAL)
    b = b[(b.depth == DEPTH) & (b.features == FEATS)
          & b.level.isin(AT_LEVELS)].copy()
    b["defence"] = "standard"
    cols = ["dataset", "activation", "seed", "defence", "level", "eps",
            "attack", "asr", "n_correct", "n_hit", "clean_f1"]
    parts = [b[cols], a[(a.depth == DEPTH) & (a.features == FEATS)][cols]]
    if pa is not None:
        parts.append(pa[(pa.depth == DEPTH) & (pa.features == FEATS)
                        & (pa.defence == AT_ROLE)][cols])
    d = pd.concat(parts, ignore_index=True)
    d = d[d.attack.isin(AT_ATTACKS + ["union"])].copy()
    d["eps"] = d.eps.astype(float).round(3)
    fr = pd.concat([f] + ([pf] if pf is not None else []), ignore_index=True)
    lab = (("cic", "CSE-CIC-IDS2018"), ("unsw", "UNSW-NB15"))
    acts = (("relu", "ReLU"), ("elu", "ELU "), ("tanh", "Tanh"))
    NAME = {"standard": "Standard", AT_ROLE: "PGD-AT", CAAT_ROLE: "CA-AT"}
    tcol = at_tpr_col(AT_TPR_TABLE)

    def q(df, **kw):
        m = np.ones(len(df), bool)
        for col, v in kw.items():
            m &= (df[col] == v).values
        return df[m]

    def missing(ds, dfc, need_fpr):
        miss = 0
        need = {(lv, round(e, 3), x) for lv in AT_LEVELS for e in EPS_LIST
                for x in AT_ATTACKS + ["union"]}
        for ac in ACTIVATIONS:
            for s in SEEDS:
                r = q(d, dataset=ds, activation=ac, seed=s, defence=dfc)
                miss += len(need - set(zip(r.level.astype(int), r.eps,
                                           r.attack)))
                if need_fpr:
                    miss += int(len(q(fr, dataset=ds, activation=ac, seed=s,
                                      defence=dfc)) != 1)
        return miss

    def sv(ds, dfc, lv, e, col="asr"):
        r = q(d, dataset=ds, defence=dfc, level=lv, eps=round(e, 3),
              attack="union")
        return np.array([float(r[r.seed == s][col].mean()) for s in SEEDS])

    def pooled(ds, dfc, lv, e):
        r = q(d, dataset=ds, defence=dfc, level=lv, eps=round(e, 3),
              attack="union")
        return int(r.n_hit.sum()), int(r.n_correct.sum())

    def per_model(ds, dfc, lv, e, col="n_hit"):
        r = q(d, dataset=ds, defence=dfc, level=lv, eps=round(e, 3),
              attack="union").set_index(["activation", "seed"])[col]
        return np.array([float(r.loc[(ac, s)]) for ac in ACTIVATIONS
                         for s in SEEDS])

    def f1s(ds, dfc, ac=None):
        """clean macro-F1 per seed as stored with the attack rows (4 dp)."""
        r = q(d, dataset=ds, defence=dfc)
        if ac is not None:
            r = r[r.activation == ac]
        g = r.groupby(["activation", "seed"]).clean_f1
        if (g.nunique() != 1).any():
            _stop(f"{ds} {dfc}: clean F1 differs between rows of one model")
        m = g.first().reset_index()
        return np.array([float(m[m.seed == s].clean_f1.mean()) for s in SEEDS])

    def fs(ds, dfc, col):
        r = q(fr, dataset=ds, defence=dfc)
        return np.array([float(r[r.seed == s][col].mean()) for s in SEEDS])

    print(f"%% CA-AT = PGD-AT (eps={AT_EPS}, {AT_STEPS} inner steps, {AT_RATIO} "
          f"of each batch, one Adam, same seed and split) with the inner attack "
          f"confined to the L{CAAT_MASK_LEVEL} mask. Standard = rows of "
          f"attacks_final.csv; PGD-AT = the advtrain stage's rows.")
    if pa is None or pf is None:
        print("%% the advtrain stage has NOT been run: PGD-AT is absent below, "
              "and so is the standard model's operating characteristic (it is "
              "measured by that stage).")
    for ds, L in lab:
        if missing(ds, "standard", False) or missing(ds, CAAT_ROLE, True):
            print(f"\n%% stage caat: {L} is INCOMPLETE ("
                  f"{missing(ds, 'standard', False)} standard and "
                  f"{missing(ds, CAAT_ROLE, True)} {CAAT_ROLE} measurements "
                  f"missing); nothing is printed for it")
            continue
        has_at = (pa is not None and pf is not None
                  and not missing(ds, AT_ROLE, True))
        has_stdf = (pf is not None
                    and not missing(ds, "standard", True))
        if pa is not None and not has_at:
            print(f"\n%% {L}: the advtrain stage is INCOMPLETE for this "
                  f"dataset; PGD-AT is left out")
        REG = ["standard"] + ([AT_ROLE] if has_at else []) + [CAAT_ROLE]
        nmod = len(ACTIVATIONS) * len(SEEDS)

        print(f"\n%% ---- CA-AT table 1, {L}: worst-case ASR (per-flow union "
              f"of {', '.join(AT_ATTACKS)}); per regime: mean$\\pm$SD over the "
              f"seeds of the activation-averaged rate & evaded / "
              f"originally-correct flows pooled over the {nmod} models")
        print("%% Level & Budget & " + " & ".join(
            f"{NAME[x]} & {NAME[x]} flows" for x in REG) + " \\\\")
        print(f"\\multicolumn{{{2 + 2 * len(REG)}}}{{l}}{{\\textit{{{L}}}}} \\\\")
        for lv in AT_LEVELS:
            for e in EPS_LIST:
                cells = []
                for x in REG:
                    h, n = pooled(ds, x, lv, e)
                    cells += [at_ms(sv(ds, x, lv, e)), f"{h:,}/{n:,}"]
                print(f"L{lv} & $\\epsilon={e}$ & " + " & ".join(cells)
                      + " \\\\")
        print("\\hline")

        print(f"%% ---- CA-AT table 2, {L}: cost. Regime & clean macro-F1 & "
              f"ROC-AUC & FPR at TPR = {AT_TPR_TABLE} & FPR at threshold "
              f"{AT_THRESHOLD} & TPR at threshold {AT_THRESHOLD} & epochs "
              f"mean (min--max); mean$\\pm$SD over the seeds of the "
              f"activation-averaged value")
        for x in REG:
            if x == "standard" and not has_stdf:
                oc = ["--"] * 5
            else:
                r = q(fr, dataset=ds, defence=x)
                oc = [at_ms(fs(ds, x, col), 5) for col in
                      ("roc_auc", tcol, "fpr_at_thr", "tpr_at_thr")]
                oc.append(f"{r.epochs.mean():.1f} ({int(r.epochs.min())}--"
                          f"{int(r.epochs.max())})")
            print(f"{NAME[x]} & {at_ms(f1s(ds, x), 4)} & " + " & ".join(oc)
                  + " \\\\")
        print("\\hline")
        print("%% clean macro-F1 by activation (layout of tab:advtrain_cost): "
              + " | ".join(NAME[x] for x in REG) + " | cost of CA-AT = "
              "Standard - CA-AT (unrounded means)")
        for ac, A in acts:
            v = {x: f1s(ds, x, ac) for x in REG}
            print(f"{A} & " + " & ".join(at_ms(v[x]) for x in REG)
                  + f" & {v['standard'].mean() - v[CAAT_ROLE].mean():.3f} \\\\")
        gap = max(abs(round(x.clean_f1, 4) - float(
            q(d, dataset=ds, activation=x.activation, seed=x.seed,
              defence=CAAT_ROLE).clean_f1.iloc[0]))
            for x in q(f, dataset=ds).itertuples())
        print(f"%% clean F1 of the caatfpr rows vs the caatattacks rows: max "
              f"difference {gap:.4f} (same models)")

        print(f"%% ---- derived statements, {L}")
        m = lambda x, lv, e: sv(ds, x, lv, e).mean()
        for e in EPS_LIST:
            s2, c2 = m("standard", 2, e), m(CAAT_ROLE, 2, e)
            hs, ns = pooled(ds, "standard", 2, e)
            hc, nc = pooled(ds, CAAT_ROLE, 2, e)
            txt = (f"  [D1] L2 eps={e}: standard {s2:.4f} ({hs:,}/{ns:,} flows)"
                   f" -> CA-AT {c2:.4f} ({hc:,}/{nc:,}), reduction "
                   f"{s2 - c2:+.4f}")
            if has_at:
                p2 = m(AT_ROLE, 2, e)
                hp, npg = pooled(ds, AT_ROLE, 2, e)
                txt += (f"; PGD-AT {p2:.4f} ({hp:,}/{npg:,}), reduction "
                        f"{s2 - p2:+.4f}; CA-AT achieves "
                        + (f"{100 * (s2 - c2) / (s2 - p2):.1f}% of PGD-AT's "
                           f"absolute L2 reduction" if s2 - p2 > 0 else
                           "an undefined share (PGD-AT's reduction is not "
                           "positive)"))
            print(txt)
        f0, f1c = f1s(ds, "standard"), f1s(ds, CAAT_ROLE)
        dif = f1c - f0
        sd0 = f0.std(ddof=SD_DDOF)
        print(f"  [D2] clean macro-F1 (activation-averaged): standard "
              f"{f0.mean():.4f}, CA-AT {f1c.mean():.4f}; CA-AT - standard = "
              f"{dif.mean():+.4f} (per seed " + ", ".join(
                  f"{v:+.4f}" for v in dif) + f"); seed SD of the standard "
              f"models {sd0:.4f}, of CA-AT {f1c.std(ddof=SD_DDOF):.4f}; "
              f"|difference| / standard seed SD = "
              + (f"{abs(dif.mean()) / sd0:.2f}" if sd0 > 0 else "undefined"))
        if has_at:
            fp = f1s(ds, AT_ROLE)
            print(f"       PGD-AT {fp.mean():.4f}; PGD-AT - standard = "
                  f"{(fp - f0).mean():+.4f}")
        if has_at:
            tot_c = np.zeros(nmod); tot_p = np.zeros(nmod)
            for e in list(EPS_LIST) + ["all budgets summed"]:
                if isinstance(e, str):
                    hc_, hp_ = tot_c, tot_p
                else:
                    hc_ = per_model(ds, CAAT_ROLE, 2, e)
                    hp_ = per_model(ds, AT_ROLE, 2, e)
                    tot_c += hc_; tot_p += hp_
                more, less = int((hc_ > hp_).sum()), int((hc_ < hp_).sum())
                tie = nmod - more - less
                print(f"  [D3] L2 {'eps=' + str(e) if not isinstance(e, str) else e}"
                      f": of the {nmod} models, CA-AT has MORE evaded flows "
                      f"than PGD-AT in {more}, fewer in {less}, the same "
                      f"number in {tie}; two-sided sign test on the "
                      f"{more + less} untied models p = "
                      f"{caat_sign_p(more, more + less):.4f}; evaded flows "
                      f"CA-AT {int(hc_.sum())} vs PGD-AT {int(hp_.sum())}")
        else:
            print("  [D3] needs PGD-AT (advtrain stage): not printed")
        for e in EPS_LIST:
            s0, c0 = m("standard", 0, e), m(CAAT_ROLE, 0, e)
            hs, ns = pooled(ds, "standard", 0, e)
            hc, nc = pooled(ds, CAAT_ROLE, 0, e)
            rs = per_model(ds, "standard", 0, e, "asr")
            rc = per_model(ds, CAAT_ROLE, 0, e, "asr")
            lo, hi = int((rc < rs).sum()), int((rc > rs).sum())
            txt = (f"  [D4] L0 eps={e}: standard {s0:.4f} ({hs:,}/{ns:,}) vs "
                   f"CA-AT {c0:.4f} ({hc:,}/{nc:,}), CA-AT - standard = "
                   f"{c0 - s0:+.4f}; CA-AT lower in {lo} of {nmod} models, "
                   f"higher in {hi}, sign test p = "
                   f"{caat_sign_p(lo, lo + hi):.4f}")
            if has_at:
                txt += f"; PGD-AT {m(AT_ROLE, 0, e):.4f}"
            print(txt)
        npool = q(f, dataset=ds).n_pool.unique()
        if len(npool) != 1:
            _stop(f"{ds}: the pool size differs between parts: {npool}")
        N = int(npool[0])
        for x in REG:
            r = q(d, dataset=ds, defence=x, attack="union")
            tpr = r.groupby(["activation", "seed"]).n_correct.first().mean() / N
            urd = {}
            for lv in AT_LEVELS:
                urd[lv] = []
                for e in EPS_LIST:
                    y = r[(r.level == lv) & (r.eps == round(e, 3))]
                    urd[lv].append(float(((y.n_correct - y.n_hit) / N).mean()))
            print(f"  [D5] {NAME[x]}: share of the {N:,}-flow pool classified "
                  f"correctly {tpr:.4f}; still flagged after the adversary, "
                  f"(n_c - n_h)/N, at eps = {EPS_LIST}: " + "; ".join(
                      f"L{lv} " + ", ".join(f"{v:.3f}" for v in urd[lv])
                      for lv in AT_LEVELS))


STAGES.append(("caat", stage_caat, summarize_caat))

# ======== stage capu: code ========
# Table cuquant (UNSW-NB15, L2), four capability variants of the SAME base L2 perturbations.
CU_QCOLS = ["dataset", "activation", "seed", "variant", "eps", "attack", "asr",
            "n_correct", "n_hit", "n_free", "n_int_free", "n_pool_tcp",
            "n_correct_tcp", "n_hit_tcp", "attack_seed", "elapsed_s"]
CU_QKEY = CU_QCOLS[:6]
CU_VARIANTS = ["three", "whole", "tcp", "wholetcp"]


def cu_tcp_flows(c, sc):
    if CU_TCP_FLAG not in c.names:
        _stop(f"{CU_TCP_FLAG} is not among the {FEATS} selected features of "
              f"{c.ds}; the TCP flows cannot be identified.")
    j = c.names.index(CU_TCP_FLAG)
    g = int(c.cols[j])
    raw = lambda X: (X[:, j].astype(np.float64) * float(sc["scale"][g])
                     + float(sc["mean"][g]))
    vals = np.unique(np.round(raw(c.Xte), 4))
    if len(vals) != 2:
        _stop(f"{CU_TCP_FLAG} takes {len(vals)} distinct values on the test "
              f"set, expected 2 (a one-hot indicator).")
    if sc["source"] == "kaggle" and not np.allclose(vals, [0.0, 1.0],
                                                    atol=1e-3):
        _stop(f"{CU_TCP_FLAG} in raw units takes the values {vals.tolist()}, "
              f"expected 0 and 1.")
    return raw(c.Xe) > float(vals.mean())


def cu_untouch(X, adv, rows, cols):
    """The perturbation removed on the given coordinates of the given flows."""
    out = adv.astype(np.float32).copy()
    if rows.any() and len(cols):
        ix = np.ix_(np.where(rows)[0], np.asarray(cols, dtype=int))
        out[ix] = X.astype(np.float32)[ix]
    return out


def cu_stage(c):
    if c.ds != CU_DATASET:
        return
    sc = ca_scaler(c.ds, c.data)
    scale_k, int_k = sc["scale"][c.cols], sc["is_int_all"][c.cols]
    free = c.masks[2]
    n_free = int(free.sum())
    q_idx = [i for i in range(len(c.names)) if free[i] > 0 and int_k[i]]
    for nm in CU_TCP_ONLY:
        if nm not in c.names or free[c.names.index(nm)] <= 0:
            _stop(f"{nm} is not a free L2 feature of {c.ds}.")
    t_idx = [c.names.index(nm) for nm in CU_TCP_ONLY]
    is_tcp = cu_tcp_flows(c, sc)
    non_tcp = ~is_tcp
    have = stage_have("cuquant", CU_QKEY)
    base = ca_base_flows(c)
    npz = stage_npz("cuquant", c)
    saved = npz_load(npz)
    kb = dict(dataset=c.ds, activation=c.act, seed=c.seed, n_correct=c.n_ok,
              n_free=n_free, n_pool_tcp=int(is_tcp.sum()),
              n_correct_tcp=int((c.correct & is_tcp).sum()))
    nint = {"three": 0, "whole": len(q_idx), "tcp": 0, "wholetcp": len(q_idx)}

    def put(v, eps, a, h, s, el):
        key = _k(c.ds, c.act, c.seed, v, round(eps, 3), a)
        if key not in have:
            stage_append("cuquant", CU_QCOLS, dict(
                kb, variant=v, eps=round(eps, 3), attack=a,
                asr=round(float(h.sum() / c.n_ok), 4), n_hit=int(h.sum()),
                n_int_free=nint[v], n_hit_tcp=int((h & is_tcp).sum()),
                attack_seed=s, elapsed_s=round(el, 1)))
            have.add(key)

    print(f"    capu: {n_free} free L2 features, integer-valued: "
          f"{[c.names[i] for i in q_idx]}; TCP flows in the pool: "
          f"{int(is_tcp.sum())} of {len(c.Xe)}", flush=True)
    for eps in EPS_LIST:
        if all(_k(c.ds, c.act, c.seed, v, round(eps, 3), "union") in have
               for v in CU_VARIANTS):
            continue
        union = {v: np.zeros(len(c.Xe), bool) for v in CU_VARIANTS}
        for a in CA_ATTACKS:
            t0 = time.time()
            s = _seed_of(c.ds, DEPTH, FEATS, c.act, c.seed, 2, eps, a)
            adv = ca_attack(c, a, eps, free, s)
            h3 = ca_hits(c, adv, eps, free, f"three {a}")
            if not np.array_equal(h3, base[f"L2_e{eps}_{a}"]):
                _stop(f"{c.part} eps={eps} {a}: the regenerated L2 attack "
                      f"evades {int(h3.sum())} flows, the base run recorded "
                      f"{int(base[f'L2_e{eps}_{a}'].sum())} (or other flows). "
                      f"The base attack is not being reproduced; nothing was "
                      f"written for this cell.")
            t1 = time.time()
            advw = ca_quantise(c.Xe, adv, scale_k, q_idx)
            H = {"three": h3,
                 "whole": ca_hits(c, advw, eps, free, f"whole {a}"),
                 "tcp": ca_hits(c, cu_untouch(c.Xe, adv, non_tcp, t_idx),
                                eps, free, f"tcp {a}"),
                 "wholetcp": ca_hits(c, cu_untouch(c.Xe, advw, non_tcp, t_idx),
                                     eps, free, f"wholetcp {a}")}
            t2 = time.time()
            for v in CU_VARIANTS:
                saved[f"{v}_e{eps}_{a}"] = H[v]
                union[v] |= H[v]
                put(v, eps, a, H[v], s, (t1 - t0) if v == "three" else (t2 - t1))
        if not np.array_equal(union["three"], base[f"L2_e{eps}_union"]):
            _stop(f"{c.part} eps={eps}: regenerated L2 union differs from the "
                  f"base run's.")
        for v in CU_VARIANTS:
            saved[f"{v}_e{eps}_union"] = union[v]
        npz_save(npz, saved)
        for v in CU_VARIANTS:
            put(v, eps, "union", union[v], 0, 0.0)
        print(f"    capu eps={eps}: " + "  ".join(
            f"{v} {union[v].sum() / c.n_ok:.4f} ({int(union[v].sum())})"
            for v in CU_VARIANTS) + f"  of {c.n_ok} flows", flush=True)


def cu_summarize():
    q = stage_read("cuquant", CU_QKEY)
    if q is None:
        print("%% stage capu: not run yet")
        return
    b = ca_base_rows()
    if b is None:
        print("%% stage capu: the base run has not written any rows yet")
        return
    L = dict(CA_LAB)[CU_DATASET]
    q = q.copy()
    q["eps"] = q.eps.round(3)
    q["rate"] = q.n_hit / q.n_correct
    # THREE is the base L2 row.
    rep = q[q.variant == "three"]
    bb = b[(b.dataset == CU_DATASET) & (b.level == 2)].copy()
    bb["eps"] = bb.eps.round(3)
    m = rep.merge(bb, on=["activation", "seed", "eps", "attack"],
                  suffixes=("", "_base"))
    bad = int(((m.n_hit != m.n_hit_base)
               | (m.n_correct != m.n_correct_base)).sum())
    if bad or len(m) != len(rep):
        _stop(f"cuquant: {bad} regenerated THREE rows differ from the base L2 "
              f"rows ({len(rep) - len(m)} have no base row).")
    print(f"\n%% ---- cuquant: {L}, L2 worst case (union of the four attacks) "
          f"when the base L2 perturbations are restricted. three | whole "
          f"units | TCP-only fields {CU_TCP_ONLY} untouched on non-TCP flows "
          f"| both; mean$\\pm$SD over seeds of the activation-averaged value")
    print(f"%% THREE = the base L2 rows; the stage regenerated all "
          f"{len(rep)} of them identically (flow counts equal). The other "
          f"three columns restrict finished perturbations (lower bounds).")
    u = q[q.attack == "union"]
    n_models = len(ACTIVATIONS) * len(SEEDS)
    n = len(ca_pick(u, variant="wholetcp", eps=EPS_LIST[0]))
    if n != n_models:
        print(f"%% !! {n} of {n_models} models present -- INCOMPLETE")
    T = {}
    for e in EPS_LIST:
        T[e] = {v: ca_seedavg(ca_pick(u, variant=v, eps=e), col="rate")
                for v in CU_VARIANTS}
        print(f"$\\epsilon={e}$ & " + " & ".join(
            ca_ms(T[e][v]) for v in CU_VARIANTS) + " \\\\")
    print("\\hline")
    print("%% the same, 4 decimals, with the base tab:levels L2 value and the "
          "share of the unconstrained (L0) worst case retained")
    for e in EPS_LIST:
        l2 = float(np.mean(ca_seedavg(ca_pick(b, dataset=CU_DATASET, level=2,
                                              eps=e, attack="union"),
                                      col="rate")))
        l0 = float(np.mean(ca_seedavg(ca_pick(b, dataset=CU_DATASET, level=0,
                                              eps=e, attack="union"),
                                      col="rate")))
        print(f"  eps={e}: base L2 {l2:.4f}, L0 {l0:.4f}; " + "; ".join(
            f"{v} {float(np.mean(T[e][v])):.4f} "
            f"({100 * float(np.mean(T[e][v])) / l0:.1f}% of L0)"
            for v in CU_VARIANTS))
    print("%% pooled over the models: evaded flows of correctly classified "
          "flows, and how many of the evaded flows are TCP")
    for e in EPS_LIST:
        parts = []
        for v in CU_VARIANTS:
            r = ca_pick(u, variant=v, eps=e)
            parts.append(f"{v} {int(r.n_hit.sum())} of "
                         f"{int(r.n_correct.sum())} ({int(r.n_hit_tcp.sum())} "
                         f"TCP; per model {ca_span(r.n_hit)})")
        print(f"  eps={e}: " + "; ".join(parts))
    print("%% by activation, mean$\\pm$SD over seeds: " + " | ".join(
        CU_VARIANTS))
    for e in EPS_LIST:
        for act in ACTIVATIONS:
            print(f"  eps={e} {act}: " + " | ".join(
                ca_ms(ca_pick(u, variant=v, eps=e, activation=act)
                      .sort_values("seed").rate.values) for v in CU_VARIANTS))
    print("%% by attack, activation-averaged, mean$\\pm$SD over seeds: "
          + " | ".join(CU_VARIANTS))
    for e in EPS_LIST:
        for a in CA_ATTACKS:
            print(f"  eps={e} {a}: " + " | ".join(
                ca_ms(ca_seedavg(ca_pick(q, variant=v, eps=e, attack=a),
                                 col="rate")) for v in CU_VARIANTS))
    r = ca_pick(u, variant="three", eps=EPS_LIST[0])
    print(f"%% protocol: TCP flows in the malicious pool {ca_span(r.n_pool_tcp)}"
          f" per model; of the correctly classified flows "
          f"{int(r.n_correct_tcp.sum())} of {int(r.n_correct.sum())} are TCP "
          f"({100 * r.n_correct_tcp.sum() / r.n_correct.sum():.1f}%; per model "
          f"{ca_span(r.n_correct_tcp)} of {ca_span(r.n_correct)}); "
          f"{ca_span(ca_pick(u, variant='whole', eps=EPS_LIST[0]).n_int_free)}"
          f" of {ca_span(r.n_free)} free coordinates integer-valued")
    un = stage_read("caunits", CA_UKEY)
    if un is None:
        print("%% caunits: not run yet (raw-unit size of the budget)")
        return
    g = un[(un.dataset == CU_DATASET) & (un.free_l2 == 1)]
    for e in EPS_LIST:
        print(f"  eps={e} in raw units: " + ", ".join(
            f"{x.feature} {e * x.scale:.4g}"
            f"{' (integer-valued)' if x.integer_all else ''}"
            for x in g.itertuples()))


STAGES.append(("capu", cu_stage, cu_summarize))

# ======== stage fgsm: code ========
FG_COLS = ["dataset", "depth", "features", "activation", "seed", "eps",
           "attack", "asr", "n_correct", "n_hit", "n_hit_pgd", "n_hit_bim",
           "n_not_pgd_bim", "n_hit_union", "n_not_union", "clean_f1",
           "attack_seed", "elapsed_s"]
FG_KEY = FG_COLS[:7]
FG_GRID = ["dataset", "depth", "features", "activation", "seed", "eps"]


def fg_attack(clf, X, eps):
    from art.attacks.evasion import FastGradientMethod
    atk = FastGradientMethod(clf, **_kw(
        FastGradientMethod, norm=np.inf, eps=eps,
        eps_step=FG_EPS_STEP_FACTOR * eps, targeted=False, num_random_init=0,
        minimal=False, batch_size=BATCH_SIZE))
    return atk.generate(x=X.astype(np.float32))


def fg_cfgs(ds, act):
    """The (depth, features) pairs of the base run for this part."""
    cfgs = [(d, FEATS) for d in DEPTHS]
    if act == "relu":
        cfgs += [(1, k) for k in K_SWEEP[ds] if k != FEATS]
    return cfgs


def fg_stage(c):
    Xtr_f, ytr, Xte_f, yte, order, names = c.data
    have = stage_have("fgsm", FG_KEY)
    for depth, k in fg_cfgs(c.ds, c.act):
        if all(_k(c.ds, depth, k, c.act, c.seed, round(e, 3), "fgsm") in have
               for e in EPS_LIST):
            continue
        cols = np.asarray(order)[:k]
        Xtr, Xte = Xtr_f[:, cols], Xte_f[:, cols]
        Xe, ye = Xte_f[c.pool][:, cols].astype(np.float32), yte[c.pool]
        model, f1, _ = get_model(c.ds, depth, k, c.act, c.seed, Xtr, ytr, Xte,
                                 yte, load_only=True)
        clf = _art_clf(model, Xtr, DEVICE)
        tag = f"{c.ds}_d{depth}_f{k}_{c.act}_s{c.seed}"
        p = os.path.join(RES_DIR, "flows", f"{tag}.npz")
        if not os.path.exists(p):
            _stop(f"{p} is missing: the base run has not saved the per-flow "
                  f"outputs of this model. Run the part without RERUN_STAGES "
                  f"first.")
        base = dict(np.load(p))
        correct = _probs(model, Xe, DEVICE).argmax(1) == ye
        if not (np.array_equal(base["pool"], c.pool)
                and np.array_equal(base["correct"], correct)):
            _stop(f"{tag}: the pool or the correctly classified flows differ "
                  f"from those the base run saved; nothing was written.")
        n_ok = max(int(correct.sum()), 1)
        npz = os.path.join(RES_DIR, "flows", f"fgsm_{tag}.npz")
        saved = npz_load(npz)
        msg = []
        for eps in EPS_LIST:
            if _k(c.ds, depth, k, c.act, c.seed, round(eps, 3), "fgsm") in have:
                continue
            s = _seed_of("fgsm", c.ds, depth, k, c.act, c.seed, eps)
            _reseed(s)
            t0 = time.time()
            adv = _timed(lambda: fg_attack(clf, Xe, eps))
            if not np.isfinite(adv).all():
                _stop(f"FGSM returned non-finite values ({tag} eps={eps}); "
                      f"nothing was written for this cell.")
            _, h, _ = hits_of(model, adv, Xe, ye, DEVICE, eps, None)
            hp, hb = base[f"L0_e{eps}_pgd"], base[f"L0_e{eps}_bim"]
            hu = base[f"L0_e{eps}_union"]
            saved[f"L0_e{eps}_fgsm"] = h
            npz_save(npz, saved)
            stage_append("fgsm", FG_COLS, dict(
                dataset=c.ds, depth=depth, features=k, activation=c.act,
                seed=c.seed, eps=round(eps, 3), attack="fgsm",
                asr=round(float(h.sum() / n_ok), 4), n_correct=n_ok,
                n_hit=int(h.sum()), n_hit_pgd=int(hp.sum()),
                n_hit_bim=int(hb.sum()), n_not_pgd_bim=int((h & ~(hp | hb)).sum()),
                n_hit_union=int(hu.sum()), n_not_union=int((h & ~hu).sum()),
                clean_f1=round(float(f1), 4), attack_seed=s,
                elapsed_s=round(time.time() - t0, 1)))
            have.add(_k(c.ds, depth, k, c.act, c.seed, round(eps, 3), "fgsm"))
            msg.append(f"eps={eps} FGSM {h.sum() / n_ok:.3f} (PGD "
                       f"{hp.sum() / n_ok:.3f})")
        print(f"    fgsm d{depth} k{k}: " + "  ".join(msg), flush=True)


def fg_summarize():
    d = stage_read("fgsm", FG_KEY)
    if d is None:
        print("%% stage fgsm: not run yet")
        return
    if not os.path.exists(OUT_FINAL):
        _stop(f"{OUT_FINAL} is missing: the PGD and BIM columns are read from "
              f"it. Run RERUN_PART=summary (it writes that file first).")
    b = pd.read_csv(OUT_FINAL)
    b = b[b.level == 0].copy()
    b["eps"] = b.eps.round(3)
    b["rate"] = b.n_hit / b.n_correct
    d = d.copy()
    d["eps"] = d.eps.round(3)
    d["rate"] = d.n_hit / d.n_correct
    NAME = {"relu": "ReLU", "elu": "ELU", "tanh": "Tanh"}
    # every FGSM row must sit on a base PGD row of the same model and budget.
    m = d.merge(b[b.attack == "pgd"], on=FG_GRID, suffixes=("", "_b"))
    bad = int(((m.n_correct != m.n_correct_b) | (m.n_hit_pgd != m.n_hit_b)).sum())
    if bad or len(m) != len(d):
        _stop(f"fgsm: {bad} rows disagree with the base PGD rows of the same "
              f"model ({len(d) - len(m)} have no base row).")
    n_want = sum(len(fg_cfgs(ds, a)) for ds, _ in CA_LAB for a in ACTIVATIONS
                 ) * len(SEEDS) * len(EPS_LIST)
    print(f"\n%% ---- fgsm: FGSM (one step of size eps, no random start) at L0 "
          f"on the saved standard models; {len(d)} of {n_want} measurements; "
          f"each row agrees with the base PGD row of the same model and "
          f"budget in its denominator and PGD flow count")
    if len(d) != n_want:
        print(f"%% !! {len(d)} of {n_want} measurements present -- INCOMPLETE")

    def vals(ds, depth, k, act, e, attack):
        src = d if attack == "fgsm" else b
        r = src[(src.dataset == ds) & (src.depth == depth) & (src.features == k)
                & (src.activation == act) & (src.eps == round(e, 3))
                & (src.attack == attack)].sort_values("seed")
        return r.rate.values

    print("%% 30 features, mean$\\pm$SD over seeds. Columns: for ReLU, ELU "
          "and Tanh in turn, FGSM | PGD | BIM (PGD and BIM are the base rows)")
    for ds, L in CA_LAB:
        for e in EPS_LIST:
            print(f"\\multicolumn{{10}}{{l}}{{\\textit{{{L}, "
                  f"$\\epsilon = {e}$}}}} \\\\")
            for depth in DEPTHS:
                print(f"Model {depth} & " + " & ".join(
                    ca_ms(vals(ds, depth, FEATS, act, e, a))
                    for act in ACTIVATIONS for a in ("fgsm", "pgd", "bim"))
                    + " \\\\")
        print("\\hline")
    print("%% Model 1, ReLU, by number of features: FGSM | PGD | BIM at each "
          "budget, mean$\\pm$SD over seeds")
    for ds, L in CA_LAB:
        print(f"\\multicolumn{{10}}{{l}}{{\\textit{{{L}}}}} \\\\")
        for k in sorted(set(K_SWEEP[ds]) | {FEATS}):
            print(f"{k} & " + " & ".join(
                ca_ms(vals(ds, 1, k, "relu", e, a))
                for e in EPS_LIST for a in ("fgsm", "pgd", "bim")) + " \\\\")
    print("\\hline")
    ds0, dp0, k0, a0 = FG_REF_CFG
    print(f"%% the undefended configuration of Fig. 16 of the study whose "
          f"design is followed: {dict(CA_LAB)[ds0]}, Model {dp0}, {k0} "
          f"features, {NAME[a0]} (that study reports attack success below "
          f"{FG_REF_LEVEL} there at eps = {EPS_LIST[-1]})")
    for e in EPS_LIST:
        four = b[(b.dataset == ds0) & (b.depth == dp0) & (b.features == k0)
                 & (b.activation == a0) & (b.eps == round(e, 3))
                 & b.attack.isin(["pgd", "bim", "spsa", "hsj_inf"])]
        f4 = four.groupby("seed").rate.mean().values
        print(f"  eps={e}: FGSM {ca_ms(vals(ds0, dp0, k0, a0, e, 'fgsm'))} "
              f"(per seed " + " ".join(
                  f"{v:.4f}" for v in vals(ds0, dp0, k0, a0, e, "fgsm"))
              + f") | PGD {ca_ms(vals(ds0, dp0, k0, a0, e, 'pgd'))} | BIM "
              f"{ca_ms(vals(ds0, dp0, k0, a0, e, 'bim'))} | mean of the four "
              f"primary attacks {ca_ms(f4)}")
    print(f"%% how many saved models have an FGSM success rate below "
          f"{FG_REF_LEVEL}, and the lowest and highest FGSM rate of a model")
    for ds, L in CA_LAB:
        for e in EPS_LIST:
            r = d[(d.dataset == ds) & (d.eps == round(e, 3))]
            r30 = r[r.features == FEATS]
            r1 = r[(r.depth == 1) & (r.activation == "relu")]
            print(f"  {ds} eps={e}: all {int((r.rate < FG_REF_LEVEL).sum())} of "
                  f"{len(r)} models (rates {r.rate.min():.4f} to "
                  f"{r.rate.max():.4f}); 30 features "
                  f"{int((r30.rate < FG_REF_LEVEL).sum())} of {len(r30)}; "
                  f"Model 1 ReLU, every feature count "
                  f"{int((r1.rate < FG_REF_LEVEL).sum())} of {len(r1)} "
                  f"(rates {r1.rate.min():.4f} to {r1.rate.max():.4f})")
    print("%% FGSM against the base attacks on the same flows, pooled over "
          "the models")
    for ds, L in CA_LAB:
        for e in EPS_LIST:
            r = d[(d.dataset == ds) & (d.eps == round(e, 3))]
            N = int(r.n_correct.sum())
            print(f"  {ds} eps={e}: FGSM {int(r.n_hit.sum())} of {N} "
                  f"({r.n_hit.sum() / N:.4f}); PGD {int(r.n_hit_pgd.sum())} "
                  f"({r.n_hit_pgd.sum() / N:.4f}); BIM "
                  f"{int(r.n_hit_bim.sum())} ({r.n_hit_bim.sum() / N:.4f}); "
                  f"flows FGSM evades that neither PGD nor BIM does "
                  f"{int(r.n_not_pgd_bim.sum())}, that the base union does not "
                  f"{int(r.n_not_union.sum())}; models with FGSM above PGD "
                  f"{int((r.n_hit > r.n_hit_pgd).sum())} of {len(r)}; mean "
                  f"over models of PGD minus FGSM "
                  f"{float(((r.n_hit_pgd - r.n_hit) / r.n_correct).mean()):.4f}"
                  f", largest "
                  f"{float(((r.n_hit_pgd - r.n_hit) / r.n_correct).max()):.4f}")


STAGES.append(("fgsm", fg_stage, fg_summarize))

# ======== stage capd: code ========
# Table cddir, L2, both datasets.
CD_COLS = ["dataset", "activation", "seed", "variant", "eps", "attack", "asr",
           "n_correct", "n_hit", "n_free", "n_onesided", "n_projected",
           "attack_seed", "elapsed_s"]
CD_KEY = CD_COLS[:6]


def cd_box(c, eps, variant):
    """Lowest and highest allowed displacement of each of the 30 features."""
    free = np.asarray(c.masks[2]).reshape(-1) > 0
    lo = np.where(free, -eps, 0.0).astype(np.float64)
    hi = np.where(free, eps, 0.0).astype(np.float64)
    if variant == "addonly":
        for nm in CD_ADD_ONLY[c.ds]:
            if nm not in c.names or not free[c.names.index(nm)]:
                _stop(f"{nm} is not a free L2 feature of {c.ds}.")
            lo[c.names.index(nm)] = 0.0
    elif variant == "up":
        lo[free] = 0.0
    elif variant == "down":
        hi[free] = 0.0
    elif variant != "sym":
        _stop(f"unknown direction variant {variant!r}")
    return lo, hi


def cd_onehot(clf, X):
    p = clf.predict(X.astype(np.float32), batch_size=BATCH_SIZE)
    return np.eye(p.shape[1], dtype=np.float32)[p.argmax(1)]


def cd_grad(clf, X, eps, lo, hi, mask, which):
    """PGD / BIM of the base run inside the box [x + lo, x + hi]."""
    from art.attacks.evasion import BasicIterativeMethod, ProjectedGradientDescent
    free = np.asarray(mask).reshape(-1) > 0
    half = np.where(free, (hi - lo) / 2.0, eps).astype(np.float32)
    centre = (X.astype(np.float64) + (lo + hi) / 2.0).astype(np.float32)
    kw = dict(eps=half, eps_step=half / FGSM_EPS_STEP_DIV,
              max_iter=PGD_BIM_MAX_ITER, batch_size=BATCH_SIZE, verbose=False)
    atk = (ProjectedGradientDescent(clf, num_random_init=PGD_RANDOM_INIT, **kw)
           if which == "pgd" else BasicIterativeMethod(clf, **kw))
    return atk.generate(x=centre, y=cd_onehot(clf, X), mask=np.broadcast_to(
        mask, X.shape).astype(np.float32).copy())


def cd_spsa(model, X, y, eps, lo, hi, seed, mask):
    """a_spsa generalised to a box."""
    rng = np.random.default_rng(seed)
    x0 = np.ascontiguousarray(X, np.float32)
    lo32, hi32 = lo.astype(np.float32), hi.astype(np.float32)
    x = (x0 + (lo32 + hi32) / 2.0).astype(np.float32)
    m = np.asarray(mask, np.float32)
    alpha = ((hi32 - lo32) / 2.0) / SPSA_STEPS * 2.5
    for _ in range(SPSA_STEPS):
        g = np.zeros_like(x)
        for _ in range(SPSA_NDIRS):
            d = rng.choice([-1.0, 1.0], size=x.shape).astype(np.float32) * m
            g += ((_ce(model, x + SPSA_C * d, y, DEVICE) -
                   _ce(model, x - SPSA_C * d, y, DEVICE))
                  / (2 * SPSA_C))[:, None] * d
        x = x + alpha * np.sign(g) * m
        x = np.clip(x, x0 + lo32, x0 + hi32)
    return x.astype(np.float32)


def cd_attack(c, a, eps, lo, hi, mask, s):
    sub = c.substitute() if a == "transfer" else None    # loaded BEFORE seeding
    fn = {"pgd": lambda: cd_grad(c.clf, c.Xe, eps, lo, hi, mask, "pgd"),
          "bim": lambda: cd_grad(c.clf, c.Xe, eps, lo, hi, mask, "bim"),
          "spsa": lambda: cd_spsa(c.model, c.Xe, c.ye, eps, lo, hi, s, mask),
          "transfer": lambda: cd_grad(sub, c.Xe, eps, lo, hi, mask, "pgd")}[a]
    _reseed(s)
    adv = _timed(fn)
    if not np.isfinite(adv).all():
        _stop(f"{a} returned non-finite values ({c.part} eps={eps}); nothing "
              f"was written for this cell.")
    return adv


def cd_score(c, adv, lo, hi):
    d = adv.astype(np.float64) - c.Xe.astype(np.float64)
    dp = np.clip(d, lo, hi)
    n_proj = int((np.abs(d - dp) > CD_TOL).sum())
    advp = (c.Xe.astype(np.float64) + dp).astype(np.float32)
    h = c.correct & (_probs(c.model, advp, DEVICE).argmax(1) != c.ye)
    return h, n_proj


def cd_lattice(c, lo, hi):
    free = np.where(np.asarray(c.masks[2]).reshape(-1) > 0)[0]
    axes = np.meshgrid(*[np.linspace(lo[j], hi[j], CA_ENUM_GRID) for j in free],
                       indexing="ij")
    offs = np.unique(np.stack([a.reshape(-1) for a in axes], 1)
                     .astype(np.float32), axis=0)
    lo_c, hi_c = float(c.Xtr.min()), float(c.Xtr.max())     # as _art_clf
    n, d = c.Xe.shape
    out = np.zeros(n, bool)
    for i in range(0, n, CA_ENUM_CHUNK):
        sl = slice(i, i + CA_ENUM_CHUNK)
        xb, yb, ok = c.Xe[sl], c.ye[sl], c.correct[sl]
        pts = np.repeat(xb[:, None, :], len(offs), 1).astype(np.float32)
        pts[:, :, free] += offs[None]
        pts = np.clip(pts, lo_c, hi_c)
        w = (_probs(c.model, pts.reshape(-1, d), DEVICE).argmax(1)
             .reshape(pts.shape[0], pts.shape[1]) != yb[:, None])
        out[sl] = w.any(1) & ok
    return out, len(offs)


def cd_stage(c):
    mask = np.asarray(c.masks[2]).reshape(-1).astype(np.float32)
    n_free = int(mask.sum())
    have = stage_have("cddir", CD_KEY)
    npz = stage_npz("cddir", c)
    saved = npz_load(npz)
    enum = n_free <= CA_ENUM_MAX_FREE
    kb = dict(dataset=c.ds, activation=c.act, seed=c.seed, n_correct=c.n_ok,
              n_free=n_free)

    def put(v, eps, a, h, n_one, n_proj, s, el):
        key = _k(c.ds, c.act, c.seed, v, round(eps, 3), a)
        if key not in have:
            stage_append("cddir", CD_COLS, dict(
                kb, variant=v, eps=round(eps, 3), attack=a,
                asr=round(float(h.sum() / c.n_ok), 4), n_hit=int(h.sum()),
                n_onesided=n_one, n_projected=n_proj, attack_seed=s,
                elapsed_s=round(el, 1)))
            have.add(key)

    for eps in EPS_LIST:
        msg = []
        for v in CD_VARIANTS:
            last = "lattice" if enum else "union"
            if _k(c.ds, c.act, c.seed, v, round(eps, 3), last) in have:
                continue
            lo, hi = cd_box(c, eps, v)
            n_one = int(((lo == 0.0) != (hi == 0.0)).sum())
            union, tot_proj = np.zeros(len(c.Xe), bool), 0
            for a in CA_ATTACKS:
                # sym uses the seed of the base L2 row, so that it can be compared with it.
                s = (_seed_of(c.ds, DEPTH, FEATS, c.act, c.seed, 2, eps, a)
                     if v == "sym" else
                     _seed_of("capd", v, c.ds, c.act, c.seed, eps, a))
                t0 = time.time()
                adv = cd_attack(c, a, eps, lo, hi, mask, s)
                h, n_proj = cd_score(c, adv, lo, hi)
                saved[f"{v}_e{eps}_{a}"] = h
                union |= h
                tot_proj += n_proj
                put(v, eps, a, h, n_one, n_proj, s, time.time() - t0)
            saved[f"{v}_e{eps}_union"] = union
            npz_save(npz, saved)
            put(v, eps, "union", union, n_one, tot_proj, 0, 0.0)
            txt = f"{v} {union.sum() / c.n_ok:.4f}"
            if enum:
                t0 = time.time()
                lat, n_pts = cd_lattice(c, lo, hi)
                saved[f"{v}_e{eps}_lattice"] = lat
                npz_save(npz, saved)
                put(v, eps, "lattice", lat, n_one, 0, 0, time.time() - t0)
                txt += f" (lattice {lat.sum() / c.n_ok:.4f}, {n_pts} points)"
            msg.append(txt)
        if msg:
            print(f"    capd eps={eps}: " + "  ".join(msg)
                  + f"  of {c.n_ok} flows", flush=True)


def cd_summarize():
    q = stage_read("cddir", CD_KEY)
    if q is None:
        print("%% stage capd: not run yet")
        return
    b = ca_base_rows()
    if b is None:
        print("%% stage capd: the base run has not written any rows yet")
        return
    q = q.copy()
    q["eps"] = q.eps.round(3)
    q["rate"] = q.n_hit / q.n_correct
    n_models = len(ACTIVATIONS) * len(SEEDS)
    print("\n%% ---- cddir: L2 worst case (union of the four attacks) with the "
          "perturbation confined to one direction INSIDE the attacks. sym "
          "(this stage's code path, symmetric box) | addonly | up | down; "
          "mean$\\pm$SD over seeds of the activation-averaged value")
    for ds, L in CA_LAB:
        print(f"%% {L}: addonly = may only increase: {CD_ADD_ONLY[ds]}; the "
              f"other source-side features stay symmetric")
    u = q[q.attack == "union"]
    for ds, L in CA_LAB:
        g = u[u.dataset == ds]
        n = len(ca_pick(g, variant=CD_VARIANTS[-1], eps=EPS_LIST[-1]))
        if n != n_models:
            print(f"%% !! {ds}: {n} of {n_models} models present -- INCOMPLETE")
        print(f"\\multicolumn{{5}}{{l}}{{\\textit{{{L}}}}} \\\\")
        for e in EPS_LIST:
            print(f"$\\epsilon={e}$ & " + " & ".join(
                ca_ms(ca_seedavg(ca_pick(g, variant=v, eps=e), col="rate"))
                for v in CD_VARIANTS) + " \\\\")
        print("\\hline")
    print("%% the same, 4 decimals, with the base tab:levels L2 and L0 values, "
          "and each variant as a share of the base L2 value and of L0")
    for ds, L in CA_LAB:
        for e in EPS_LIST:
            l2 = float(np.mean(ca_seedavg(ca_pick(
                b, dataset=ds, level=2, eps=e, attack="union"), col="rate")))
            l0 = float(np.mean(ca_seedavg(ca_pick(
                b, dataset=ds, level=0, eps=e, attack="union"), col="rate")))
            parts = []
            for v in CD_VARIANTS:
                x = float(np.mean(ca_seedavg(ca_pick(
                    u, dataset=ds, variant=v, eps=e), col="rate")))
                parts.append(f"{v} {x:.4f} ({100 * x / l2:.1f}% of base L2, "
                             f"{100 * x / l0:.1f}% of L0)")
            print(f"  {ds} eps={e}: base L2 {l2:.4f}, L0 {l0:.4f}; "
                  + "; ".join(parts))
    # the symmetric variant through this stage's code path against the base rows.
    print("%% check of this stage's code path: its symmetric variant against "
          "the base L2 rows of the same model, budget and attack")
    bb = b[b.level == 2].copy()
    bb["eps"] = bb.eps.round(3)
    rep = q[(q.variant == "sym") & (q.attack != "lattice")]
    m = rep.merge(bb, on=["dataset", "activation", "seed", "eps", "attack"],
                  suffixes=("", "_base"))
    for ds, L in CA_LAB:
        g = m[m.dataset == ds]
        dif = (g.n_hit - g.n_hit_base)
        print(f"  {ds}: {int((dif == 0).sum())} of {len(g)} rows have the same "
              f"flow count as the base row; largest difference "
              f"{int(dif.abs().max()) if len(g) else 0} flows; same "
              f"denominators in {int((g.n_correct == g.n_correct_base).sum())}"
              f" of {len(g)}; union rows identical in "
              f"{int((dif[g.attack == 'union'] == 0).sum())} of "
              f"{int((g.attack == 'union').sum())}")
    print("%% pooled over the models: evaded flows of correctly classified "
          "flows (union), per model range in brackets")
    for ds, L in CA_LAB:
        for e in EPS_LIST:
            parts = []
            for v in CD_VARIANTS:
                r = ca_pick(u, dataset=ds, variant=v, eps=e)
                parts.append(f"{v} {int(r.n_hit.sum())} of "
                             f"{int(r.n_correct.sum())} [{ca_span(r.n_hit)}]")
            print(f"  {ds} eps={e}: " + "; ".join(parts))
    print("%% by activation, mean$\\pm$SD over seeds: " + " | ".join(CD_VARIANTS))
    for ds, L in CA_LAB:
        for e in EPS_LIST:
            for act in ACTIVATIONS:
                print(f"  {ds} eps={e} {act}: " + " | ".join(
                    ca_ms(ca_pick(u, dataset=ds, variant=v, eps=e,
                                  activation=act).sort_values("seed")
                          .rate.values) for v in CD_VARIANTS))
    print("%% by attack, activation-averaged, mean$\\pm$SD over seeds: "
          + " | ".join(CD_VARIANTS))
    for ds, L in CA_LAB:
        for e in EPS_LIST:
            for a in CA_ATTACKS:
                print(f"  {ds} eps={e} {a}: " + " | ".join(
                    ca_ms(ca_seedavg(ca_pick(q, dataset=ds, variant=v, eps=e,
                                             attack=a), col="rate"))
                    for v in CD_VARIANTS))
    lat = q[q.attack == "lattice"]
    if len(lat):
        en = stage_read("caenum", CA_EKEY)
        print(f"%% the same boxes enumerated on a {CA_ENUM_GRID}-per-axis "
              f"lattice (no attack): " + " | ".join(CD_VARIANTS)
              + "; then the attack union of the same variant")
        for ds in sorted(lat.dataset.unique()):
            for e in EPS_LIST:
                print(f"  {ds} eps={e} lattice: " + " | ".join(
                    ca_ms(ca_seedavg(ca_pick(lat, dataset=ds, variant=v, eps=e),
                                     col="rate"), 4) for v in CD_VARIANTS)
                    + "  || union: " + " | ".join(
                    ca_ms(ca_seedavg(ca_pick(u, dataset=ds, variant=v, eps=e),
                                     col="rate"), 4) for v in CD_VARIANTS))
                parts = []
                for v in CD_VARIANTS:
                    r = ca_pick(lat, dataset=ds, variant=v, eps=e)
                    parts.append(f"{v} {int(r.n_hit.sum())}")
                txt = f"  {ds} eps={e} lattice flows of " \
                      f"{int(ca_pick(lat, dataset=ds, variant='sym', eps=e).n_correct.sum())}: " \
                      + "; ".join(parts)
                if en is not None:
                    x = en[(en.dataset == ds) & (en.eps.round(3) == round(e, 3))]
                    txt += (f"; the capa enumeration of the symmetric box "
                            f"recorded {int(x.n_hit_exact.sum())}")
                print(txt)
    r = q[q.attack != "lattice"]
    print(f"%% displacements outside the box before the projection that is "
          f"scored (more than {CD_TOL}): " + "; ".join(
              f"{v} {int(r[r.variant == v].n_projected.sum())}"
              for v in CD_VARIANTS)
          + " (flow, coordinate) pairs over all attack rows")
    for ds, L in CA_LAB:
        g = u[u.dataset == ds]
        print(f"  {ds}: one-sided coordinates per variant: " + ", ".join(
            f"{v} {ca_span(g[g.variant == v].n_onesided)}" for v in CD_VARIANTS)
            + f" of {ca_span(g.n_free)} free")


STAGES.append(("capd", cd_stage, cd_summarize))

# ======== stage rvstd1: code ========
# Table rvstd1fpr (the column layout of atfpr).
def rv1_settings(c):
    return dict(optimiser="one Adam for the whole run", adv_ratio=0.0, lr=LR,
                batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, chunk=CHUNK,
                patience=PATIENCE, tol=TOL, val_fraction=VAL_FRACTION,
                split_seed=SEED, train_seed=int(c.seed))


def rv1_train(model, Xtrn, ytrn, Xval, yval, tag):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    Xt = torch.tensor(Xtrn, dtype=torch.float32)
    yt = torch.tensor(ytrn, dtype=torch.long)
    n, best, stale, done, t0 = len(Xt), -1.0, 0, 0, time.time()
    while done < MAX_EPOCHS:
        model.train()
        for _ in range(CHUNK):
            perm = torch.randperm(n)
            for i in range(0, n, BATCH_SIZE):
                idx = perm[i:i + BATCH_SIZE]
                opt.zero_grad()
                F.cross_entropy(model(Xt[idx].to(DEVICE)),
                                yt[idx].to(DEVICE)).backward()
                opt.step()
        done += CHUNK
        f1 = macro_f1(model, Xval, yval, DEVICE)
        print(f"      {tag}: {done} epochs, validation F1 {f1:.4f} "
              f"[{time.time() - t0:.0f}s]", flush=True)
        if f1 > best + TOL:
            best, stale = f1, 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.eval()
    return best, done


def rv1_get_model(c):
    """Trained once, saved, afterwards only loaded (the pattern of at_get_model)."""
    k = c.Xtr.shape[1]
    p = model_path(c.ds, DEPTH, k, c.act, c.seed, RV1_ROLE)
    cfg = rv1_settings(c)
    if os.path.exists(p):
        blob = torch.load(p, map_location=DEVICE, weights_only=False)
        if blob.get("train") != cfg:
            _stop(f"{p} was trained with other settings than this file's:\n"
                  f"  saved   {blob.get('train')}\n  current {cfg}")
        model = build_model(DEPTH, k, c.act).to(DEVICE)
        model.load_state_dict(blob["state_dict"]); model.eval()
        f1 = macro_f1(model, c.Xte, c.yte, DEVICE)
        if abs(f1 - blob["clean_f1"]) > 1e-9:
            _stop(f"{p} re-measures clean F1 {f1:.6f}, saved "
                  f"{blob['clean_f1']:.6f}.")
        return model, f1, int(blob["epochs"]), "saved"
    Xtrn, Xval, ytrn, yval = c.split()
    _reseed(cfg["train_seed"]); t0 = time.time()
    model = build_model(DEPTH, k, c.act).to(DEVICE)
    best, epochs = rv1_train(model, Xtrn, ytrn, Xval, yval,
                             f"{RV1_ROLE} {c.part}")
    model.eval()
    f1 = macro_f1(model, c.Xte, c.yte, DEVICE)
    os.makedirs(MODEL_DIR, exist_ok=True)
    tmp = p + f".tmp{os.getpid()}"
    torch.save({"state_dict": model.state_dict(), "clean_f1": float(f1),
                "epochs": int(epochs), "val_best": float(best), "ds": c.ds,
                "depth": DEPTH, "features": k, "activation": c.act,
                "seed": c.seed, "role": RV1_ROLE, "train": cfg,
                "train_s": round(time.time() - t0, 1),
                "script": "rerun_all.py"}, tmp)
    os.replace(tmp, p)
    return model, f1, int(epochs), "trained"


def stage_rvstd1(c):
    if _k(c.ds, c.act, c.seed, RV1_ROLE) in stage_have("rvstd1fpr", AT_FKEY):
        print(f"    skip {RV1_ROLE} {c.part}: done", flush=True)
        return
    t0 = time.time()
    model, f1, epochs, src = rv1_get_model(c)
    correct = _probs(model, c.Xe, DEVICE).argmax(1) == c.ye
    row = at_operating_row(c, model, RV1_ROLE, f1, epochs, correct)
    stage_append("rvstd1fpr", AT_FCOLS, row)
    print(f"    {RV1_ROLE} {c.part}: {src}, {epochs} epochs, clean F1 "
          f"{f1:.4f} (standard {c.f1:.4f}), AUC {row['roc_auc']:.5f}, at "
          f"{AT_THRESHOLD}: FPR {row['fpr_at_thr']:.5f} TPR "
          f"{row['tpr_at_thr']:.5f} [{time.time() - t0:.0f}s]", flush=True)


def rv_ms(v, nd=4):
    v = np.asarray(v, float)
    return f"{v.mean():.{nd}f}+-{v.std(ddof=SD_DDOF):.{nd}f}"


def summarize_rvstd1():
    r = stage_read("rvstd1fpr", AT_FKEY)
    f = stage_read("atfpr", AT_FKEY)
    if r is None or f is None:
        print("%% stage rvstd1: not run yet (or advtrain not run)")
        return
    d = pd.concat([f, r], ignore_index=True)
    print("%% rvstd1: standard Model 4 (fresh Adam per block of epochs),\n"
          f"%% {RV1_ROLE} (one Adam for the whole run, as PGD-AT), PGD-AT.\n"
          "%% mean+-SD over seeds; rows = models in each cell.")
    cols = ["clean_f1", "roc_auc", "fpr_at_thr", "tpr_at_thr",
            at_tpr_col(AT_TPR_TABLE), "epochs"]
    for ds in DATASETS:
        for ac in ACTIVATIONS + ["all"]:
            x = (d[d.dataset == ds] if ac == "all" else
                 d[(d.dataset == ds) & (d.activation == ac)])
            for dfc in ("standard", RV1_ROLE, AT_ROLE):
                y = x[x.defence == dfc]
                if not len(y):
                    continue
                print(f"  {ds:4s} {ac:4s} {dfc:9s} n={len(y):2d}  " + "  ".join(
                    f"{cn} {rv_ms(y[cn])}" for cn in cols))
            p = x.pivot_table(index=["activation", "seed"], columns="defence",
                              values="clean_f1")
            if {RV1_ROLE, "standard", AT_ROLE} <= set(p.columns):
                p = p.dropna()
                for a_, b_ in ((RV1_ROLE, "standard"), ("standard", AT_ROLE),
                               (RV1_ROLE, AT_ROLE)):
                    dd = p[a_] - p[b_]
                    print(f"      clean F1 {a_} - {b_}: {rv_ms(dd)}  "
                          f"min {dd.min():.4f} max {dd.max():.4f}  "
                          f"positive {int((dd > 0).sum())} negative "
                          f"{int((dd < 0).sum())} of {len(dd)}  sign-test p "
                          f"{caat_sign_p(int((dd > 0).sum()), int((dd != 0).sum())):.4g}")


STAGES.append(("rvstd1", stage_rvstd1, summarize_rvstd1))

# ======== stage rvspsam: code ========
RV2_COLS = ["dataset", "depth", "features", "activation", "seed", "level",
            "eps", "attack", "asr", "n_correct", "n_hit", "n_hit_spsa",
            "n_new_vs_spsa", "n_hit_union", "n_union_with", "n_saturated",
            "clean_f1", "attack_seed", "elapsed_s"]
RV2_KEY = RV2_COLS[:8]


def rv_margin_loss(model, X, y, device):
    """Per-flow logit margin z_(1-y) - z_y of the two-class network."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            z = model(torch.tensor(X[i:i + BATCH_SIZE], dtype=torch.float32,
                                   device=device)).double().cpu().numpy()
            yy = np.asarray(y[i:i + BATCH_SIZE]).astype(int)
            r = np.arange(len(yy))
            out.append(z[r, 1 - yy] - z[r, yy])
    return np.concatenate(out)


def rv_spsa_margin(model, X, y, eps, seed, mask):
    """a_spsa ITSELF with _ce rebound to rv_margin_loss for this one call (restored whatever happens)."""
    global _ce
    if RV2_LOSS != "margin":
        _stop(f"unknown RV2_LOSS {RV2_LOSS!r}")
    keep, calls = _ce, [0]

    def counted(m, Z, yy, dev):
        calls[0] += 1
        return rv_margin_loss(m, Z, yy, dev)
    _ce = counted
    try:
        adv = a_spsa(model, X, y, eps, seed, mask)
    finally:
        _ce = keep
    if calls[0] != 2 * SPSA_STEPS * SPSA_NDIRS:
        _stop(f"margin SPSA issued {calls[0]} loss queries per flow, expected "
              f"{2 * SPSA_STEPS * SPSA_NDIRS}.")
    return adv


def rv2_seed(ds, depth, k, act, seed, lv, eps):
    return _seed_of("rvspsam", ds, depth, k, act, seed, lv, eps)


def stage_rvspsam(c):
    Xtr_f, ytr, Xte_f, yte, order, names = c.data
    have = stage_have("rvspsam", RV2_KEY)
    for depth, k in fg_cfgs(c.ds, c.act):
        levels = LEVELS if (depth, k) == (DEPTH, FEATS) else [0]
        if all(_k(c.ds, depth, k, c.act, c.seed, lv, round(e, 3), "spsam")
               in have for lv in levels for e in EPS_LIST):
            continue
        cols = np.asarray(order)[:k]
        Xtr, Xte = Xtr_f[:, cols], Xte_f[:, cols]
        Xe, ye = Xte_f[c.pool][:, cols].astype(np.float32), yte[c.pool]
        model, f1, _ = get_model(c.ds, depth, k, c.act, c.seed, Xtr, ytr, Xte,
                                 yte, load_only=True)
        tag = f"{c.ds}_d{depth}_f{k}_{c.act}_s{c.seed}"
        p = os.path.join(RES_DIR, "flows", f"{tag}.npz")
        if not os.path.exists(p):
            _stop(f"{p} is missing: the base run has not saved the per-flow "
                  f"outputs of this model.")
        base = dict(np.load(p))
        correct = _probs(model, Xe, DEVICE).argmax(1) == ye
        if not (np.array_equal(base["pool"], c.pool)
                and np.array_equal(base["correct"], correct)):
            _stop(f"{tag}: the pool or the correctly classified flows differ "
                  f"from those the base run saved; nothing was written.")
        n_ok = max(int(correct.sum()), 1)
        n_sat = int((correct & (_ce(model, Xe, ye, DEVICE) == 0.0)).sum())
        npz = os.path.join(RES_DIR, "flows", f"rvspsam_{tag}.npz")
        saved = npz_load(npz)
        msg = []
        for lv in levels:
            mask = c.masks[lv] if (depth, k) == (DEPTH, FEATS) else None
            for eps in EPS_LIST:
                if _k(c.ds, depth, k, c.act, c.seed, lv, round(eps, 3),
                      "spsam") in have:
                    continue
                s = rv2_seed(c.ds, depth, k, c.act, c.seed, lv, eps)
                _reseed(s)
                t0 = time.time()
                adv = _timed(lambda: rv_spsa_margin(model, Xe, ye, eps, s, mask))
                _, h, v = hits_of(model, adv, Xe, ye, DEVICE, eps, mask)
                if v:
                    _stop(f"capability violation: {v} frozen coordinates moved "
                          f"(rvspsam {tag} L{lv} eps={eps})")
                hs, hu = base[f"L{lv}_e{eps}_spsa"], base[f"L{lv}_e{eps}_union"]
                saved[f"L{lv}_e{eps}_spsam"] = h
                npz_save(npz, saved)
                stage_append("rvspsam", RV2_COLS, dict(
                    dataset=c.ds, depth=depth, features=k, activation=c.act,
                    seed=c.seed, level=lv, eps=round(eps, 3), attack="spsam",
                    asr=round(float(h.sum() / n_ok), 4), n_correct=n_ok,
                    n_hit=int(h.sum()), n_hit_spsa=int(hs.sum()),
                    n_new_vs_spsa=int((h & ~hs).sum()),
                    n_hit_union=int(hu.sum()), n_union_with=int((hu | h).sum()),
                    n_saturated=n_sat, clean_f1=round(float(f1), 4),
                    attack_seed=s, elapsed_s=round(time.time() - t0, 1)))
                have.add(_k(c.ds, depth, k, c.act, c.seed, lv, round(eps, 3),
                            "spsam"))
                msg.append(f"L{lv} eps={eps} {hs.sum() / n_ok:.3f}->"
                           f"{h.sum() / n_ok:.3f}")
        print(f"    rvspsam d{depth} k{k} (SPSA base->margin; {n_sat} of "
              f"{n_ok} flows saturated): " + "  ".join(msg), flush=True)


def rv_base_rows():
    """Every base row, one per measurement, with the unrounded rate."""
    files = sorted(f for f in glob.glob(os.path.join(RES_DIR, "attacks_*.csv"))
                   if f != OUT_FINAL)
    if not files:
        return None
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d = d.drop_duplicates(_KEY, keep="last").reset_index(drop=True)
    d["rate"] = d.n_hit / d.n_correct
    d["eps"] = d.eps.astype(float).round(3)
    return d


def summarize_rvspsam():
    r = stage_read("rvspsam", RV2_KEY)
    b = rv_base_rows()
    if r is None or b is None:
        print("%% stage rvspsam: not run yet")
        return
    r = r.copy()
    r["eps"] = r.eps.astype(float).round(3)
    for cn in ("n_hit", "n_hit_spsa", "n_hit_union", "n_union_with",
               "n_saturated"):
        r["r_" + cn[2:]] = r[cn] / r.n_correct
    print("%% rvspsam (a): Model 4, SPSA with the base loss and with the margin "
          "loss (same\n%% directions, budget and projection); 'sat' = share of "
          "correctly classified flows whose\n%% base SPSA loss is exactly 0; "
          "union = base union of PGD, BIM, SPSA, transfer.")
    m4 = r[(r.depth == DEPTH) & (r.features == FEATS)]
    for ds in DATASETS:
        for lv in LEVELS:
            for eps in EPS_LIST:
                for ac in ACTIVATIONS + ["all"]:
                    x = m4[(m4.dataset == ds) & (m4.level == lv)
                           & (m4.eps == round(eps, 3))]
                    if ac != "all":
                        x = x[x.activation == ac]
                    if not len(x):
                        continue
                    up = x.n_union_with - x.n_hit_union
                    print(f"  {ds:4s} C{lv} eps={eps} {ac:4s} n={len(x):2d}  "
                          f"SPSA base {rv_ms(x.r_hit_spsa)}  margin "
                          f"{rv_ms(x.r_hit)}  sat {rv_ms(x.r_saturated)}  "
                          f"union {rv_ms(x.r_hit_union)} -> "
                          f"{rv_ms(x.r_union_with)}  (models with a larger "
                          f"union {int((up > 0).sum())}, flows added "
                          f"{int(up.sum())})")
    print("\n%% rvspsam (b): mean of the four primary attacks (PGD, BIM, SPSA, "
          "HopSkipJump l_inf\n%% within eps) per model, with the base SPSA and "
          "with the margin SPSA in its place; L0.")
    b0 = b[(b.level == 0) & b.attack.isin(["pgd", "bim", "spsa", "hsj_inf"])]
    key = ["dataset", "depth", "features", "activation", "seed", "eps"]
    p = b0.pivot_table(index=key, columns="attack", values="rate").reset_index()
    q = r[r.level == 0][key + ["r_hit"]]
    p = p.merge(q, on=key, how="inner")
    p["mean4"] = p[["pgd", "bim", "spsa", "hsj_inf"]].mean(1)
    p["mean4m"] = p[["pgd", "bim", "r_hit", "hsj_inf"]].mean(1)
    for title, sel in (("depth sweep, 30 features", p.features == FEATS),
                       ("feature sweep, Model 1 ReLU",
                        (p.depth == 1) & (p.activation == "relu"))):
        print(f"  -- {title}")
        x = p[sel]
        for (ds, ac, dp, k, eps), y in x.groupby(
                ["dataset", "activation", "depth", "features", "eps"]):
            print(f"  {ds:4s} {ac:4s} d{dp} k{k:3d} eps={eps}  n={len(y):2d}  "
                  f"mean4 base {rv_ms(y.mean4)}  margin {rv_ms(y.mean4m)}  "
                  f"SPSA {rv_ms(y.spsa)} -> {rv_ms(y.r_hit)}")


STAGES.append(("rvspsam", stage_rvspsam, summarize_rvspsam))

# ======== stage rvthr: code ========
RV3_OCOLS = ["dataset", "activation", "seed", "defence", "tune", "fpr_target",
             "bias", "p_threshold", "fpr_val", "tpr_val", "fpr_test",
             "tpr_test", "clean_f1", "n_pool", "n_pool_detected", "elapsed_s"]
RV3_OKEY = RV3_OCOLS[:6]
RV3_COLS = ["dataset", "activation", "seed", "defence", "tune", "fpr_target",
            "level", "eps", "attack", "asr", "n_correct", "n_hit", "n_pool",
            "det_after", "attack_seed", "elapsed_s"]
RV3_KEY = RV3_COLS[:9]


class RvShift(nn.Module):
    """The network with its threshold moved."""
    def __init__(self, m, b):
        super().__init__()
        self.m = m
        self.register_buffer("shift", torch.tensor([float(b), 0.0],
                                                   dtype=torch.float32))

    def forward(self, x):
        return self.m(x) + self.shift


def rv_margins(model, X):
    """z_mal - z_ben per flow (float64 of the float32 logits)."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            z = model(torch.tensor(X[i:i + BATCH_SIZE], dtype=torch.float32,
                                   device=DEVICE)).double().cpu().numpy()
            out.append(z[:, 1] - z[:, 0])
    return np.concatenate(out)


def rv3_bias(m_benign, target):
    s = np.sort(np.asarray(m_benign, float))
    j = int(np.floor(target * len(s)))
    return float(s[len(s) - j - 1]) if j < len(s) else float(s[0])


def rv_model(c, dfc):
    """(model, clf, correct, n_ok) of a SAVED Model 4 network."""
    if dfc == "standard":
        return c.model, c.clf, c.correct, c.n_ok
    p = model_path(c.ds, DEPTH, c.Xtr.shape[1], c.act, c.seed, dfc)
    if not os.path.exists(p):
        _stop(f"{p} is missing: run the "
              f"{'advtrain' if dfc == AT_ROLE else 'caat'} stage of this part "
              f"first.")
    model = (at_get_model(c) if dfc == AT_ROLE else caat_get_model(c))[0]
    clf = _art_clf(model, c.Xtr, DEVICE)
    correct = _probs(model, c.Xe, DEVICE).argmax(1) == c.ye
    return model, clf, correct, max(int(correct.sum()), 1)


def rv_attack(c, model, clf, lv, eps, a, s, tag):
    """ONE base attack on the pool (as at_attack, with the model's name in the violation message)."""
    mask = c.masks[lv]
    sub = c.substitute() if a == "transfer" else None   # loaded before seeding
    _reseed(s)
    fn = {"pgd": lambda: _ref_q(clf, c.Xe, eps, mask, "pgd"),
          "bim": lambda: _ref_q(clf, c.Xe, eps, mask, "bim"),
          "spsa": lambda: a_spsa(model, c.Xe, c.ye, eps, s, mask),
          "spsam": lambda: rv_spsa_margin(model, c.Xe, c.ye, eps, s, mask),
          "transfer": lambda: a_transfer_saved(sub, c.Xe, eps, mask)}[a]
    adv = _timed(fn)
    _, h, v = hits_of(model, adv, c.Xe, c.ye, DEVICE, eps, mask)
    if v:
        _stop(f"capability violation: {v} frozen coordinates moved "
              f"({c.part} {tag} L{lv} eps={eps} {a})")
    return h


def stage_rvthr(c):
    have = stage_have("rvthr", RV3_KEY)
    haveo = stage_have("rvthrop", RV3_OKEY)
    Xtrn, Xval, ytrn, yval = c.split()
    yv, yt = (np.asarray(yval) != 0), (np.asarray(c.yte) != 0)
    npz = stage_npz("rvthr", c)
    saved = npz_load(npz)
    for dfc in RV3_DEFENCES:
        todo = []
        for tune in RV3_TUNE:
            for t in RV3_FPR[c.ds]:
                ko = _k(c.ds, c.act, c.seed, dfc, tune, t)
                ka = [_k(c.ds, c.act, c.seed, dfc, tune, t, lv, round(e, 3),
                         "union") for lv in AT_LEVELS for e in EPS_LIST]
                if not (ko in haveo and all(x in have for x in ka)):
                    todo.append((tune, t))
        if not todo:
            print(f"    skip rvthr {dfc} {c.part}: done", flush=True)
            continue
        model = rv_model(c, dfc)[0]
        for tune, t in todo:
            t0 = time.time()
            Xb = Xval[~yv] if tune == "val" else c.Xte[~yt]
            b = rv3_bias(rv_margins(model, Xb), t)
            w = RvShift(model, b).to(DEVICE).eval()
            clf = _art_clf(w, c.Xtr, DEVICE)
            pv = _probs(w, Xval, DEVICE).argmax(1) == 1
            pt = _probs(w, c.Xte, DEVICE).argmax(1) == 1
            correct = _probs(w, c.Xe, DEVICE).argmax(1) == c.ye
            n_ok, n_pool = max(int(correct.sum()), 1), len(c.ye)
            kb = dict(dataset=c.ds, activation=c.act, seed=c.seed, defence=dfc,
                      tune=tune, fpr_target=t)
            if _k(c.ds, c.act, c.seed, dfc, tune, t) not in haveo:
                stage_append("rvthrop", RV3_OCOLS, dict(
                    kb, bias=b, p_threshold=float(1.0 / (1.0 + np.exp(-b))),
                    fpr_val=float(pv[~yv].mean()), tpr_val=float(pv[yv].mean()),
                    fpr_test=float(pt[~yt].mean()),
                    tpr_test=float(pt[yt].mean()),
                    clean_f1=float(macro_f1(w, c.Xte, c.yte, DEVICE)),
                    n_pool=n_pool, n_pool_detected=int(correct.sum()),
                    elapsed_s=round(time.time() - t0, 1)))
            msg = []
            for lv in AT_LEVELS:
                for eps in EPS_LIST:
                    if _k(c.ds, c.act, c.seed, dfc, tune, t, lv, round(eps, 3),
                          "union") in have:
                        continue
                    union, rows = np.zeros(n_pool, bool), []
                    for a in AT_ATTACKS:
                        s = _seed_of("rvthr", c.ds, c.act, c.seed, dfc, tune, t,
                                     lv, eps, a)
                        t2 = time.time()
                        h = rv_attack(c, w, clf, lv, eps, a, s,
                                      f"rvthr {dfc} {tune} {t}")
                        union |= h
                        rows.append((a, h, s, time.time() - t2))
                    rows.append(("union", union, 0, 0.0))
                    for a, h, s, el in rows:
                        saved[f"{dfc}_{tune}_{t}_L{lv}_e{eps}_{a}"] = h
                    saved[f"{dfc}_{tune}_{t}_correct"] = correct
                    npz_save(npz, saved)
                    for a, h, s, el in rows:
                        stage_append("rvthr", RV3_COLS, dict(
                            kb, level=lv, eps=round(eps, 3), attack=a,
                            asr=round(float(h.sum() / n_ok), 4), n_correct=n_ok,
                            n_hit=int(h.sum()), n_pool=n_pool,
                            det_after=float((correct.sum() - h.sum()) / n_pool),
                            attack_seed=s, elapsed_s=round(el, 1)))
                    msg.append(f"C{lv}/{eps} {union.sum() / n_ok:.3f}")
            print(f"    rvthr {dfc} {tune} FPR {t}: test FPR "
                  f"{pt[~yt].mean():.4f} TPR {pt[yt].mean():.4f}, pool "
                  f"{int(correct.sum())}/{n_pool}; union " + " ".join(msg)
                  + f" [{time.time() - t0:.0f}s]", flush=True)


def rv_default_rows():
    """Union rows at the default threshold for the three networks."""
    b = rv_base_rows()
    parts = []
    if b is not None:
        x = b[(b.depth == DEPTH) & (b.features == FEATS)].copy()
        x["defence"] = "standard"
        parts.append(x)
    for st, role in (("atattacks", AT_ROLE), ("caatattacks", CAAT_ROLE)):
        y = stage_read(st, AT_KEY)
        if y is not None:
            y = y[y.defence == role].copy()
            y["eps"] = y.eps.astype(float).round(3)
            parts.append(y)
    if not parts:
        return None
    d = pd.concat(parts, ignore_index=True)
    return d[d.attack == "union"]


def summarize_rvthr():
    o = stage_read("rvthrop", RV3_OKEY)
    a = stage_read("rvthr", RV3_KEY)
    if o is None or a is None:
        print("%% stage rvthr: not run yet")
        return
    a = a.copy()
    a["eps"] = a.eps.astype(float).round(3)
    a["rate"] = a.n_hit / a.n_correct
    u = a[a.attack == "union"]
    dflt = rv_default_rows()
    pool = {ds: int(o[o.dataset == ds].n_pool.iloc[0])
            for ds in DATASETS if (o.dataset == ds).any()}
    print("%% rvthr: networks compared at the same false-positive rate. "
          "'union' = worst case\n%% over PGD, BIM, SPSA, transfer among the "
          "flows detected at that threshold;\n%% 'det' = share of the WHOLE "
          "pool still detected after the attack.\n%% Reference rows: the "
          "default threshold (argmax, P(malicious) >= 0.5).")
    for ds in DATASETS:
        if ds not in pool:
            continue
        for ac in ACTIVATIONS + ["all"]:
            sel = (lambda d: d[d.dataset == ds] if ac == "all" else
                   d[(d.dataset == ds) & (d.activation == ac)])
            if dflt is not None:
                x = sel(dflt)
                for dfc in RV3_DEFENCES:
                    for lv in AT_LEVELS:
                        y = x[(x.defence == dfc) & (x.level == lv)]
                        if not len(y):
                            continue
                        print(f"  {ds:4s} {ac:4s} {dfc:9s} default      C{lv}  "
                              + "  ".join(
                                  f"eps={e}: union {rv_ms(z.n_hit / z.n_correct)}"
                                  f" det {rv_ms((z.n_correct - z.n_hit) / pool[ds])}"
                                  for e in EPS_LIST
                                  for z in [y[y.eps == round(e, 3)]] if len(z)))
            for tune in RV3_TUNE:
                for t in RV3_FPR[ds]:
                    for dfc in RV3_DEFENCES:
                        oo = sel(o)
                        oo = oo[(oo.defence == dfc) & (oo.tune == tune)
                                & np.isclose(oo.fpr_target, t)]
                        if not len(oo):
                            continue
                        print(f"  {ds:4s} {ac:4s} {dfc:9s} {tune:4s} FPR {t:<5}  "
                              f"test FPR {rv_ms(oo.fpr_test)} TPR "
                              f"{rv_ms(oo.tpr_test)} F1 {rv_ms(oo.clean_f1)} "
                              f"val FPR {rv_ms(oo.fpr_val)} pool detected "
                              f"{rv_ms(oo.n_pool_detected / oo.n_pool)}")
                        uu = sel(u)
                        uu = uu[(uu.defence == dfc) & (uu.tune == tune)
                                & np.isclose(uu.fpr_target, t)]
                        for lv in AT_LEVELS:
                            y = uu[uu.level == lv]
                            print(f"        C{lv}  " + "  ".join(
                                f"eps={e}: union {rv_ms(z.rate)} det "
                                f"{rv_ms(z.det_after)}"
                                for e in EPS_LIST
                                for z in [y[y.eps == round(e, 3)]] if len(z)))
    print("\n%% rvthr paired: det(PGD-AT) - det(standard) and det(SS-AT) - "
          "det(standard), same\n%% activation and seed, same threshold rule; "
          "positive = the defended network detects more.")
    for ds in DATASETS:
        for tune in RV3_TUNE:
            for t in RV3_FPR.get(ds, []):
                for lv in AT_LEVELS:
                    for eps in EPS_LIST:
                        x = u[(u.dataset == ds) & (u.tune == tune)
                              & np.isclose(u.fpr_target, t) & (u.level == lv)
                              & (u.eps == round(eps, 3))]
                        p = x.pivot_table(index=["activation", "seed"],
                                          columns="defence",
                                          values="det_after").dropna()
                        if "standard" not in p.columns:
                            continue
                        txt = []
                        for dfc in (AT_ROLE, CAAT_ROLE):
                            if dfc in p.columns:
                                dd = p[dfc] - p["standard"]
                                txt.append(
                                    f"{dfc} {rv_ms(dd)} (+{int((dd > 0).sum())}"
                                    f"/-{int((dd < 0).sum())} of {len(dd)}, p "
                                    f"{caat_sign_p(int((dd > 0).sum()), int((dd != 0).sum())):.3g})")
                        print(f"  {ds:4s} {tune:4s} FPR {t:<5} C{lv} eps={eps}: "
                              + "  ".join(txt))


STAGES.append(("rvthr", stage_rvthr, summarize_rvthr))

# ======== stage rvstrong: code ========
RV4_COLS = ["dataset", "activation", "seed", "defence", "level", "eps",
            "attack", "asr", "n_correct", "n_hit", "n_free", "n_diff_base",
            "source", "attack_seed", "elapsed_s"]
RV4_KEY = RV4_COLS[:7]


def rv4_base_seed(c, dfc, lv, eps, a):
    if dfc == "standard":
        return _seed_of(c.ds, DEPTH, FEATS, c.act, c.seed, lv, eps, a)
    return (at_attack_seed if dfc == AT_ROLE else caat_attack_seed)(
        c, lv, eps, a)


def rv4_base_npz(c, dfc):
    if dfc == "standard":
        return os.path.join(RES_DIR, "flows", f"{c.ds}_d{DEPTH}_f{FEATS}_"
                            f"{c.act}_s{c.seed}.npz")
    return stage_npz("advtrain" if dfc == AT_ROLE else "caat", c)


def rv4_hsj_seed(c, dfc):
    k = c.Xtr.shape[1]
    if dfc == "standard":
        return _seed_of(c.ds, DEPTH, k, c.act, c.seed, "hsj", AT_HSJ[0])
    if dfc == AT_ROLE:
        return _seed_of("advtrain", c.ds, DEPTH, k, c.act, c.seed, AT_ROLE,
                        "hsj", AT_HSJ[0])
    return _seed_of("rvstrong", c.ds, DEPTH, k, c.act, c.seed, dfc, "hsj",
                    AT_HSJ[0])


def rv4_spsam_seed(c, dfc, lv, eps):
    if dfc == "standard":                 # the very draw of stage rvspsam
        return rv2_seed(c.ds, DEPTH, FEATS, c.act, c.seed, lv, eps)
    return _seed_of("rvstrong", c.ds, c.act, c.seed, dfc, lv, eps, "spsam")


def rv4_lattice(c, model, correct, eps):
    """cd_lattice for a given network, symmetric box."""
    free = np.where(np.asarray(c.masks[2]).reshape(-1) > 0)[0]
    axes = np.meshgrid(*[np.linspace(-eps, eps, CA_ENUM_GRID) for _ in free],
                       indexing="ij")
    offs = np.unique(np.stack([a.reshape(-1) for a in axes], 1)
                     .astype(np.float32), axis=0)
    lo_c, hi_c = float(c.Xtr.min()), float(c.Xtr.max())
    n, d = c.Xe.shape
    out = np.zeros(n, bool)
    for i in range(0, n, CA_ENUM_CHUNK):
        sl = slice(i, i + CA_ENUM_CHUNK)
        xb, yb, ok = c.Xe[sl], c.ye[sl], correct[sl]
        pts = np.repeat(xb[:, None, :], len(offs), 1).astype(np.float32)
        pts[:, :, free] += offs[None]
        pts = np.clip(pts, lo_c, hi_c)
        w = (_probs(model, pts.reshape(-1, d), DEVICE).argmax(1)
             .reshape(pts.shape[0], pts.shape[1]) != yb[:, None])
        out[sl] = w.any(1) & ok
    return out, len(offs)


def stage_rvstrong(c):
    if (sorted(set(RV4_REPORT)) != RV4_REPORT or RV4_REPORT[0] != 1
            or RV4_REPORT[-1] != RV4_RESTARTS):
        _stop("RV4_REPORT must be increasing, start at 1 and end at "
              "RV4_RESTARTS.")
    have = stage_have("rvstrong", RV4_KEY)
    npz = stage_npz("rvstrong", c)
    saved = npz_load(npz)
    hsj = f"hsj_{AT_HSJ[0]}"
    for dfc in RV4_DEFENCES:
        key = lambda lv, eps, a: _k(c.ds, c.act, c.seed, dfc, lv,
                                    round(eps, 3), a)
        if all(key(lv, e, "strong") in have for lv in AT_LEVELS
               for e in EPS_LIST):
            print(f"    skip rvstrong {dfc} {c.part}: done", flush=True)
            continue
        model, clf, correct, n_ok = rv_model(c, dfc)
        base = npz_load(rv4_base_npz(c, dfc))
        if base and not (np.array_equal(base.get("pool"), c.pool) and
                         np.array_equal(base.get("correct"), correct)):
            _stop(f"rvstrong {dfc} {c.part}: the pool or the correctly "
                  f"classified flows differ from the saved base result.")
        hs, hsrc, hseed = {}, "", rv4_hsj_seed(c, dfc)
        if 0 in AT_LEVELS and not all(key(0, e, "strong") in have
                                      for e in EPS_LIST):
            kk = [f"L0_e{e}_{hsj}{x}" for e in EPS_LIST for x in ("", "_proj")]
            kr = [f"{dfc}_L0_e{e}_{hsj}" for e in EPS_LIST]
            if all(x in base for x in kk):
                hsrc = "saved"
                for e in EPS_LIST:
                    hs[e] = base[f"L0_e{e}_{hsj}"] | base[f"L0_e{e}_{hsj}_proj"]
            elif all(x in saved for x in kr):
                hsrc = "run"
                for e in EPS_LIST:
                    hs[e] = saved[f"{dfc}_L0_e{e}_{hsj}"]
            else:
                hsrc, t2 = "run", time.time()
                raw, req, sets = at_hsj(c, model, clf, correct, hseed)
                for e in EPS_LIST:
                    hs[e] = sets[e][0] | sets[e][1]
                    saved[f"{dfc}_L0_e{e}_{hsj}"] = hs[e]
                npz_save(npz, saved)
                print(f"    rvstrong {dfc} HopSkipJump: " + " ".join(
                    f"eps={e} {hs[e].sum() / n_ok:.3f}" for e in EPS_LIST)
                    + f" [{time.time() - t2:.0f}s]", flush=True)
        for lv in AT_LEVELS:
            mask = c.masks[lv]
            nfree = FEATS if mask is None else int(mask.sum())
            for eps in EPS_LIST:
                if key(lv, eps, "strong") in have:
                    continue
                t1, rows, comp = time.time(), [], {}

                def add(a, h, nd, src, s, el):
                    rows.append((a, h, nd, src, s, el))
                for a in AT_ATTACKS:
                    s = rv4_base_seed(c, dfc, lv, eps, a)
                    t2 = time.time()
                    h = rv_attack(c, model, clf, lv, eps, a, s, f"rvstrong {dfc}")
                    ref = base.get(f"L{lv}_e{eps}_{a}")
                    comp[a] = h
                    add(a, h, -1 if ref is None else int((ref != h).sum()),
                        "rerun", s, time.time() - t2)
                bu = np.logical_or.reduce([comp[a] for a in AT_ATTACKS])
                ref = base.get(f"L{lv}_e{eps}_union")
                add("base_union", bu, -1 if ref is None else int((ref != bu).sum()),
                    "rerun", 0, 0.0)
                s = rv4_spsam_seed(c, dfc, lv, eps)
                t2 = time.time()
                hm = rv_attack(c, model, clf, lv, eps, "spsam", s,
                               f"rvstrong {dfc}")
                add("spsam", hm, -1, "run", s, time.time() - t2)
                cum, t2 = comp["pgd"].copy(), time.time()
                if 1 in RV4_REPORT:
                    add("pgd_r1", cum.copy(), -1, "rerun",
                        rv4_base_seed(c, dfc, lv, eps, "pgd"), 0.0)
                for i in range(1, RV4_RESTARTS):
                    si = _seed_of("rvstrong", c.ds, c.act, c.seed, dfc, lv, eps,
                                  "pgd", i)
                    cum |= rv_attack(c, model, clf, lv, eps, "pgd", si,
                                     f"rvstrong {dfc} start {i}")
                    if i + 1 in RV4_REPORT:
                        add(f"pgd_r{i + 1}", cum.copy(), -1, "run", 0,
                            time.time() - t2)
                strong = bu | hm | cum
                if lv == 0 and eps in hs:
                    add(hsj, hs[eps], -1, hsrc, hseed, 0.0)
                    strong = strong | hs[eps]
                if lv == 2 and nfree <= CA_ENUM_MAX_FREE:
                    t2 = time.time()
                    lat, n_pts = rv4_lattice(c, model, correct, eps)
                    add("lattice", lat, -1, f"{n_pts} points", 0,
                        time.time() - t2)
                    strong = strong | lat
                add("strong", strong, -1, "union", 0, time.time() - t1)
                for a, h, nd, src, s, el in rows:
                    saved[f"{dfc}_L{lv}_e{eps}_{a}"] = h
                saved[f"{dfc}_correct"] = correct
                saved["pool"] = c.pool
                npz_save(npz, saved)
                for a, h, nd, src, s, el in rows:
                    stage_append("rvstrong", RV4_COLS, dict(
                        dataset=c.ds, activation=c.act, seed=c.seed,
                        defence=dfc, level=lv, eps=round(eps, 3), attack=a,
                        asr=round(float(h.sum() / n_ok), 4), n_correct=n_ok,
                        n_hit=int(h.sum()), n_free=nfree, n_diff_base=nd,
                        source=src, attack_seed=s, elapsed_s=round(el, 1)))
                have.add(key(lv, eps, "strong"))
                nd = sum(r[2] for r in rows if r[2] > 0)
                print(f"    rvstrong {dfc} C{lv} eps={eps}: base union "
                      f"{bu.sum() / n_ok:.4f} -> strong {strong.sum() / n_ok:.4f}"
                      f" (margin SPSA {hm.sum() / n_ok:.4f}, PGD x"
                      f"{RV4_RESTARTS} {cum.sum() / n_ok:.4f}); flows differing "
                      f"from the saved base: {nd} [{time.time() - t1:.0f}s]",
                      flush=True)


def summarize_rvstrong():
    d = stage_read("rvstrong", RV4_KEY)
    if d is None:
        print("%% stage rvstrong: not run yet")
        return
    d = d.copy()
    d["eps"] = d.eps.astype(float).round(3)
    d["rate"] = d.n_hit / d.n_correct
    hsj = f"hsj_{AT_HSJ[0]}"
    chk = d[d.attack.isin(AT_ATTACKS + ["base_union"]) & (d.n_diff_base >= 0)]
    print("%% rvstrong: re-run of the base attacks against the saved base "
          f"per-flow results: {int(chk.n_diff_base.sum())} differing flows in "
          f"{len(chk)} compared rows ({int((chk.n_diff_base > 0).sum())} rows "
          f"with any difference); "
          f"{int((d.attack.isin(AT_ATTACKS) & (d.n_diff_base < 0)).sum())} "
          "re-run rows had no saved base result to compare with.")
    if (chk.n_diff_base > 0).any():
        print(chk[chk.n_diff_base > 0][["dataset", "activation", "seed",
                                        "defence", "level", "eps", "attack",
                                        "n_diff_base"]].to_string(index=False))
    print("%% mean+-SD of attack success over the models of each cell "
          "(n = models).")
    comps = (["base_union", "spsam"] + [f"pgd_r{R}" for R in RV4_REPORT]
             + [hsj, "lattice", "strong"])
    for ds in DATASETS:
        for lv in AT_LEVELS:
            for eps in EPS_LIST:
                for ac in ACTIVATIONS + ["all"]:
                    x = d[(d.dataset == ds) & (d.level == lv)
                          & (d.eps == round(eps, 3))]
                    if ac != "all":
                        x = x[x.activation == ac]
                    for dfc in RV4_DEFENCES:
                        y = x[x.defence == dfc]
                        if not len(y):
                            continue
                        p = y.pivot_table(index=["activation", "seed"],
                                          columns="attack", values="n_hit")
                        up = (p["strong"] - p["base_union"]) if {
                            "strong", "base_union"} <= set(p.columns) else None
                        print(f"  {ds:4s} C{lv} eps={eps} {ac:4s} {dfc:9s} " +
                              "  ".join(f"{a} {rv_ms(y[y.attack == a].rate)}"
                                        for a in comps if (y.attack == a).any())
                              + ("" if up is None else
                                 f"  | strong>base on {int((up > 0).sum())} of "
                                 f"{len(up)} models, flows added {int(up.sum())}"
                                 f" (max {int(up.max())})"))
    print("\n%% rvstrong: share of the PGD-AT reduction kept by SS-AT, "
          "(S_std - S_ss) / (S_std - S_pgd),\n%% from the mean attack success "
          "over the models of each cell; base union and strong.")
    for ds in DATASETS:
        for lv in AT_LEVELS:
            for ac in ACTIVATIONS + ["all"]:
                txt = []
                for eps in EPS_LIST:
                    x = d[(d.dataset == ds) & (d.level == lv)
                          & (d.eps == round(eps, 3))]
                    if ac != "all":
                        x = x[x.activation == ac]
                    v = {}
                    for a in ("base_union", "strong"):
                        m = {dfc: x[(x.defence == dfc) & (x.attack == a)].rate
                             for dfc in RV4_DEFENCES}
                        if all(len(m[dfc]) for dfc in RV4_DEFENCES):
                            s0, s1, s2 = (m["standard"].mean(),
                                          m[AT_ROLE].mean(), m[CAAT_ROLE].mean())
                            v[a] = ((s0 - s2) / (s0 - s1) if s0 != s1
                                    else float("nan"))
                    if v:
                        txt.append(f"eps={eps}: base {v.get('base_union', float('nan')):.4f}"
                                   f" strong {v.get('strong', float('nan')):.4f}")
                if txt:
                    print(f"  {ds:4s} C{lv} {ac:4s} " + "  ".join(txt))
    print("\n%% rvstrong: evasions pooled over the models and the budgets "
          "(flow-budget pairs), and the\n%% per-model comparison SS-AT vs "
          "PGD-AT (sum over budgets), base union and strong.")
    for ds in DATASETS:
        for lv in AT_LEVELS:
            x = d[(d.dataset == ds) & (d.level == lv)]
            if not len(x):
                continue
            txt = []
            for a in ("base_union", "strong"):
                y = x[x.attack == a]
                tot = {dfc: int(y[y.defence == dfc].n_hit.sum())
                       for dfc in RV4_DEFENCES}
                p = y.pivot_table(index=["activation", "seed"],
                                  columns="defence", values="n_hit",
                                  aggfunc="sum")
                cmp = ""
                if {AT_ROLE, CAAT_ROLE} <= set(p.columns):
                    dd = (p[CAAT_ROLE] - p[AT_ROLE]).dropna()
                    cmp = (f" SS-AT more on {int((dd > 0).sum())}, fewer on "
                           f"{int((dd < 0).sum())} of {len(dd)} models (p "
                           f"{caat_sign_p(int((dd > 0).sum()), int((dd != 0).sum())):.3g})")
                txt.append(f"{a}: " + " ".join(f"{k_} {v_}" for k_, v_ in
                                               tot.items()) + cmp)
            print(f"  {ds:4s} C{lv}: " + " | ".join(txt))


STAGES.append(("rvstrong", stage_rvstrong, summarize_rvstrong))


# ======== stage rvhsjc2: code ========
RV5_COLS = ["dataset", "activation", "seed", "defence", "level", "eps", "attack",
            "asr", "n_correct", "n_hit", "n_hit_union", "n_new_vs_union",
            "n_hit_strong", "n_new_vs_strong", "n_free", "required_linf",
            "attack_seed", "elapsed_s"]
RV5_KEY = RV5_COLS[:7]


class _hsj_exact_interpolation:
    """During the with-block, ART HopSkipJump interpolates as x + a * (x_adv - x) instead of (1 - a) * x + a * x_adv."""
    def __enter__(self):
        from art.attacks.evasion import hop_skip_jump as _H
        self._cls = _H.HopSkipJump
        self._orig = self._cls.__dict__["_interpolate"]

        def _interp(current_sample, original_sample, alpha, norm):
            if norm == 2:
                return original_sample + alpha * (current_sample - original_sample)
            return np.clip(current_sample, original_sample - alpha,
                           original_sample + alpha)
        self._cls._interpolate = staticmethod(_interp)
        return self

    def __exit__(self, *exc):
        self._cls._interpolate = self._orig
        return False


def rv5_clip_check(clf, X, mask, where):
    """Stop if a frozen pool value lies outside the classifier clip range."""
    lo, hi = (float(v) for v in np.asarray(clf.clip_values, np.float64).ravel()[:2])
    fr = np.asarray(mask).reshape(-1) == 0
    Xf = np.asarray(X, np.float64)[:, fr]
    n_out = int(((Xf < lo) | (Xf > hi)).sum())
    if n_out:
        _stop(f"{where}: {n_out} frozen pool values lie outside the classifier "
              f"clip range [{lo:.4g}, {hi:.4g}]; HopSkipJump would move them.")


def rv5_hsj_masked(clf, X, mask, seed):
    from art.attacks.evasion import HopSkipJump
    atk = HopSkipJump(classifier=clf, targeted=False, norm=AT_HSJ[1],
                      max_iter=HSJ_MAX_ITER, max_eval=HSJ_MAX_EVAL,
                      init_eval=HSJ_INIT_EVAL, batch_size=128,
                      verbose=HSJ_VERBOSE)
    _orig, _n = np.random.RandomState, [0]

    def _seeded(*a, **k):
        if a or k:
            return _orig(*a, **k)
        _n[0] += 1
        return _orig((seed + _n[0]) % (2 ** 32))
    rv5_clip_check(clf, X, mask, "rv5_hsj_masked")
    np.random.RandomState = _seeded
    try:
        with _hsj_exact_interpolation():
            adv = atk.generate(x=X.astype(np.float32),
                               mask=np.asarray(mask, np.float32).reshape(-1))
    finally:
        np.random.RandomState = _orig
    ok = np.isfinite(adv).all(1)
    return np.where(ok[:, None], adv, X).astype(np.float32)


# Side table rvhsjc2round.
RV5_ULPS = 4
RV5R_COLS = ["dataset", "activation", "seed", "defence", "level",
             "n_frozen_rounded", "max_abs_dev", "n_frozen", "n_flows"]
RV5R_KEY = RV5R_COLS[:5]


def rv5_restore_rounding(raw, X, frozen, part, dfc, lv):
    """(raw with the frozen coordinates restored, number restored, largest absolute move)."""
    d = np.abs(raw.astype(np.float64) - X.astype(np.float64))[:, frozen]
    moved = d > 1e-6
    n_moved = int(moved.sum())
    if n_moved == 0:
        return raw, 0, 0.0
    tol = RV5_ULPS * np.finfo(np.float32).eps * np.abs(X.astype(np.float64)[:, frozen])
    real = moved & (d > tol)
    if real.any():
        xs = X.astype(np.float64)[:, frozen][real]
        _stop(f"capability violation: HopSkipJump moved {int(real.sum())} "
              f"frozen coordinates by more than {RV5_ULPS} float32 units in "
              f"the last place ({part} {dfc} L{lv}): largest move "
              f"{float(d[real].max()):.3g} at x = "
              f"{float(xs[np.argmax(d[real])]):.4g}; nothing was written.")
    out = raw.copy()
    out[:, frozen] = X[:, frozen]
    return out, n_moved, float(d.max())


# Per-flow driver with a time cap.
RV5_FLOW_CAP = 600                  # seconds per flow; the median is 0.3 s
RV5_RESEED_STEP = 1_000_003         # counter and seed offset after a capped flow
RV5C_COLS = ["dataset", "activation", "seed", "defence", "level", "n_flows",
             "n_capped", "capped_flows", "cap_s"]
RV5C_KEY = RV5C_COLS[:5]


def rv5_hsj_masked_capped(clf, X, mask, seed, cap=RV5_FLOW_CAP):
    """rv5_hsj_masked driven one flow at a time with a per-flow time cap."""
    from art.attacks.evasion import HopSkipJump
    atk = HopSkipJump(classifier=clf, targeted=False, norm=AT_HSJ[1],
                      max_iter=HSJ_MAX_ITER, max_eval=HSJ_MAX_EVAL,
                      init_eval=HSJ_INIT_EVAL, batch_size=128,
                      verbose=HSJ_VERBOSE)
    m = np.asarray(mask, np.float32).reshape(-1)
    X = X.astype(np.float32)
    out, capped = X.copy(), []
    _orig, _n = np.random.RandomState, [0]

    def _seeded(*a, **k):
        if a or k:
            return _orig(*a, **k)
        _n[0] += 1
        return _orig((seed + _n[0]) % (2 ** 32))
    rv5_clip_check(clf, X, mask, "rv5_hsj_masked_capped")
    np.random.RandomState = _seeded
    try:
        with _hsj_exact_interpolation():
            for i in range(len(X)):
                try:
                    adv = _run_with_timeout(
                        lambda: atk.generate(x=X[i:i + 1], mask=m), cap)
                    out[i] = np.where(np.isfinite(adv[0]), adv[0], X[i])
                except _Timeout:
                    capped.append(i)
                    _n[0] = RV5_RESEED_STEP * (i + 1)
                    np.random.seed((seed + RV5_RESEED_STEP * (i + 1)) % (2 ** 32))
    finally:
        np.random.RandomState = _orig
    return out, capped


def stage_rvhsjc2(c):
    lv = RV5_LEVEL
    mask = c.masks[lv]
    if mask is None:
        _stop(f"RV5_LEVEL={lv} has no mask.")
    nfree = int(np.asarray(mask).sum())
    have = stage_have("rvhsjc2", RV5_KEY)
    npz = stage_npz("rvhsjc2", c)
    saved = npz_load(npz)
    hsj = f"hsj_{AT_HSJ[0]}"
    for dfc in RV5_DEFENCES:
        key = lambda eps, a: _k(c.ds, c.act, c.seed, dfc, lv, round(eps, 3), a)
        if all(key(e, hsj + x) in have for e in EPS_LIST for x in ("", "_proj")):
            print(f"    skip rvhsjc2 {dfc} {c.part}: done", flush=True)
            continue
        model, clf, correct, n_ok = rv_model(c, dfc)
        strong = npz_load(stage_npz("rvstrong", c))
        ku = f"{dfc}_L{lv}_e{EPS_LIST[0]}_base_union"
        if ku not in strong:
            _stop(f"rvstrong results for {dfc} {c.part} are missing ({ku}): run "
                  f"stage rvstrong first.")
        if not (np.array_equal(strong.get("pool"), c.pool)
                and np.array_equal(strong.get(f"{dfc}_correct"), correct)):
            _stop(f"rvhsjc2 {dfc} {c.part}: pool or correctly classified flows "
                  f"differ from those of stage rvstrong.")
        s = _seed_of("rvhsjc2", c.ds, DEPTH, FEATS, c.act, c.seed, dfc, "hsj",
                     AT_HSJ[0], lv)
        _reseed(s); t0 = time.time()
        raw, capped = rv5_hsj_masked_capped(clf, c.Xe, mask, s)
        if not stage_have("rvhsjc2cap", RV5C_KEY) >= {
                _k(c.ds, c.act, c.seed, dfc, lv)}:
            stage_append("rvhsjc2cap", RV5C_COLS, dict(
                dataset=c.ds, activation=c.act, seed=c.seed, defence=dfc,
                level=lv, n_flows=int(len(c.Xe)), n_capped=len(capped),
                capped_flows=" ".join(str(i) for i in capped),
                cap_s=RV5_FLOW_CAP))
        frozen = np.asarray(mask).reshape(-1) == 0
        raw, n_round, max_dev = rv5_restore_rounding(raw, c.Xe, frozen,
                                                     c.part, dfc, lv)
        if not stage_have("rvhsjc2round", RV5R_KEY) >= {
                _k(c.ds, c.act, c.seed, dfc, lv)}:
            stage_append("rvhsjc2round", RV5R_COLS, dict(
                dataset=c.ds, activation=c.act, seed=c.seed, defence=dfc,
                level=lv, n_frozen_rounded=n_round, max_abs_dev=max_dev,
                n_frozen=int(frozen.sum()), n_flows=int(len(c.Xe))))
        dist = np.abs(raw - c.Xe).max(1)
        wrong = _probs(model, raw, DEVICE).argmax(1) != c.ye
        evaded = correct & wrong
        req = float(np.median(dist[evaded])) if evaded.any() else float("nan")
        el = round((time.time() - t0) / len(EPS_LIST), 1)
        saved[f"{dfc}_{hsj}_raw"] = raw
        msg = []
        for eps in EPS_LIST:
            within = evaded & (dist <= eps + 1e-6)
            proj = (c.Xe + np.clip(raw - c.Xe, -eps, eps)).astype(np.float32)
            _, hp, v = hits_of(model, proj, c.Xe, c.ye, DEVICE, eps, mask)
            if v:
                _stop(f"capability violation after projection ({c.part} {dfc})")
            hu = strong[f"{dfc}_L{lv}_e{eps}_base_union"]
            hs = strong[f"{dfc}_L{lv}_e{eps}_strong"]
            for a, h in ((hsj, within), (hsj + "_proj", hp)):
                saved[f"{dfc}_L{lv}_e{eps}_{a}"] = h
                if key(eps, a) in have:
                    continue
                stage_append("rvhsjc2", RV5_COLS, dict(
                    dataset=c.ds, activation=c.act, seed=c.seed, defence=dfc,
                    level=lv, eps=round(eps, 3), attack=a,
                    asr=round(float(h.sum() / n_ok), 4), n_correct=n_ok,
                    n_hit=int(h.sum()), n_hit_union=int(hu.sum()),
                    n_new_vs_union=int((h & ~hu).sum()),
                    n_hit_strong=int(hs.sum()),
                    n_new_vs_strong=int((h & ~hs).sum()), n_free=nfree,
                    required_linf=round(req, 4), attack_seed=s, elapsed_s=el))
            msg.append(f"eps={eps} {within.sum() / n_ok:.4f}/{hp.sum() / n_ok:.4f}"
                       f" (union {hu.sum() / n_ok:.4f}, new {int((hp & ~hu).sum())})")
        saved["pool"], saved[f"{dfc}_correct"] = c.pool, correct
        npz_save(npz, saved)
        print(f"    rvhsjc2 {dfc} C{lv} HopSkipJump within/projected: "
              + "  ".join(msg) + f"  required l_inf {req:.3f}  frozen "
              f"coordinates restored from rounding {n_round} (max {max_dev:.2e})"
              f"  flows capped at {RV5_FLOW_CAP}s: {len(capped)}"
              f" [{time.time() - t0:.0f}s]", flush=True)


def summarize_rvhsjc2():
    d = stage_read("rvhsjc2", RV5_KEY)
    if d is None:
        print("%% stage rvhsjc2: not run yet")
        return
    d = d.copy()
    d["eps"] = d.eps.astype(float).round(3)
    for cn in ("n_hit", "n_hit_union", "n_hit_strong"):
        d["r" + cn[1:]] = d[cn] / d.n_correct
    hsj = f"hsj_{AT_HSJ[0]}"
    print("%% rvhsjc2: HopSkipJump with the C2 mask inside the search, scored "
          "within eps and\n%% projected; 'union' = four-attack union, 'strong' = "
          "stronger worst case of stage rvstrong,\n%% both on the same flows. "
          "Mean+-SD over the models of each cell; 'new' = flows the attack\n%% "
          "evades that the union / the strong worst case does not, summed over "
          "the models.")
    for ds in DATASETS:
        for eps in EPS_LIST:
            for ac in ACTIVATIONS + ["all"]:
                x = d[(d.dataset == ds) & (d.eps == eps)]
                if ac != "all":
                    x = x[x.activation == ac]
                for dfc in RV5_DEFENCES:
                    y = x[x.defence == dfc]
                    if not len(y):
                        continue
                    w, p = y[y.attack == hsj], y[y.attack == hsj + "_proj"]
                    print(f"  {ds:4s} eps={eps} {ac:4s} {dfc:9s} n={len(w):2d}  "
                          f"within {rv_ms(w.r_hit)}  projected {rv_ms(p.r_hit)}  "
                          f"union {rv_ms(w.r_hit_union)}  strong "
                          f"{rv_ms(w.r_hit_strong)}  | new vs union: within "
                          f"{int(w.n_new_vs_union.sum())}, projected "
                          f"{int(p.n_new_vs_union.sum())}; new vs strong: "
                          f"{int(w.n_new_vs_strong.sum())}, "
                          f"{int(p.n_new_vs_strong.sum())}  (models with any: "
                          f"{int((p.n_new_vs_strong > 0).sum())}; max "
                          f"{int(p.n_new_vs_strong.max())})")
    print("\n%% rvhsjc2: models on which the projected C2 HopSkipJump exceeds the "
          "four-attack union")
    p = d[d.attack == hsj + "_proj"]
    e = p[p.n_hit > p.n_hit_union]
    print(f"  {len(e)} of {len(p)} (model, budget, defence) cells; by dataset and "
          "defence: " + ", ".join(f"{k[0]}/{k[1]} {v}" for k, v in
                                  e.groupby(["dataset", "defence"]).size().items())
          + ("" if not len(e) else "\n" + e[["dataset", "activation", "seed",
                                             "defence", "eps", "n_hit",
                                             "n_hit_union", "n_hit_strong"]]
             .to_string(index=False)))
    print("\n%% rvhsjc2: required l_inf (median over the evaded flows of the C2 "
          "search), mean over the models")
    w = d[d.attack == hsj]
    for (ds, dfc), y in w.groupby(["dataset", "defence"]):
        y = y.drop_duplicates(["activation", "seed"])
        print(f"  {ds:4s} {dfc:9s} {rv_ms(y.required_linf.astype(float))}")
    cp = stage_read("rvhsjc2cap", RV5C_KEY)
    print("\n%% rvhsjc2: flows whose masked search did not finish within the "
          "per-flow cap and were scored as not evaded\n%% (lower bound); calls "
          "run with the batch driver before the cap existed have no record")
    if cp is None:
        print("  (no call was run with the per-flow driver)")
    else:
        for _, y in cp.iterrows():
            print(f"  {y.dataset:4s} {y.activation:4s} seed {y.seed:2d} "
                  f"{y.defence:9s} capped {int(y.n_capped)} of {int(y.n_flows)} "
                  f"flows at {int(y.cap_s)}s"
                  + (f": {y.capped_flows}" if int(y.n_capped) else ""))
    r = stage_read("rvhsjc2round", RV5R_KEY)
    print("\n%% rvhsjc2: frozen coordinates moved by float32 rounding in the "
          "library's initial binary search\n%% and restored before scoring "
          "(see the comment above rv5_restore_rounding); models run before "
          "this\n%% check was added had none (the run stopped on any move)")
    if r is None:
        print("  (no model needed a restoration)")
    else:
        for (ds, dfc), y in r.groupby(["dataset", "defence"]):
            print(f"  {ds:4s} {dfc:9s} models recorded {len(y):2d}, with a "
                  f"restoration {int((y.n_frozen_rounded > 0).sum()):2d}, "
                  f"coordinates restored {int(y.n_frozen_rounded.sum())} of "
                  f"{int((y.n_frozen * y.n_flows).sum()):,} frozen, largest move "
                  f"{float(y.max_abs_dev.max()):.2e}")


STAGES.append(("rvhsjc2", stage_rvhsjc2, summarize_rvhsjc2))


# ======== stage cxcomb: code ========
CX_COLS = ["dataset", "activation", "seed", "variant", "eps", "attack", "asr",
           "n_correct", "n_hit", "n_free", "n_int_free", "n_tcp_only_free",
           "n_nontcp_flows", "n_totals_moved", "n_diff_base", "attack_seed",
           "elapsed_s"]
CX_KEY = CX_COLS[:6]
CXF_COLS = ["dataset", "activation", "seed", "variant", "eps", "attack",
            "feature", "n_hit", "n_moved", "frac_moved", "median_abs_raw",
            "max_abs_raw"]
CXF_KEY = CXF_COLS[:7]
CX_VARIANTS = ["sym", "comb"]
CX_MOVE_TOL = 1e-6      # |raw displacement| above this counts as "moved"
CX_BOX_TOL = 1e-5       # a free coordinate outside its box by more stops the part
# fields that exist only in TCP.
CX_TCP_ONLY = {"unsw": ["swin", "stcpb", "trans_depth"],
               "cic": ["Init Fwd Win Bytes", "Fwd PSH Flags", "Fwd RST Flags",
                       "Fwd URG Flags"]}
CX_TCP_NAME = {"unsw": "tcp", "cic": "TCP"}     # raw protocol value of TCP
# sender-set count -> whole-flow count it is part of.
CX_FLAG_TOTALS = {"cic": {"Fwd PSH Flags": "PSH Flag Count",
                          "Fwd RST Flags": "RST Flag Count",
                          "Fwd URG Flags": "URG Flag Count"},
                  "unsw": {}}


def cx_groups(c, eps):
    """(name, flow indices, lo, hi, one-row mask) of the comb attack."""
    free = np.asarray(c.masks[2]).reshape(-1) > 0
    lo, hi = cd_box(c, eps, "addonly")
    proto = np.asarray(get_meta(c.ds)["proto_te"])[c.pool]
    is_tcp = np.char.lower(proto.astype(str)) == CX_TCP_NAME[c.ds].lower()
    tcp_cols = [c.names.index(n) for n in CX_TCP_ONLY.get(c.ds, [])
                if n in c.names and free[c.names.index(n)]]
    m_all = free.astype(np.float32)
    m_non = m_all.copy(); m_non[tcp_cols] = 0.0
    lo_n, hi_n = lo.copy(), hi.copy()
    lo_n[tcp_cols] = 0.0; hi_n[tcp_cols] = 0.0
    out = []
    if is_tcp.any():
        out.append(("tcp", np.where(is_tcp)[0], lo, hi, m_all))
    if (~is_tcp).any():
        out.append(("nontcp", np.where(~is_tcp)[0], lo_n, hi_n, m_non))
    return out, tcp_cols, int((~is_tcp).sum())


def cx_attack(c, a, eps, lo, hi, mask, s, idx):
    """cd_attack on a subset of the pool (one group of cx_groups)."""
    X, y = c.Xe[idx], c.ye[idx]
    sub = c.substitute() if a == "transfer" else None    # loaded BEFORE seeding
    fn = {"pgd": lambda: cd_grad(c.clf, X, eps, lo, hi, mask, "pgd"),
          "bim": lambda: cd_grad(c.clf, X, eps, lo, hi, mask, "bim"),
          "spsa": lambda: cd_spsa(c.model, X, y, eps, lo, hi, s, mask),
          "transfer": lambda: cd_grad(sub, X, eps, lo, hi, mask, "pgd")}[a]
    _reseed(s)
    adv = _timed(fn)
    if not np.isfinite(adv).all():
        _stop(f"{a} returned non-finite values ({c.part} eps={eps}, cxcomb); "
              f"nothing was written for this cell.")
    return np.asarray(adv, np.float32)


def cx_flag_totals(c, adv, sc):
    pairs = CX_FLAG_TOTALS.get(c.ds, {})
    free = np.asarray(c.masks[2]).reshape(-1) > 0
    out = adv.astype(np.float64).copy()
    cols = []
    for fwd, tot in pairs.items():
        if fwd not in c.names or tot not in c.names:
            continue
        jf, jt = c.names.index(fwd), c.names.index(tot)
        if not free[jf]:
            continue
        gf, gt = int(c.cols[jf]), int(c.cols[jt])
        d_raw = (out[:, jf] - c.Xe[:, jf].astype(np.float64)) * float(sc["scale"][gf])
        out[:, jt] = c.Xe[:, jt].astype(np.float64) + d_raw / float(sc["scale"][gt])
        cols.append(jt)
    return out.astype(np.float32), cols


def cx_score(c, adv, lo, hi, extra_cols, what):
    """Per-flow success with the checks the base scoring does."""
    free = np.asarray(c.masks[2]).reshape(-1) > 0
    d = adv.astype(np.float64) - c.Xe.astype(np.float64)
    frozen = ~free
    frozen[list(extra_cols)] = False
    viol = int((np.abs(d[:, frozen]) > 1e-6).sum())
    out_box = int(((d[:, free] < lo[free] - CX_BOX_TOL)
                   | (d[:, free] > hi[free] + CX_BOX_TOL)).sum())
    if viol or out_box:
        _stop(f"capability violation in cxcomb ({c.part} {what}): {viol} frozen "
              f"coordinates moved, {out_box} free coordinates outside the box.")
    return c.correct & (_probs(c.model, adv, DEVICE).argmax(1) != c.ye)


def cx_feature_rows(c, adv, h, scale_k, v, eps, a):
    if not h.any():
        return []
    d = (adv.astype(np.float64) - c.Xe.astype(np.float64))[h]
    raw = np.abs(d) * np.asarray(scale_k, np.float64).reshape(1, -1)
    out, nh = [], int(h.sum())
    for j, nm in enumerate(c.names):
        moved = raw[:, j] > CX_MOVE_TOL
        if not moved.any():
            continue
        out.append(dict(dataset=c.ds, activation=c.act, seed=c.seed, variant=v,
                        eps=round(eps, 3), attack=a, feature=nm, n_hit=nh,
                        n_moved=int(moved.sum()),
                        frac_moved=round(float(moved.mean()), 4),
                        median_abs_raw=float(np.median(raw[moved, j])),
                        max_abs_raw=float(raw[moved, j].max())))
    return out


def cx_stage(c):
    sc = ca_scaler(c.ds, c.data)
    scale_k = np.asarray(sc["scale"])[c.cols]
    int_k = np.asarray(sc["is_int_all"])[c.cols]
    free = np.asarray(c.masks[2]).reshape(-1) > 0
    n_free = int(free.sum())
    q_idx = [i for i in range(len(c.names)) if free[i] and int_k[i]]
    have, havef = stage_have("cxcomb", CX_KEY), stage_have("cxfeat", CXF_KEY)
    base = ca_base_flows(c)
    for eps in EPS_LIST:
        for v in CX_VARIANTS:
            if _k(c.ds, c.act, c.seed, v, round(eps, 3), "union") in have:
                continue
            if v == "sym":
                lo, hi = cd_box(c, eps, "sym")
                groups, tcp_cols, n_non = None, [], 0
            else:
                groups, tcp_cols, n_non = cx_groups(c, eps)
                lo, hi = cd_box(c, eps, "addonly")
            union = np.zeros(len(c.Xe), bool)
            union_adv = c.Xe.astype(np.float32).copy()
            n_tot = 0
            for a in CA_ATTACKS:
                t0 = time.time()
                if v == "sym":
                    s = _seed_of(c.ds, DEPTH, FEATS, c.act, c.seed, 2, eps, a)
                    raw_adv = cd_attack(c, a, eps, lo, hi,
                                        free.astype(np.float32), s)
                    d = np.clip(raw_adv.astype(np.float64)
                                - c.Xe.astype(np.float64), lo, hi)
                    adv = (c.Xe.astype(np.float64) + d).astype(np.float32)
                    extra = []
                else:
                    s = _seed_of("cxcomb", "comb", c.ds, c.act, c.seed, eps, a)
                    adv = c.Xe.astype(np.float32).copy()
                    for gname, idx, glo, ghi, gm in groups:
                        if gm.sum() == 0:
                            continue          # nothing this group may move
                        sg = _seed_of("cxcomb", "comb", c.ds, c.act, c.seed,
                                      eps, a, gname)
                        g_adv = cx_attack(c, a, eps, glo, ghi, gm, sg, idx)
                        d = np.clip(g_adv.astype(np.float64)
                                    - c.Xe[idx].astype(np.float64), glo, ghi)
                        adv[idx] = (c.Xe[idx].astype(np.float64) + d).astype(
                            np.float32)
                    adv = ca_quantise(c.Xe, adv, scale_k, q_idx)
                    adv, extra = cx_flag_totals(c, adv, sc)
                    n_tot = max(n_tot, int((np.abs(adv[:, extra]
                                                   - c.Xe[:, extra]) > 1e-6).any(1).sum())
                                if extra else 0)
                h = cx_score(c, adv, lo, hi, extra, f"{v} eps={eps} {a}")
                nd = (int((h != np.asarray(base[f"L2_e{eps}_{a}"], bool)).sum())
                      if v == "sym" and f"L2_e{eps}_{a}" in base else -1)
                new = h & ~union
                union_adv[new] = adv[new]
                union |= h
                stage_append("cxcomb", CX_COLS, dict(
                    dataset=c.ds, activation=c.act, seed=c.seed, variant=v,
                    eps=round(eps, 3), attack=a,
                    asr=round(float(h.sum() / c.n_ok), 4), n_correct=c.n_ok,
                    n_hit=int(h.sum()), n_free=n_free,
                    n_int_free=len(q_idx) if v == "comb" else 0,
                    n_tcp_only_free=len(tcp_cols), n_nontcp_flows=n_non,
                    n_totals_moved=n_tot if v == "comb" else 0,
                    n_diff_base=nd, attack_seed=s,
                    elapsed_s=round(time.time() - t0, 1)))
                for row in cx_feature_rows(c, adv, h, scale_k, v, eps, a):
                    fk = _k(c.ds, c.act, c.seed, v, round(eps, 3), a,
                            row["feature"])
                    if fk not in havef:
                        stage_append("cxfeat", CXF_COLS, row)
                        havef.add(fk)
            for row in cx_feature_rows(c, union_adv, union, scale_k, v, eps,
                                       "union"):
                fk = _k(c.ds, c.act, c.seed, v, round(eps, 3), "union",
                        row["feature"])
                if fk not in havef:
                    stage_append("cxfeat", CXF_COLS, row)
                    havef.add(fk)
            nd_u = (int((union != np.asarray(base[f"L2_e{eps}_union"], bool)).sum())
                    if v == "sym" and f"L2_e{eps}_union" in base else -1)
            stage_append("cxcomb", CX_COLS, dict(
                dataset=c.ds, activation=c.act, seed=c.seed, variant=v,
                eps=round(eps, 3), attack="union",
                asr=round(float(union.sum() / c.n_ok), 4), n_correct=c.n_ok,
                n_hit=int(union.sum()), n_free=n_free,
                n_int_free=len(q_idx) if v == "comb" else 0,
                n_tcp_only_free=len(tcp_cols), n_nontcp_flows=n_non,
                n_totals_moved=n_tot if v == "comb" else 0,
                n_diff_base=nd_u, attack_seed=0, elapsed_s=0))
            have.add(_k(c.ds, c.act, c.seed, v, round(eps, 3), "union"))
            print(f"    cxcomb {v} eps={eps}: union {union.sum() / c.n_ok:.4f}"
                  + (f" (differs from base L2 on {nd_u} flows)"
                     if v == "sym" else ""), flush=True)


def cx_summarize():
    for ds in DATASETS:
        pm = os.path.join(DATA_DIR, f"{ds}_meta.npz")
        if os.path.exists(pm):
            m = np.load(pm, allow_pickle=False)
            if "flagid_names" in m.files and len(m["flagid_names"]):
                print(f"%% {ds}: flag-count identity in the raw data "
                      f"(share of rows): " + ", ".join(
                          f"{n}: total = fwd + bwd {e:.4f}, total >= fwd {g:.4f}"
                          for n, e, g in zip(m["flagid_names"], m["flagid_eq"],
                                             m["flagid_ge"])))
    r, f = stage_read("cxcomb", CX_KEY), stage_read("cxfeat", CXF_KEY)
    if r is None:
        print("%% stage cxcomb: not run yet")
        return
    if not os.path.exists(OUT_FINAL):
        print(f"%% stage cxcomb: {OUT_FINAL} is missing (the base summary "
              f"writes it)")
        return
    b = pd.read_csv(OUT_FINAL)
    c0 = b[(b.depth == DEPTH) & (b.features == FEATS) & (b.level == 0)
           & (b.attack == "union")]
    c2 = b[(b.depth == DEPTH) & (b.features == FEATS) & (b.level == 2)
           & (b.attack == "union")]
    key = ["dataset", "activation", "seed"]
    print("%% cxcomb: worst case (union of PGD, BIM, SPSA, transfer) as a "
          "percentage of the unconstrained (C0) worst case of the same "
          "network, mean over networks; comb is a LOWER bound (units and "
          "flag totals act on the output)")
    for ds in DATASETS:
        for eps in EPS_LIST:
            e = round(eps, 3)
            den = c0[(c0.dataset == ds) & (c0.eps.round(3) == e)][key + ["asr"]]
            row = []
            base2 = c2[(c2.dataset == ds) & (c2.eps.round(3) == e)][key + ["asr"]]
            mm = den.merge(base2, on=key, suffixes=("_c0", "_x"))
            if len(mm):
                row.append(f"C2 {100 * (mm.asr_x / mm.asr_c0).mean():5.1f}%")
            for v in CX_VARIANTS:
                x = r[(r.dataset == ds) & (r.eps.round(3) == e)
                      & (r.attack == "union") & (r.variant == v)][key + ["asr"]]
                mm = den.merge(x, on=key, suffixes=("_c0", "_x"))
                if len(mm):
                    row.append(f"{v} {100 * (mm.asr_x / mm.asr_c0).mean():5.1f}%"
                               f" (n={len(mm)})")
            if row:
                print(f"  {ds:4s} eps={eps}: " + " | ".join(row))
    s = r[(r.variant == "sym") & (r.n_diff_base >= 0)]
    if len(s):
        print(f"%% reproduction check (sym against the base L2 rows): "
              f"{int((s.n_diff_base > 0).sum())} of {len(s)} cells differ, "
              f"{int(s.n_diff_base.sum())} flows in all")
    cmb = r[(r.variant == "comb") & (r.attack == "union")]
    if len(cmb):
        print(f"%% comb: TCP-only free fields {sorted(cmb.n_tcp_only_free.unique().tolist())}"
              f", non-TCP pool flows {sorted(cmb.n_nontcp_flows.unique().tolist())}"
              f", flows whose flag totals were recomputed (max per network) "
              f"{int(cmb.n_totals_moved.max())}")
    if f is None:
        return
    print("\n%% cxfeat (per-feature breakdown): of the flows the worst case "
          "evades, the share whose perturbation moves each feature, mean over "
          "networks; median |change| in raw units")
    g = f[f.attack == "union"]
    for v in CX_VARIANTS:
        for ds in DATASETS:
            for eps in EPS_LIST:
                x = g[(g.variant == v) & (g.dataset == ds)
                      & (g.eps.round(3) == round(eps, 3))]
                if not len(x):
                    continue
                print(f"  {v} {ds:4s} eps={eps}: " + "; ".join(
                    f"{nm} {grp.frac_moved.mean():.0%} "
                    f"({grp.median_abs_raw.median():.3g})"
                    for nm, grp in x.groupby("feature")))


STAGES.append(("cxcomb", cx_stage, cx_summarize))


# ======== stage fam: code ========
FAM_COLS = ["dataset", "activation", "seed", "level", "eps", "family",
            "n_pool", "n_correct", "n_hit"]
FAM_KEY = FAM_COLS[:6]
FAMP_COLS = ["dataset", "protocol", "n_pool"]
FAMP_KEY = FAMP_COLS[:2]
FAM_LEVELS = [0, 2]


_NA_TOKENS = {"", "#n/a", "#n/a n/a", "#na", "-1.#ind", "-1.#qnan", "-nan",
              "1.#ind", "1.#qnan", "<na>", "n/a", "na", "nan", "null", "none"}


def _fam_safe(x):
    """A family or protocol name as written to the stage CSV."""
    x = str(x).strip()
    if x == "Backdoors":
        x = "Backdoor"
    return f"[{x}]" if x.lower() in _NA_TOKENS else x


def stage_fam(c):
    npz = os.path.join(RES_DIR, "flows", f"{c.ds}_d4_f30_{c.act}_s{c.seed}.npz")
    if not os.path.exists(npz):
        _stop(f"stage fam: {npz} not found; the base attacks of {c.part} "
              f"must have run.")
    z = np.load(npz)
    if not np.array_equal(z["pool"], c.pool):
        _stop(f"stage fam: the pool saved in {npz} is not make_pool's.")
    meta = get_meta(c.ds)
    fam = np.array([_fam_safe(x) for x in np.asarray(meta["fam_te"])[c.pool]])
    proto = np.array([_fam_safe(x) for x in np.asarray(meta["proto_te"])[c.pool]])
    correct = z["correct"].astype(bool)
    have = stage_have("fam", FAM_KEY)
    for lv in FAM_LEVELS:
        for eps in EPS_LIST:
            k = f"L{lv}_e{eps}_union"
            if k not in z.files:
                _stop(f"stage fam: {k} not in {npz}.")
            u = z[k].astype(bool)
            for f in sorted(set(fam.tolist())):
                if _k(c.ds, c.act, c.seed, lv, round(eps, 3), f) in have:
                    continue
                m = fam == f
                stage_append("fam", FAM_COLS, dict(
                    dataset=c.ds, activation=c.act, seed=c.seed, level=lv,
                    eps=round(eps, 3), family=f, n_pool=int(m.sum()),
                    n_correct=int((m & correct).sum()),
                    n_hit=int((m & u).sum())))
    havep = stage_have("famproto", FAMP_KEY)
    for pr in sorted(set(proto.tolist())):
        if _k(c.ds, pr) not in havep:
            stage_append("famproto", FAMP_COLS, dict(
                dataset=c.ds, protocol=pr, n_pool=int((proto == pr).sum())))


def summarize_fam():
    for ds in DATASETS:
        pm = os.path.join(DATA_DIR, f"{ds}_meta.npz")
        if os.path.exists(pm):
            m = np.load(pm, allow_pickle=False)
            print(f"%% {ds}: deduplication {int(m['n_before']):,} -> "
                  f"{int(m['n_after']):,} rows ({int(m['n_dup_removed']):,} "
                  f"exact duplicates; {int(m['n_conflict_rows']):,} rows of "
                  f"{int(m['n_conflict_vectors']):,} vectors with both labels); "
                  f"malicious {int(m['n_mal_before']):,} -> "
                  f"{int(m['n_mal_after']):,}; vectors with more than one "
                  f"family {int(m['n_family_ambiguous']):,}; test "
                  f"{int(m['n_test']):,} ({int(m['n_test_mal']):,} malicious), "
                  f"undersampled training {int(m['n_train']):,}; test flows "
                  f"identical to a training flow: {int(m['overlap_all']):,} "
                  f"(all features), {int(m['overlap_top30']):,} (top-{FEATS})")
    d = stage_read("fam", FAM_KEY)
    p = stage_read("famproto", FAMP_KEY)
    if d is None:
        print("%% stage fam: not run yet")
        return
    if p is not None:
        for ds in DATASETS:
            q = p[p.dataset == ds].sort_values("n_pool", ascending=False)
            if len(q):
                print(f"%% {ds} pool by protocol: " + ", ".join(
                    f"{r.protocol} {int(r.n_pool)}" for r in q.itertuples()))
    sd = lambda v: float(np.std(v, ddof=SD_DDOF)) if len(v) > 1 else 0.0
    print("%% worst case by attack family: mean (SD) over seeds of the "
          "activation-averaged rate n_hit / n_correct; C2/C0 from the means")
    for ds in DATASETS:
        q = d[d.dataset == ds]
        if q.empty:
            continue
        comp = (q[(q.level == 0) & (q.eps.round(3) == round(EPS_LIST[0], 3))]
                .groupby("family").n_pool.max().sort_values(ascending=False))
        print(f"%% {ds} pool by family: " + ", ".join(
            f"{f} {int(n)}" for f, n in comp.items()))
        for f in comp.index:
            row = []
            for eps in EPS_LIST:
                r = {}
                for lv in FAM_LEVELS:
                    g = q[(q.family == f) & (q.level == lv)
                          & (q.eps.round(3) == round(eps, 3)) & (q.n_correct > 0)]
                    per = (g.assign(rate=g.n_hit / g.n_correct)
                           .groupby("seed").rate.mean())
                    r[lv] = (per.mean(), sd(per)) if len(per) else (np.nan, 0.0)
                ratio = (r[2][0] / r[0][0]) if r[0][0] else np.nan
                row.append(f"eps={eps}: C0 {r[0][0]:.3f}({r[0][1]:.3f}) "
                           f"C2 {r[2][0]:.3f}({r[2][1]:.3f}) C2/C0 {ratio:.0%}")
            print(f"  {ds:4s} {f:28s} n={int(comp[f]):4d} | " + " | ".join(row))


STAGES.append(("fam", stage_fam, summarize_fam))


# ---- STAGE cxenum: C2 lattice on CSE-CIC-IDS2018 ----
CXE_GRID = 5
CXE_ROWS = 200_000          # (flow, point) rows per forward batch (speed only)
CXE_COLS = ["dataset", "activation", "seed", "level", "eps", "n_free", "grid",
            "n_points", "n_correct", "n_hit_lattice", "n_hit_corner",
            "n_hit_union", "n_union_not_lattice", "n_lattice_not_union",
            "asr_lattice", "asr_union", "elapsed_s"]
CXE_KEY = CXE_COLS[:5]


def cxe_offsets(eps, k, grid):
    import itertools
    axes = np.meshgrid(*([np.linspace(-eps, eps, grid)] * k), indexing="ij")
    lat = np.stack([a.reshape(-1) for a in axes], 1).astype(np.float32)
    cor = np.array(list(itertools.product((-eps, eps), repeat=k)), np.float32)
    return lat, cor


def cxe_enumerate(c, eps, free, grid=CXE_GRID):
    """Per-flow sets over the correctly classified flows."""
    lat, cor = cxe_offsets(eps, len(free), grid)
    offs = np.concatenate([lat, cor], 0)
    lo, hi = float(c.Xtr.min()), float(c.Xtr.max())          # as _art_clf
    n, d = c.Xe.shape
    hit, hitc = np.zeros(n, bool), np.zeros(n, bool)
    idx = np.where(c.correct)[0]
    step = max(1, CXE_ROWS // len(offs))
    for s in range(0, len(idx), step):
        ii = idx[s:s + step]
        pts = np.repeat(c.Xe[ii][:, None, :], len(offs), 1).astype(np.float32)
        pts[:, :, free] += offs[None]
        np.clip(pts, lo, hi, out=pts)
        w = (_probs(c.model, pts.reshape(-1, d), DEVICE).argmax(1)
             .reshape(len(ii), len(offs)) != c.ye[ii][:, None])
        hit[ii] = w.any(1)
        hitc[ii] = w[:, len(lat):].any(1)
    return hit, hitc, len(offs)


def stage_cxenum(c):
    if c.ds != "cic":
        return                      # UNSW-NB15 is enumerated by stage capa
    free = np.where(np.asarray(c.masks[2]) > 0)[0]
    have = stage_have("cxenum", CXE_KEY)
    base = ca_base_flows(c)
    if not np.array_equal(base.get("pool"), c.pool):
        _stop(f"cxenum {c.part}: the saved base flows use a different pool.")
    npz = stage_npz("cxenum", c)
    saved = npz_load(npz)
    for eps in EPS_LIST:
        e = round(eps, 3)
        if _k(c.ds, c.act, c.seed, 2, e) in have:
            continue
        ukey = f"L2_e{eps}_union"
        if ukey not in base:
            _stop(f"cxenum {c.part}: {ukey} is not in the saved base flows.")
        union = base[ukey].astype(bool)
        t0 = time.time()
        hit, hitc, n_pts = cxe_enumerate(c, eps, free)
        saved[f"lattice_e{eps}"], saved[f"corner_e{eps}"] = hit, hitc
        npz_save(npz, saved)
        stage_append("cxenum", CXE_COLS, dict(
            dataset=c.ds, activation=c.act, seed=c.seed, level=2, eps=e,
            n_free=len(free), grid=CXE_GRID, n_points=n_pts, n_correct=c.n_ok,
            n_hit_lattice=int(hit.sum()), n_hit_corner=int(hitc.sum()),
            n_hit_union=int(union.sum()),
            n_union_not_lattice=int((union & ~hit).sum()),
            n_lattice_not_union=int((hit & ~union).sum()),
            asr_lattice=round(float(hit.sum() / c.n_ok), 4),
            asr_union=round(float(union.sum() / c.n_ok), 4),
            elapsed_s=round(time.time() - t0, 1)))
        have.add(_k(c.ds, c.act, c.seed, 2, e))
        print(f"    C2 lattice eps={eps}: {int(hit.sum())} of {c.n_ok} flows "
              f"({n_pts} points); corners {int(hitc.sum())}; attack union "
              f"{int(union.sum())}", flush=True)


def summarize_cxenum():
    d = stage_read("cxenum", CXE_KEY)
    if d is None:
        print("%% cxenum: not run yet"); return
    print(f"%% cxenum: CSE-CIC-IDS2018 C2, {CXE_GRID}-point lattice per free axis "
          f"plus corners, clipped to the training range; mean+-SD over seeds of "
          f"the activation-averaged rate; flows pooled over the models")
    for e, g in d.groupby("eps"):
        lat = g.groupby("seed").asr_lattice.mean(); uni = g.groupby("seed").asr_union.mean()
        print(f"  eps={e}: lattice {lat.mean():.4f}+-{lat.std(ddof=1):.4f} | "
              f"union {uni.mean():.4f}+-{uni.std(ddof=1):.4f} | lattice - union "
              f"{lat.mean() - uni.mean():+.4f} | flows: lattice {int(g.n_hit_lattice.sum())}, "
              f"union {int(g.n_hit_union.sum())}, union not lattice "
              f"{int(g.n_union_not_lattice.sum())}, lattice not union "
              f"{int(g.n_lattice_not_union.sum())} of {int(g.n_correct.sum())} "
              f"({len(g)} models)")
        for a, h in g.groupby("activation"):
            print(f"      {a:4s}: lattice {h.asr_lattice.mean():.4f}  union "
                  f"{h.asr_union.mean():.4f}  corners {(h.n_hit_corner / h.n_correct).mean():.4f}")


STAGES.append(("cxenum", stage_cxenum, summarize_cxenum))


# ======== stage atfgsm: code ========
ATFG_COLS = ["dataset", "activation", "seed", "defence", "eps", "attack",
             "asr", "n_correct", "n_hit", "n_hit_pgd", "n_hit_bim",
             "n_not_pgd_bim", "attack_seed", "elapsed_s"]
ATFG_KEY = ATFG_COLS[:6]


def atfg_stage(c):
    dfc = AT_ROLE
    have = stage_have("atfgsm", ATFG_KEY)
    key = lambda eps: _k(c.ds, c.act, c.seed, dfc, round(eps, 3), "fgsm")
    if all(key(e) in have for e in EPS_LIST):
        print(f"    skip atfgsm {c.part}: done", flush=True)
        return
    model, clf, correct, n_ok = rv_model(c, dfc)
    base = npz_load(rv4_base_npz(c, dfc))
    if not base:
        _stop(f"atfgsm {c.part}: {rv4_base_npz(c, dfc)} is missing; the "
              f"advtrain stage has not saved its per-flow results.")
    if not (np.array_equal(base.get("pool"), c.pool)
            and np.array_equal(base.get("correct"), correct)):
        _stop(f"atfgsm {c.part}: the pool or the correctly classified flows "
              f"differ from those the advtrain stage saved; nothing was written.")
    npz = stage_npz("atfgsm", c)
    saved = npz_load(npz)
    msg = []
    for eps in EPS_LIST:
        if key(eps) in have:
            continue
        s = _seed_of("atfgsm", c.ds, c.act, c.seed, dfc, eps)
        _reseed(s)
        t0 = time.time()
        adv = _timed(lambda: fg_attack(clf, c.Xe, eps))
        if not np.isfinite(adv).all():
            _stop(f"FGSM returned non-finite values (atfgsm {c.part} "
                  f"eps={eps}); nothing was written for this cell.")
        _, h, _ = hits_of(model, adv, c.Xe, c.ye, DEVICE, eps, None)
        hp, hb = base[f"L0_e{eps}_pgd"], base[f"L0_e{eps}_bim"]
        saved[f"{dfc}_L0_e{eps}_fgsm"] = h
        saved["pool"] = c.pool
        npz_save(npz, saved)
        stage_append("atfgsm", ATFG_COLS, dict(
            dataset=c.ds, activation=c.act, seed=c.seed, defence=dfc,
            eps=round(eps, 3), attack="fgsm",
            asr=round(float(h.sum() / n_ok), 4), n_correct=n_ok,
            n_hit=int(h.sum()), n_hit_pgd=int(hp.sum()), n_hit_bim=int(hb.sum()),
            n_not_pgd_bim=int((h & ~(hp | hb)).sum()), attack_seed=s,
            elapsed_s=round(time.time() - t0, 1)))
        have.add(key(eps))
        msg.append(f"eps={eps} FGSM {h.sum() / n_ok:.3f} (PGD {hp.sum() / n_ok:.3f},"
                   f" BIM {hb.sum() / n_ok:.3f})")
    print(f"    atfgsm {dfc} {c.part}: " + "  ".join(msg), flush=True)


def atfg_summarize():
    d = stage_read("atfgsm", ATFG_KEY)
    if d is None:
        print("%% stage atfgsm: not run yet")
        return
    d = d.copy()
    for a in ("", "_pgd", "_bim"):
        d["r" + a] = d["n_hit" + a] / d.n_correct
    d["mean3"] = (d.r + d.r_pgd + d.r_bim) / 3
    ms = lambda v: f"{v.mean():.3f}$\\pm${v.std(ddof=1):.3f}"
    print(f"%% atfgsm: FGSM at L0 on the saved PGD-AT Model 4 networks (30 features); "
          f"{len(d)} measurements. PGD and BIM from the advtrain per-flow sets. "
          f"Mean+-SD over seeds; 'all' = activation-averaged per seed.")
    print("%% columns: FGSM | BIM | PGD | mean of the three | flows FGSM evades that "
          "neither PGD nor BIM does")
    for ds in DATASETS:
        for ac in ACTIVATIONS + ["all"]:
            x = d[d.dataset == ds] if ac == "all" else d[(d.dataset == ds) & (d.activation == ac)]
            for eps, g0 in x.groupby("eps"):
                g = g0.groupby("seed")[["r", "r_bim", "r_pgd", "mean3"]].mean()
                print(f"  {ds:4s} {ac:4s} eps={eps}: FGSM {ms(g.r)} | BIM {ms(g.r_bim)} | "
                      f"PGD {ms(g.r_pgd)} | mean {ms(g.mean3)} | FGSM-only flows "
                      f"{int(g0.n_not_pgd_bim.sum())} ({len(g0)} networks)", flush=True)


STAGES.append(("atfgsm", atfg_stage, atfg_summarize))


# ---- later-stage code goes above this line ----


def _stages_selected():
    want = [x.strip() for x in RERUN_STAGES.split(",") if x.strip()]
    unknown = [x for x in want if x not in [n for n, _, _ in STAGES]]
    if unknown:
        _stop(f"RERUN_STAGES names unknown stage(s) {unknown}; known: "
              f"{[n for n, _, _ in STAGES]}")
    return [s for s in STAGES if not want or s[0] in want]


def run_stages(ds, act, seed, data):
    sel = _stages_selected()
    if not sel:
        return
    c = part_ctx(ds, act, seed, data)
    for name, run_fn, _ in sel:
        t0 = time.time()
        print(f"  == stage {name} ({c.part})", flush=True)
        run_fn(c)
        print(f"  == stage {name} done [{time.time() - t0:.0f}s]", flush=True)


def summarize_stages():
    for name, _, summ in _stages_selected():
        print(f"\n%% ======================= stage: {name}")
        summ()


def run_parts():
    print(">>> rerun_all starting", flush=True)
    print("=" * 74, flush=True)
    print(json.dumps(env_report()), flush=True)
    print(f"threads per process: {torch.get_num_threads()}   part: "
          f"{RERUN_PART or 'all'}   -> {OUT}", flush=True)
    print("=" * 74, flush=True)
    if RERUN_PART == "data":
        fails = []
        for ds in DATASETS:                 # build every dataset first ...
            out = get_data(ds, check=False)
        for ds in DATASETS:                 # ... then check every one
            out = get_data(ds, check=False)
            try:
                preflight(ds, out[5], out[4])
            except RuntimeError as e:
                fails.append(str(e))
        if fails:
            _stop("\n\n".join(fails))
        print("\n>>> data frozen in data_final/. Now start the 18 parts.")
        return
    have = set()
    for f in _files_to_consult("attacks"):
        have |= done_keys(f)
        print(f"  {len(done_keys(f))} rows already in {f}", flush=True)
    print("", flush=True)
    data = {}
    for ds, act, seed in PARTS:
        if ds not in data:
            data = {ds: get_data(ds)}
        run_part(ds, act, seed, have, data[ds])
        run_stages(ds, act, seed, data[ds])
    print("\n>>> part done", flush=True)


def _ms(v):
    v = np.asarray(v, float)
    return f"{v.mean():.3f}$\\pm${v.std(ddof=SD_DDOF):.3f}"


def summarize():
    files = sorted(f for f in glob.glob(os.path.join(RES_DIR, "attacks_*.csv"))
                   if f != OUT_FINAL)
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # A cell interrupted part-way is recomputed on resume, so its finished attack rows can appear twice.
    dup = d[d.duplicated(_KEY, keep=False)]
    if len(dup):
        bad = dup.groupby(_KEY).asr.nunique()
        if (bad > 1).any():
            _stop(f"{int((bad > 1).sum())} measurements were repeated with "
                  f"DIFFERENT values:\n{bad[bad > 1].head(20)}")
        print(f"  {len(dup)} repeated rows from resumed cells, all identical; "
              f"one kept per measurement")
        d = d.drop_duplicates(_KEY, keep="last").reset_index(drop=True)
    d.to_csv(OUT_FINAL, index=False)
    n_models = 2 * 3 * 3 * 5 + 3 * (len(K_SWEEP["cic"]) + len(K_SWEEP["unsw"]))
    print(f"combined {len(files)} file(s) -> {OUT_FINAL}: {len(d)} rows, "
          f"{d.groupby(['dataset', 'depth', 'features', 'activation', 'seed']).ngroups}"
          f" of {n_models} models")
    acts = ["relu", "elu", "tanh"]
    lab = (("cic", "CSE-CIC-IDS2018"), ("unsw", "UNSW-NB15"))
    A4 = ["pgd", "bim", "spsa", "hsj_inf"]          # the four primary attacks

    def q(**kw):
        m = np.ones(len(d), bool)
        for c, v in kw.items():
            m &= np.isclose(d[c], v) if c == "eps" else (d[c] == v).values
        return d[m]

    def mean4(ds, depth, k, act, seed, eps):
        r = q(dataset=ds, depth=depth, features=k, activation=act, seed=seed,
              level=0, eps=eps)
        v = r[r.attack.isin(A4)]
        if len(v) != 4:
            _stop(f"missing attacks for {ds} d{depth} k{k} {act} s{seed} eps={eps}")
        return v.asr.mean()

    print("\n%% ---- tab:partd_f1 (clean macro-F1, 30 features)")
    for ds, L in lab:
        print(f"\\multicolumn{{6}}{{l}}{{\\textit{{{L}}}}} \\\\")
        for a, A in (("elu", "ELU "), ("relu", "ReLU"), ("tanh", "Tanh")):
            cells = []
            for dp in DEPTHS:
                r = q(dataset=ds, depth=dp, features=FEATS, activation=a, level=0,
                      eps=0.3, attack="pgd")
                cells.append(_ms(r.sort_values("seed").clean_f1))
            print(f"{A} & " + " & ".join(cells) + " \\\\")
        print("\\hline")

    print("\n%% ---- tab:both_depth_eps (ReLU, mean of PGD, BIM, SPSA, HopSkipJump)")
    for ds, L in lab:
        print(f"\\multicolumn{{4}}{{l}}{{\\textit{{{L}}}}} \\\\")
        for dp in DEPTHS:
            print(f"Model {dp} & " + " & ".join(
                _ms([mean4(ds, dp, FEATS, "relu", s, e) for s in SEEDS])
                for e in EPS_LIST) + " \\\\")
        print("\\hline")

    print("\n%% ---- tab:partd_eps (mean of four attacks, averaged over depths)")
    for ds, L in lab:
        print(f"\\multicolumn{{4}}{{l}}{{\\textit{{{L}}}}} \\\\")
        for a, A in (("elu", "ELU "), ("relu", "ReLU"), ("tanh", "Tanh")):
            print(f"{A} & " + " & ".join(
                _ms([np.mean([mean4(ds, dp, FEATS, a, s, e) for dp in DEPTHS])
                     for s in SEEDS]) for e in EPS_LIST) + " \\\\")
        print("\\hline")

    print("\n%% ---- tab:features_both (Model 1, ReLU; last column = mean over eps)")
    for ds, L in lab:
        print(f"\\multicolumn{{5}}{{l}}{{\\textit{{{L}}}}} \\\\")
        for k in sorted(set(K_SWEEP[ds]) | {FEATS}):
            per = {e: [mean4(ds, 1, k, "relu", s, e) for s in SEEDS]
                   for e in EPS_LIST}
            allm = [np.mean([per[e][i] for e in EPS_LIST])
                    for i in range(len(SEEDS))]
            print(f"{k} & " + " & ".join(_ms(per[e]) for e in EPS_LIST)
                  + f" & {_ms(allm)} \\\\")
        print("\\hline")

    print("\n%% ---- tab:depth_act (bold = lowest, separated from the next by "
          "more than both SDs)")
    for e in EPS_LIST:
        print(f"\\multicolumn{{7}}{{l}}{{\\textit{{$\\epsilon = {e}$}}}} \\\\")
        for dp in DEPTHS:
            cells = []
            for ds in ("cic", "unsw"):
                v = {a: np.array([mean4(ds, dp, FEATS, a, s, e) for s in SEEDS])
                     for a in acts}
                ms = {a: (v[a].mean(), v[a].std(ddof=SD_DDOF)) for a in acts}
                o = sorted(acts, key=lambda a: ms[a][0])
                gap = ms[o[1]][0] - ms[o[0]][0]
                sep = gap > ms[o[0]][1] and gap > ms[o[1]][1]
                cells += [(f"\\textbf{{{_ms(v[a])}}}" if sep and a == o[0]
                           else _ms(v[a])) for a in acts]
            print(f"Model {dp} & " + " & ".join(cells) + " \\\\")

    m4 = dict(depth=4, features=FEATS)
    print("\n%% ---- tab:per_attack (Table 4): Model 4, L0, eps = 0.3")
    for ds, L in lab:
        print(f"\\multicolumn{{5}}{{l}}{{\\textit{{{L}}}}} \\\\")
        for a, A in (("relu", "ReLU"), ("elu", "ELU "), ("tanh", "Tanh")):
            print(f"{A} & " + " & ".join(
                _ms(q(dataset=ds, activation=a, level=0, eps=0.3, attack=x,
                      **m4).sort_values("seed").asr)
                for x in ("pgd", "bim", "hsj_inf", "spsa")) + " \\\\")
        print("\\hline")

    print("\n%% ---- tab:hsj_corrected (Table 5): Model 4, L0, eps = 0.3; means "
          "over seeds. PGD | l_inf within | l_inf projected | l_2 within | "
          "l_2 projected | required l_inf (l_inf search)")
    for ds, L in lab:
        print(f"\\multicolumn{{7}}{{l}}{{\\textit{{{L}}}}} \\\\")
        for a, A in (("relu", "ReLU"), ("elu", "ELU "), ("tanh", "Tanh")):
            g = lambda x: q(dataset=ds, activation=a, level=0, eps=0.3,
                            attack=x, **m4)
            vals = [g(x).asr.mean() for x in
                    ("pgd", "hsj_inf", "hsj_inf_proj", "hsj_l2", "hsj_l2_proj")]
            vals.append(g("hsj_inf").required_linf.mean())
            print(f"{A} & " + " & ".join(f"{v:.3f}" for v in vals) + " \\\\")
        print("\\hline")
    print("%% per seed, HopSkipJump l_inf within eps = 0.3:")
    t = q(level=0, eps=0.3, attack="hsj_inf", **m4)
    print(t.pivot_table(index=["dataset", "activation"], columns="seed",
                        values="asr").round(4).to_string())

    def actavg(ds, lv, e, atk):
        return [q(dataset=ds, level=lv, eps=e, attack=atk, seed=s,
                  **m4).asr.mean() for s in SEEDS]

    print("\n%% ---- tab:levels (union of PGD, BIM, SPSA, transfer; "
          "activation-averaged)")
    for ds, L in lab:
        print(f"\\multicolumn{{5}}{{l}}{{\\textit{{{L}}}}} \\\\")
        for e in EPS_LIST:
            v = {lv: actavg(ds, lv, e, "union") for lv in LEVELS}
            print(f"$\\epsilon={e}$ & " + " & ".join(_ms(v[lv]) for lv in LEVELS)
                  + f" & {100 * np.mean(v[2]) / np.mean(v[0]):.0f}\\% \\\\")
        print("\\hline")

    print("\n%% ---- tab:levels_attack (eps = 0.3 is the table; all budgets shown)")
    for e in EPS_LIST:
        print(f"%% eps = {e}")
        for ds, L in lab:
            print(f"\\multicolumn{{5}}{{l}}{{\\textit{{{L}}}}} \\\\")
            for atk, A in (("pgd", "PGD"), ("bim", "BIM"), ("spsa", "SPSA"),
                           ("transfer", "Transfer")):
                v = {lv: actavg(ds, lv, e, atk) for lv in LEVELS}
                print(f"{A} & " + " & ".join(_ms(v[lv]) for lv in LEVELS)
                      + f" & {np.mean(v[0]) - np.mean(v[2]):.3f} \\\\")
            print("\\hline")

    print("\n%% ---- tab:levels_act, MLP columns (CNN column: later stage)")
    for ds, L in lab:
        print(f"\\multicolumn{{6}}{{l}}{{\\textit{{{L}}}}} \\\\")
        for e in EPS_LIST:
            for i, lv in enumerate((0, 2)):
                head = (f"$\\epsilon = {e}$ & L{lv}" if i == 0
                        else f"                 & L{lv}")
                print(head + " & " + " & ".join(
                    _ms(q(dataset=ds, level=lv, eps=e, attack="union",
                          activation=a, **m4).sort_values("seed").asr)
                    for a in acts) + " & CNN \\\\")
        print("\\hline")

    print("\n%% ---- consistency: the same measurement is ONE row everywhere")
    for ds in ("cic", "unsw"):
        for a in acts:
            r = q(dataset=ds, activation=a, level=0, eps=0.3, attack="pgd", **m4)
            print(f"  {ds:4s} {a:4s} Model 4 PGD eps=0.3: {r.asr.mean():.3f} "
                  f"(Table 4 = Table 5 = Table 7/B.1 inputs = levels L0)")
    print("\n%% ---- clean F1 of the Model 4 networks vs the earlier Table 5 models")
    r = q(level=0, eps=0.3, attack="pgd", **m4)
    for x in r.sort_values(["dataset", "activation", "seed"]).itertuples():
        old = OLD_TABLE5_F1.get((x.dataset, x.activation, int(x.seed)))
        if old is None:
            continue
        print(f"  {x.dataset:4s} {x.activation:4s} s{int(x.seed):<2d}: "
              f"{x.clean_f1:.4f}  earlier {old:.4f}"
              + ("  same" if abs(x.clean_f1 - old) < 5.01e-5 else "  different"))


# ---- parallel launcher ----
_LAUNCHED = False


def _n_parallel():
    """Parts to run at once."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    try:
        with open("/sys/fs/cgroup/cpu.max") as fh:
            quota, period = fh.read().split()[:2]
        if quota != "max":
            n = min(n, max(1, int(float(quota) / float(period))))
    except (OSError, ValueError):
        pass
    try:
        gb = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2 ** 30
        try:                                # container memory limit
            with open("/sys/fs/cgroup/memory.max") as fh:
                lim = fh.read().strip()
            if lim != "max":
                gb = min(gb, int(lim) / 2 ** 30)
        except (OSError, ValueError):
            pass
        n = min(n, max(1, int(gb // 2)))
    except (OSError, ValueError, AttributeError):
        pass
    return max(1, min(n, len(PARTS)))


def _worker_file():
    """The file each part process runs."""
    f = globals().get("__file__")
    if f and os.path.exists(f):
        return os.path.abspath(f)
    try:
        import IPython
        src = IPython.get_ipython().user_ns["_ih"][-1]
    except Exception:
        return None
    if "def run_part(" not in src or "def launch_parallel(" not in src:
        return None
    try:
        compile(src, "rerun_all_worker.py", "exec")
    except SyntaxError:
        return None                         # e.g. a %magic line in the cell
    tmp = f"rerun_all_worker.py.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(src)
    os.replace(tmp, "rerun_all_worker.py")
    return os.path.abspath("rerun_all_worker.py")


def _tail(path, n=1):
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 6000))
            lines = fh.read().decode("utf-8", "replace").splitlines()
        return [x for x in lines if x.strip()][-n:]
    except OSError:
        return []


def launch_parallel(worker, n):
    """data -> the 18 parts, n at a time -> summary."""
    global _LAUNCHED
    _LAUNCHED = True
    now = lambda: time.strftime("%H:%M")

    def spawn(part, log):
        fh = open(log, "w", encoding="utf-8")
        p = subprocess.Popen(
            [sys.executable, worker], stdout=fh, stderr=subprocess.STDOUT,
            env=dict(os.environ, RERUN_PART=part, PYTHONUNBUFFERED="1",
                     RERUN_LAUNCHER_PID=str(os.getpid())))
        return p, fh, log, time.time()

    print(">>> rerun_all starting (parallel launcher)", flush=True)
    print("=" * 74, flush=True)
    print(json.dumps(env_report()), flush=True)
    print(f"{len(PARTS)} parts, {n} at a time, 1 thread each. Each part writes "
          f"run_all_<part>.out\nand {RES_DIR}/attacks_<part>.csv. Parts run "
          f"side by side change no number.", flush=True)
    print("=" * 74, flush=True)

    p, fh, log, t0 = spawn("data", "run_all_data.out")
    rc = p.wait(); fh.close()
    if rc != 0:
        print("\n".join(_tail(log, 30)), flush=True)
        _stop(f"the data step failed (see {log}). Nothing else was started.")
    for line in _tail(log, 12):
        if "data hash" in line:
            print(line, flush=True)
    print(f"[{now()}] data ready ({(time.time() - t0) / 60:.0f} min)", flush=True)

    todo = sorted(PARTS, key=lambda x: (x[0] != "cic", x[1] != "relu"))
    todo = [f"{d}-{a}-{s}" for d, a, s in todo]
    running, failed, n_done, beat = {}, [], 0, time.time()
    try:
        while todo or running:
            while todo and len(running) < n:
                part = todo.pop(0)
                running[part] = spawn(part, f"run_all_{part}.out")
                print(f"[{now()}] started  {part}", flush=True)
            time.sleep(5)
            for part in list(running):
                p, fh, log, t0 = running[part]
                rc = p.poll()
                if rc is None:
                    continue
                fh.close(); del running[part]
                mins = (time.time() - t0) / 60
                if rc == 0 and any("part done" in x for x in _tail(log, 6)):
                    n_done += 1
                    print(f"[{now()}] finished {part} in {mins:.0f} min "
                          f"({n_done} of {len(PARTS)} done)", flush=True)
                else:
                    failed.append(part)
                    print(f"[{now()}] FAILED   {part} after {mins:.0f} min; "
                          f"last lines of {log}:", flush=True)
                    print("\n".join("      " + x for x in _tail(log, 12)),
                          flush=True)
            if running and time.time() - beat >= HEARTBEAT:
                beat = time.time()
                print(f"[{now()}] {n_done} of {len(PARTS)} done, "
                      f"{len(running)} running, {len(todo)} waiting:", flush=True)
                for part, (p, fh, log, t0) in running.items():
                    last = (_tail(log, 1) or [""])[0].strip()[:90]
                    print(f"      {part:13s} {(time.time() - t0) / 60:4.0f} min"
                          f"  {last}", flush=True)
    finally:
        for part, (p, fh, log, t0) in running.items():
            p.terminate(); fh.close()       # interrupted: stop what is running
    if failed:
        _stop(f"{len(failed)} part(s) failed: {', '.join(failed)}. The other "
              f"parts are finished and saved. Fix the cause shown above and "
              f"run this again: finished work is skipped.")

    with open("rerun_summary.txt", "w", encoding="utf-8") as fh:
        rc = subprocess.run(
            [sys.executable, worker], stdout=fh, stderr=subprocess.STDOUT,
            env=dict(os.environ, RERUN_PART="summary")).returncode
    with open("rerun_summary.txt", encoding="utf-8") as fh:
        print(fh.read(), flush=True)
    if rc != 0:
        _stop("the summary step failed (see rerun_summary.txt above).")
    print(f"[{now()}] all {len(PARTS)} parts done; tables are in "
          f"rerun_summary.txt", flush=True)


def main():
    if RERUN_PART == "summary":
        print("%% environment (ART version etc.): " + json.dumps(env_report()),
              flush=True)
        summarize()
        summarize_stages()
    elif RERUN_PART == "" and PARALLEL != 1:
        n = min(PARALLEL, len(PARTS)) if PARALLEL > 1 else _n_parallel()
        worker = _worker_file() if n > 1 else None
        if n > 1 and worker:
            launch_parallel(worker, n)
        else:
            print("running the parts one after another in this process ("
                  + ("only one CPU core or too little memory for more"
                     if n <= 1 else "could not save this cell as a file")
                  + ").", flush=True)
            run_parts()
    else:
        run_parts()


# ---- the report is a FILE, not a notebook ----
LOG = f"report_rerun_{RERUN_PART or 'all'}.txt"


class _Tee:
    def __init__(self, path, stream):
        self.f = open(path, "w", encoding="utf-8", buffering=1)
        self.stream = stream

    def write(self, s):
        self.stream.write(s)
        try:
            self.f.write(s)
        except Exception:
            pass
        return len(s)

    def flush(self):
        self.stream.flush()
        try:
            self.f.flush()
        except Exception:
            pass

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self.stream, name)


_orig_stdout = sys.stdout
sys.stdout = _Tee(LOG, _orig_stdout)
_ok = False
try:
    main()
    _ok = True
except Exception as _e:
    import traceback
    traceback.print_exc(file=sys.stdout)
    print(f"\n>>> STOPPED: {type(_e).__name__}: {_e}", flush=True)
    print(f"{OUT} is written row-by-row and the model cache is intact. "
          f"Re-paste this cell: finished cells are skipped.", flush=True)
    raise
finally:
    sys.stdout.flush()
    sys.stdout.close()
    sys.stdout = _orig_stdout
    if _ok:
        print(f"\nDone: "
              f"{OUT_FINAL if (_LAUNCHED or RERUN_PART == 'summary') else OUT}"
              f" and {LOG}" + (", rerun_summary.txt" if _LAUNCHED else ""))
    else:
        print(f"\nIncomplete. The reason is printed above and is also in {LOG}.")
