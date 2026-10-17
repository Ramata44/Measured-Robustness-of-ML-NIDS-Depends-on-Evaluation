# strong_lattice_cic.py -- CSE-CIC-IDS2018, C2, standard detectors: add the 5^6 lattice to the
# stronger-attack set, per flow, as is done on UNSW-NB15.  Reads only the saved
# per-flow files (no attack, no training, no model loading); about one minute.
import os, glob, base64, zipfile
import numpy as np, pandas as pd
from IPython.display import HTML, display

FLOWS, OUT, ZIP = os.path.join("results_final", "flows"), "strong_lattice_cic.csv", "rerun_results_strong_lattice.zip"
rows = []
for f in sorted(glob.glob(os.path.join(FLOWS, "cxenum_cic_*_s*.npz"))):
    tag = os.path.basename(f)[len("cxenum_"):]                  # cic_<act>_s<seed>.npz
    g = os.path.join(FLOWS, "rvstrong_" + tag)
    if not os.path.isfile(g):
        raise SystemExit(f"STOPPED: {g} is missing")
    L, S = dict(np.load(f)), dict(np.load(g))
    _, act, s = tag[:-4].split("_")
    corr = S["standard_correct"].astype(bool)
    for eps in [0.1, 0.3, 0.5]:
        lat = L[f"lattice_e{eps}"].astype(bool)
        st = S[f"standard_L2_e{eps}_strong"].astype(bool)
        bu = S[f"standard_L2_e{eps}_base_union"].astype(bool)
        if not (lat.shape == st.shape == corr.shape):
            raise SystemExit(f"STOPPED: shapes differ in {tag}")
        if (lat & ~corr).any() or (st & ~corr).any():
            raise SystemExit(f"STOPPED: a hit on a misclassified flow in {tag}")
        n = int(corr.sum())
        rows.append(dict(dataset="cic", activation=act, seed=int(s[1:]), level=2, eps=eps,
                         n_correct=n, n_base_union=int(bu.sum()), n_strong=int(st.sum()),
                         n_lattice=int(lat.sum()), n_strong_or_lattice=int((st | lat).sum()),
                         n_lattice_not_strong=int((lat & ~st).sum()),
                         n_strong_not_lattice=int((st & ~lat).sum())))
d = pd.DataFrame(rows)
if len(d) != 90:
    raise SystemExit(f"STOPPED: expected 90 rows (30 networks x 3 budgets), got {len(d)}")
d.to_csv(OUT, index=False)
for eps, g in d.groupby("eps"):
    r = lambda c: (g[c] / g.n_correct).groupby(g.seed).mean()
    print(f"eps={eps}: strong {r('n_strong').mean():.4f}+-{r('n_strong').std(ddof=1):.4f} | "
          f"lattice {r('n_lattice').mean():.4f} | strong or lattice "
          f"{r('n_strong_or_lattice').mean():.4f}+-{r('n_strong_or_lattice').std(ddof=1):.4f} | "
          f"flows lattice-not-strong {g.n_lattice_not_strong.sum()}, strong-not-lattice {g.n_strong_not_lattice.sum()}")
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(OUT)
b64 = base64.b64encode(open(ZIP, "rb").read()).decode()
display(HTML(f'<a download="{ZIP}" href="data:application/zip;base64,{b64}" '
             f'style="font-size:22px;font-weight:bold">CLICK HERE to download {ZIP}</a>'))
