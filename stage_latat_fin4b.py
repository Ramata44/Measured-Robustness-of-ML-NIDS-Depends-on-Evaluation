# Revision stages latat and fin4b for mlnids_robustness_evaluation.py.
#
# latat  CSE-CIC-IDS2018, capability C2: the lattice of the feasible box
#        (5^6 points and 2^6 corners) on the saved PGD-AT and SS-AT Model 4
#        networks, combined per flow with the stronger-attack set.
# fin4b  CSE-CIC-IDS2018, capability C2: the whole-unit and all-combined
#        restrictions with Fwd Seg Size Min moved in 4-byte steps.
#
# Usage: paste this file into mlnids_robustness_evaluation.py above the line
# "# ---- later-stage code goes above this line ----", then run
#     python run_stage.py latat
#     python run_stage.py fin4b
# Requires the saved results of stages capa, cxcomb and rvstrong.

FIN_STEP_FEATURE = "Fwd Seg Size Min"
FIN_STEP_BYTES = 4
LA_DEFENCES = [AT_ROLE, CAAT_ROLE]
LA_COLS = ["dataset", "activation", "seed", "defence", "level", "eps",
           "n_correct", "n_points", "n_lattice", "n_corner", "n_base_union",
           "n_strong", "n_strong_or_lattice", "n_lattice_not_strong",
           "elapsed_s"]
LA_KEY = LA_COLS[:6]
F4_COLS = ["dataset", "activation", "seed", "variant", "eps", "attack", "asr",
           "n_correct", "n_hit", "n_c0_union", "n_check_diff", "elapsed_s"]
F4_KEY = F4_COLS[:6]
F4F_COLS = ["dataset", "activation", "seed", "variant", "eps", "feature",
            "n_hit", "n_moved", "frac_moved"]
F4F_KEY = F4F_COLS[:6]


def la_enumerate(c, model, correct, eps, free):
    """cxe_enumerate for a given network: per flow, any lattice point / any
    corner misclassified (correctly classified flows only)."""
    lat, cor = cxe_offsets(eps, len(free), CXE_GRID)
    offs = np.concatenate([lat, cor], 0)
    lo, hi = float(c.Xtr.min()), float(c.Xtr.max())
    n, d = c.Xe.shape
    hit, hitc = np.zeros(n, bool), np.zeros(n, bool)
    idx = np.where(correct)[0]
    step = max(1, CXE_ROWS // len(offs))
    for s in range(0, len(idx), step):
        ii = idx[s:s + step]
        pts = np.repeat(c.Xe[ii][:, None, :], len(offs), 1).astype(np.float32)
        pts[:, :, free] += offs[None]
        np.clip(pts, lo, hi, out=pts)
        w = (_probs(model, pts.reshape(-1, d), DEVICE).argmax(1)
             .reshape(len(ii), len(offs)) != c.ye[ii][:, None])
        hit[ii] = w.any(1)
        hitc[ii] = w[:, len(lat):].any(1)
    return hit, hitc, len(offs)


def stage_latat(c):
    if c.ds != "cic":
        return
    free = np.where(np.asarray(c.masks[2]) > 0)[0]
    have = stage_have("latat", LA_KEY)
    strong = npz_load(stage_npz("rvstrong", c))
    npz = stage_npz("latat", c)
    saved = npz_load(npz)
    for dfc in LA_DEFENCES:
        todo = [e for e in EPS_LIST
                if _k(c.ds, c.act, c.seed, dfc, 2, round(e, 3)) not in have]
        if not todo:
            continue
        model, clf, correct, n_ok = rv_model(c, dfc)
        if f"{dfc}_correct" not in strong or not np.array_equal(
                strong[f"{dfc}_correct"].astype(bool), correct):
            _stop(f"latat {c.part} {dfc}: the rvstrong flows are missing or "
                  f"belong to other correctly classified flows.")
        for eps in todo:
            ks, kb = f"{dfc}_L2_e{eps}_strong", f"{dfc}_L2_e{eps}_base_union"
            if ks not in strong or kb not in strong:
                _stop(f"latat {c.part}: {ks} is not in the rvstrong flows.")
            st, bu = strong[ks].astype(bool), strong[kb].astype(bool)
            t0 = time.time()
            lat, cor, n_pts = la_enumerate(c, model, correct, eps, free)
            if (lat & ~correct).any() or (st & ~correct).any():
                _stop(f"latat {c.part} {dfc}: a hit on a misclassified flow.")
            saved[f"{dfc}_lattice_e{eps}"], saved[f"{dfc}_corner_e{eps}"] = lat, cor
            npz_save(npz, saved)
            stage_append("latat", LA_COLS, dict(
                dataset=c.ds, activation=c.act, seed=c.seed, defence=dfc,
                level=2, eps=round(eps, 3), n_correct=n_ok, n_points=n_pts,
                n_lattice=int(lat.sum()), n_corner=int(cor.sum()),
                n_base_union=int(bu.sum()), n_strong=int(st.sum()),
                n_strong_or_lattice=int((st | lat).sum()),
                n_lattice_not_strong=int((lat & ~st).sum()),
                elapsed_s=round(time.time() - t0, 1)))
            have.add(_k(c.ds, c.act, c.seed, dfc, 2, round(eps, 3)))
            print(f"    latat {dfc} C2 eps={eps}: lattice {int(lat.sum())}, "
                  f"stronger {int(st.sum())}, either {int((st | lat).sum())} "
                  f"of {n_ok} flows ({n_pts} points)", flush=True)


def summarize_latat():
    d = stage_read("latat", LA_KEY)
    if d is None:
        print("%% latat: not run yet"); return
    print("%% latat: CSE-CIC-IDS2018 C2, defended Model 4; mean+-SD over seeds "
          "of the activation-averaged rate (as Table advtrain)")
    for (dfc, e), g in d.groupby(["defence", "eps"]):
        out = []
        for col in ["n_base_union", "n_strong", "n_lattice", "n_strong_or_lattice"]:
            r = (g[col] / g.n_correct).groupby(g.seed).mean()
            out.append(f"{col[2:]} {r.mean():.3f}+-{r.std(ddof=SD_DDOF):.3f}")
        print(f"  {dfc:7s} eps={e}: " + " | ".join(out) +
              f" | flows lattice-not-stronger {int(g.n_lattice_not_strong.sum())}"
              f" ({len(g)} networks)")


def f4_quantise(X, adv, scale, quant_idx, steps):
    """ca_quantise with a raw step per quantised coordinate (1 = ca_quantise):
    the raw displacement is truncated toward zero to a multiple of the step."""
    d = (adv.astype(np.float64) - X.astype(np.float64))
    if len(quant_idx):
        qi = np.asarray(quant_idx, dtype=int)
        st = np.asarray(steps, np.float64)
        raw = d[:, qi] * scale[None, qi]
        d[:, qi] = np.trunc(raw / st[None]) * st[None] / scale[None, qi]
    return (X.astype(np.float32) + d.astype(np.float32))


def f4_saved_cx(c):
    """n_hit of the saved cxcomb 'comb' rows of this part."""
    fs = glob.glob(os.path.join(RES_DIR, "cxcomb_*.csv"))
    if not fs:
        _stop("fin4b: no cxcomb rows are saved.")
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d = d[(d.dataset == c.ds) & (d.activation == c.act) & (d.seed == c.seed)
          & (d.variant == "comb")].drop_duplicates(["eps", "attack"])
    return {(round(float(r.eps), 3), r.attack): int(r.n_hit)
            for r in d.itertuples()}


def stage_fin4b(c):
    if c.ds != "cic":
        return
    if FIN_STEP_FEATURE not in c.names:
        _stop(f"fin4b: {FIN_STEP_FEATURE} is not among the selected features.")
    sc = ca_scaler(c.ds, c.data)
    scale_k = np.asarray(sc["scale"])[c.cols]
    int_k = np.asarray(sc["is_int_all"])[c.cols]
    free = np.asarray(c.masks[2]).reshape(-1) > 0
    q_idx = [i for i in range(len(c.names)) if free[i] and int_k[i]]
    j4 = c.names.index(FIN_STEP_FEATURE)
    if j4 not in q_idx:
        _stop(f"fin4b: {FIN_STEP_FEATURE} is not a free integer feature at C2.")
    st1 = [1.0] * len(q_idx)
    st4 = [float(FIN_STEP_BYTES) if i == j4 else 1.0 for i in q_idx]
    have, havef = stage_have("fin4b", F4_KEY), stage_have("fin4bfeat", F4F_KEY)
    base = ca_base_flows(c)
    cq = npz_load(stage_npz("caquant", c))
    cx = f4_saved_cx(c)
    free7 = c.masks[2]
    for eps in EPS_LIST:
        e3 = round(eps, 3)
        if all(_k(c.ds, c.act, c.seed, v, e3, "union") in have
               for v in ("whole4", "comb4")):
            continue
        n_c0 = int(np.asarray(base[f"L0_e{eps}_union"], bool).sum())
        U = {v: np.zeros(len(c.Xe), bool) for v in ("whole1", "whole4", "comb1", "comb4")}
        Uadv = {v: c.Xe.astype(np.float32).copy() for v in ("comb4",)}
        groups, tcp_cols, n_non = cx_groups(c, eps)
        lo, hi = cd_box(c, eps, "addonly")
        rows = []
        for a in CA_ATTACKS:
            t0 = time.time()
            # whole units
            s7 = _seed_of(c.ds, DEPTH, FEATS, c.act, c.seed, 2, eps, a)
            adv7 = ca_attack(c, a, eps, free7, s7)
            h7 = ca_hits(c, adv7, eps, free7, f"fin4b seven {a}")
            if not np.array_equal(h7, np.asarray(base[f"L2_e{eps}_{a}"], bool)):
                _stop(f"fin4b {c.part} eps={eps} {a}: the base L2 attack is not "
                      f"reproduced; nothing was written for this cell.")
            h1 = ca_hits(c, f4_quantise(c.Xe, adv7, scale_k, q_idx, st1),
                         eps, free7, f"fin4b whole1 {a}")
            ref = cq.get(f"whole_e{eps}_{a}")
            if ref is None or not np.array_equal(h1, np.asarray(ref, bool)):
                _stop(f"fin4b {c.part} eps={eps} {a}: the 1-byte whole-unit "
                      f"result differs from the saved caquant flows.")
            h4 = ca_hits(c, f4_quantise(c.Xe, adv7, scale_k, q_idx, st4),
                         eps, free7, f"fin4b whole4 {a}")
            # all combined
            raw = c.Xe.astype(np.float32).copy()
            for gname, idx, glo, ghi, gm in groups:
                if gm.sum() == 0:
                    continue
                sg = _seed_of("cxcomb", "comb", c.ds, c.act, c.seed, eps, a, gname)
                g_adv = cx_attack(c, a, eps, glo, ghi, gm, sg, idx)
                d = np.clip(g_adv.astype(np.float64)
                            - c.Xe[idx].astype(np.float64), glo, ghi)
                raw[idx] = (c.Xe[idx].astype(np.float64) + d).astype(np.float32)
            out = {}
            for v, st in (("comb1", st1), ("comb4", st4)):
                adv = f4_quantise(c.Xe, raw, scale_k, q_idx, st)
                adv, extra = cx_flag_totals(c, adv, sc)
                out[v] = (adv, cx_score(c, adv, lo, hi, extra, f"fin4b {v} eps={eps} {a}"))
            if out["comb1"][1].sum() != cx.get((e3, a), -1):
                _stop(f"fin4b {c.part} eps={eps} {a}: the 1-byte combined result "
                      f"({int(out['comb1'][1].sum())} flows) differs from the "
                      f"saved cxcomb row ({cx.get((e3, a))}).")
            new = out["comb4"][1] & ~U["comb4"]
            Uadv["comb4"][new] = out["comb4"][0][new]
            for v, h, nd in (("whole1", h1, 0), ("whole4", h4, -1),
                             ("comb1", out["comb1"][1], 0),
                             ("comb4", out["comb4"][1], -1)):
                U[v] |= h
                rows.append(dict(dataset=c.ds, activation=c.act, seed=c.seed,
                                 variant=v, eps=e3, attack=a,
                                 asr=round(float(h.sum() / c.n_ok), 4),
                                 n_correct=c.n_ok, n_hit=int(h.sum()),
                                 n_c0_union=n_c0, n_check_diff=nd,
                                 elapsed_s=round(time.time() - t0, 1)))
        if U["comb1"].sum() != cx.get((e3, "union"), -1):
            _stop(f"fin4b {c.part} eps={eps}: the 1-byte combined union differs "
                  f"from the saved cxcomb row.")
        if not np.array_equal(U["whole1"], np.asarray(cq[f"whole_e{eps}_union"], bool)):
            _stop(f"fin4b {c.part} eps={eps}: the 1-byte whole-unit union differs "
                  f"from the saved caquant flows.")
        for v in U:
            rows.append(dict(dataset=c.ds, activation=c.act, seed=c.seed,
                             variant=v, eps=e3, attack="union",
                             asr=round(float(U[v].sum() / c.n_ok), 4),
                             n_correct=c.n_ok, n_hit=int(U[v].sum()),
                             n_c0_union=n_c0,
                             n_check_diff=0 if v.endswith("1") else -1,
                             elapsed_s=0.0))
        for r in cx_feature_rows(c, Uadv["comb4"], U["comb4"], scale_k,
                                 "comb4", eps, "union"):
            fk = _k(c.ds, c.act, c.seed, "comb4", e3, r["feature"])
            if fk not in havef:
                stage_append("fin4bfeat", F4F_COLS, {k: r[k] for k in F4F_COLS})
                havef.add(fk)
        for r in rows:
            if _k(*[r[k] for k in F4_KEY]) not in have:
                stage_append("fin4b", F4_COLS, r)
                have.add(_k(*[r[k] for k in F4_KEY]))
        print(f"    fin4b eps={eps}: whole {int(U['whole1'].sum())} -> "
              f"{int(U['whole4'].sum())}, combined {int(U['comb1'].sum())} -> "
              f"{int(U['comb4'].sum())} flows (1-byte -> {FIN_STEP_BYTES}-byte "
              f"{FIN_STEP_FEATURE}); C0 {n_c0}", flush=True)


def summarize_fin4b():
    d = stage_read("fin4b", F4_KEY)
    if d is None:
        print("%% fin4b: not run yet"); return
    u = d[d.attack == "union"]
    print(f"%% fin4b: CSE-CIC-IDS2018, Table c2checks entries (mean worst case "
          f"/ mean C0 success x 100, {u.groupby('variant').size().max() // 3} "
          f"networks); {FIN_STEP_FEATURE} step 1 vs {FIN_STEP_BYTES} bytes")
    for v in ("whole1", "whole4", "comb1", "comb4"):
        txt = []
        for e, g in u[u.variant == v].groupby("eps"):
            txt.append(f"eps={e}: {100 * (g.n_hit / g.n_correct).mean() / (g.n_c0_union / g.n_correct).mean():.1f}")
        print(f"  {v:6s} " + "  ".join(txt))
    f = stage_read("fin4bfeat", F4F_KEY)
    if f is not None:
        print("%% fin4b: comb4 union, share of evaded flows that move each feature")
        for (e, nm), g in f.groupby(["eps", "feature"]):
            print(f"  eps={e} {nm}: {100 * g.n_moved.sum() / g.n_hit.sum():.1f}% "
                  f"of {int(g.n_hit.sum())} flows")


STAGES.append(("latat", stage_latat, summarize_latat))
STAGES.append(("fin4b", stage_fin4b, summarize_fin4b))
