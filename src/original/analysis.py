"""Prespecified analysis (protocol v0 §6-§7). Reads only targets.csv and the sealed logits.

Unit: one parent photo (one target box). Pairing: the same bootstrap resample of parents is used for
every model, view and policy. Seeded models (M1, M3): per-parent correctness = mean over the 3 seeds.

PRIMARY    I_core = Δ_M1 − Δ_M2, where Δ_m = acc(C) − acc(F); PlantDoc primary cohort, resize policy,
           full 38-class label space.
SECONDARY  (Holm family) Δ_M1, Δ_M2; I_core masked label space; I_core sensitivity cohort; I_core per
           box-fraction tertile (3); I_context = (M−F) contrast; I_scale = (C−M) contrast; I_ext = Δ_M3 − Δ_M4;
           I_core letterbox policy; I_core FieldPlant replication cohort.
Decision rule with δ (percentage points; default 5): negligible / meaningful / nonzero-below-δ / inconclusive.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

DELTA_PP = 5.0
B = 10_000
RNG_SEED = 20260928
CORE = ("M1_mobilenetv3_large", "M2_dinov2_small")
EXT = ("M3_convnext_tiny", "M4_dinov2_base")
MODELS = ("M1_mobilenetv3_large", "M2_dinov2_small", "M3_convnext_tiny", "M4_dinov2_base")
SEEDED = {"M1_mobilenetv3_large", "M3_convnext_tiny"}
SEEDS = (20260928, 20260929, 20260930)


# ------------------------------------------------------------------ statistics
def outcome(point, lo, hi, delta=DELTA_PP):
    """Four mutually exclusive outcomes (protocol §7). Inputs in percentage points."""
    if -delta < lo and hi < delta:
        return "negligible"
    excl0 = lo > 0 or hi < 0
    if excl0 and abs(point) >= delta:
        return "meaningful_difference"
    if excl0:
        return "nonzero_below_or_near_delta"
    return "inconclusive"


def boot_ci(x: np.ndarray, idx: np.ndarray):
    """x: per-parent values (n,), idx: (B, n) resample indices. Returns mean, 2.5%, 97.5% (x100 = pp)."""
    bm = x[idx].mean(1)
    return 100 * x.mean(), 100 * np.percentile(bm, 2.5), 100 * np.percentile(bm, 97.5)


def signflip_p(x: np.ndarray, rng: np.random.Generator, n=B):
    """Two-sided sign-flip permutation p-value for mean(x)=0 (paired differences)."""
    obs = abs(x.mean())
    if obs == 0:
        return 1.0
    s = rng.choice(np.array([-1.0, 1.0]), size=(n, len(x)))
    return float((1 + (np.abs((s * x).mean(1)) >= obs - 1e-12).sum()) / (n + 1))


def holm(pvals: dict) -> dict:
    ks = sorted(pvals, key=lambda k: pvals[k]); m = len(ks); out, run = {}, 0.0
    for i, k in enumerate(ks):
        run = max(run, min(1.0, (m - i) * pvals[k])); out[k] = run
    return out


def softmax(L):
    L = L - np.nanmax(np.where(np.isfinite(L), L, -np.inf), 1, keepdims=True)
    e = np.where(np.isfinite(L), np.exp(L), 0.0)
    return e / e.sum(1, keepdims=True)


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any():
            e += s.mean() * abs(correct[s].mean() - conf[s].mean())
    return float(e)


def macro_f1(pred, y, labels):
    f = []
    for c in labels:
        tp = ((pred == c) & (y == c)).sum(); fp = ((pred == c) & (y != c)).sum(); fn = ((pred != c) & (y == c)).sum()
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


# ------------------------------------------------------------------ data
def load(out: Path, tpath: Path):
    targets = list(csv.DictReader(open(tpath)))
    L = {}
    classes = None
    for m in MODELS:
        files = [out / "logits" / (f"{m}_seed{s}.npz" if m in SEEDED else f"{m}.npz") for s in (SEEDS if m in SEEDED else (None,))]
        L[m] = []
        for f in files:
            z = np.load(f, allow_pickle=False)
            if classes is None:
                classes = [str(c) for c in z["classes"]]; ids = [str(i) for i in z["ids"]]
            assert [str(c) for c in z["classes"]] == classes and [str(i) for i in z["ids"]] == ids, f"id/class mismatch in {f}"
            L[m].append(z["logits"])
    pos = {k: i for i, k in enumerate(ids)}
    return targets, classes, L, pos


class Cohort:
    """Correctness arrays for one (dataset, flag, policy, label space)."""
    def __init__(self, targets, classes, L, pos, dataset, flag, policy, space, mask_classes=None):
        self.rows = [t for t in targets if t["dataset"] == dataset and t[flag] == "1"]
        self.n = len(self.rows)
        cidx = {c: i for i, c in enumerate(classes)}
        self.y = np.array([cidx[t["pv_class"]] for t in self.rows])
        self.labels = sorted(set(self.y.tolist()))
        allowed = None
        if space == "masked":
            allowed = np.zeros(len(classes), bool); allowed[[cidx[c] for c in mask_classes]] = True
        self.host = np.array([c.split("___")[0] for c in classes])
        self.corr, self.pred, self.conf = {}, {}, {}
        for m, mats in L.items():
            per_seed = []
            preds, confs = [], []
            for M in mats:
                vv = {}
                for v in ("F", "C", "M"):
                    X = M[[pos[f"{t['target_id']}|{v}|{policy}"] for t in self.rows]].copy()
                    if allowed is not None:
                        X[:, ~allowed] = -np.inf
                    P = softmax(X)
                    p = P.argmax(1)
                    vv[v] = (p == self.y).astype(float)
                    preds.append((v, p)); confs.append((v, P.max(1)))
                per_seed.append(vv)
            self.corr[m] = {v: np.mean([s[v] for s in per_seed], 0) for v in ("F", "C", "M")}
            self.corr[m]["_seeds"] = per_seed
            self.pred[m] = preds; self.conf[m] = confs
        self.box_frac = np.array([float(t["box_frac"]) for t in self.rows])

    def d(self, m, a="C", b="F"):
        return self.corr[m][a] - self.corr[m][b]


def contrasts(c: Cohort):
    """Per-parent vectors for every named estimand in this cohort."""
    return {
        "Delta_M1": c.d(CORE[0]), "Delta_M2": c.d(CORE[1]),
        "Delta_M3": c.d(EXT[0]), "Delta_M4": c.d(EXT[1]),
        "I_core": c.d(CORE[0]) - c.d(CORE[1]),
        "I_context": c.d(CORE[0], "M", "F") - c.d(CORE[1], "M", "F"),
        "I_scale": c.d(CORE[0], "C", "M") - c.d(CORE[1], "C", "M"),
        "I_ext": c.d(EXT[0]) - c.d(EXT[1]),
        "Context_M1": c.d(CORE[0], "M", "F"), "Context_M2": c.d(CORE[1], "M", "F"),
        "Scale_M1": c.d(CORE[0], "C", "M"), "Scale_M2": c.d(CORE[1], "C", "M"),
    }


def summarise(x, idx, rng, delta):
    pt, lo, hi = boot_ci(x, idx)
    return {"estimate_pp": round(float(pt), 3), "ci95_pp": [round(float(lo), 3), round(float(hi), 3)],
            "p_signflip": signflip_p(x, rng), "outcome": outcome(pt, lo, hi, delta), "n_parents": int(len(x))}


def describe(c: Cohort, idx, classes):
    acc = {}
    for m in MODELS:
        acc[m] = {}
        for v in ("F", "C", "M"):
            pt, lo, hi = boot_ci(c.corr[m][v], idx)
            preds = [p for vv, p in c.pred[m] if vv == v]; confs = [q for vv, q in c.conf[m] if vv == v]
            f1 = float(np.mean([macro_f1(p, c.y, c.labels) for p in preds]))
            host_ret = float(np.mean([(c.host[p] == c.host[c.y]).mean() for p in preds]))
            within = float(np.mean([((p == c.y).sum() / max(1, (c.host[p] == c.host[c.y]).sum())) for p in preds]))
            e = float(np.mean([ece(q, (p == c.y).astype(float)) for p, q in zip(preds, confs)]))
            acc[m][v] = {"acc_pp": round(pt, 2), "ci95_pp": [round(lo, 2), round(hi, 2)], "macro_f1": round(f1, 4),
                         "host_retention": round(host_ret, 4), "within_host_acc": round(within, 4), "ece": round(e, 4)}
        # paired transitions C vs F, per seed
        acc[m]["transitions_C_vs_F_per_seed"] = [
            {"both_right": int(((s["F"] == 1) & (s["C"] == 1)).sum()), "F_right_C_wrong": int(((s["F"] == 1) & (s["C"] == 0)).sum()),
             "F_wrong_C_right": int(((s["F"] == 0) & (s["C"] == 1)).sum()), "both_wrong": int(((s["F"] == 0) & (s["C"] == 0)).sum())}
            for s in c.corr[m]["_seeds"]]
    per_class = []
    for k in c.labels:
        s = c.y == k
        row = {"pv_class": classes[k], "n": int(s.sum())}
        for m in CORE:
            row[f"Delta_{m}_pp"] = round(100 * c.d(m)[s].mean(), 2)
        row["I_core_pp"] = round(100 * (c.d(CORE[0]) - c.d(CORE[1]))[s].mean(), 2)
        per_class.append(row)
    ccls = np.array([r["I_core_pp"] for r in per_class])
    seeds_I = [round(100 * float(((sd["C"] - sd["F"]) - c.d(CORE[1])).mean()), 3) for sd in c.corr[CORE[0]]["_seeds"]]
    return {"n_parents": c.n, "n_classes": len(c.labels), "accuracy": acc, "per_class": per_class,
            "I_core_class_balanced_pp": round(float(ccls.mean()), 3), "I_core_per_M1_seed_pp": seeds_I}


# ------------------------------------------------------------------ main entry
def run(out: Path, tpath: Path, delta: float = DELTA_PP):
    targets, classes, L, pos = load(out, tpath)
    rng = np.random.default_rng(RNG_SEED)
    res = {"delta_pp": delta, "bootstrap_B": B, "rng_seed": RNG_SEED, "cohorts": {}}

    def cohort(ds, flag, policy="resize", space="full38"):
        sel = [t for t in targets if t["dataset"] == ds and t[flag] == "1"]
        if not sel:
            return None
        mask = sorted({t["pv_class"] for t in sel})
        return Cohort(targets, classes, L, pos, ds, flag, policy, space, mask)

    def idx_for(n):
        return np.random.default_rng(RNG_SEED).integers(0, n, size=(B, n), dtype=np.int32)

    pd = cohort("plantdoc", "primary")
    idx = idx_for(pd.n)
    C = contrasts(pd)
    primary = summarise(C["I_core"], idx, rng, delta)
    sec = {"Delta_M1": summarise(C["Delta_M1"], idx, rng, delta), "Delta_M2": summarise(C["Delta_M2"], idx, rng, delta),
           "I_context": summarise(C["I_context"], idx, rng, delta), "I_scale": summarise(C["I_scale"], idx, rng, delta),
           "I_ext": summarise(C["I_ext"], idx, rng, delta)}
    pdm = cohort("plantdoc", "primary", space="masked")
    sec["I_core_masked"] = summarise(contrasts(pdm)["I_core"], idx, rng, delta)
    pdl = cohort("plantdoc", "primary", policy="letterbox")
    sec["I_core_letterbox"] = summarise(contrasts(pdl)["I_core"], idx, rng, delta)
    t1, t2 = np.quantile(pd.box_frac, [1 / 3, 2 / 3])
    for name, s in (("small", pd.box_frac <= t1), ("medium", (pd.box_frac > t1) & (pd.box_frac <= t2)), ("large", pd.box_frac > t2)):
        x = C["I_core"][s]
        sec[f"I_core_boxfrac_{name}"] = summarise(x, idx_for(len(x)), rng, delta)
    res["box_frac_tertile_cuts"] = [round(float(t1), 5), round(float(t2), 5)]
    pds = cohort("plantdoc", "sensitivity")
    if pds is not None and pds.n != pd.n:
        sec["I_core_sensitivity"] = summarise(contrasts(pds)["I_core"], idx_for(pds.n), rng, delta)
    fp = cohort("fieldplant", "primary")
    if fp is not None:
        fidx = idx_for(fp.n)
        sec["I_core_fieldplant"] = summarise(contrasts(fp)["I_core"], fidx, rng, delta)
        res["cohorts"]["fieldplant_primary"] = describe(fp, fidx, classes)
        res["fieldplant_all_contrasts"] = {k: summarise(v, fidx, rng, delta) for k, v in contrasts(fp).items()}
    fpe = cohort("fieldplant", "sensitivity")
    if fpe is not None and fpe.n:
        res["fieldplant_exact_only_descriptive"] = {k: summarise(v, idx_for(fpe.n), rng, delta)
                                                    for k, v in contrasts(fpe).items() if k in ("I_core", "Delta_M1", "Delta_M2")}
    adj = holm({k: v["p_signflip"] for k, v in sec.items()})
    for k in sec:
        sec[k]["p_holm"] = adj[k]
    res["primary"] = {"estimand": "I_core", "cohort": "plantdoc_primary", "policy": "resize", "label_space": "full38", **primary}
    res["secondary"] = sec
    res["plantdoc_all_contrasts"] = {k: summarise(v, idx, rng, delta) for k, v in C.items()}
    res["cohorts"]["plantdoc_primary"] = describe(pd, idx, classes)
    rdir = out / "results"; rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "results.json").write_text(json.dumps(res, indent=1))
    with open(rdir / "secondary_table.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["estimand", "estimate_pp", "ci_lo", "ci_hi", "p_signflip", "p_holm", "outcome", "n"])
        w.writerow(["PRIMARY I_core", primary["estimate_pp"], *primary["ci95_pp"], primary["p_signflip"], "", primary["outcome"], primary["n_parents"]])
        for k, v in sec.items():
            w.writerow([k, v["estimate_pp"], *v["ci95_pp"], v["p_signflip"], round(v["p_holm"], 5), v["outcome"], v["n_parents"]])
    print(json.dumps({"primary": res["primary"], "secondary": {k: (v["estimate_pp"], v["ci95_pp"], v["outcome"]) for k, v in sec.items()}}, indent=1))
    return res
