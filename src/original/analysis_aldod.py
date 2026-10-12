"""Prespecified analysis for the ALDOD single-crop check (PROTOCOL_ALDOD_v1 §5-§6).
Reads only cohort/targets.csv and the sealed logits. Unit: one distinct photo (one target box).

PRIMARY    D = I_healthy − I_diseased, I_g = Δ_M1,g − Δ_M2,g, Δ = acc(C) − acc(F); primary cohort
           (box fraction ≤ 0.50), direct resize, 38-class output. Seeded models: correctness averaged over seeds.
           95% percentile bootstrap, photos resampled within group (B = 10,000, seed 20260928);
           two-sided label-permutation p (10,000). Reproduced if CI > 0; reversal if CI < 0; magnitude by the
           main study's four-outcome rule (δ = 5 pp).
SECONDARY  (Holm, family of 5) I_healthy, I_diseased (sign-flip p), D_ext (M3−M4), D letterbox, D apple-only output.
DESCRIPTIVE accuracies, per-disease I, per-seed D, box-fraction tertiles, large-box cohort, transitions, host retention.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import analysis as A                          # sealed main-study helpers (outcome, signflip_p, holm, load)

APPLE = ("Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy")
VIEWS = ("F", "C")


class Cohort:
    def __init__(self, targets, classes, L, pos, flag, policy, space):
        self.rows = [t for t in targets if t[flag] == "1"]
        cidx = {c: i for i, c in enumerate(classes)}
        self.y = np.array([cidx[t["pv_class"]] for t in self.rows])
        self.h = np.array([t["group"] == "healthy" for t in self.rows])
        self.label = np.array([t["label"] for t in self.rows])
        self.box = np.array([float(t["box_frac"]) for t in self.rows])
        allowed = None
        if space == "apple":
            allowed = np.zeros(len(classes), bool); allowed[[cidx[c] for c in APPLE]] = True
        host = np.array([c.split("___")[0] for c in classes])
        self.corr, self.seed_corr, self.pred, self.host_ret = {}, {}, {}, {}
        for m, mats in L.items():
            per_seed, preds = [], []
            for M in mats:
                vv, pp = {}, {}
                for v in VIEWS:
                    X = M[[pos[f"{t['target_id']}|{v}|{policy}"] for t in self.rows]].copy()
                    if allowed is not None:
                        X[:, ~allowed] = -np.inf
                    p = X.argmax(1)
                    vv[v] = (p == self.y).astype(float); pp[v] = p
                per_seed.append(vv); preds.append(pp)
            self.corr[m] = {v: np.mean([s[v] for s in per_seed], 0) for v in VIEWS}
            self.seed_corr[m] = per_seed; self.pred[m] = preds
            self.host_ret[m] = {v: np.mean([(host[p[v]] == "Apple") for p in preds], 0) for v in VIEWS}

    def d(self, m, seed=None):
        c = self.corr[m] if seed is None else self.seed_corr[m][seed]
        return c["C"] - c["F"]


def boot_idx(nh, nd, seed=A.RNG_SEED, B=A.B):
    rng = np.random.default_rng(seed)
    return rng.integers(0, nh, size=(B, nh), dtype=np.int32), rng.integers(0, nd, size=(B, nd), dtype=np.int32)


def perm_p(x, h, seed=A.RNG_SEED, n=A.B):
    obs = abs(x[h].mean() - x[~h].mean())
    rng = np.random.default_rng(seed); cnt = 0
    for _ in range(n):
        hp = rng.permutation(h)
        cnt += abs(x[hp].mean() - x[~hp].mean()) >= obs - 1e-12
    return float((1 + cnt) / (n + 1))


def ci(bm, pt):
    return {"estimate_pp": round(100 * float(pt), 3),
            "ci95_pp": [round(100 * float(np.percentile(bm, 2.5)), 3), round(100 * float(np.percentile(bm, 97.5)), 3)]}


def D_stats(x, h, ih, idd, with_p=True):
    xh, xd = x[h], x[~h]
    r = ci(xh[ih].mean(1) - xd[idd].mean(1), xh.mean() - xd.mean())
    r["outcome"] = A.outcome(r["estimate_pp"], *r["ci95_pp"])
    lo, hi = r["ci95_pp"]
    r["direction"] = "reproduced (CI > 0)" if lo > 0 else ("reversal (CI < 0)" if hi < 0 else "not reproduced (CI includes 0)")
    if with_p:
        r["p_permutation"] = perm_p(x, h)
    r["n_healthy"], r["n_diseased"] = int(h.sum()), int((~h).sum())
    return r


def I_stats(x, idx, rng):
    r = ci(x[idx].mean(1), x.mean())
    r["outcome"] = A.outcome(r["estimate_pp"], *r["ci95_pp"]); r["p_signflip"] = A.signflip_p(x, rng); r["n"] = int(len(x))
    return r


def run(out: Path, tpath: Path):
    targets, classes, L, pos = A.load(out, tpath)
    res = {"protocol": "PROTOCOL_ALDOD_v1", "primary": None, "secondary": {}, "descriptive": {}}
    C = {(f, p, s): Cohort(targets, classes, L, pos, f, p, s)
         for f, p, s in (("primary", "resize", "full38"), ("primary", "letterbox", "full38"),
                         ("primary", "resize", "apple"), ("largebox", "resize", "full38"))
         if any(t[f] == "1" for t in targets)}
    c = C[("primary", "resize", "full38")]
    ih, idd = boot_idx(int(c.h.sum()), int((~c.h).sum()))
    x = c.d(A.CORE[0]) - c.d(A.CORE[1])
    res["primary"] = {"estimand": "D = I_healthy - I_diseased (M1 vs M2), primary cohort, resize, 38-class", **D_stats(x, c.h, ih, idd)}
    rng = np.random.default_rng(A.RNG_SEED)
    sec = {"I_healthy": I_stats(x[c.h], ih, rng), "I_diseased": I_stats(x[~c.h], idd, rng)}
    xe = c.d(A.EXT[0]) - c.d(A.EXT[1])
    sec["D_ext_M3_M4"] = D_stats(xe, c.h, ih, idd)
    for key, name in ((("primary", "letterbox", "full38"), "D_letterbox"), (("primary", "resize", "apple"), "D_apple_only_output")):
        cc = C[key]; sec[name] = D_stats(cc.d(A.CORE[0]) - cc.d(A.CORE[1]), cc.h, ih, idd)
    ps = {k: v.get("p_permutation", v.get("p_signflip")) for k, v in sec.items()}
    for k, p in A.holm(ps).items():
        sec[k]["p_holm"] = p
    res["secondary"] = sec
    # descriptive
    desc = {}
    desc["accuracy_pct"] = {m: {g: {v: round(100 * float(c.corr[m][v][mask].mean()), 2) for v in VIEWS}
                                for g, mask in (("healthy", c.h), ("diseased", ~c.h))} for m in A.MODELS}
    desc["host_retention_pct"] = {m: {g: {v: round(100 * float(c.host_ret[m][v][mask].mean()), 2) for v in VIEWS}
                                      for g, mask in (("healthy", c.h), ("diseased", ~c.h))} for m in A.MODELS}
    desc["Delta_by_group_pp"] = {m: {g: round(100 * float(c.d(m)[mask].mean()), 2) for g, mask in (("healthy", c.h), ("diseased", ~c.h))}
                                 for m in A.MODELS}
    per_dis = {}
    for lab in sorted(set(c.label[~c.h])):
        s = c.label == lab; xs = x[s]
        bi = np.random.default_rng(A.RNG_SEED).integers(0, len(xs), size=(A.B, len(xs)), dtype=np.int32)
        per_dis[lab] = {**ci(xs[bi].mean(1), xs.mean()), "n": int(s.sum()),
                        "Delta_M1_pp": round(100 * float(c.d(A.CORE[0])[s].mean()), 2), "Delta_M2_pp": round(100 * float(c.d(A.CORE[1])[s].mean()), 2)}
    desc["I_per_disease"] = per_dis
    desc["D_per_M1_seed"] = {str(A.SEEDS[k]): round(100 * float((lambda z: z[c.h].mean() - z[~c.h].mean())(c.d(A.CORE[0], k) - c.d(A.CORE[1]))), 2)
                             for k in range(len(A.SEEDS))}
    cuts = np.quantile(c.box, [1 / 3, 2 / 3])
    tert = {}
    for k, (lo, hi) in enumerate(((-1, cuts[0]), (cuts[0], cuts[1]), (cuts[1], 2))):
        s = (c.box > lo) & (c.box <= hi)
        tert[f"T{k+1}"] = {"box_frac_range": [round(float(max(lo, 0)), 3), round(float(min(hi, 1)), 3)],
                           "n_healthy": int((s & c.h).sum()), "n_diseased": int((s & ~c.h).sum()),
                           "I_healthy_pp": round(100 * float(x[s & c.h].mean()), 2) if (s & c.h).any() else None,
                           "I_diseased_pp": round(100 * float(x[s & ~c.h].mean()), 2) if (s & ~c.h).any() else None}
    desc["box_fraction_tertiles"] = tert
    if ("largebox", "resize", "full38") in C:
        cl = C[("largebox", "resize", "full38")]
        if cl.h.any() and (~cl.h).any():
            jh, jd = boot_idx(int(cl.h.sum()), int((~cl.h).sum()))
            xl = cl.d(A.CORE[0]) - cl.d(A.CORE[1])
            desc["largebox_cohort"] = {"D": D_stats(xl, cl.h, jh, jd, with_p=False),
                                       "I_healthy_pp": round(100 * float(xl[cl.h].mean()), 2),
                                       "I_diseased_pp": round(100 * float(xl[~cl.h].mean()), 2)}
    trans = {}
    for m in A.CORE:
        s0 = c.seed_corr[m][0]
        trans[m] = {g: {"wrong_to_right": int(((s0["F"] == 0) & (s0["C"] == 1) & mask).sum()),
                        "right_to_wrong": int(((s0["F"] == 1) & (s0["C"] == 0) & mask).sum())}
                    for g, mask in (("healthy", c.h), ("diseased", ~c.h))}
    desc["transitions_first_seed"] = trans
    res["descriptive"] = desc
    rdir = out / "results"; rdir.mkdir(exist_ok=True)
    A_json = rdir / "aldod_results.json"
    A_json.write_text(json.dumps(res, indent=1))
    p = res["primary"]
    print(f"\nPRIMARY  D = {p['estimate_pp']:+.1f} pp  95% CI [{p['ci95_pp'][0]:+.1f}, {p['ci95_pp'][1]:+.1f}]  "
          f"p = {p['p_permutation']:.4f}  -> {p['direction']}; magnitude: {p['outcome']}  (n = {p['n_healthy']} healthy, {p['n_diseased']} diseased)")
    for k, v in sec.items():
        print(f"  {k:22s} {v['estimate_pp']:+6.1f} [{v['ci95_pp'][0]:+.1f}, {v['ci95_pp'][1]:+.1f}]  p_holm {v['p_holm']:.4f}  {v['outcome']}")
    print("written:", A_json)
    return res
