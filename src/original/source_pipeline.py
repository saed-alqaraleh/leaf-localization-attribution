#!/usr/bin/env python3
"""Source-only (PlantVillage) training pipeline for the paired crop-vs-full study.

Stages (restartable; each writes _STAGE_SUCCESS.json; reruns skip finished work):
  fetch     PlantVillage colour images at the pinned commit (reuses n3_pilot.stage_fetch)
  manifest  v2 manifest: leaf-map groups + byte-identical merge + perceptual-hash (pHash)
            grouping of images without a leaf-map id, then a grouped stratified split
  cnn       fine-tune a CNN (per model x seed), select checkpoint by PlantVillage val accuracy,
            save val/test logits and metrics                      [M1 MobileNetV3-L, M3 ConvNeXt-T]
  probe     frozen backbone features (CLS token) + multinomial logistic regression, C chosen on
            PlantVillage val, save probe weights, val/test logits and metrics  [M2 DINOv2-S, M4 DINOv2-B]
  summary   collect all source metrics into source_summary.json

No PlantDoc or FieldPlant image is read by this file.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from n3_pilot import (LEAFMAP_ALIASES, PV_COMMIT, atomic_json, done, leaf_key, log, mark,  # noqa: E402
                      stage_fetch)

CNN_MODELS = {"M1_mobilenetv3_large": "mobilenetv3_large_100.ra_in1k",
              "M3_convnext_tiny": "convnext_tiny.fb_in1k"}
PROBE_MODELS = {"M2_dinov2_small": "vit_small_patch14_dinov2.lvd142m",
                "M4_dinov2_base": "vit_base_patch14_dinov2.lvd142m"}
SEEDS = [20260928, 20260929, 20260930]
SPLIT_SEED = 20260928
PHASH_MAX_HAMMING = 6          # fixed before inspecting the data; near-duplicates only
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
PROBE_C_GRID = [0.01, 0.1, 1.0, 10.0]
IMG = 224
PRETRAINED = True   # set False only by --random-init (smoke tests without internet)


# ------------------------------------------------------------------ pHash
def phash64(path: Path) -> int:
    """Standard DCT perceptual hash: 32x32 greyscale, 2-D DCT-II, 8x8 low band, median threshold."""
    import numpy as np
    from PIL import Image
    from scipy.fft import dct
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    d = dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
    bits = (d > np.median(d)).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _popcount64(x):
    import numpy as np
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


# ------------------------------------------------------------------ manifest v2
def stage_manifest_v2(repo: Path, out: Path, seed: int = SPLIT_SEED, fractions=(0.70, 0.15, 0.15),
                      max_ham: int = PHASH_MAX_HAMMING) -> Path:
    import numpy as np
    sdir = out / "manifest_v2"
    sdir.mkdir(parents=True, exist_ok=True)
    mpath = sdir / "plantvillage_manifest_v2.csv"
    if done(sdir) and mpath.exists():
        log("manifest", f"already complete: {mpath}")
        return mpath
    leafmap = json.loads((repo / "leaf-map.json").read_text())
    files = sorted(p for p in (repo / "raw" / "color").rglob("*") if p.is_file())
    rows, t0 = [], time.time()
    for i, p in enumerate(files, 1):
        cls = p.parent.name
        ids = [x for x in leafmap.get(leaf_key(p.name), [])
               if LEAFMAP_ALIASES.get(x.split(":::", 1)[0], x.split(":::", 1)[0]) == cls]
        rows.append({"relpath": str(p.relative_to(repo)), "class": cls,
                     "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                     "phash": f"{phash64(p):016x}",
                     "group": f"leaf:{ids[0]}" if ids else f"single:{cls}/{p.name}",
                     "group_source": "leafmap" if ids else "singleton"})
        if i % 5000 == 0 or i == len(files):
            log("manifest", f"hashed {i}/{len(files)} ({time.time()-t0:.0f}s)")

    parent: dict[str, str] = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # 1) byte-identical files share a group
    first: dict[str, str] = {}
    for r in rows:
        find(r["group"]); union(first.setdefault(r["sha256"], r["group"]), r["group"])
    # 2) pHash near-duplicates among images WITHOUT a leaf-map id, within the same class
    phash_pairs = 0
    by_cls: dict[str, list] = {}
    for r in rows:
        if r["group_source"] == "singleton":
            by_cls.setdefault(r["class"], []).append(r)
    for cls, recs in sorted(by_cls.items()):
        h = np.array([int(r["phash"], 16) for r in recs], dtype=np.uint64)
        for i in range(len(recs) - 1):
            dist = _popcount64(h[i] ^ h[i + 1:])
            for j in np.nonzero(dist <= max_ham)[0]:
                union(recs[i]["group"], recs[i + 1 + int(j)]["group"]); phash_pairs += 1
    for r in rows:
        root = find(r["group"])
        if root != r["group"]:
            r["group"] = root
            r["group_source"] += "+merged"
    cls_by_hash: dict[str, set] = {}
    for r in rows:
        cls_by_hash.setdefault(r["sha256"], set()).add(r["class"])
    for r in rows:
        r["label_conflict"] = int(len(cls_by_hash[r["sha256"]]) > 1)

    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    gcls: dict[str, list] = {}
    for g, recs in groups.items():
        maj = max(sorted({x["class"] for x in recs}), key=lambda c: sum(x["class"] == c for x in recs))
        gcls.setdefault(maj, []).append(g)
    rng = random.Random(seed)
    for cls in sorted(gcls):
        keys = sorted(gcls[cls]); rng.shuffle(keys)
        n, acc = sum(len(groups[k]) for k in keys), 0
        for k in keys:
            frac = (acc + len(groups[k]) / 2) / n
            split = "train" if frac < fractions[0] else ("val" if frac < fractions[0] + fractions[1] else "test")
            for r in groups[k]:
                r["split"] = split
            acc += len(groups[k])
    seen: dict[str, str] = {}
    for r in rows:
        if seen.setdefault(r["group"], r["split"]) != r["split"]:
            raise RuntimeError(f"group {r['group']} spans splits")
    tmp = mpath.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    os.replace(tmp, mpath)

    sizes_by_cls = {}
    for cls in sorted(by_cls):
        gs = [len(v) for k, v in groups.items() if v[0]["class"] == cls]
        sizes_by_cls[cls] = {"n_images": sum(gs), "n_groups": len(gs), "largest_group": max(gs)}
    summary = {
        "n_images": len(rows), "n_groups": len(groups),
        "group_source_counts": {s: sum(r["group_source"] == s for r in rows) for s in sorted({r["group_source"] for r in rows})},
        "phash_max_hamming": max_ham, "phash_pairs_merged": phash_pairs,
        "phash_classes_group_sizes": sizes_by_cls,
        "n_label_conflict_files": sum(r["label_conflict"] for r in rows),
        "n_multiclass_groups": sum(1 for g in groups.values() if len({x["class"] for x in g}) > 1),
        "split_counts": {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")},
        "n_classes": len({r["class"] for r in rows}), "seed": seed, "commit": PV_COMMIT,
        "manifest_sha256": hashlib.sha256(mpath.read_bytes()).hexdigest(),
    }
    atomic_json(sdir / "manifest_summary.json", summary)
    log("manifest", json.dumps({k: v for k, v in summary.items() if k != "phash_classes_group_sizes"}))
    mark(sdir, {"manifest_sha256": summary["manifest_sha256"]})
    return mpath


# ------------------------------------------------------------------ shared data utils
def load_manifest(mpath: Path, limit_per_split: int | None, seed: int):
    rows = list(csv.DictReader(open(mpath)))
    classes = sorted({r["class"] for r in rows})
    if limit_per_split:   # smoke tests only: stratified-ish subsample
        rng = random.Random(seed)
        out = []
        for s in ("train", "val", "test"):
            sub = [r for r in rows if r["split"] == s]
            rng.shuffle(sub); out += sub[:limit_per_split]
        rows = out
    return rows, classes


def eval_transform():
    from torchvision import transforms as T
    # Protocol v0 §4: direct resize to 224x224 (no centre crop), ImageNet normalisation.
    return T.Compose([T.Resize((IMG, IMG), antialias=True), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def train_transform():
    from torchvision import transforms as T
    return T.Compose([T.RandomResizedCrop(IMG, scale=(0.6, 1.0), antialias=True), T.RandomHorizontalFlip(),
                      T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def make_dataset(repo: Path, recs, classes, tf):
    from PIL import Image
    from torch.utils.data import Dataset
    cidx = {c: i for i, c in enumerate(classes)}

    class DS(Dataset):
        def __len__(self):
            return len(recs)
        def __getitem__(self, i):
            with Image.open(repo / recs[i]["relpath"]) as im:
                return tf(im.convert("RGB")), cidx[recs[i]["class"]]
    return DS()


def metrics(logits, y, n_classes):
    import numpy as np
    pred = logits.argmax(1)
    acc = float((pred == y).mean())
    f1s = []
    for c in range(n_classes):
        tp = int(((pred == c) & (y == c)).sum()); fp = int(((pred == c) & (y != c)).sum()); fn = int(((pred != c) & (y == c)).sum())
        if tp + fn == 0:
            continue
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return {"n": int(len(y)), "accuracy": round(acc, 5), "macro_f1": round(float(np.mean(f1s)), 5)}


def device(allow_cpu: bool):
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu:
        return torch.device("cpu")
    raise RuntimeError("No GPU visible. In Colab: Runtime > Change runtime type > GPU.")


def env_info():
    import torch, timm, torchvision
    return {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
            "timm": timm.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}


# ------------------------------------------------------------------ CNN fine-tuning
def stage_cnn(repo, mpath, out, key, seed, epochs, batch, workers, limit, allow_cpu):
    import numpy as np
    import torch, timm
    from torch.utils.data import DataLoader
    sdir = out / "cnn" / f"{key}_seed{seed}"
    sdir.mkdir(parents=True, exist_ok=True)
    if done(sdir):
        log(key, f"seed {seed} already complete"); return
    random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed)
    dev = device(allow_cpu); amp = dev.type == "cuda"
    rows, classes = load_manifest(mpath, limit, seed)
    tr = [r for r in rows if r["split"] == "train" and r["label_conflict"] == "0"]
    va = [r for r in rows if r["split"] == "val"]; te = [r for r in rows if r["split"] == "test"]
    g = torch.Generator(); g.manual_seed(seed)
    dl_tr = DataLoader(make_dataset(repo, tr, classes, train_transform()), batch_size=batch, shuffle=True,
                       num_workers=workers, pin_memory=amp, drop_last=True, generator=g, persistent_workers=workers > 0)
    model = timm.create_model(CNN_MODELS[key], pretrained=PRETRAINED, num_classes=len(classes)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    steps_ep = len(dl_tr); total = epochs * steps_ep; warm = steps_ep
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    crit = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    start, best, hist = 0, -1.0, []
    last = sdir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"]); start, best, hist = ck["epoch"] + 1, ck["best"], ck["hist"]
        torch.set_rng_state(ck["rng_cpu"]); g.set_state(ck["rng_gen"])
        log(key, f"resumed at epoch {start}")

    def predict(recs):
        dl = DataLoader(make_dataset(repo, recs, classes, eval_transform()), batch_size=batch * 2,
                        num_workers=workers, pin_memory=amp)
        model.eval(); outs, ys = [], []
        with torch.inference_mode(), torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
            for x, y in dl:
                outs.append(model(x.to(dev, non_blocking=True)).float().cpu()); ys.append(y)
        return torch.cat(outs).numpy(), torch.cat(ys).numpy()

    for ep in range(start, epochs):
        model.train(); t0 = time.time(); tot = 0.0
        for x, y in dl_tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        lv, yv = predict(va); mv = metrics(lv, yv, len(classes))
        hist.append({"epoch": ep, "train_loss": round(tot / steps_ep, 4), "val": mv, "sec": round(time.time() - t0, 1)})
        log(key, f"seed {seed} epoch {ep+1}/{epochs} loss {tot/steps_ep:.4f} val_acc {mv['accuracy']:.4f} ({time.time()-t0:.0f}s)")
        if mv["accuracy"] > best:
            best = mv["accuracy"]
            torch.save({"model": model.state_dict(), "epoch": ep, "classes": classes}, sdir / "best.tmp")
            os.replace(sdir / "best.tmp", sdir / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep, "best": best, "hist": hist,
                    "rng_cpu": torch.get_rng_state(), "rng_gen": g.get_state()}, sdir / "last.tmp")
        os.replace(sdir / "last.tmp", last)

    ck = torch.load(sdir / "best.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    lv, yv = predict(va); lt, yt = predict(te)
    np.savez_compressed(sdir / "source_logits.npz", val_logits=lv, val_y=yv, test_logits=lt, test_y=yt,
                        val_relpath=np.array([r["relpath"] for r in va]), test_relpath=np.array([r["relpath"] for r in te]))
    res = {"model_key": key, "timm_name": CNN_MODELS[key], "seed": seed, "epochs": epochs, "best_epoch": ck["epoch"],
           "val": metrics(lv, yv, len(classes)), "test": metrics(lt, yt, len(classes)), "history": hist,
           "classes": classes, "env": env_info(), "limit_per_split": limit,
           "best_pt_sha256": hashlib.sha256((sdir / "best.pt").read_bytes()).hexdigest()}
    atomic_json(sdir / "result.json", res)
    last.unlink(missing_ok=True)   # keep best.pt only (saves Drive space)
    mark(sdir, {"val_acc": res["val"]["accuracy"], "test_acc": res["test"]["accuracy"]})


# ------------------------------------------------------------------ frozen features + linear probe
def extract_features(repo, recs, classes, name, batch, workers, dev):
    import numpy as np
    import torch, timm
    from torch.utils.data import DataLoader
    amp = dev.type == "cuda"
    fx = timm.create_model(name, pretrained=PRETRAINED, num_classes=0, img_size=IMG).to(dev).eval()
    dl = DataLoader(make_dataset(repo, recs, classes, eval_transform()), batch_size=batch,
                    num_workers=workers, pin_memory=amp)
    feats, ys = [], []
    with torch.inference_mode(), torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
        for x, y in dl:
            feats.append(fx(x.to(dev, non_blocking=True)).float().cpu()); ys.append(y)
    return torch.cat(feats).numpy().astype(np.float32), torch.cat(ys).numpy()


def fit_probe(Xtr, ytr, Xva, yva, n_classes, seed):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    z = lambda X: (X - mu) / sd
    trials = []
    for C in PROBE_C_GRID:
        clf = LogisticRegression(C=C, max_iter=3000, tol=1e-4, random_state=seed)
        clf.fit(z(Xtr), ytr)
        lv = full_logits(clf, z(Xva), n_classes)
        trials.append((metrics(lv, yva, n_classes)["accuracy"], -C, C, clf))  # ties -> smaller C
    trials.sort(key=lambda t: (t[0], t[1]), reverse=True)
    acc, _, C, clf = trials[0]
    return {"C": C, "mean": mu, "std": sd, "clf": clf,
            "grid": [{"C": t[2], "val_acc": t[0]} for t in sorted(trials, key=lambda t: t[2])]}


def full_logits(clf, Z, n_classes):
    """Decision function mapped onto all classes (a class absent from training gets -inf)."""
    import numpy as np
    d = clf.decision_function(Z)
    if d.ndim == 1:
        d = np.stack([-d, d], 1)
    out = np.full((Z.shape[0], n_classes), -np.inf, dtype=np.float64)
    out[:, clf.classes_] = d
    return out


def stage_probe(repo, mpath, out, key, batch, workers, limit, allow_cpu):
    import numpy as np
    sdir = out / "probe" / key
    sdir.mkdir(parents=True, exist_ok=True)
    if done(sdir):
        log(key, "already complete"); return
    dev = device(allow_cpu)
    rows, classes = load_manifest(mpath, limit, SPLIT_SEED)
    parts = {s: [r for r in rows if r["split"] == s and (s != "train" or r["label_conflict"] == "0")]
             for s in ("train", "val", "test")}
    fpath = sdir / "features.npz"
    if fpath.exists():
        F = dict(np.load(fpath))
    else:
        F, t0 = {}, time.time()
        for s, recs in parts.items():
            F[f"{s}_X"], F[f"{s}_y"] = extract_features(repo, recs, classes, PROBE_MODELS[key], batch, workers, dev)
            log(key, f"features {s}: {F[f'{s}_X'].shape} ({time.time()-t0:.0f}s)")
        np.savez(sdir / "features.tmp.npz", **F); os.replace(sdir / "features.tmp.npz", fpath)
    p = fit_probe(F["train_X"], F["train_y"], F["val_X"], F["val_y"], len(classes), SPLIT_SEED)
    z = lambda X: (X - p["mean"]) / p["std"]
    lv = full_logits(p["clf"], z(F["val_X"]), len(classes)); lt = full_logits(p["clf"], z(F["test_X"]), len(classes))
    np.savez_compressed(sdir / "probe.npz", mean=p["mean"], std=p["std"], coef=p["clf"].coef_,
                        intercept=p["clf"].intercept_, classes_idx=p["clf"].classes_, C=p["C"])
    np.savez_compressed(sdir / "source_logits.npz", val_logits=lv, val_y=F["val_y"], test_logits=lt, test_y=F["test_y"],
                        val_relpath=np.array([r["relpath"] for r in parts["val"]]),
                        test_relpath=np.array([r["relpath"] for r in parts["test"]]))
    res = {"model_key": key, "timm_name": PROBE_MODELS[key], "feature_dim": int(F["train_X"].shape[1]),
           "C_selected": p["C"], "C_grid": p["grid"], "val": metrics(lv, F["val_y"], len(classes)),
           "test": metrics(lt, F["test_y"], len(classes)), "classes": classes, "env": env_info(),
           "limit_per_split": limit, "probe_npz_sha256": hashlib.sha256((sdir / "probe.npz").read_bytes()).hexdigest()}
    atomic_json(sdir / "result.json", res)
    fpath.unlink()   # features are re-creatable; keep Drive small
    mark(sdir, {"val_acc": res["val"]["accuracy"], "test_acc": res["test"]["accuracy"], "C": p["C"]})


def stage_summary(out):
    res = {}
    for f in sorted(out.glob("cnn/*/result.json")) + sorted(out.glob("probe/*/result.json")):
        r = json.loads(f.read_text())
        res[f.parent.name] = {"val": r["val"], "test": r["test"], **({"best_epoch": r["best_epoch"]} if "best_epoch" in r else {"C": r["C_selected"]})}
    atomic_json(out / "source_summary.json", res)
    log("summary", json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stages", default="fetch,manifest,cnn,probe,summary")
    ap.add_argument("--cnn-models", default=",".join(CNN_MODELS))
    ap.add_argument("--probe-models", default=",".join(PROBE_MODELS))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2))
    ap.add_argument("--limit-per-split", type=int, default=None, help="SMOKE TEST ONLY")
    ap.add_argument("--allow-cpu", action="store_true", help="SMOKE TEST ONLY")
    ap.add_argument("--random-init", action="store_true", help="SMOKE TEST ONLY: no pretrained weights")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True); a.data_root.mkdir(parents=True, exist_ok=True)
    st = a.stages.split(",")
    global PRETRAINED
    PRETRAINED = not a.random_init
    atomic_json(a.out / f"run_config_{int(time.time())}.json",
                {**{k: str(v) for k, v in vars(a).items()}, "pv_commit": PV_COMMIT, "cnn_models": CNN_MODELS,
                 "probe_models": PROBE_MODELS, "pretrained": not a.random_init, "phash_max_hamming": PHASH_MAX_HAMMING, "argv": sys.argv})
    repo = stage_fetch(a.data_root) if "fetch" in st else a.data_root / "PlantVillage-Dataset"
    mpath = stage_manifest_v2(repo, a.out) if "manifest" in st else a.out / "manifest_v2" / "plantvillage_manifest_v2.csv"
    if "cnn" in st:
        for key in [k for k in a.cnn_models.split(",") if k]:
            for seed in map(int, a.seeds.split(",")):
                stage_cnn(repo, mpath, a.out, key, seed, a.epochs, a.batch, a.workers, a.limit_per_split, a.allow_cpu)
    if "probe" in st:
        for key in [k for k in a.probe_models.split(",") if k]:
            stage_probe(repo, mpath, a.out, key, a.batch, a.workers, a.limit_per_split, a.allow_cpu)
    if "summary" in st:
        stage_summary(a.out)
    log("done", f"outputs in {a.out}")


if __name__ == "__main__":
    main()
