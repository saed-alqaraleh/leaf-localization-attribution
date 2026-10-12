#!/usr/bin/env python3
"""Gate N3 timing pilot (source data only).

Stages (each writes a _STAGE_SUCCESS.json marker; reruns skip completed stages):
  fetch    - PlantVillage colour images at a pinned commit of spMohanty/PlantVillage-Dataset
  manifest - image manifest, SHA-256 per file, leaf-group IDs, grouped train/val/test split
  timing   - MobileNetV3-Large fine-tuning throughput and DINOv2-S/14 frozen feature throughput

No PlantDoc or FieldPlant data is read. Nothing here is a result for the paper; it only
measures compute so that the protocol can fix the number of seeds and optional models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PV_REPO = "https://github.com/spMohanty/PlantVillage-Dataset"
PV_COMMIT = "7f7ecc7e1eaca78107e3affe7cb5abd9427e139a"
# The source leaf map names apple black rot by its other common name (frogeye leaf spot).
LEAFMAP_ALIASES = {"Apple_Frogeye Spot": "Apple___Black_rot"}
MODELS = {
    "mobilenetv3_large": "mobilenetv3_large_100.ra_in1k",
    "dinov2_small": "vit_small_patch14_dinov2.lvd142m",
}


def log(stage: str, msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [{stage}] {msg}", flush=True)


def atomic_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def done(stage_dir: Path) -> bool:
    return (stage_dir / "_STAGE_SUCCESS.json").exists()


def mark(stage_dir: Path, info: dict) -> None:
    info = {**info, "finished_utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(stage_dir / "_STAGE_SUCCESS.json", info)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    log("sh", " ".join(cmd))
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(cmd)}")


# ---------------------------------------------------------------- fetch
def stage_fetch(data_root: Path) -> Path:
    """Sparse, blob-filtered checkout of raw/color and leaf-map.json at the pinned commit."""
    repo = data_root / "PlantVillage-Dataset"
    sdir = data_root / "_fetch"
    sdir.mkdir(parents=True, exist_ok=True)
    if done(sdir) and (repo / "raw" / "color").is_dir():
        log("fetch", f"already complete: {repo}")
        return repo
    if not (repo / ".git").exists():
        run(["git", "clone", "--filter=blob:none", "--no-checkout", PV_REPO, str(repo)])
    run(["git", "-c", "gc.auto=0", "sparse-checkout", "init", "--no-cone"], cwd=repo)
    run(["git", "-c", "gc.auto=0", "sparse-checkout", "set", "raw/color/", "leaf-map.json"], cwd=repo)
    run(["git", "-c", "gc.auto=0", "checkout", PV_COMMIT], cwd=repo)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if head != PV_COMMIT:
        raise RuntimeError(f"checked out {head}, expected {PV_COMMIT}")
    n = sum(1 for _ in (repo / "raw" / "color").rglob("*") if _.is_file())
    log("fetch", f"{n} colour image files at {head}")
    if n != 54305:
        raise RuntimeError(f"expected 54,305 colour images at the pinned commit, found {n}")
    mark(sdir, {"commit": head, "n_images": n})
    return repo


# ---------------------------------------------------------------- manifest
def leaf_key(filename: str) -> str:
    # Mirrors leaf_grouping/aggregate_map.py in the source repository.
    return filename.split("___", 1)[-1].rsplit(".", 1)[0].lower().strip()


def stage_manifest(repo: Path, out: Path, seed: int, fractions=(0.70, 0.15, 0.15)) -> Path:
    sdir = out / "manifest"
    sdir.mkdir(parents=True, exist_ok=True)
    mpath = sdir / "plantvillage_manifest.csv"
    if done(sdir) and mpath.exists():
        log("manifest", f"already complete: {mpath}")
        return mpath
    import csv

    leafmap = json.loads((repo / "leaf-map.json").read_text())
    rows = []
    files = sorted(p for p in (repo / "raw" / "color").rglob("*") if p.is_file())
    t0 = time.time()
    for i, p in enumerate(files, 1):
        cls = p.parent.name
        key = leaf_key(p.name)
        # Leaf-map keys are filename suffixes and can repeat across classes, so only accept
        # leaf ids whose class prefix matches this image's class; otherwise a singleton group.
        ids = [i for i in leafmap.get(key, []) if LEAFMAP_ALIASES.get(i.split(":::", 1)[0], i.split(":::", 1)[0]) == cls]
        group = f"leaf:{ids[0]}" if ids else f"single:{cls}/{p.name}"
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        rows.append({"relpath": str(p.relative_to(repo)), "class": cls, "sha256": h,
                     "group": group, "group_source": "leafmap" if ids else "singleton"})
        if i % 5000 == 0 or i == len(files):
            log("manifest", f"hashed {i}/{len(files)} ({time.time()-t0:.0f}s)")

    # Union-find over leaf-map groups and byte-identical files, so duplicates never span splits.
    parent: dict[str, str] = {}
    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    first_by_hash: dict[str, str] = {}
    for r in rows:
        find(r["group"])
        g0 = first_by_hash.setdefault(r["sha256"], r["group"])
        union(g0, r["group"])
    for r in rows:
        root = find(r["group"])
        if root != r["group"]:
            r["group"], r["group_source"] = root, r["group_source"] + "+merged"

    # Byte-identical files carrying different class labels are label conflicts: flag them.
    cls_by_hash: dict[str, set] = {}
    for r in rows:
        cls_by_hash.setdefault(r["sha256"], set()).add(r["class"])
    for r in rows:
        r["label_conflict"] = int(len(cls_by_hash[r["sha256"]]) > 1)

    # Grouped split, stratified by each group's majority class; groups shuffled deterministically.
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    by_cls: dict[str, list[str]] = {}
    for g, recs in groups.items():
        maj = max(sorted({x["class"] for x in recs}), key=lambda c: sum(x["class"] == c for x in recs))
        by_cls.setdefault(maj, []).append(g)
    rng = random.Random(seed)
    for cls in sorted(by_cls):
        keys = sorted(by_cls[cls])
        rng.shuffle(keys)
        n = sum(len(groups[k]) for k in keys)
        acc = 0
        for k in keys:
            frac = (acc + len(groups[k]) / 2) / n  # group midpoint keeps split sizes close to target
            split = "train" if frac < fractions[0] else ("val" if frac < fractions[0] + fractions[1] else "test")
            for r in groups[k]:
                r["split"] = split
            acc += len(groups[k])
    seen: dict[str, str] = {}
    for r in rows:
        if seen.setdefault(r["group"], r["split"]) != r["split"]:
            raise RuntimeError(f"group {r['group']} spans splits")
    with open(mpath.with_suffix(".tmp"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    os.replace(mpath.with_suffix(".tmp"), mpath)
    summary = {
        "n_images": len(rows),
        "n_groups": len({r["group"] for r in rows}),
        "group_source_counts": {s: sum(r["group_source"] == s for r in rows) for s in sorted({r["group_source"] for r in rows})},
        "n_label_conflict_files": sum(r["label_conflict"] for r in rows),
        "n_multiclass_groups": sum(1 for g in groups.values() if len({x["class"] for x in g}) > 1),
        "split_counts": {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")},
        "n_classes": len({r["class"] for r in rows}), "seed": seed, "commit": PV_COMMIT,
    }
    atomic_json(sdir / "manifest_summary.json", summary)
    log("manifest", json.dumps(summary))
    mark(sdir, summary)
    return mpath


# ---------------------------------------------------------------- timing
def stage_timing(repo: Path, mpath: Path, out: Path, train_steps: int, feat_images: int,
                 batch: int, workers: int, seed: int) -> None:
    sdir = out / "timing"
    sdir.mkdir(parents=True, exist_ok=True)
    if done(sdir):
        log("timing", f"already complete: {sdir/'timing.json'}")
        return
    import csv
    import torch
    import timm
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms as T

    torch.manual_seed(seed)
    random.seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU visible. In Colab: Runtime > Change runtime type > GPU.")
    dev = torch.device("cuda")
    rows = list(csv.DictReader(open(mpath)))
    classes = sorted({r["class"] for r in rows})
    cidx = {c: i for i, c in enumerate(classes)}
    train = [r for r in rows if r["split"] == "train" and r["label_conflict"] == "0"]

    class DS(Dataset):
        def __init__(self, recs, tf):
            self.recs, self.tf = recs, tf
        def __len__(self):
            return len(self.recs)
        def __getitem__(self, i):
            r = self.recs[i]
            im = Image.open(repo / r["relpath"]).convert("RGB")
            return self.tf(im), cidx[r["class"]]

    norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    tf_train = T.Compose([T.RandomResizedCrop(224, scale=(0.7, 1.0)), T.RandomHorizontalFlip(), T.ToTensor(), norm])
    tf_eval = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), norm])
    res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "timm": timm.__version__,
           "python": platform.python_version(), "batch": batch, "workers": workers,
           "n_train_images": len(train), "n_classes": len(classes)}

    # MobileNetV3-Large fine-tuning throughput (AMP).
    name = MODELS["mobilenetv3_large"]
    model = timm.create_model(name, pretrained=True, num_classes=len(classes)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda")
    dl = DataLoader(DS(train, tf_train), batch_size=batch, shuffle=True, num_workers=workers,
                    pin_memory=True, drop_last=True, persistent_workers=workers > 0)
    model.train()
    it = iter(dl)
    warm = 10
    for step in range(train_steps + warm):
        if step == warm:
            torch.cuda.synchronize(); t0 = time.time()
        x, y = next(it)
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = torch.nn.functional.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if (step + 1) % 25 == 0:
            log("timing", f"mobilenetv3 step {step+1}/{train_steps+warm} loss {loss.item():.3f}")
    torch.cuda.synchronize()
    sec = time.time() - t0
    ips = train_steps * batch / sec
    res["mobilenetv3_large"] = {"timm_name": name, "train_images_per_s": round(ips, 1),
                                "est_min_per_epoch": round(len(train) / ips / 60, 2),
                                "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    log("timing", json.dumps(res["mobilenetv3_large"]))
    del model, opt; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    # DINOv2-S/14 frozen feature extraction throughput at 224 px (FP16 inference).
    name = MODELS["dinov2_small"]
    fx = timm.create_model(name, pretrained=True, num_classes=0, img_size=224).to(dev).eval()
    sub = random.Random(seed).sample(rows, min(feat_images, len(rows)))
    dl = DataLoader(DS(sub, tf_eval), batch_size=batch, num_workers=workers, pin_memory=True)
    n = 0
    torch.cuda.synchronize(); t0 = time.time()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for x, _ in dl:
            f = fx(x.to(dev, non_blocking=True)); n += x.shape[0]
    torch.cuda.synchronize()
    sec = time.time() - t0
    ips = n / sec
    res["dinov2_small"] = {"timm_name": name, "feature_dim": int(f.shape[1]), "images_per_s": round(ips, 1),
                           "est_min_all_54305_images": round(54305 / ips / 60, 2),
                           "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    log("timing", json.dumps(res["dinov2_small"]))
    atomic_json(sdir / "timing.json", res)
    mark(sdir, {"timing": str(sdir / "timing.json")})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True, help="fast local disk for images, e.g. /content/data")
    ap.add_argument("--out", type=Path, required=True, help="persistent output dir, e.g. Drive")
    ap.add_argument("--stages", default="fetch,manifest,timing")
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--train-steps", type=int, default=150)
    ap.add_argument("--feat-images", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2))
    a = ap.parse_args()
    a.data_root.mkdir(parents=True, exist_ok=True)
    a.out.mkdir(parents=True, exist_ok=True)
    stages = a.stages.split(",")
    atomic_json(a.out / "run_config.json", {**{k: str(v) for k, v in vars(a).items()}, "pv_commit": PV_COMMIT,
                                            "models": MODELS, "argv": sys.argv})
    repo = stage_fetch(a.data_root) if "fetch" in stages else a.data_root / "PlantVillage-Dataset"
    mpath = stage_manifest(repo, a.out, a.seed) if "manifest" in stages else a.out / "manifest" / "plantvillage_manifest.csv"
    if "timing" in stages:
        stage_timing(repo, mpath, a.out, a.train_steps, a.feat_images, a.batch, a.workers, a.seed)
    log("done", f"outputs in {a.out}")


if __name__ == "__main__":
    main()
