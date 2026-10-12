#!/usr/bin/env python3
"""Field evaluation for the paired crop-vs-full study (protocol v0 §2-§8).

Stages (restartable; each writes _STAGE_SUCCESS.json):
  fetch    PlantDoc detection repository at the pinned commit (FieldPlant is supplied as an extracted folder)
  cohorts  frozen target list: one parent photo + one target box per row, cohort flags, source-file hash check
  views    for every target: F (full), C (crop, 10% margin), M (full with outside-box masked) x
           {resize (primary), letterbox (sensitivity)} -> 224x224 PNG; hashes in views_manifest.csv
  freeze   record SHA-256 of protocol, code, targets, views and all source checkpoints -> PROTOCOL_FROZEN.json
  infer    SEALED: verifies PROTOCOL_FROZEN.json, then computes 38-class logits for every model/seed/view.
           Refuses to run if anything hashed at freeze time has changed. Never overwrites existing logits.
  analyze  all prespecified estimands (see analysis.py), from saved logits only

No image is passed through a model before `infer`, and `infer` cannot run before `freeze`.
EXIF orientation is NOT applied: PlantDoc boxes are defined on raw pixels (checked visually), FieldPlant
had EXIF stripped by Roboflow, and PlantVillage training read raw pixels as well.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PD_REPO = "https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset"
PD_COMMIT = "4730a233a555b30ee98e0879c63ad25d82407455"
IMG = 224
MARGIN = 0.10
MEAN_RGB = (124, 116, 104)          # ImageNet mean x 255, rounded
VIEWS = ("F", "C", "M")
POLICIES = ("resize", "letterbox")
CNN_KEYS = ("M1_mobilenetv3_large", "M3_convnext_tiny")
PROBE_KEYS = ("M2_dinov2_small", "M4_dinov2_base")
TIMM = {"M1_mobilenetv3_large": "mobilenetv3_large_100.ra_in1k", "M3_convnext_tiny": "convnext_tiny.fb_in1k",
        "M2_dinov2_small": "vit_small_patch14_dinov2.lvd142m", "M4_dinov2_base": "vit_base_patch14_dinov2.lvd142m"}
SEEDS = (20260928, 20260929, 20260930)
MIN_PER_CLASS = 30


# ------------------------------------------------------------------ helpers
def log(stage, msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [{stage}] {msg}", flush=True)


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(p: Path, obj):
    t = p.with_suffix(p.suffix + ".tmp"); t.write_text(json.dumps(obj, indent=1, default=str)); os.replace(t, p)


def atomic_csv(p: Path, rows):
    t = p.with_suffix(p.suffix + ".tmp")
    with open(t, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    os.replace(t, p)


def done(d: Path) -> bool:
    return (d / "_STAGE_SUCCESS.json").exists()


def mark(d: Path, info: dict):
    atomic_json(d / "_STAGE_SUCCESS.json", {**info, "finished_utc": datetime.now(timezone.utc).isoformat()})


# ------------------------------------------------------------------ fetch
def stage_fetch(data_root: Path) -> Path:
    repo = data_root / "PlantDoc-Object-Detection-Dataset"
    if (repo / ".git").exists() and subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() == PD_COMMIT:
        log("fetch", f"PlantDoc already at {PD_COMMIT[:8]}"); return repo
    data_root.mkdir(parents=True, exist_ok=True)
    if not (repo / ".git").exists():
        subprocess.run(["git", "clone", "-q", PD_REPO, str(repo)], check=True)
    subprocess.run(["git", "checkout", "-q", PD_COMMIT], cwd=repo, check=True)
    log("fetch", f"PlantDoc at {PD_COMMIT[:8]}")
    return repo


# ------------------------------------------------------------------ cohorts
def load_decisions(p: Path, allow_pending: bool) -> dict:
    d = json.loads(p.read_text())
    pend = [k for k, v in d["approximate_mappings"].items() if v not in (True, False)]
    if pend and not allow_pending:
        raise SystemExit(f"pathologist decisions pending for: {pend}. Fill mapping_decisions.json (true/false).")
    return d


def stage_cohorts(pd_repo: Path, fp_root: Path | None, inputs: Path, out: Path, decisions: dict) -> Path:
    sdir = out / "cohorts"; sdir.mkdir(parents=True, exist_ok=True)
    tpath = sdir / "targets.csv"
    if done(sdir) and tpath.exists():
        log("cohorts", f"already complete: {tpath}"); return tpath
    appr = {k for k, v in decisions["approximate_mappings"].items() if v is True}
    rows = []
    # PlantDoc
    cw = {r["plantdoc_label"]: r for r in csv.DictReader(open(inputs / "plantdoc_to_plantvillage_crosswalk_DRAFT.csv"))}
    sparse = set(decisions["plantdoc_sparse_excluded"])
    approx_pd = set(decisions["plantdoc_approximate_labels"])
    for r in csv.DictReader(open(inputs / "plantdoc_eligible_parents_DRAFT.csv")):
        lab = r["labels"]
        if lab not in cw or lab in sparse:
            continue
        status = "approximate" if lab in approx_pd else "exact"
        rows.append({"dataset": "plantdoc", "relpath": f"{r['split'].upper()}/{r['file']}", "sha256": r["sha256"],
                     "label": lab, "pv_class": cw[lab]["plantvillage_class"], "mapping": status,
                     "width": int(r["w"]), "height": int(r["h"]),
                     "x1": float(r["box_x1"]), "y1": float(r["box_y1"]), "x2": float(r["box_x2"]), "y2": float(r["box_y2"]),
                     "box_frac": float(r["box_frac"]),
                     "primary": int(status == "exact"),
                     "sensitivity": int(status == "exact" or lab in appr)})
    # FieldPlant
    if fp_root is not None:
        fmap = decisions["fieldplant_map"]
        fp = [r for r in csv.DictReader(open(inputs / "fieldplant_eligible_parents_DRAFT.csv")) if r["label"] in fmap]
        ok_lab = {l for l, m in fmap.items() if m["mapping"] == "exact" or l in appr}
        counts = {}
        for r in fp:
            if r["label"] in ok_lab:
                counts[r["label"]] = counts.get(r["label"], 0) + 1
        rep = {l for l, n in counts.items() if n >= MIN_PER_CLASS}
        exact_rep = {l for l in rep if fmap[l]["mapping"] == "exact"}
        for r in fp:
            lab = r["label"]
            rows.append({"dataset": "fieldplant", "relpath": r["file_name"], "sha256": r["sha256"], "label": lab,
                         "pv_class": fmap[lab]["pv_class"], "mapping": fmap[lab]["mapping"],
                         "width": int(r["width"]), "height": int(r["height"]),
                         "x1": float(r["box_x1"]), "y1": float(r["box_y1"]), "x2": float(r["box_x2"]), "y2": float(r["box_y2"]),
                         "box_frac": float(r["box_frac"]),
                         "primary": int(lab in rep), "sensitivity": int(lab in exact_rep)})
    rows = [r for r in rows if r["primary"] or r["sensitivity"]]   # unused rows (rejected mappings) are not evaluated
    rows.sort(key=lambda r: (r["dataset"], r["relpath"]))
    for i, r in enumerate(rows):
        r["target_id"] = f"{'PD' if r['dataset']=='plantdoc' else 'FP'}{i:05d}"
    # verify the image files are exactly the inventoried bytes
    bad = []
    for i, r in enumerate(rows, 1):
        root = pd_repo if r["dataset"] == "plantdoc" else fp_root
        p = root / r["relpath"]
        if not p.exists() or sha_file(p) != r["sha256"]:
            bad.append(r["relpath"])
        if i % 500 == 0:
            log("cohorts", f"hash-checked {i}/{len(rows)}")
    if bad:
        raise RuntimeError(f"{len(bad)} target files missing or changed, e.g. {bad[:3]}")
    rows = [{"target_id": r.pop("target_id"), **r} for r in rows]
    atomic_csv(tpath, rows)
    summ = {}
    for ds in ("plantdoc", "fieldplant"):
        for flag in ("primary", "sensitivity"):
            sel = [r for r in rows if r["dataset"] == ds and r[flag] == 1]
            if sel:
                per = {}
                for r in sel:
                    per[r["label"]] = per.get(r["label"], 0) + 1
                summ[f"{ds}_{flag}"] = {"n": len(sel), "n_labels": len(per), "per_label": dict(sorted(per.items()))}
    summ["approved_approximations"] = sorted(appr)
    summ["targets_sha256"] = sha_file(tpath)
    atomic_json(sdir / "cohort_summary.json", summ)
    log("cohorts", json.dumps({k: (v["n"] if isinstance(v, dict) and "n" in v else v) for k, v in summ.items()}))
    mark(sdir, {"targets_sha256": summ["targets_sha256"], "n_targets": len(rows)})
    return tpath


# ------------------------------------------------------------------ views
def make_views(im, box):
    """im: PIL RGB (raw orientation). box: (x1,y1,x2,y2) in pixels. Returns dict (view, policy) -> 224x224 PIL."""
    from PIL import Image, ImageDraw
    W, H = im.size
    x1, y1, x2, y2 = box
    mx, my = MARGIN * (x2 - x1), MARGIN * (y2 - y1)
    ex = (max(0, int(round(x1 - mx))), max(0, int(round(y1 - my))),
          min(W, int(round(x2 + mx))), min(H, int(round(y2 + my))))
    if ex[2] - ex[0] < 2 or ex[3] - ex[1] < 2:
        raise ValueError(f"degenerate enlarged box {ex}")
    full = im
    crop = im.crop(ex)
    masked = Image.new("RGB", (W, H), MEAN_RGB); masked.paste(im.crop(ex), (ex[0], ex[1]))
    out = {}
    for v, x in (("F", full), ("C", crop), ("M", masked)):
        out[(v, "resize")] = x.resize((IMG, IMG), Image.Resampling.BILINEAR)
        s = max(x.size)
        sq = Image.new("RGB", (s, s), MEAN_RGB); sq.paste(x, ((s - x.size[0]) // 2, (s - x.size[1]) // 2))
        out[(v, "letterbox")] = sq.resize((IMG, IMG), Image.Resampling.BILINEAR)
    return out, ex


def png_bytes(im) -> bytes:
    b = io.BytesIO(); im.save(b, format="PNG", optimize=False, compress_level=6); return b.getvalue()


def stage_views(pd_repo: Path, fp_root: Path | None, tpath: Path, out: Path) -> Path:
    import numpy as np
    from PIL import Image
    sdir = out / "views"; sdir.mkdir(parents=True, exist_ok=True)
    vpath = sdir / "views_manifest.csv"
    if done(sdir) and vpath.exists():
        log("views", f"already complete: {vpath}"); return vpath
    targets = list(csv.DictReader(open(tpath)))
    part = sdir / "views_manifest.partial.csv"
    have = {}
    if part.exists():
        for r in csv.DictReader(open(part)):
            have[(r["target_id"], r["view"], r["policy"])] = r
    fields = ["target_id", "view", "policy", "png", "png_sha256", "pixels_sha256", "ex_x1", "ex_y1", "ex_x2", "ex_y2", "src_w", "src_h"]
    new_file = not part.exists()
    f = open(part, "a", newline=""); w = csv.DictWriter(f, fieldnames=fields)
    if new_file:
        w.writeheader()
    t0 = time.time()
    for i, t in enumerate(targets, 1):
        if all((t["target_id"], v, p) in have for v in VIEWS for p in POLICIES):
            continue
        if t["dataset"] == "fieldplant" and fp_root is None:
            raise SystemExit("FieldPlant targets present but --fieldplant-root not given")
        root = pd_repo if t["dataset"] == "plantdoc" else fp_root
        with Image.open(root / t["relpath"]) as im0:
            im = im0.convert("RGB")          # raw orientation; EXIF intentionally not applied
        if im.size != (int(t["width"]), int(t["height"])):
            raise RuntimeError(f"{t['relpath']}: size {im.size} != manifest {(t['width'], t['height'])}")
        views, ex = make_views(im, tuple(float(t[k]) for k in ("x1", "y1", "x2", "y2")))
        for (v, p), vim in views.items():
            rel = Path(t["dataset"]) / p / v / f"{t['target_id']}.png"
            dst = sdir / rel; dst.parent.mkdir(parents=True, exist_ok=True)
            b = png_bytes(vim)
            tmp = dst.with_suffix(".tmp"); tmp.write_bytes(b); os.replace(tmp, dst)
            w.writerow({"target_id": t["target_id"], "view": v, "policy": p, "png": str(rel),
                        "png_sha256": hashlib.sha256(b).hexdigest(),
                        "pixels_sha256": hashlib.sha256(np.asarray(vim, dtype=np.uint8).tobytes()).hexdigest(),
                        "ex_x1": ex[0], "ex_y1": ex[1], "ex_x2": ex[2], "ex_y2": ex[3], "src_w": im.size[0], "src_h": im.size[1]})
        f.flush()
        if i % 100 == 0 or i == len(targets):
            log("views", f"{i}/{len(targets)} targets ({time.time()-t0:.0f}s)")
    f.close()
    rows = list(csv.DictReader(open(part)))
    keyset = {(r["target_id"], r["view"], r["policy"]) for r in rows}
    need = {(t["target_id"], v, p) for t in targets for v in VIEWS for p in POLICIES}
    if need - keyset:
        raise RuntimeError(f"{len(need - keyset)} views missing")
    # keep the last row per key (a crash can leave a duplicate); order deterministically
    last = {}
    for r in rows:
        last[(r["target_id"], r["view"], r["policy"])] = r
    rows = [last[k] for k in sorted(need)]
    for r in rows:   # verify files on disk
        if sha_file(sdir / r["png"]) != r["png_sha256"]:
            raise RuntimeError(f"hash mismatch {r['png']}")
    atomic_csv(vpath, rows)
    mark(sdir, {"views_manifest_sha256": sha_file(vpath), "n_views": len(rows)})
    log("views", f"{len(rows)} views written and verified")
    return vpath


# ------------------------------------------------------------------ freeze
def checkpoint_files(src: Path) -> list[Path]:
    fs = []
    for k in CNN_KEYS:
        for s in SEEDS:
            fs += [src / "cnn" / f"{k}_seed{s}" / "best.pt", src / "cnn" / f"{k}_seed{s}" / "result.json"]
    for k in PROBE_KEYS:
        fs += [src / "probe" / k / "probe.npz", src / "probe" / k / "result.json"]
    fs.append(src / "manifest_v2" / "manifest_summary.json")
    return fs


def freeze_record(protocol: Path, tpath: Path, vpath: Path, src: Path, decisions_path: Path) -> dict:
    code = {p.name: sha_file(p) for p in (HERE / "field_eval.py", HERE / "analysis.py")}
    ck = {str(p.relative_to(src)): sha_file(p) for p in checkpoint_files(src)}
    return {"protocol": {"path": str(protocol), "sha256": sha_file(protocol)}, "code": code,
            "targets_sha256": sha_file(tpath), "views_manifest_sha256": sha_file(vpath),
            "mapping_decisions_sha256": sha_file(decisions_path), "checkpoints": ck}


def stage_freeze(protocol: Path, tpath: Path, vpath: Path, src: Path, decisions_path: Path, out: Path):
    fpath = out / "PROTOCOL_FROZEN.json"
    if fpath.exists():
        log("freeze", f"already frozen: {fpath} (not modified)"); return
    missing = [str(p) for p in checkpoint_files(src) if not p.exists()]
    if missing:
        raise SystemExit(f"missing source checkpoints: {missing[:4]}")
    rec = freeze_record(protocol, tpath, vpath, src, decisions_path)
    rec["frozen_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(fpath, rec)
    log("freeze", f"frozen: {fpath}\n  protocol sha256 {rec['protocol']['sha256']}")


# ------------------------------------------------------------------ infer (sealed)
def verify_frozen(protocol, tpath, vpath, src, decisions_path, out) -> dict:
    fpath = out / "PROTOCOL_FROZEN.json"
    if not fpath.exists():
        raise SystemExit("REFUSED: run the `freeze` stage first (after the protocol is approved).")
    frozen = json.loads(fpath.read_text())
    now = freeze_record(protocol, tpath, vpath, src, decisions_path)
    diffs = [k for k in ("protocol", "code", "targets_sha256", "views_manifest_sha256", "mapping_decisions_sha256", "checkpoints")
             if json.dumps(frozen[k], sort_keys=True) != json.dumps(now[k], sort_keys=True)]
    if diffs:
        raise SystemExit(f"REFUSED: files changed since freeze: {diffs}")
    return frozen


def stage_infer(protocol, tpath, vpath, src, decisions_path, out, batch, workers, allow_cpu, random_init=False):
    import numpy as np
    import torch, timm
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms as T
    if random_init and not allow_cpu:
        raise SystemExit("--random-init is for CPU tests only")
    frozen = verify_frozen(protocol, tpath, vpath, src, decisions_path, out)
    sdir = out / "logits"; sdir.mkdir(parents=True, exist_ok=True)
    if done(sdir):
        log("infer", "already complete (sealed run finished; nothing recomputed)"); return
    dev = torch.device("cuda") if torch.cuda.is_available() else (torch.device("cpu") if allow_cpu else None)
    if dev is None:
        raise SystemExit("No GPU visible. In Colab: Runtime > Change runtime type > GPU.")
    amp = dev.type == "cuda"
    vrows = list(csv.DictReader(open(vpath)))
    vdir = vpath.parent
    tf = T.Compose([T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    class DS(Dataset):
        def __len__(self):
            return len(vrows)
        def __getitem__(self, i):
            with Image.open(vdir / vrows[i]["png"]) as im:
                assert im.size == (IMG, IMG)
                return tf(im.convert("RGB"))
    dl = lambda: DataLoader(DS(), batch_size=batch, num_workers=workers, pin_memory=amp)
    ids = np.array([f"{r['target_id']}|{r['view']}|{r['policy']}" for r in vrows])
    classes = json.loads((src / "cnn" / f"{CNN_KEYS[0]}_seed{SEEDS[0]}" / "result.json").read_text())["classes"]
    jobs = [(k, s) for k in CNN_KEYS for s in SEEDS] + [(k, None) for k in PROBE_KEYS]
    for k, s in jobs:
        name = f"{k}_seed{s}" if s else k
        dst = sdir / f"{name}.npz"
        if dst.exists():
            log("infer", f"{name}: exists, kept"); continue
        t0 = time.time()
        if s:
            ck = torch.load(src / "cnn" / name / "best.pt", map_location="cpu", weights_only=False)
            assert ck["classes"] == classes
            m = timm.create_model(TIMM[k], pretrained=False, num_classes=len(classes))
            m.load_state_dict(ck["model"]); m = m.to(dev).eval()
            outs = []
            with torch.inference_mode(), torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                for x in dl():
                    outs.append(m(x.to(dev, non_blocking=True)).float().cpu())
            L = torch.cat(outs).numpy().astype(np.float64)
        else:
            pr = np.load(src / "probe" / k / "probe.npz")
            assert json.loads((src / "probe" / k / "result.json").read_text())["classes"] == classes
            m = timm.create_model(TIMM[k], pretrained=not random_init, num_classes=0, img_size=IMG).to(dev).eval()
            feats = []
            with torch.inference_mode(), torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                for x in dl():
                    feats.append(m(x.to(dev, non_blocking=True)).float().cpu())
            Z = (torch.cat(feats).numpy() - pr["mean"]) / pr["std"]
            d = Z @ pr["coef"].T + pr["intercept"]
            L = np.full((Z.shape[0], len(classes)), -np.inf); L[:, pr["classes_idx"]] = d
        if not np.isfinite(L[:, np.isfinite(L).any(0)]).all():
            raise RuntimeError(f"{name}: non-finite logits")
        tmp = sdir / f"{name}.tmp.npz"
        np.savez_compressed(tmp, logits=L, ids=ids, classes=np.array(classes))
        os.replace(tmp, dst)
        log("infer", f"{name}: {L.shape} in {time.time()-t0:.0f}s")
        del m
        if amp:
            torch.cuda.empty_cache()
    import torchvision
    rec = {"frozen": frozen, "random_init_TEST_ONLY": random_init, "env": {"python": platform.python_version(), "torch": torch.__version__,
           "torchvision": torchvision.__version__, "timm": timm.__version__,
           "gpu": torch.cuda.get_device_name(0) if amp else "cpu"},
           "logits_sha256": {p.name: sha_file(p) for p in sorted(sdir.glob("*.npz"))}}
    atomic_json(sdir / "sealed_run_record.json", rec)
    mark(sdir, {"n_files": len(rec["logits_sha256"])})


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True, help="local disk for the PlantDoc clone")
    ap.add_argument("--fieldplant-root", type=Path, default=None, help="extracted FieldPlant folder containing the images (…/train)")
    ap.add_argument("--inputs", type=Path, default=HERE / "inputs", help="frozen input tables + mapping_decisions.json")
    ap.add_argument("--source", type=Path, help="source_training output dir (checkpoints)")
    ap.add_argument("--protocol", type=Path, help="frozen protocol file")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stages", default="fetch,cohorts,views")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2))
    ap.add_argument("--allow-pending", action="store_true", help="DRAFT ONLY: build cohorts before pathologist decisions")
    ap.add_argument("--allow-cpu", action="store_true", help="TESTS ONLY")
    ap.add_argument("--random-init", action="store_true", help="TESTS ONLY (needs --allow-cpu): probe backbones without pretrained weights")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    st = a.stages.split(",")
    dpath = a.inputs / "mapping_decisions.json"
    atomic_json(a.out / f"run_config_{int(time.time())}.json", {**{k: str(v) for k, v in vars(a).items()}, "argv": sys.argv})
    pd_repo = stage_fetch(a.data_root) if "fetch" in st else a.data_root / "PlantDoc-Object-Detection-Dataset"
    tpath = a.out / "cohorts" / "targets.csv"
    if "cohorts" in st:
        if a.allow_pending and "freeze" in st:
            raise SystemExit("--allow-pending cannot be combined with freeze")
        tpath = stage_cohorts(pd_repo, a.fieldplant_root, a.inputs, a.out, load_decisions(dpath, a.allow_pending))
    vpath = a.out / "views" / "views_manifest.csv"
    if "views" in st:
        vpath = stage_views(pd_repo, a.fieldplant_root, tpath, a.out)
    if "freeze" in st:
        load_decisions(dpath, allow_pending=False)
        stage_freeze(a.protocol, tpath, vpath, a.source, dpath, a.out)
    if "infer" in st:
        stage_infer(a.protocol, tpath, vpath, a.source, dpath, a.out, a.batch, a.workers, a.allow_cpu, a.random_init)
    if "analyze" in st:
        if not done(a.out / "logits"):
            raise SystemExit("NOT YET: analysis needs the sealed logits. Run Stage A, B and C first (in that order).")
        sys.path.insert(0, str(HERE))
        import analysis
        analysis.run(a.out, tpath)
    log("done", f"outputs in {a.out}")


if __name__ == "__main__":
    main()
