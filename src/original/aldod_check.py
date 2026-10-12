#!/usr/bin/env python3
"""ALDOD single-crop secondary check (PROTOCOL_ALDOD_v1.md). Restartable stages:

  cohort   eligibility cascade on the Pascal-VOC copy of ALDOD -> cohort/targets.csv + cohort/cascade.json
           (single-label photos, one per pHash group, label-conflict groups and reference matches removed,
           target = largest box of the label, primary cohort = box fraction <= 0.50)
  views    F and C views x {resize, letterbox}, rendered in chunks of 50 targets, each saved as one zip on Drive
           (resume granularity = one chunk). Shardable: '--shard 0/2' and '--shard 1/2' can run in two Colab
           notebooks at once; '--shard all' finishes whatever is missing. The fast renderer is checked
           pixel-for-pixel against the sealed field_eval.make_views on the first target of every chunk.
  hash     SHA-256 of the ALDOD archive (can run in the second notebook while the first renders)
  materialize  unpack chunks to local disk, verify every PNG hash, write views_manifest.csv (run automatically
           before freeze/infer; repeated on a fresh Colab VM)
  freeze   refuses if the protocol still has unfilled fields; records SHA-256 of protocol, this code, the sealed
           field_eval.py/analysis.py, targets, cascade, views manifest, mapping, ALDOD archive and checkpoints
           (checkpoints must equal the main study's sealed hashes) -> PROTOCOL_FROZEN.json
  infer    SEALED: re-verifies everything, then calls the unchanged field_eval.stage_infer (same code as the
           main study's sealed run). One npz per model on Drive; finished models are skipped after a restart.
  analyze  prespecified estimands (analysis_aldod.py) from the cohort file and logits only

No image is passed through a model before `infer`, and `infer` cannot run before `freeze`.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import field_eval as FE                      # the sealed main-study code, unchanged (hash checked at freeze)

SEALED_FIELD_EVAL = "f013bf0ec8ea6f08917ca4dd14d8a728a33d90e997753b30953eb7ce1fd76c43"
SEALED_ANALYSIS = "5cb6b99fd1fa980c723af7c32e6a5f11a5e35182b9cd00d36cfe191a9d15f012"
MAP = {"apple_healthy": ("Apple___healthy", "healthy"), "apple_scab": ("Apple___Apple_scab", "diseased"),
       "apple_black_rot": ("Apple___Black_rot", "diseased"), "apple_rust": ("Apple___Cedar_apple_rust", "diseased")}
BOX_CAP = 0.50
MAX_HAM = 6
VIEWS = ("F", "C")
POLICIES = ("resize", "letterbox")
VOC_SUB = Path("VOCdevkit/VOCdevkit/VOC2012")
log = FE.log


# ------------------------------------------------------------------ pHash (identical to candidate_inventory.py)
def _phash_file(path):
    import numpy as np
    from PIL import Image
    from scipy.fft import dct
    with Image.open(path) as im:
        size = im.size
        im.draft("RGB", (256, 256))
        a = np.asarray(im.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    d = dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
    bits = (d > np.median(d)).flatten()
    return int("".join("1" if b else "0" for b in bits), 2), size


def _popcount64(x):
    import numpy as np
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


def _probe(path):
    try:
        h, size = _phash_file(path)
        return {"phash": h, "w": size[0], "h": size[1], "ok": True}
    except Exception as e:
        return {"ok": False, "error": type(e).__name__}


# ------------------------------------------------------------------ cohort
def parse_voc(xmlp: Path):
    t = ET.parse(xmlp).getroot()
    W = int(float(t.findtext("size/width") or 0)); H = int(float(t.findtext("size/height") or 0))
    boxes = []
    for o in t.iter("object"):
        b = o.find("bndbox")
        if b is None:
            continue
        x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append((o.findtext("name").strip(), x1, y1, x2, y2))
    return W, H, boxes


def stage_cohort(root: Path, refs: list[Path], out: Path, workers: int) -> Path:
    import numpy as np
    sdir = out / "cohort"; sdir.mkdir(parents=True, exist_ok=True)
    tpath = sdir / "targets.csv"
    if FE.done(sdir) and tpath.exists():
        log("cohort", f"already complete: {tpath}"); return tpath
    voc = root / VOC_SUB
    imgs = sorted((voc / "JPEGImages").glob("*.jpg"))
    casc = {"jpeg_files": len(imgs)}
    log("cohort", f"{len(imgs)} JPEGs; fingerprinting with {workers} workers")
    with Pool(workers) as pool:
        probes = pool.map(_probe, [str(p) for p in imgs], chunksize=16)
    rows, drop = [], defaultdict(int)
    for p, pr in zip(imgs, probes):
        xmlp = voc / "Annotations" / (p.stem + ".xml")
        if not pr["ok"]:
            drop["decode_error"] += 1; continue
        if not xmlp.exists():
            drop["no_annotation"] += 1; continue
        W, H, boxes = parse_voc(xmlp)
        if (W, H) != (pr["w"], pr["h"]):
            drop["annotation_size_mismatch"] += 1; continue
        good = []
        for lab, x1, y1, x2, y2 in boxes:
            x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(W), x2), min(float(H), y2)
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                good.append((lab, x1, y1, x2, y2))
        if not good:
            drop["no_valid_box"] += 1; continue
        labs = {g[0] for g in good}
        if not labs <= set(MAP):
            drop["unknown_label"] += 1; continue
        if len(labs) > 1:
            drop["multi_label_photo"] += 1; continue
        lab = labs.pop()
        x1, y1, x2, y2 = max(((g[3] - g[1]) * (g[4] - g[2]), g[1:]) for g in good)[1]
        rows.append({"file": p.name, "label": lab, "phash": pr["phash"], "w": W, "h": H, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                     "n_boxes": len(good), "box_frac": (x2 - x1) * (y2 - y1) / (W * H)})
    casc.update({f"excluded_{k}": v for k, v in drop.items()}); casc["single_label_photos"] = len(rows)
    # near-duplicate groups over ALL decodable single-label photos
    Hs = np.array([r["phash"] for r in rows], dtype=np.uint64)
    parent = list(range(len(rows)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(len(rows) - 1):
        for j in np.nonzero(_popcount64(Hs[i] ^ Hs[i + 1:]) <= MAX_HAM)[0]:
            a, b = find(i), find(i + 1 + int(j))
            if a != b:
                parent[b] = a
    groups = defaultdict(list)
    for k in range(len(rows)):
        groups[find(k)].append(k)
    # reference matches (PlantVillage, PlantDoc, FieldPlant)
    ref_hit = np.zeros(len(rows), bool); per_ref = {}
    for ref in refs:
        z = np.load(ref, allow_pickle=False)
        for src in sorted({s.split("::")[0] for s in z.files if s.endswith("::phash")}):
            R = z[f"{src}::phash"].astype(np.uint64); n = 0
            for k in range(len(rows)):
                if _popcount64(R ^ Hs[k]).min() <= MAX_HAM:
                    ref_hit[k] = True; n += 1
            per_ref[src] = n
    casc["reference_matches_by_source"] = per_ref
    casc["near_duplicate_groups"] = len(groups)
    targets, n_conflict, n_ref = [], 0, 0
    for g, ks in sorted(groups.items(), key=lambda kv: min(rows[k]["file"] for k in kv[1])):
        if len({rows[k]["label"] for k in ks}) > 1:
            n_conflict += 1; continue
        if ref_hit[ks].any():
            n_ref += 1; continue
        r = rows[min(ks, key=lambda k: rows[k]["file"])]
        pv, grp = MAP[r["label"]]
        targets.append({"target_id": f"aldod_{Path(r['file']).stem}", "dataset": "aldod",
                        "relpath": str(VOC_SUB / "JPEGImages" / r["file"]), "width": r["w"], "height": r["h"],
                        "label": r["label"], "pv_class": pv, "group": grp,
                        "x1": r["x1"], "y1": r["y1"], "x2": r["x2"], "y2": r["y2"], "box_frac": round(r["box_frac"], 6),
                        "n_boxes_label": r["n_boxes"], "group_size": len(ks), "phash": f"{r['phash']:016x}",
                        "primary": int(r["box_frac"] <= BOX_CAP), "largebox": int(r["box_frac"] > BOX_CAP)})
    casc["excluded_groups_label_conflict"] = n_conflict; casc["excluded_groups_reference_match"] = n_ref
    casc["distinct_photos_eligible"] = len(targets)
    for c in ("primary", "largebox"):
        for grp in ("healthy", "diseased"):
            casc[f"{c}_{grp}"] = sum(1 for t in targets if t[c] == 1 and t["group"] == grp)
        casc[f"{c}_by_label"] = {l: sum(1 for t in targets if t[c] == 1 and t["label"] == l) for l in MAP}
    FE.atomic_csv(tpath, targets)
    FE.atomic_json(sdir / "cascade.json", casc)
    FE.mark(sdir, {"targets_sha256": FE.sha_file(tpath), "n_targets": len(targets)})
    log("cohort", json.dumps(casc, indent=1))
    return tpath


# ------------------------------------------------------------------ views (chunked, shardable, resumable)
CHUNK = 50                                   # targets per chunk; one chunk = one zip on Drive (resume granularity)


def fast_views(im, box):
    """F and C views only, with exactly the operations of the sealed FE.make_views (M is skipped for speed).
    Equality with FE.make_views is verified on the first target of every chunk."""
    from PIL import Image
    W, H = im.size
    x1, y1, x2, y2 = box
    mx, my = FE.MARGIN * (x2 - x1), FE.MARGIN * (y2 - y1)
    ex = (max(0, int(round(x1 - mx))), max(0, int(round(y1 - my))),
          min(W, int(round(x2 + mx))), min(H, int(round(y2 + my))))
    if ex[2] - ex[0] < 2 or ex[3] - ex[1] < 2:
        raise ValueError(f"degenerate enlarged box {ex}")
    out = {}
    for v, x in (("F", im), ("C", im.crop(ex))):
        out[(v, "resize")] = x.resize((FE.IMG, FE.IMG), Image.Resampling.BILINEAR)
        s = max(x.size)
        sq = Image.new("RGB", (s, s), FE.MEAN_RGB); sq.paste(x, ((s - x.size[0]) // 2, (s - x.size[1]) // 2))
        out[(v, "letterbox")] = sq.resize((FE.IMG, FE.IMG), Image.Resampling.BILINEAR)
    return out, ex


def _pix(vim):
    import numpy as np
    return hashlib.sha256(np.asarray(vim, dtype=np.uint8).tobytes()).hexdigest()


def _view_job(args):
    root, t, check = args
    from PIL import Image
    with Image.open(Path(root) / t["relpath"]) as im0:
        im = im0.convert("RGB")                     # raw orientation, as in the main study
    if im.size != (int(t["width"]), int(t["height"])):
        raise RuntimeError(f"{t['relpath']}: size {im.size} != cohort {(t['width'], t['height'])}")
    box = tuple(float(t[k]) for k in ("x1", "y1", "x2", "y2"))
    views, ex = fast_views(im, box)
    if check:                                       # pixel-identical to the sealed implementation?
        ref, ex_ref = FE.make_views(im, box)
        for k, vim in views.items():
            if ex != ex_ref or _pix(vim) != _pix(ref[k]):
                raise RuntimeError(f"fast_views differs from sealed make_views for {t['target_id']} {k}")
    res = []
    for (v, p), vim in sorted(views.items()):
        res.append((v, p, FE.png_bytes(vim), _pix(vim)))
    return t, ex, im.size, res


def chunks_of(targets):
    return [targets[i:i + CHUNK] for i in range(0, len(targets), CHUNK)]


def stage_views(root: Path, tpath: Path, out: Path, workers: int, shard: str):
    """Render chunk zips into <out>/views_chunks/. shard 'k/N' = chunks j with j % N == k; 'all' = every
    missing chunk (in reverse order, so a sweep overlaps as little as possible with a running shard)."""
    import zipfile, io, tempfile
    cdir = out / "views_chunks"; cdir.mkdir(parents=True, exist_ok=True)
    targets = list(csv.DictReader(open(tpath)))
    ch = chunks_of(targets)
    if shard == "all":
        mine = list(range(len(ch)))[::-1]
    else:
        k, n = (int(x) for x in shard.split("/")); mine = [j for j in range(len(ch)) if j % n == k]
    todo = [j for j in mine if not (cdir / f"chunk_{j:04d}.zip").exists()]
    log("views", f"shard {shard}: {len(mine)} chunks assigned, {len(mine) - len(todo)} already on Drive, {len(todo)} to render "
                 f"({len(targets)} targets in {len(ch)} chunks; {workers} workers)")
    fields = ["target_id", "view", "policy", "png", "png_sha256", "pixels_sha256", "ex_x1", "ex_y1", "ex_x2", "ex_y2", "src_w", "src_h"]
    t0 = time.time(); done_t = 0
    with Pool(workers) as pool:
        for n_done, j in enumerate(todo, 1):
            dst = cdir / f"chunk_{j:04d}.zip"
            if dst.exists():                        # rendered meanwhile by another notebook
                continue
            jobs = [(str(root), t, i == 0) for i, t in enumerate(ch[j])]
            buf = io.BytesIO(); rows = []
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
                for t, ex, size, res in pool.imap(_view_job, jobs):
                    for v, p, b, pix in res:
                        rel = f"aldod/{p}/{v}/{t['target_id']}.png"
                        z.writestr(rel, b)
                        rows.append({"target_id": t["target_id"], "view": v, "policy": p, "png": rel,
                                     "png_sha256": hashlib.sha256(b).hexdigest(), "pixels_sha256": pix,
                                     "ex_x1": ex[0], "ex_y1": ex[1], "ex_x2": ex[2], "ex_y2": ex[3], "src_w": size[0], "src_h": size[1]})
                s = io.StringIO(); w = csv.DictWriter(s, fieldnames=fields); w.writeheader(); w.writerows(rows)
                z.writestr("rows.csv", s.getvalue())
            tmp = cdir / f".chunk_{j:04d}.{os.getpid()}.tmp"
            tmp.write_bytes(buf.getvalue()); os.replace(tmp, dst)
            done_t += len(ch[j]); el = time.time() - t0
            left = sum(len(ch[x]) for x in todo[n_done:])
            log("views", f"chunk {j} saved ({n_done}/{len(todo)}); {done_t / el:.1f} targets/s; ETA this shard {left / max(done_t / el, 1e-9) / 60:.0f} min")
    missing = [j for j in range(len(ch)) if not (cdir / f"chunk_{j:04d}.zip").exists()]
    log("views", f"shard {shard} finished. Chunks still missing overall: {len(missing)} of {len(ch)}")
    return missing


def stage_materialize(tpath: Path, out: Path, local: Path) -> Path:
    """Unpack all chunk zips to fast local disk, verify every PNG hash, write the deterministic manifest locally
    (the path handed to the sealed inference) and a copy on Drive. Skips work that is already verified."""
    import zipfile
    cdir = out / "views_chunks"
    targets = list(csv.DictReader(open(tpath)))
    n_ch = len(chunks_of(targets))
    missing = [j for j in range(n_ch) if not (cdir / f"chunk_{j:04d}.zip").exists()]
    if missing:
        raise SystemExit(f"NOT YET: {len(missing)} view chunks missing (e.g. {missing[:5]}). Finish the views stage.")
    local.mkdir(parents=True, exist_ok=True)
    vpath = local / "views_manifest.csv"; drive_copy = out / "views" / "views_manifest.csv"
    if vpath.exists() and (local / ".verified").exists() and drive_copy.exists() and FE.sha_file(vpath) == FE.sha_file(drive_copy):
        log("materialize", f"already verified: {vpath}"); return vpath
    rows = {}
    for j in range(n_ch):
        mk = local / f".chunk_{j:04d}.ok"
        with zipfile.ZipFile(cdir / f"chunk_{j:04d}.zip") as z:
            part = list(csv.DictReader(io.TextIOWrapper(z.open("rows.csv"), encoding="utf-8")))
            if not mk.exists():
                z.extractall(local, members=[r["png"] for r in part]); mk.write_text("ok")
        for r in part:
            rows[(r["target_id"], r["view"], r["policy"])] = r
    need = {(t["target_id"], v, p) for t in targets for v in VIEWS for p in POLICIES}
    if need != set(rows):
        raise RuntimeError(f"view set mismatch: missing {len(need - set(rows))}, extra {len(set(rows) - need)}")
    out_rows = [rows[k] for k in sorted(need)]
    bad = [r["png"] for r in out_rows if FE.sha_file(local / r["png"]) != r["png_sha256"]]
    if bad:
        for j in range(n_ch):
            (local / f".chunk_{j:04d}.ok").unlink(missing_ok=True)
        raise RuntimeError(f"{len(bad)} PNG hash mismatches after unpacking (e.g. {bad[:3]}); rerun to unpack again")
    FE.atomic_csv(vpath, out_rows)
    drive_copy.parent.mkdir(parents=True, exist_ok=True)
    if not drive_copy.exists():
        FE.atomic_csv(drive_copy, out_rows)
    elif FE.sha_file(drive_copy) != FE.sha_file(vpath):
        raise SystemExit("REFUSED: rebuilt view manifest differs from the one on Drive (chunks changed?)")
    (local / ".verified").write_text("ok")
    log("materialize", f"{len(out_rows)} views unpacked and verified -> {vpath} (sha256 {FE.sha_file(vpath)[:12]}...)")
    return vpath


def stage_hash_archive(archive: Path, out: Path):
    rec = out / "aldod_archive_sha256.json"
    if rec.exists():
        log("hash", f"already recorded: {rec}"); return
    t0 = time.time(); h = FE.sha_file(archive)
    FE.atomic_json(rec, {"path": str(archive), "size_bytes": archive.stat().st_size, "sha256": h,
                         "hashed_utc": datetime.now(timezone.utc).isoformat()})
    log("hash", f"ALDOD archive sha256 {h} ({time.time()-t0:.0f}s)")


# ------------------------------------------------------------------ freeze / infer
def extras(out, tpath) -> dict:
    arch = out / "aldod_archive_sha256.json"
    return {"aldod_code": {p.name: FE.sha_file(p) for p in (HERE / "aldod_check.py", HERE / "analysis_aldod.py")},
            "cascade_sha256": FE.sha_file(tpath.parent / "cascade.json"),
            "aldod_archive": json.loads(arch.read_text()) if arch.exists() else None}


def check_sealed(src: Path, test: bool):
    fe, an = FE.sha_file(HERE / "field_eval.py"), FE.sha_file(HERE / "analysis.py")
    if (fe, an) != (SEALED_FIELD_EVAL, SEALED_ANALYSIS):
        raise SystemExit("REFUSED: field_eval.py / analysis.py differ from the sealed main-study code")
    sealed = json.loads((HERE / "sealed_checkpoints.json").read_text())["checkpoints"]
    now = {str(p.relative_to(src)): FE.sha_file(p) for p in FE.checkpoint_files(src)}
    if now != sealed:
        if test:
            log("sealed", "TEST MODE: checkpoints differ from the sealed main study (expected for smoke checkpoints)")
        else:
            bad = [k for k in sealed if now.get(k) != sealed[k]]
            raise SystemExit(f"REFUSED: checkpoints differ from the main study's sealed hashes: {bad[:4]}")


def stage_freeze(a, tpath, vpath, mpath):
    fpath = a.out / "PROTOCOL_FROZEN.json"
    if fpath.exists():
        log("freeze", f"already frozen: {fpath} (not modified)"); return
    if "⟦to fill⟧" in a.protocol.read_text():
        raise SystemExit("REFUSED: fill in §9 of the protocol (licence quote, approval) before freezing")
    if not FE.done(tpath.parent):
        raise SystemExit("REFUSED: run the cohort stage first")
    if not a.test and not (a.out / "aldod_archive_sha256.json").exists():
        raise SystemExit("REFUSED: run the hash stage first (records the ALDOD archive SHA-256)")
    check_sealed(a.source, a.test)
    rec = FE.freeze_record(a.protocol, tpath, vpath, a.source, mpath)
    rec.update(extras(a.out, tpath)); rec["test_mode"] = a.test
    rec["frozen_utc"] = datetime.now(timezone.utc).isoformat()
    FE.atomic_json(fpath, rec)
    log("freeze", f"frozen: {fpath}\n  protocol sha256 {rec['protocol']['sha256']}")


def stage_infer(a, tpath, vpath, mpath):
    fpath = a.out / "PROTOCOL_FROZEN.json"
    if not fpath.exists():
        raise SystemExit("REFUSED: run the `freeze` stage first (after the protocol is approved).")
    frozen = json.loads(fpath.read_text())
    now = extras(a.out, tpath)
    diffs = [k for k in now if json.dumps(frozen.get(k), sort_keys=True) != json.dumps(now[k], sort_keys=True)]
    if diffs:
        raise SystemExit(f"REFUSED: files changed since freeze: {diffs}")
    if frozen.get("test_mode") != a.test:
        raise SystemExit("REFUSED: test mode differs from the freeze record")
    check_sealed(a.source, a.test)
    # the unchanged sealed inference of the main study: re-verifies protocol, code, targets, views manifest,
    # mapping and checkpoints against the freeze record; writes one npz per model and skips finished ones
    FE.stage_infer(a.protocol, tpath, vpath, a.source, mpath, a.out, a.batch, a.workers, a.allow_cpu, a.random_init)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aldod-root", type=Path, help="extracted ALDOD folder (contains VOCdevkit/), local disk")
    ap.add_argument("--ref", type=Path, action="append", default=[], help="reference pHash npz (PlantVillage+PlantDoc, FieldPlant)")
    ap.add_argument("--archive", type=Path, help="the downloaded ALDOD Kaggle zip (hash stage)")
    ap.add_argument("--source", type=Path, help="source_training output dir (checkpoints)")
    ap.add_argument("--protocol", type=Path, default=HERE / "PROTOCOL_ALDOD_v1.md")
    ap.add_argument("--out", type=Path, required=True, help="durable output dir (Google Drive)")
    ap.add_argument("--local-views", type=Path, default=Path("/content/aldod_views"), help="fast local disk for unpacked views")
    ap.add_argument("--stages", default="cohort,views")
    ap.add_argument("--shard", default="all", help="views: 'k/N' renders chunks j %% N == k; 'all' renders every missing chunk")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--allow-cpu", action="store_true", help="TESTS ONLY")
    ap.add_argument("--random-init", action="store_true", help="TESTS ONLY")
    ap.add_argument("--test", action="store_true", help="TESTS ONLY: allow non-sealed (smoke) checkpoints")
    a = ap.parse_args()
    if (a.allow_cpu or a.random_init) and not a.test:
        raise SystemExit("--allow-cpu / --random-init are for tests only (add --test)")
    a.out.mkdir(parents=True, exist_ok=True)
    st = a.stages.split(",")
    FE.atomic_json(a.out / f"run_config_{int(time.time())}_{os.getpid()}.json", {**{k: str(v) for k, v in vars(a).items()}, "argv": sys.argv})
    mpath = HERE / "aldod_mapping.json"
    tpath = a.out / "cohort" / "targets.csv"; vpath = a.local_views / "views_manifest.csv"
    if "cohort" in st:
        if not a.ref:
            raise SystemExit("cohort needs --ref reference_phash.npz and --ref FieldPlant_phash.npz")
        tpath = stage_cohort(a.aldod_root, a.ref, a.out, a.workers)
    if "views" in st:
        if not FE.done(tpath.parent):
            raise SystemExit("NOT YET: the cohort stage has not finished")
        stage_views(a.aldod_root, tpath, a.out, a.workers, a.shard)
    if "hash" in st:
        stage_hash_archive(a.archive, a.out)
    if any(s in st for s in ("materialize", "freeze", "infer")):
        vpath = stage_materialize(tpath, a.out, a.local_views)
    if "freeze" in st:
        stage_freeze(a, tpath, vpath, mpath)
    if "infer" in st:
        stage_infer(a, tpath, vpath, mpath)
    if "analyze" in st:
        if not FE.done(a.out / "logits"):
            raise SystemExit("NOT YET: analysis needs the sealed logits (freeze, then infer).")
        import analysis_aldod
        analysis_aldod.run(a.out, tpath)
    log("done", f"outputs in {a.out}")


if __name__ == "__main__":
    main()
