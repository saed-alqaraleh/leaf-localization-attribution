"""Independent paired-view calculation/audit utilities.

This is a clean-room verification implementation, NOT the original study pipeline.
Inputs are keyed model predictions produced by the authors' fixed checkpoints.
No data are downloaded, training is performed, or results are reconstructed here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any

REQUIRED = {"dataset", "image_id", "model", "seed", "true_label", "pred_full", "pred_crop"}


def load_predictions(path: Path, plant_map: dict[str, str]) -> list[dict[str, str]]:
    rows = []
    seen = set()
    image_truth = {}
    with path.open(encoding="utf-8") as inp:
        for lineno, raw in enumerate(inp, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict) or not REQUIRED.issubset(row):
                raise ValueError(f"{path}:{lineno}: expected keys {sorted(REQUIRED)}")
            if not all(isinstance(row[k], str) and row[k] for k in REQUIRED):
                raise ValueError(f"{path}:{lineno}: required fields must be nonempty strings")
            for k in ("true_label", "pred_full", "pred_crop"):
                if row[k] not in plant_map:
                    raise ValueError(f"{path}:{lineno}: {k}={row[k]!r} is absent from plant map")
            if "health" in row and row["health"] not in ("healthy", "diseased"):
                raise ValueError(f"{path}:{lineno}: invalid health state")
            key = (row["dataset"], row["image_id"], row["model"], row["seed"])
            if key in seen:
                raise ValueError(f"{path}:{lineno}: duplicate dataset/image/model/seed key {key}")
            seen.add(key)
            photo = (row["dataset"], row["image_id"])
            current = (row["true_label"], row.get("health"))
            if photo in image_truth and image_truth[photo] != current:
                raise ValueError(f"{path}:{lineno}: inconsistent true label or health for photo {photo}")
            image_truth[photo] = current
            rows.append(row)
    if not rows:
        raise ValueError("Prediction file contains no records")
    return rows


def validate_panel(rows: list[dict[str, str]]) -> None:
    """Every model must have the same seeds for all its photographs."""
    by_group: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for r in rows:
        by_group[(r["dataset"], r["model"])][r["image_id"]].add(r["seed"])
    for group, photos in by_group.items():
        seed_patterns = set(map(frozenset, photos.values()))
        if len(seed_patterns) != 1:
            raise ValueError(f"Unbalanced image-by-seed panel for {group}")


def accuracy_decomposition(rows: list[dict[str, str]], plant_map: dict[str, str]) -> dict[str, float | int | None]:
    """A=P*W, and delta A=delta P*W_F+P_C*delta W (fraction units internally)."""
    n = len(rows)
    correct = {}
    plants = {}
    for v in ("full", "crop"):
        correct[v] = sum(r[f"pred_{v}"] == r["true_label"] for r in rows)
        plants[v] = sum(plant_map[r[f"pred_{v}"]] == plant_map[r["true_label"]] for r in rows)
    pf, pc = plants["full"] / n, plants["crop"] / n
    wf = correct["full"] / plants["full"] if plants["full"] else 0.0
    wc = correct["crop"] / plants["crop"] if plants["crop"] else 0.0
    af, ac = correct["full"] / n, correct["crop"] / n
    # Both equalities must hold even when no predicted plant identity is correct.
    assert abs(pf * wf - af) < 1e-12 and abs(pc * wc - ac) < 1e-12
    plant_term = (pc - pf) * wf
    condition_term = pc * (wc - wf)
    assert abs((ac - af) - (plant_term + condition_term)) < 1e-12
    symmetric_plant = (pc - pf) * (wf + wc) / 2
    symmetric_condition = (wc - wf) * (pf + pc) / 2
    assert abs((ac - af) - (symmetric_plant + symmetric_condition)) < 1e-12
    return {
        "prediction_records": n,
        "photographs": len({r["image_id"] for r in rows}),
        "accuracy_full_pct": 100 * af,
        "accuracy_crop_pct": 100 * ac,
        "delta_accuracy_pp": 100 * (ac - af),
        "plant_full_pct": 100 * pf,
        "plant_crop_pct": 100 * pc,
        "within_plant_full_pct": 100 * wf,
        "within_plant_crop_pct": 100 * wc,
        "plant_contribution_pp": 100 * plant_term,
        "within_plant_contribution_pp": 100 * condition_term,
        "symmetric_plant_pp": 100 * symmetric_plant,
        "symmetric_within_plant_pp": 100 * symmetric_condition,
        "plant_share_pct": 100 * plant_term / (ac - af) if abs(ac - af) > 1e-12 else None,
    }


def photo_effects(rows: list[dict[str, str]]) -> dict[str, float]:
    by_photo: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_photo[r["image_id"]].append(float(r["pred_crop"] == r["true_label"]) - float(r["pred_full"] == r["true_label"]))
    return {photo: sum(ds) / len(ds) for photo, ds in by_photo.items()}


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def bootstrap_ci_pp(values: list[float], repeats: int, seed: int) -> list[float] | None:
    """Illustrative independent percentile CI; never interpreted as original study resample."""
    if repeats <= 0:
        return None
    rng = random.Random(seed)
    n = len(values)
    samples = [100 * sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(repeats)]
    return [percentile(samples, 0.025), percentile(samples, 0.975)]


def analyse(rows: list[dict[str, str]], plant_map: dict[str, str],
            repeats: int = 0, seed: int = 0) -> dict[str, Any]:
    validate_panel(rows)
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["model"])].append(row)
    stats = {}
    effects = {}
    for (dataset, model), subset in sorted(groups.items()):
        key = f"{dataset}/{model}"
        stats[key] = accuracy_decomposition(subset, plant_map)
        effects[key] = photo_effects(subset)
        # Same means whether pooling the seeds or averaging the photo-level deltas.
        assert abs(stats[key]["delta_accuracy_pp"] - 100 * sum(effects[key].values()) / len(effects[key])) < 1e-9
        stats[key]["delta_bootstrap_ci_pp"] = bootstrap_ci_pp(list(effects[key].values()), repeats, seed)
    interactions = {}
    health_by_dataset = {}
    for r in rows:
        if "health" in r:
            health_by_dataset[(r["dataset"], r["image_id"])] = r["health"]
    for dataset in sorted({r["dataset"] for r in rows}):
        for cnn, probe in (("M1", "M2"), ("M3", "M4")):
            left, right = f"{dataset}/{cnn}", f"{dataset}/{probe}"
            if left not in effects or right not in effects:
                continue
            if effects[left].keys() != effects[right].keys():
                raise ValueError(f"Missing/extra image IDs across paired systems {left} and {right}")
            values = [effects[left][k] - effects[right][k] for k in sorted(effects[left])]
            outcome = {
                "photographs": len(values),
                "interaction_pp": 100 * sum(values) / len(values),
                "bootstrap_ci_pp": bootstrap_ci_pp(values, repeats, seed),
            }
            health = {k: health_by_dataset[(dataset, k)] for k in effects[left]
                      if (dataset, k) in health_by_dataset}
            if health and len(health) != len(values):
                raise ValueError(f"Health metadata incomplete for {dataset}")
            if health and set(health.values()) == {"healthy", "diseased"}:
                vals_by_health = {h: [effects[left][k]-effects[right][k] for k in sorted(health) if health[k] == h]
                                  for h in ("healthy", "diseased")}
                dh = sum(vals_by_health["healthy"]) / len(vals_by_health["healthy"])
                dd = sum(vals_by_health["diseased"]) / len(vals_by_health["diseased"])
                outcome["health_difference_D_pp"] = 100 * (dh - dd)
                outcome["healthy_n"] = len(vals_by_health["healthy"])
                outcome["diseased_n"] = len(vals_by_health["diseased"])
                if repeats:
                    rng = random.Random(seed)
                    h = vals_by_health["healthy"]
                    d = vals_by_health["diseased"]
                    draws = [100 * (sum(h[rng.randrange(len(h))] for _ in h)/len(h)
                                  - sum(d[rng.randrange(len(d))] for _ in d)/len(d)) for _ in range(repeats)]
                    outcome["health_D_bootstrap_ci_pp"] = [percentile(draws, .025), percentile(draws, .975)]
            interactions[f"{dataset}/{cnn}-{probe}"] = outcome
    return {"source": "independent computation on supplied keyed predictions",
            "bootstrap_draws": repeats, "bootstrap_seed": seed,
            "models": stats, "interactions": interactions}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, required=True, help="Author-supplied keyed JSONL predictions")
    ap.add_argument("--plant-map", type=Path, required=True, help="JSON: 38-class label -> host plant")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=0, help="Number of independent resamples (0 disables)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed of THIS audit (not source study)")
    args = ap.parse_args()
    if args.bootstrap < 0:
        ap.error("--bootstrap must be >= 0")
    mapping = json.loads(args.plant_map.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or not mapping or not all(isinstance(k, str) and isinstance(v, str) and v for k, v in mapping.items()):
        ap.error("Plant map must be a nonempty JSON object of label -> plant strings")
    rows = load_predictions(args.predictions, mapping)
    result = analyse(rows, mapping, args.bootstrap, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"Audited {len(rows)} prediction rows; wrote {args.output}")


if __name__ == "__main__":
    main()
