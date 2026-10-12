"""Synthetic arithmetic fixtures, not study datasets or experimental outcomes."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paired_view_audit import accuracy_decomposition, analyse, load_predictions, validate_panel


PLANTS = {"tomato_h": "tomato", "tomato_b": "tomato", "apple_h": "apple", "apple_s": "apple"}


def case(image, model, seed, truth, full, crop, dataset="synthetic"):
    return {"dataset": dataset, "image_id": image, "model": model, "seed": seed,
            "true_label": truth, "pred_full": full, "pred_crop": crop}


class PairedViewTests(unittest.TestCase):
    def test_exact_decomposition_and_plant_recovery(self):
        rows = [case("a", "M1", "s1", "tomato_h", "apple_h", "tomato_h"),
                case("b", "M1", "s1", "tomato_b", "tomato_h", "tomato_b"),
                case("c", "M1", "s1", "apple_h", "apple_h", "apple_s")]
        s = accuracy_decomposition(rows, PLANTS)
        self.assertAlmostEqual(s["accuracy_full_pct"], 100 / 3)
        self.assertAlmostEqual(s["accuracy_crop_pct"], 200 / 3)
        self.assertAlmostEqual(s["plant_contribution_pp"] + s["within_plant_contribution_pp"], s["delta_accuracy_pp"])
        self.assertAlmostEqual(s["symmetric_plant_pp"] + s["symmetric_within_plant_pp"], s["delta_accuracy_pp"])

    def test_system_interaction_and_balanced_seeds(self):
        rows = []
        for seed in ("s1", "s2", "s3"):
            rows += [case("a", "M1", seed, "tomato_h", "apple_h", "tomato_h"),
                     case("b", "M1", seed, "apple_h", "apple_h", "apple_h")]
        rows += [case("a", "M2", "probe", "tomato_h", "tomato_h", "tomato_h"),
                 case("b", "M2", "probe", "apple_h", "apple_h", "apple_h")]
        r = analyse(rows, PLANTS, repeats=64, seed=7)
        self.assertAlmostEqual(r["interactions"]["synthetic/M1-M2"]["interaction_pp"], 50)
        self.assertEqual(r["models"]["synthetic/M1"]["photographs"], 2)
        self.assertEqual(len(r["interactions"]["synthetic/M1-M2"]["bootstrap_ci_pp"]), 2)

    def test_health_stratified_difference(self):
        rows = [case("healthy1", "M1", "s1", "tomato_h", "apple_h", "tomato_h"),
                case("diseased1", "M1", "s1", "tomato_b", "tomato_b", "tomato_b"),
                case("healthy1", "M2", "probe", "tomato_h", "tomato_h", "tomato_h"),
                case("diseased1", "M2", "probe", "tomato_b", "tomato_b", "tomato_b")]
        for row in rows:
            row["health"] = "healthy" if row["image_id"] == "healthy1" else "diseased"
        result = analyse(rows, PLANTS, repeats=32, seed=4)
        out = result["interactions"]["synthetic/M1-M2"]
        self.assertAlmostEqual(out["health_difference_D_pp"], 100)
        self.assertEqual((out["healthy_n"], out["diseased_n"]), (1, 1))
        self.assertEqual(out["health_D_bootstrap_ci_pp"], [100, 100])

    def test_cross_system_image_mismatch_rejected(self):
        rows = [case("a", "M1", "s1", "tomato_h", "tomato_h", "tomato_h"),
                case("b", "M2", "probe", "tomato_h", "tomato_h", "tomato_h")]
        with self.assertRaisesRegex(ValueError, "Missing/extra"):
            analyse(rows, PLANTS)

    def test_unequal_seed_panels_rejected(self):
        rows = [case("a", "M1", "s1", "tomato_h", "tomato_h", "tomato_h"),
                case("b", "M1", "s2", "tomato_h", "tomato_h", "tomato_h")]
        with self.assertRaisesRegex(ValueError, "Unbalanced"):
            validate_panel(rows)

    def test_duplicate_input_rejected(self):
        row = case("a", "M1", "s1", "tomato_h", "tomato_h", "tomato_h")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_predictions(path, PLANTS)

    def test_plant_map_missing_label_rejected(self):
        row = case("a", "M1", "s1", "tomato_h", "other", "tomato_h")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "absent from plant map"):
                load_predictions(path, PLANTS)


if __name__ == "__main__":
    unittest.main()
