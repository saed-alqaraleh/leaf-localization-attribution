# Reproducibility and verification

## Dataset-independent verification

The standard audit installs NumPy and checks the shipped author-supplied scripts and original aggregate results, the independent attribution tests, publication-table transcriptions and the SHA-256 inventory. It does not retrain networks or reconstruct image-level analyses from unpublished labels.

```bash
python -m pip install -r requirements.txt
python scripts/verify_release.py
```

The verifier checks frozen source-file hashes, the original numerical estimates of the PlantDoc primary interaction and independent held-out ALDOD contrast, the matched cohort counts and the internal arithmetic of the published tables. It invokes the original attribution-unit tests without replacing any frozen output.

## Original experiment implementations

The original source training, paired-view evaluation and analysis modules appear unchanged under `src/original/`:

| File | Role |
| --- | --- |
| `source_pipeline.py` | PlantVillage grouping, model fitting and checkpoint selection |
| `n3_pilot.py` | Source-only timing and input preparation support |
| `field_eval.py` | PlantDoc/FieldPlant cohort, view, freeze and inference pipeline |
| `aldod_check.py` | ALDOD overlap, cohort and independent evaluation preparation |
| `analysis.py` | Photograph-paired bootstrap and analysis for Protocol 1 |
| `analysis_aldod.py` | Health-stratified replication analysis for Protocol 2 |

Re-executing the original inference requires upstream photographs/annotations and the original checkpoint files. Original code retains the recorded CLI arguments and protocol freeze checks. The additional full-stack dependencies are documented separately in `requirements-full.txt`; these are environment guidance rather than a promise that all upstream versions and assets have been reinstalled.

## Frozen record identity

`provenance/frozen_output_inventory.json` inventories the 16 author-retained model-output arrays with recorded byte counts, SHA-256 and array schemas (no photograph identifiers). `provenance/protocol_record_registry.json` records the digests of the two original frozen protocol and sealed-run records (which contain machine-specific locations and are not included).

`results/frozen/` contains unmodified aggregate Protocol 1 and Protocol 2 outputs. `results/reported_*.csv` records manuscript-level rounded tables. The two are distinguishable and are checked together. The original paper data, model weights and source evaluation records remain independently maintained.

## Scope of verification

Running the command above verifies the release package and the specified scientific assertions against frozen aggregate records. Exact image-level bootstrap replay and full source-model re-training are separate operations that require the original image-keyed predictions, cohort targets, and model checkpoints. The public repository makes no claim to regenerate those outputs without these inputs.
