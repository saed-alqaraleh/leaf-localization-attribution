# Data sources and preparation

The study uses PlantVillage as the 38-class source domain, and tests fixed systems on PlantDoc (2,185 photographs), FieldPlant (1,273) and an independent ALDOD apple cohort (2,984). The field denominator is 6,442 parent photographs. Separate original data are obtained from upstream providers listed in [third-party notices](../THIRD_PARTY_NOTICES.md).

Source-domain preparation uses the original `src/original/source_pipeline.py` with a grouped split designed to avoid repeat-leaf and duplicate leakage: 54,305 photographs, 20,591 groups and train/validation/test sizes of 38,001 / 8,161 / 8,143. The source and target data remain external; no direct-download service or proprietary copy is supplied by this repository.

PlantDoc evaluation uses its detection annotations and a 10% expansion of the selected leaf box, with matched full, cropped and masked views. FieldPlant is evaluated on the eligible five-label cohort and has two precedent-based class mappings. ALDOD uses the separately frozen four-label primary apple cohort with a selected box-area fraction at most 0.5. The protocol-specific evaluation scripts contain preprocessing and selection logic. Image-level cohort manifests, sealed prediction arrays and CNN checkpoints are held by the authors and are not present in this public-facing release.

The accompanying statistical summaries and checksum provenance support independent verification of the numerical evidence that can be checked without redistribution of restricted image-level artifacts. See [reproducibility](../REPRODUCIBILITY.md).
