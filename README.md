# Leaf localization primarily restores plant identity in field disease recognition

**Paired-view attribution of plant identity and disease-condition recognition under laboratory-to-field shift**

Research implementation, archived aggregate experimental results, independent attribution checks, and provenance utilities accompanying *Leaf localization primarily restores plant identity in field disease recognition*.

The study examines what leaf localization restores when plant-disease classifiers encounter field photographs after laboratory training. With fixed recognition checkpoints, full photographs and annotated leaf crops are compared within the same image. Joint plant-condition accuracy is decomposed into plant-identification and within-plant condition contributions; fine-tuned CNNs are contrasted with frozen DINOv2 representations.

The evaluation comprises **6,442 parent photographs** across PlantDoc (2,185), FieldPlant (1,273) and held-out ALDOD apple images (2,984). The prespecified PlantDoc system–view interaction is **9.2 percentage points** (95% CI 7.4–11.1). Plant identification accounts for **66–112%** of the six reported CNN crop gains. The held-out primary healthy-versus-diseased interaction difference reaches **20.8 points** (95% CI 16.8–24.6). These are manuscript-reported results preserved alongside their original aggregate records.

## Installation and verification

Python 3.10+ with NumPy supports the dataset-independent verification path:

```bash
git clone https://github.com/saed-alqaraleh/leaf-localization-attribution.git
cd leaf-localization-attribution
python -m pip install -r requirements.txt
python scripts/verify_release.py
```

The command verifies frozen source-code and result-record hashes, checks selected principal numerical findings against the original aggregate records, reconciles manuscript-transcribed evidence tables and runs the independent prediction-level attribution unit tests. No upstream dataset or model checkpoint is needed for this verification mode.

## Study design

| Recognition system | Training or adaptation | Paired evaluation |
| --- | --- | --- |
| MobileNetV3-Large | PlantVillage fine-tuning, three seeds | Full, crop and applicable masked views |
| ConvNeXt-Tiny | PlantVillage fine-tuning, three seeds | Full, crop and applicable masked views |
| DINOv2-S/14 | Frozen features with multinomial logistic-regression probe | Fixed-checkpoint view comparisons |
| DINOv2-B/14 | Frozen features with multinomial logistic-regression probe | Fixed-checkpoint view comparisons |

Each field photograph contributes matched views to the same frozen system. The method distinguishes changes in host identification from changes in diagnosis conditional on the correct host; uncertainty is measured with paired parent-level bootstrap resampling. An independent ALDOD study evaluates a health-stratified interaction fixed before held-out model inference.

## Research components

| Location | Contents |
| --- | --- |
| [`src/original/`](src/original/) | Six byte-preserved author-supplied training, field evaluation and analysis scripts |
| [`src/paired_view_audit.py`](src/paired_view_audit.py) | Independent image-keyed attribution analysis utilities |
| [`scripts/`](scripts/) | Package verification and numerical-table checks |
| [`results/frozen/`](results/frozen/) | Original aggregate Protocol 1 and Protocol 2 JSON results |
| [`results/`](results/) | Manuscript-transcribed accuracy, attribution and interaction tables |
| [`provenance/`](provenance/) | Original-code hashes and checksum-verified inventories of separately retained frozen artifacts |
| [`data/README.md`](data/README.md) | Upstream sources, cohort definitions and permitted preparation routes |

See the [reproduction guide](REPRODUCIBILITY.md) for the verification boundary and original pipeline components.

## Data and licensing

Original photographs, source image manifests, selected box coordinates, restricted image-keyed predictions and CNN checkpoints are not bundled. Upstream datasets and the original reference implementations remain with their publishers; the retained research outputs support audit of the included aggregate evidence. See [data provenance](data/README.md), [third-party notices](THIRD_PARTY_NOTICES.md), and [software and data rights](LICENSE_SCOPE.md).

## Citation

The associated research article is authored by **Saed Alqaraleh** (Mutah University, Jordan) and **Mohammed Madi** (Sohar University, Oman). Citation metadata for this software release appear in [`CITATION.cff`](CITATION.cff). The frozen research software release (`1.0.0`) is permanently archived on Zenodo: **[10.5281/zenodo.23323042](https://doi.org/10.5281/zenodo.23323042)**. This is the software archive DOI, not a journal-article DOI.

[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23323042.svg)](https://doi.org/10.5281/zenodo.23323042)
