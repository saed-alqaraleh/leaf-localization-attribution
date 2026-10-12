"""Inspect transcription coherence; does NOT verify experimental predictions."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_csv(name):
    with (ROOT / 'results' / name).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main():
    a, b, c, d = (read_csv(n) for n in ('reported_accuracy.csv', 'reported_attribution.csv', 'reported_heldout.csv', 'reported_interactions.csv'))
    assert len(a) == 8 and len(b) == 6 and len(c) == 2 and len(d) == 3
    assert {(r['dataset'],r['model']) for r in a} == {(d,m) for d in ['PlantDoc','FieldPlant'] for m in ['M1','M2','M3','M4']}
    for row in a:
        # Source table rounds columns separately; 0.0 may coexist with 40.1 -> 40.0.
        assert abs(float(row['crop_accuracy_pct']) - float(row['full_accuracy_pct']) - float(row['reported_delta_pp'])) <= 0.15
        assert row['provenance'] == 'manuscript_transcription_not_independent_replay'
    for row in b:
        assert abs(float(row['plant_term_pp']) + float(row['condition_term_pp']) - float(row['total_gain_pp'])) <= 0.15
        assert abs(float(row['plant_term_pp']) / float(row['total_gain_pp']) * 100 - float(row['reported_plant_share_pct'])) < 2.0
    assert {r['pair'] for r in c} == {'M1-M2','M3-M4'}
    for r in c:
        assert abs(float(r['healthy_interaction_pp']) - float(r['diseased_interaction_pp']) - float(r['reported_D_pp'])) < .15
    assert {(r['dataset'], r['pair']) for r in d} == {('PlantDoc', 'M1-M2'), ('PlantDoc', 'M3-M4'), ('FieldPlant', 'M1-M2')}
    for r in d:
        lookup = {q['model']: float(q['reported_delta_pp']) for q in a if q['dataset'] == r['dataset']}
        left, right = r['pair'].split('-')
        assert abs(lookup[left] - lookup[right] - float(r['interaction_pp'])) < .15
    print('PASS: 8 accuracy rows, 6 decompositions, 3 interactions, 2 held-out contrasts; coherent to displayed rounding.')
    print('NOT CHECKED: underlying logits, image-level predictions, bootstraps, author-protocol SHA-256 records.')


if __name__ == '__main__':
    main()
