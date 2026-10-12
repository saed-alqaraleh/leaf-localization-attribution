#!/usr/bin/env python3
"""Verify the release against original aggregate evidence and immutable file hashes."""
from __future__ import annotations
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def check(test:bool,message:str)->None:
    if not test:raise AssertionError(message)

def load(rel):return json.loads((ROOT/rel).read_text(encoding='utf8'))

def sha(rel):return hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()

def run():
    rows=list(csv.DictReader((ROOT/'provenance/RELEASE_SHA256.csv').open(newline='',encoding='utf-8')))
    paths=[r['path'] for r in rows]
    check(len(paths)==len(set(paths)) and len(paths)>10,'Invalid SHA-256 manifest')
    expected=set(paths)
    actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and not any(part in ('.git','__pycache__','.pytest_cache') for part in p.parts) and p.relative_to(ROOT).as_posix() != 'provenance/RELEASE_SHA256.csv'}
    check(expected==actual,'Missing/unexpected release files: '+repr(sorted(expected^actual)))
    for row in rows:
        p=ROOT/row['path']
        check(p.stat().st_size==int(row['bytes']) and sha(row['path'])==row['sha256'],'Release integrity mismatch: '+row['path'])
    # Aggregate analyses are original immutable records, not source reconstruction.
    p1=load('results/frozen/protocol1_results.json')
    p2=load('results/frozen/protocol2_aldod_results.json')
    assert p1['primary']['n_parents']==2185
    assert round(p1['primary']['estimate_pp'],1)==9.2
    assert [round(x,1) for x in p1['primary']['ci95_pp']]==[7.4,11.1]
    assert [round(x,1) for x in p2['primary']['ci95_pp']]==[16.8,24.6]
    assert round(p2['primary']['estimate_pp'],1)==20.8
    assert p2['primary']['n_healthy']+p2['primary']['n_diseased']==2984
    assert p1['cohorts']['fieldplant_primary']['n_parents']==1273
    assert p1['cohorts']['plantdoc_primary']['n_parents']==2185
    # Primary contrast and mask-scale partition must agree within source rounding.
    d=p1['plantdoc_all_contrasts']
    assert abs(d['I_context']['estimate_pp']+d['I_scale']['estimate_pp']-p1['primary']['estimate_pp'])<0.002
    assert abs(d['Delta_M1']['estimate_pp']-d['Delta_M2']['estimate_pp']-p1['primary']['estimate_pp'])<0.002
    r=ROOT/'scripts/check_reported_tables.py'
    x=subprocess.run([sys.executable,str(r)],cwd=ROOT,check=False)
    check(x.returncode==0,'Reported table checks failed')
    y=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,check=False)
    check(y.returncode==0,'Independent attribution tests failed')
    inventory=load('provenance/frozen_output_inventory.json')
    assert inventory['record_count']==16 and len(inventory['records'])==16
    for v in inventory['records']:
        assert v['arrays']['logits']['shape'][1]==38
        assert v['arrays']['ids']['shape'][0]==v['arrays']['logits']['shape'][0]
    print(json.dumps({'status':'PASS','sha256_checked':len(paths),'original_code_modules':6,'frozen_result_records':2,'sealed_array_schema_records':16,'analyses':'aggregate record and independent audit consistency'},indent=2))

if __name__=='__main__':
    try:run()
    except Exception as e:
        print(type(e).__name__+': '+str(e),file=sys.stderr);sys.exit(1)
