"""Evaluation harness, deliberately separated from the outcome-free method API."""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import time
import numpy as np
import pandas as pd
from sc_arc_core import HistoricalModels, fit_measured, score_queries, decide

HERE = Path(__file__).resolve().parent

def run_dataset(manifest, history, labels, config):
    build = manifest.build_dataset.iloc[0]
    models = HistoricalModels(history, config['feature_columns'], build)
    predictions, selections = [], []
    for row in manifest.itertuples():
        features = pd.read_csv(HERE/row.features, float_precision='round_trip')
        task = models.prepare(features, row.phase)
        # The oracle releases only the requested measurements to the method.
        lookup = labels[labels.task_id.eq(row.task_id)].set_index('cell_id').soh
        measured = {c:float(lookup.loc[c]) for c in task.requested_cells}
        fit = fit_measured(task, measured)
        score = score_queries(task, fit, config['thresholds'], config['lambda_soh'])
        score['task_id'] = row.task_id
        score['dataset'] = row.dataset
        score['physical_provenance'] = row.physical_provenance
        score['role'] = row.role
        predictions.append(score)
        selections.extend(dict(task_id=row.task_id,order=i,cell_id=c)
                          for i,c in enumerate(task.requested_cells))
        print('TASK', row.task_id, 'scored', len(score), flush=True)
    return pd.concat(predictions,ignore_index=True), pd.DataFrame(selections)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=HERE/'reproduced')
    parser.add_argument('--workers',type=int,default=2)
    args = parser.parse_args()
    start = time.time()
    config = json.loads((HERE/'config.json').read_text('utf-8'))
    for rel, expected in config['input_sha256'].items():
        assert hashlib.sha256((HERE/rel).read_bytes()).hexdigest() == expected, rel
    manifest = pd.read_csv(HERE/'data/task_manifest.csv')
    history = pd.read_csv(HERE/'data/history.csv')
    labels = pd.read_csv(HERE/'data/evaluation_labels.csv',float_precision='round_trip')
    frames, rosters = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_dataset,g,history,labels,config)
                   for _,g in manifest.groupby('build_dataset',sort=True)]
        for future in as_completed(futures):
            frame, roster = future.result()
            frames.append(frame); rosters.append(roster)
    scores = pd.concat(frames,ignore_index=True)
    selected = pd.concat(rosters,ignore_index=True).sort_values(['task_id','order']).reset_index(drop=True)
    expected_selection = pd.read_csv(HERE/'reference/selected_cells.csv').sort_values(['task_id','order']).reset_index(drop=True)
    pd.testing.assert_frame_equal(selected, expected_selection)
    # Outcomes are first attached after every selection, fit and score is complete.
    scored = scores.merge(labels,on=['task_id','cell_id'],validate='many_to_one')
    invalid = scored[~scored.soh.between(0,1.2)].copy()
    scored = scored[scored.soh.between(0,1.2)].copy()
    scored['wrong_candidate'] = scored.predicted_sign*(scored.soh-scored.threshold) <= 0
    dev = config['development_physical_sources']
    bg = {source: max(0.,float(g.loc[g.eligible & g.wrong_candidate,'action_score'].max()))
          if (g.eligible & g.wrong_candidate).any() else 0.
          for source,g in scored[scored.physical_provenance.isin(dev)].groupby('physical_provenance')}
    assert set(bg) == set(dev)
    bop = max(bg.values())
    marked = []
    for source,g in scored.groupby('physical_provenance'):
        cutoff = max(v for s,v in bg.items() if s != source) if source in bg else bop
        marked.append(decide(g,cutoff))
    result = pd.concat(marked,ignore_index=True)
    result['correct'] = result.issued & ~result.wrong_candidate
    result['wrong'] = result.issued & result.wrong_candidate
    ref = pd.read_csv(HERE/'reference/decisions.csv.gz',float_precision='round_trip')
    merged = result.merge(ref,on=['task_id','cell_id','threshold'],suffixes=('_new','_ref'),
                          how='outer',validate='one_to_one',indicator=True)
    assert merged._merge.eq('both').all()
    differences = {name:float(abs(merged[name+'_new']-merged[name+'_ref']).max())
                   for name in ['baseline_prediction','local_scale']}
    differences['action_score'] = float(abs(merged.action_score-merged.reference_action_score).max())
    assert max(differences.values()) < 1e-10, differences
    assert np.array_equal(merged.issued,merged.reference_issued)
    assert abs(bop-config['operating_cutoff']) < 1e-10
    summary = result.groupby('role').agg(candidates=('issued','size'),correct=('correct','sum'),wrong=('wrong','sum')).reset_index()
    expected = {'development':(7116,3949,1),'external_retrospective':(3177,1823,1)}
    for row in summary.itertuples():
        assert (row.candidates,row.correct,row.wrong) == expected[row.role]
    costs = []
    for tau,g in result.groupby('threshold'):
        costs.append(dict(threshold=tau,valid_events=3851,initial_tests=420,
            follow_up_tests=3431-int(g.issued.sum()),
            total_tests=3851-int(g.issued.sum()),correctly_avoided_tests=int(g.correct.sum()),
            wrong_decisions=int(g.wrong.sum()),avoided_fraction=float(g.correct.sum()/3851),
            scope='Conditional on available archived diagnostic inputs; acquisition cost excluded'))
    args.out.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.out/'decision_rows.csv.gz',index=False)
    selected.to_csv(args.out/'selected_cells.csv',index=False)
    invalid.to_csv(args.out/'invalid_reference_records.csv',index=False)
    summary.to_csv(args.out/'headline_results.csv',index=False)
    pd.DataFrame(costs).to_csv(args.out/'conditional_test_accounting.csv',index=False)
    pd.DataFrame([dict(source=s,score=v) for s,v in bg.items()]).to_csv(args.out/'source_scores.csv',index=False)
    report = dict(status='PASS',tasks=len(manifest),selected_cells=len(selected),
        candidates=len(result),invalid_cell_stage_events=invalid[['task_id','cell_id']].drop_duplicates().shape[0],
        all_selections_match=True,all_decisions_match=True,max_absolute_differences=differences,
        operating_cutoff=bop,seconds=time.time()-start,summary=summary.to_dict('records'))
    (args.out/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__ == '__main__':
    main()
