"""Two-stage use: select measurements first; score only after five labels arrive."""
from pathlib import Path
import argparse
from dataclasses import fields
import json
import numpy as np
import pandas as pd
from sc_arc_core import HistoricalModels, PreparedTask, fit_measured, score_queries, decide

HERE = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command',required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--features',type=Path,required=True)
    prepare.add_argument('--history',type=Path,default=HERE/'data/history.csv')
    prepare.add_argument('--dataset-id',required=True)
    prepare.add_argument('--phase',type=float,required=True)
    prepare.add_argument('--out',type=Path,required=True)
    evaluate = sub.add_parser('decide')
    evaluate.add_argument('--prepared',type=Path,required=True)
    evaluate.add_argument('--measurements',type=Path,required=True)
    evaluate.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    config = json.loads((HERE/'config.json').read_text('utf-8'))
    args.out.mkdir(parents=True,exist_ok=True)
    if args.command == 'prepare':
        models = HistoricalModels(pd.read_csv(args.history),config['feature_columns'],args.dataset_id)
        task = models.prepare(pd.read_csv(args.features,float_precision='round_trip'),args.phase)
        values = {f.name:getattr(task,f.name) for f in fields(PreparedTask)}
        values['cells'] = values['cells'].astype(str)
        np.savez_compressed(args.out/'prepared_task.npz',**values)
        pd.DataFrame({'cell_id':task.requested_cells,'soh':np.nan}).to_csv(args.out/'measurements_request.csv',index=False)
        print('Measure these five cells:', ', '.join(task.requested_cells))
    else:
        with np.load(args.prepared,allow_pickle=False) as data:
            values = {k:data[k].item() if data[k].ndim == 0 else data[k] for k in data.files}
        task = PreparedTask(**values)
        labels = pd.read_csv(args.measurements,dtype={'cell_id':str})
        if list(labels.columns) != ['cell_id','soh'] or labels.cell_id.duplicated().any():
            raise ValueError('Measurement file must contain unique cell_id,soh columns')
        fit = fit_measured(task,dict(zip(labels.cell_id,labels.soh)))
        score = score_queries(task,fit,config['thresholds'],config['lambda_soh'])
        result = decide(score,config['operating_cutoff'])
        result.to_csv(args.out/'decisions.csv',index=False)
        print(f'Scored {len(result)} threshold candidates with the fixed development cutoff.')

if __name__ == '__main__':
    main()
