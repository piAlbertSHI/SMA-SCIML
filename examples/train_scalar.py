"""Small genuine new fit on the bundled portable 63-case table; not an old benchmark rerun."""
import argparse,json
from pathlib import Path
import numpy as np
from SciML.models.v12.learning_ensemble import LearningEnsemble

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=200);p.add_argument('--seed',type=int,default=701);a=p.parse_args()
    if a.output.exists(): raise FileExistsError('Use a fresh output directory')
    d=np.load(Path(__file__).resolve().parents[1]/'data/scalar63.npz',allow_pickle=False)
    train=d['split']=='train';test=d['split']=='test'
    model=LearningEnsemble(members=3,epochs=a.epochs,seed=a.seed).fit(d['x'][train],d['y'][train],d['case_ids'][train].tolist())
    pred=model.predict_members(d['x']).mean(0);a.output.mkdir(parents=True)
    model.save(a.output/'ensemble.pt')
    result={'status':'completed_portable_training_example','seed':a.seed,'epochs':a.epochs,'members':3,'train_cases':int(train.sum()),'test_cases':int(test.sum()),'targets':d['target_names'].tolist(),'test_rmse':np.sqrt(np.mean((pred[test]-d['y'][test])**2,axis=0)).tolist(),'scope':'new exploratory six-output run, not original five-output benchmark','fresh_solver_calls':0}
    (a.output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
