"""Rerun recovered Example 3 code, recording every optimizer status.

No objective, derivative, stopping option, pruning threshold or training sample
is changed for the convergence sweep. The contour comparison reuses its seeded
N=400 fit; the two polynomial baselines use seed 42 (historical seed unknown).
"""
from pathlib import Path
import os, runpy, json, time, inspect, platform
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'../build/matplotlib'))
import numpy as np
import scipy
import matplotlib
matplotlib.use('Agg')
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent.parent
SRC=ROOT/'code/recovered_originals'
DEST=ROOT/'results/reproduction'
DEST.mkdir(parents=True,exist_ok=True)

def dump(name,obj):
    (DEST/name).write_text(json.dumps(obj,indent=2)+'\n')

def replay():
    ns=runpy.run_path(str(SRC/'example3_25112025_claude_code.py'))
    # runpy functions keep their original global dictionary.
    ns=ns['solve_method_4_for_N'].__globals__
    calls=[]
    def recorded_minimize(*args,**kwargs):
        start=time.perf_counter()
        result=minimize(*args,**kwargs)
        record={k:float(result[k]) if k=='fun' else int(result[k]) for k in ['fun','nit','nfev','status']}
        record.update(success=bool(result.success),message=str(result.message),seconds=time.perf_counter()-start)
        calls.append(record)
        print('Optimizer:',record,flush=True)
        return result
    ns['minimize']=recorded_minimize
    truth=ns['v_true_vals'];pts=ns['pts_eval']
    ricfun=ns['solve_method_2a']()
    ric=np.array([ricfun(x) for x in pts])
    summary=dict(environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,blas_threads=1),seed=42,records=[])
    np.savez_compressed(DEST/'example3_grid.npz',points=pts,truth=truth,riccati=ric,coordinates=ns['x_coords'])
    def values(fun):
        closure=inspect.getclosurevars(fun).nonlocals
        xp=closure['xp'];alpha=closure['alpha_final'];w=closure['beta_norm_final']
        def evaluate(p):
            v=np.zeros(len(p))
            # Scalar-center summation follows the recovered function's order.
            for j,z in enumerate(xp):
                kval=np.zeros(len(p));dots=1+p@z
                for l,weight in enumerate(w):
                    if weight<=1e-8: continue
                    kv=np.exp(-np.sum((p-z)**2,axis=1)/(2*[.5,1.5][l]**2)) if l<2 else dots**l
                    kval+=weight*kv
                v+=alpha[j]*kval
            return v
        out=evaluate(pts)
        indices=[0,123,777,3999,7999]
        exact=np.array([fun(pts[i]) for i in indices])
        err=float(np.max(np.abs(out[indices]-exact)))
        if not np.allclose(out[indices],exact,atol=1e-10,rtol=1e-10):
            raise AssertionError(('Prediction replay mismatch',err))
        return out,xp,alpha,w,err,float(fun(np.zeros(3)))
    for n in [50,100,200,400,800]:
        print('BEGIN N =',n,flush=True);start=time.perf_counter();calls.clear()
        fun=ns['solve_method_4_for_N'](n,verbose=True)
        v,xp,alpha,w,err,anchor=values(fun)
        rmse,maxae,rell2=ns['calculate_metrics'](v,truth)
        record=dict(n=n,rmse=float(rmse),max_abs=float(maxae),relative_l2=float(rell2),riccati_rmse=float(np.sqrt(np.mean((ric-truth)**2))),weights=w.tolist(),anchor=anchor,optimizers=list(calls),prediction_check_max_abs=err,seconds=time.perf_counter()-start)
        summary['records'].append(record)
        np.savez_compressed(DEST/f'example3_N{n}.npz',centers=xp,alpha=alpha,weights=w,values=v)
        dump('example3_replay.json',summary)
        print('COMPLETE',json.dumps(record),flush=True)
    # Historical contour code did not set a seed. Declare one rather than infer it.
    np.random.seed(42)
    base=runpy.run_path(str(SRC/'example3_22112025_gpt_code.py'))
    base=base['solve_method_2b'].__globals__;base['minimize']=recorded_minimize
    baseline_records=[];arrays={}
    for label,func in [('quadratic','solve_method_2b'),('polynomial','solve_method_3')]:
        calls.clear();name,fun=base[func]()
        v=np.array([fun(x) for x in pts]);rmse,maxae,rell2=base['calculate_metrics'](v,truth)
        baseline_records.append(dict(label=label,rmse=float(rmse),max_abs=float(maxae),relative_l2=float(rell2),optimizers=list(calls)))
        arrays[label]=v
        for key,value in inspect.getclosurevars(fun).nonlocals.items():
            if isinstance(value,np.ndarray):arrays[label+'_'+key]=value
    np.savez_compressed(DEST/'example3_baselines.npz',**arrays)
    summary['baseline_records']=baseline_records
    dump('example3_replay.json',summary)

if __name__=='__main__':
    with threadpool_limits(1):replay()
