"""Replay the recovered Maier--Stein sweep; retain and flag failed solves.
The kernel-value cache is checked against the unchanged recovered function.
"""
from pathlib import Path
import ast, json, time, os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key]='1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent.parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'build/matplotlib'))
source=ROOT/'code/recovered_originals/maier_stein_24Nov2025.py'
tree=ast.parse(source.read_text())
prefix=[];tail=[];started=False
for node in tree.body:
    if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='lam1':started=True
    (tail if started else prefix).append(node)
ns={'__name__':'replay_maier'}
exec(compile(ast.Module(body=prefix,type_ignores=[]),str(source),'exec'),ns)
original=ns['build_phi'];centers=ns['XY'];x0=ns['xbar'];n=ns['n'];N=len(centers)
kernels=ns['base_kernels'];eps=1e-5
shift=np.eye(2)*eps
print('Precomputing exact scalar kernel samples.',flush=True)
start=time.perf_counter()
plus=np.empty((7,N,N,2));minus=np.empty_like(plus)
anchor_plus=np.empty((7,N,2));anchor_minus=np.empty_like(anchor_plus)
for l,kernel in enumerate(kernels):
    for q,z in enumerate(centers):
        for a in range(2):
            anchor_plus[l,q,a]=kernel(x0+shift[a],z)
            anchor_minus[l,q,a]=kernel(x0-shift[a],z)
            for p,x in enumerate(centers):
                plus[l,p,q,a]=kernel(x+shift[a],z)
                minus[l,p,q,a]=kernel(x-shift[a],z)
F=ns['F_ms'](centers);linear=(centers-x0)@ns['evecR']
def fast_phi(beta,i):
    # Preserve the recovered code's short-beta zip semantics and operation order.
    gp=(sum(b*plus[l] for l,b in enumerate(beta))-sum(b*minus[l] for l,b in enumerate(beta)))/(2*eps)
    ga=(sum(b*anchor_plus[l] for l,b in enumerate(beta))-sum(b*anchor_minus[l] for l,b in enumerate(beta)))/(2*eps)
    K=np.zeros((N+3,N+3));Y=np.zeros(N+3)
    K[0,0]=1.;K[1:3,3:]=ga.T
    Y[1+i]=1.
    # The column-indexed drift is retained to replay the source exactly.
    K[3:,3:]=gp[:,:,0]*F[None,:,0]+gp[:,:,1]*F[None,:,1]
    Y[3:]=-ns['evals'][i]*linear[:,i]
    K+=1e-6*np.eye(N+3)
    h=np.linalg.solve(K,Y)
    return (h[3:]+linear[:,i]).reshape(n,n)

with threadpool_limits(1):
    beta=np.ones(7)/7
    checks=[]
    for weights in [beta, np.array([.05,.10,.15,.20,.10,.15,.25]), np.array([.165,.271,.565])]:
        for mode in [0,1]:
            old=original(weights,mode);new=fast_phi(weights,mode)
            checks.append(dict(beta=weights.tolist(),mode=mode,max_abs=float(np.max(abs(old-new))),relative=float(np.linalg.norm(old-new)/np.linalg.norm(old))))
    comparison={'cases':checks,'maximum_absolute_difference':max(c['max_abs'] for c in checks)}
    assert comparison['maximum_absolute_difference'] == 0., comparison
    print('Original vs cached evaluation:',comparison,'seconds',time.perf_counter()-start,flush=True)
    ns['build_phi']=fast_phi
    records=[];surfaces=[];full=[];masks=[]
    for lam in ns['lambda_list']:
        started=time.perf_counter()
        opt=ns['minimize'](ns['loss_with_penalty'],beta,args=(10.,lam),constraints=ns['cons'],method='SLSQP',options={'maxiter':200})
        bp=opt.x.copy();mask=bp>=1/7;kept=bp[mask];kept/=kept.sum()
        phi=np.c_[fast_phi(kept,0).ravel(),fast_phi(kept,1).ravel()]
        pm=ns['evecR']@np.diag(-2*ns['evals'])@ns['inv'](ns['evecR'])
        v=(.5*np.einsum('ij,jk,ik->i',phi,pm,phi)).reshape(n,n)
        X,Y=ns['Xg'],ns['Yg'];truth=.5*((X+1)**2*(X-1)**2+2*(X*X+1)*Y*Y)
        scale=float(truth.ravel()@v.ravel()/(v.ravel()@v.ravel()));v*=scale
        record=dict(sparsity=lam,active=int(sum(mask)),rmse=float(np.sqrt(np.mean((v-truth)**2))),
            beta_full=bp.tolist(),beta_kept=kept.tolist(),success=bool(opt.success),status=int(opt.status),
            message=str(opt.message),simplex_sum=float(bp.sum()),minimum_weight=float(bp.min()),feasible=bool(abs(bp.sum()-1)<1e-6 and bp.min()>=-1e-6),nfev=int(opt.nfev),nit=int(opt.nit),scale=scale,seconds=time.perf_counter()-started)
        records.append(record);surfaces.append(v);full.append(bp);masks.append(mask)
        print(json.dumps(record),flush=True)
        dest=ROOT/'results/reproduction';dest.mkdir(parents=True,exist_ok=True)
        (dest/'maier_replay.json').write_text(json.dumps({'cached_evaluation_check':comparison,'records':records},indent=2)+'\n')
        np.savez_compressed(dest/'maier_replay.npz',X=X,Y=Y,truth=truth,potentials=np.asarray(surfaces),beta_full=full,masks=masks)
