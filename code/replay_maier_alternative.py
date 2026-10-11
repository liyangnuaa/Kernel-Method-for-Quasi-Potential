"""Exact-formula replay of the second recovered 24 November script."""
from pathlib import Path
import ast,json,os,time,sys
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'mpl'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
from threadpoolctl import threadpool_limits
ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parents[1]
OUT=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT/'results/accuracy';OUT.mkdir(parents=True,exist_ok=True)
source=ROOT/'code/recovered_originals/maier_stein_24Nov2025a.py';tree=ast.parse(source.read_text());prefix=[]
for node in tree.body:
 if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='lam1':break
 prefix.append(node)
ns={'__name__':'maier_alternative'};exec(compile(ast.Module(body=prefix,type_ignores=[]),str(source),'exec'),ns)
original=ns['build_phi'];centers=ns['XY'];x0=ns['xbar'];n=ns['n'];N=len(centers);F=ns['F_ms'](centers);linear=(centers-x0)@ns['evecR']
plus=np.empty((7,N,N,2));minus=np.empty_like(plus);ap=np.empty((7,N,2));am=np.empty_like(ap)
for l,k in enumerate(ns['base_kernels']):
 for q,z in enumerate(centers):
  for a in range(2):
   e=np.eye(2)[a];ap[l,q,a]=k(x0+1e-5*e,z);am[l,q,a]=k(x0-1e-5*e,z)
   for p,x in enumerate(centers):plus[l,p,q,a]=k(x+1e-6*e,z);minus[l,p,q,a]=k(x-1e-6*e,z)
def fast(beta,i):
 gp=(sum(b*plus[l] for l,b in enumerate(beta))-sum(b*minus[l] for l,b in enumerate(beta)))/(2e-6)
 ga=(sum(b*ap[l] for l,b in enumerate(beta))-sum(b*am[l] for l,b in enumerate(beta)))/(2e-5)
 K=np.zeros((N+3,N+3));Y=np.zeros(N+3);K[0,0]=1.;K[1:3,3:]=ga.T;Y[1+i]=1.
 K[3:,3:]=gp[:,:,0]*F[None,:,0]+gp[:,:,1]*F[None,:,1];Y[3:]=-ns['evals'][i]*linear[:,i];K+=1e-6*np.eye(N+3)
 return (np.linalg.solve(K,Y)[3:]+linear[:,i]).reshape(n,n)
records=[];surfaces=[]
with threadpool_limits(1):
 checks=[float(np.max(abs(original(b,i)-fast(b,i)))) for b in [np.ones(7)/7,np.array([.165,0,.271,0,.565,0,0])] for i in [0,1]]
 assert max(checks)==0.,checks
 ns['build_phi']=fast
 for lam in ns['lambda_list']:
  t=time.perf_counter();opt=ns['minimize'](ns['loss_with_penalty'],np.ones(7)/7,args=(100.,lam),method='L-BFGS-B',bounds=ns['bounds'],options={'maxiter':200})
  b=opt.x;mask=b>=1/7;bf=np.where(mask,b,0);bf/=bf.sum() if bf.sum()>0 else 1.
  phi=np.c_[fast(bf,0).ravel(),fast(bf,1).ravel()];pm=ns['evecR']@np.diag(-2*ns['evals'])@np.linalg.inv(ns['evecR']);v=.5*np.einsum('ij,jk,ik->i',phi,pm,phi)
  X,Y=ns['Xg'],ns['Yg'];truth=.5*(X*X-1)**2+(X*X+1)*Y*Y;scale=truth.ravel()@v/(v@v);v*=scale
  r=dict(sparsity=lam,success=bool(opt.success),status=int(opt.status),message=str(opt.message),active=int(mask.sum()),raw_weights=b.tolist(),prediction_weights=bf.tolist(),rmse=float(np.sqrt(np.mean((v-truth.ravel())**2))),scale=float(scale),fun=float(opt.fun),nit=int(opt.nit),nfev=int(opt.nfev),seconds=time.perf_counter()-t)
  records.append(r);surfaces.append(v.reshape(n,n));print(json.dumps(r),flush=True)
  (OUT/'maier_alternative.json').write_text(json.dumps(dict(cached_function_max_abs=max(checks),source=source.name,records=records),indent=2)+'\n')
  np.savez_compressed(OUT/'maier_alternative.npz',X=X,Y=Y,truth=truth,values=surfaces)
