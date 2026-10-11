"""Anchored polynomial-kernel HJ experiments for the two 2D examples.

These experiments fit the potential directly in a polynomial RKHS. Polynomial spaces r=2,...,6 are components of the paper dictionary.
Local Riccati curvature is imposed exactly. A fixed domain continuation avoids
spurious HJ stationary points. All model selection uses held-out HJ residuals.
Reference potentials are evaluated only after selection.
"""
from pathlib import Path
import os,sys,json,platform,time
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
import numpy as np
from scipy.linalg import solve_continuous_lyapunov
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits
OUT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parents[1]/'results/accuracy';OUT.mkdir(parents=True,exist_ok=True)
ETA=1e-8
RADII=[.125,.25,.5,1.]

def problem(name):
 x0=np.array([-1.,0.]) if name=='maier_stein' else np.array([1.,0.]);F=np.diag([-2.,-2. if name=='maier_stein' else -1.]);P=np.linalg.inv(solve_continuous_lyapunov(F,-np.eye(2)))
 def drift(x):
  u,v=x.T
  return np.c_[u-u**3-u*v*v,-(1+u*u)*v] if name=='maier_stein' else np.c_[u-u**3,-v]
 return x0,P,drift

def reference(name,x):
 u,v=x.T
 return .5*(u*u-1)**2+((u*u+1) if name=='maier_stein' else 1)*v*v

def basis(x,x0,degree):
 # Explicit basis of the polynomial RKHS, restricted to zero second jet.
 y=x-x0;e=np.array([(i,j) for i in range(degree+1) for j in range(degree+1-i) if 3<=i+j<=degree],dtype=int).reshape(-1,2)
 v=np.prod(y[:,None]**e[None],axis=2);g=np.zeros((len(x),len(e),2))
 for d in [0,1]:
  ee=e.copy();ee[:,d]=np.maximum(ee[:,d]-1,0);g[:,:,d]=e[None,:,d]*np.prod(y[:,None]**ee[None],axis=2)
 return v,g,e

def predict(x,x0,P,degree,c):
 K,G,_=basis(x,x0,degree);y=x-x0
 return .5*np.sum(y*(y@P),axis=1)+K@c,y@P+np.einsum('nmd,m->nd',G,c)

def hj(b,g):return np.sum(b*g+.5*g*g,axis=1)

def fit(train,x0,P,drift,degree):
 _,_,e=basis(train[:1],x0,degree);coef=np.zeros(len(e));stages=[]
 if degree==2:return coef,stages
 for radius in RADII:
  x=x0+radius*(train-x0);b=drift(x);g0=(x-x0)@P;_,G,_=basis(x,x0,degree)
  s=np.maximum(np.sqrt(np.mean(G*G,axis=(0,2))),1e-8);G/=s[None,:,None]
  def residual(c):
   g=g0+np.einsum('nmd,m->nd',G,c)
   return np.r_[hj(b,g)/np.sqrt(len(x)),np.sqrt(ETA)*c/s]
  def jacobian(c):
   g=g0+np.einsum('nmd,m->nd',G,c)
   return np.vstack([np.einsum('nd,nmd->nm',b+g,G)/np.sqrt(len(x)),np.diag(np.sqrt(ETA)/s)])
  z=least_squares(residual,coef*s,jac=jacobian,max_nfev=3000,gtol=1e-10,ftol=1e-13,xtol=1e-13)
  coef=z.x/s
  stages.append(dict(radius=radius,success=bool(z.success),status=int(z.status),nfev=int(z.nfev),message=z.message,optimality=float(z.optimality),objective=float(2*z.cost)))
 return coef,stages

def metrics(name,x,x0,P,degree,c,drift):
 v,g=predict(x,x0,P,degree,c);truth=reference(name,x);r=hj(drift(x),g)
 return dict(rmse=float(np.sqrt(np.mean((v-truth)**2))),max_abs=float(max(abs(v-truth))),relative_l2=float(np.linalg.norm(v-truth)/np.linalg.norm(truth)),hj_mean_abs=float(np.mean(abs(r))),hj_max=float(max(abs(r))))

def grid(lo,hi,n):return np.array(np.meshgrid(*[np.linspace(a,b,n) for a,b in zip(lo,hi)])).reshape(2,-1).T

def run():
 records=[]
 for name in ['maier_stein','double_well']:
  x0,P,drift=problem(name);lo,hi=([-4.,-3.],[0.,3.]) if name=='maier_stein' else ([.05,-1.5],[1.5,1.5])
  for seed in [42,43,44]:
   start=time.perf_counter();rng=np.random.RandomState(seed);train=rng.uniform(lo,hi,(200,2));valid=np.random.default_rng(seed+10000).uniform(lo,hi,(2048,2));candidates=[]
   for degree in range(2,7):
    c,stages=fit(train,x0,P,drift,degree);_,g=predict(valid,x0,P,degree,c);score=float(np.sqrt(np.mean(hj(drift(valid),g)**2)))
    candidates.append(dict(degree=degree,coefficients=c.tolist(),validation_hj_rms=score,stages=stages))
   # Prefer the simplest model meeting a fixed PDE tolerance, not the best RMSE.
   acceptable=[r for r in candidates if r['validation_hj_rms']<=1e-6 and all(z['success'] for z in r['stages'])]
   chosen=min(acceptable,key=lambda r:r['degree']) if acceptable else min(candidates,key=lambda r:r['validation_hj_rms'])
   degree=chosen['degree'];c=np.array(chosen['coefficients']);test=np.random.default_rng(20260923).uniform(lo,hi,(10000,2))
   display=grid([-4,-3],[0,3],10) if name=='maier_stein' else grid([-1.5,-1.5],[1.5,1.5],101)
   spatial=grid([-2.5,-1.8],[.2,1.8],101) if name=='maier_stein' else display.copy()
   r=dict(problem=name,seed=seed,n=200,drift_queries=800,training_bounds=[lo,hi],chosen_degree=degree,candidates=candidates,display_metrics=metrics(name,display,x0,P,degree,c,drift),spatial_metrics=metrics(name,spatial,x0,P,degree,c,drift),independent_basin_metrics=metrics(name,test,x0,P,degree,c,drift),riccati_display_metrics=metrics(name,display,x0,P,2,np.array([]),drift),anchor_value=0.,anchor_gradient=[0.,0.],hessian_error=0.,seconds=time.perf_counter()-start)
   records.append(r);print(json.dumps({k:v for k,v in r.items() if k!='candidates'}),flush=True)
   np.savez_compressed(OUT/f'{name}_polynomial_seed{seed}.npz',train=train,validation=valid,display=display,spatial=spatial,test=test,x0=x0,P=P,degree=degree,coefficients=c,display_values=predict(display,x0,P,degree,c)[0],spatial_values=predict(spatial,x0,P,degree,c)[0],test_values=predict(test,x0,P,degree,c)[0])
   (OUT/'polynomial_experiments.json').write_text(json.dumps(dict(protocol='New direct scalar polynomial-kernel HJ experiments; fixed curvature; Riccati warm start; four fixed continuation radii; five candidate degrees; no analytic-reference training or selection',eta=ETA,radii=RADII,model_selection='Smallest degree with validation HJ RMS <=1e-6, otherwise lowest validation HJ RMS',records=records),indent=2)+'\n')
 # Derivative check independent of optimizer implementation.
 x0,P,b=problem('maier_stein');x=np.random.default_rng(9).uniform(-1,1,(5,2));c=np.random.default_rng(10).normal(size=len(basis(x,x0,6)[2]));_,g=predict(x,x0,P,6,c);errors=[]
 for j in range(2):
  e=np.eye(2)[j]*1e-6;fd=(predict(x+e,x0,P,6,c)[0]-predict(x-e,x0,P,6,c)[0])/(2e-6);errors.append(float(max(abs(fd-g[:,j]))))
 assert max(errors)<1e-6,errors
 (OUT/'polynomial_derivative_check.json').write_text(json.dumps(dict(max_absolute_error=max(errors),passed=True),indent=2)+'\n')
if __name__=='__main__':
 with threadpool_limits(1):run()
