"""Stable refit of the Poly(2) component selected by every literal Example 3 run.

The recovered direct scalar kernel model is retained. Replace its redundant
N kernel coefficients by the nine coefficients of its anchored quadratic.
Use exactly normalized kernel weight one. Minimize the same HJ + curvature +
coefficient L2 terms (L1 weight term is constant .05 for this one-kernel refit).
The minimum-norm N-vector is reconstructed by the pseudoinverse of the feature
map. No truth values enter fitting; local curvature comes from Lyapunov.
"""
from pathlib import Path
import os,sys,json,time
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import solve_continuous_lyapunov
from threadpoolctl import threadpool_limits
OUT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parents[1]/'results/accuracy';OUT.mkdir(parents=True,exist_ok=True)
J=np.array([[-1.,0,0],[0,-.5,-5],[0,5,-.5]])
P=np.linalg.inv(solve_continuous_lyapunov(J,-np.eye(3)))
EXP=np.array([[1,0,0],[0,1,0],[0,0,1],[2,0,0],[0,2,0],[0,0,2],[1,1,0],[1,0,1],[0,1,1]])
H=np.zeros((9,3,3))
for i in range(3):H[3+i,i,i]=2
for k,(i,j) in enumerate([(0,1),(0,2),(1,2)],6):H[k,i,j]=H[k,j,i]=1

def features(x):
 K=np.prod(x[:,None]**EXP[None],axis=2);G=np.zeros((len(x),9,3))
 for d in range(3):
  e=EXP.copy();e[:,d]=np.maximum(e[:,d]-1,0);G[:,:,d]=EXP[None,:,d]*np.prod(x[:,None]**e[None],axis=2)
 return K,G

def field(x):
 b=x@J.T;b[:,0]-=x[:,0]**3;a=np.ones_like(x);a[:,0]+=x[:,0]**2;return b,a

def metrics(x,c):
 K,G=features(x);v=K@c;g=np.einsum('nmd,m->nd',G,c);b,a=field(x);r=np.sum(b*g+.5*a*g*g,axis=1);truth=x[:,0]**2+.5*np.sum(x[:,1:]**2,axis=1)
 return dict(rmse=float(np.sqrt(np.mean((v-truth)**2))),max_abs=float(max(abs(v-truth))),relative_l2=float(np.linalg.norm(v-truth)/np.linalg.norm(truth)),hj_mean_abs=float(np.mean(abs(r))),hj_max=float(max(abs(r)))),v,g

def main():
 records=[];coords=np.linspace(-2,2,20);points=np.array(np.meshgrid(coords,coords,coords,indexing='ij')).reshape(3,-1).T;test=np.random.default_rng(20260923).uniform(-2,2,(10000,3))
 for n in [50,100,200,400,800]:
  start=time.perf_counter();centers=np.random.RandomState(42).uniform(-2,2,(n,3));K,G=features(centers)
  # (1+x.z)^2 - (1+0.z)^2 has the following feature coefficient map.
  mapping=K.T.copy();mapping[:3]*=2;mapping[6:]*=2
  inverse=np.linalg.pinv(mapping,rcond=1e-12);rank=int(np.linalg.matrix_rank(mapping));assert rank==9
  b,a=field(centers)
  def residual(c):
   g=np.einsum('nmd,m->nd',G,c);r=np.sum(b*g+.5*a*g*g,axis=1);h=np.einsum('m,mij->ij',c,H)-P
   return np.r_[r/np.sqrt(n),np.sqrt(10)*h.ravel(),.01*(inverse@c)]
  def jac(c):
   g=np.einsum('nmd,m->nd',G,c)
   return np.vstack([np.einsum('nd,nmd->nm',b+a*g,G)/np.sqrt(n),np.sqrt(10)*H.reshape(9,9).T,.01*inverse])
  initial=np.r_[np.zeros(3),np.diag(P)/2,P[0,1],P[0,2],P[1,2]]
  opt=least_squares(residual,initial,jac=jac,ftol=1e-13,xtol=1e-13,gtol=1e-10,max_nfev=1000)
  c=opt.x;alpha=inverse@c;m,v,g=metrics(points,c);tm,tv,tg=metrics(test,c)
  literal=((1+points[:50]@centers.T)**2-1)@alpha
  check=float(np.max(abs(literal-v[:50])));assert check<1e-10
  direction=np.random.default_rng(5).normal(size=9);direction/=np.linalg.norm(direction);step=1e-6;fd=(residual(c+step*direction)-residual(c-step*direction))/(2*step);jerr=float(max(abs(fd-jac(c)@direction)));assert jerr<1e-7
  r=dict(n=n,seed=42,success=bool(opt.success),status=int(opt.status),message=opt.message,nfev=int(opt.nfev),optimality=float(opt.optimality),objective=float(2*opt.cost+.05),coefficients=c.tolist(),weights=[0.,0.,1.,0.,0.,0.,0.],anchor=0.,gradient_at_anchor=c[:3].tolist(),hessian_error=float(np.linalg.norm(np.einsum('m,mij->ij',c,H)-P)),grid_metrics=m,independent_test_metrics=tm,rank=rank,kernel_prediction_check=check,jacobian_check=jerr,seconds=time.perf_counter()-start)
  assert r['success'] and r['optimality']<1e-8,r
  print(json.dumps(r),flush=True);records.append(r)
  np.savez_compressed(OUT/f'example3_stable_N{n}.npz',centers=centers,alpha=alpha,coefficients=c,points=points,values=v,gradient=g,test=test,test_values=tv,test_gradient=tg,P=P)
 (OUT/'example3_stable.json').write_text(json.dumps(dict(protocol='Poly(2) selected-kernel refit: exact amplitude normalization; subtract V(0); rank-9 feature coordinates; minimum-norm original alpha; HJ+10 curvature+1e-4 alpha norm; source L1 term constant .05; seed 42',records=records),indent=2)+'\n')
if __name__=='__main__':
 with threadpool_limits(1):main()
