"""Recovered Example 3 objective with analytic derivatives and larger budgets.
Training uses dynamics and local Riccati curvature. V(0)=0 is imposed at output.
No reference potential values are used to select parameters or stopping.
"""
from pathlib import Path
import os, sys, json, runpy, inspect, time
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'mpl'))
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve_continuous_lyapunov
from threadpoolctl import threadpool_limits
ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parents[1]
OUT=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT/'results/accuracy'
OUT.mkdir(parents=True,exist_ok=True)
ns=runpy.run_path(str(ROOT/'code/recovered_originals/example3_25112025_claude_code.py'))
ns=ns['solve_method_4_for_N'].__globals__
ns['H_true']=np.linalg.inv(solve_continuous_lyapunov(ns['J'],-ns['a'](ns['xbar'])))
checks=[];calls=[]

def analytic_objective(fun):
    c=inspect.getclosurevars(fun).nonlocals
    if 'loss' in c:
        active=c['active_mask'];c=inspect.getclosurevars(c['loss']).nonlocals
    else:active=np.ones(7,dtype=bool)
    G=c['G_val'][active];H=c['H_val'][active];b=c['bxp'];a=c['axp'];n=c['N_col']
    def value_grad(z):
        alpha=z[:n];beta=z[n:];den=beta@beta+1e-10;w=beta**2/den
        gi=np.einsum('j,lijd->lid',alpha,G);hi=np.einsum('j,ljab->lab',alpha,H)
        g=np.einsum('l,lid->id',w,gi);hh=np.einsum('l,lab->ab',w,hi)
        ag=np.einsum('iab,ib->ia',a,g);r=np.sum(b*g+.5*g*ag,axis=1);dh=hh-ns['H_true']
        value=np.mean(r*r)+10*np.sum(dh*dh)+.05*np.sum(abs(beta))+1e-4*(alpha@alpha+beta@beta)
        gg=2*r[:,None]*(b+ag)/n;gh=20*dh
        grad_alpha=np.einsum('l,lijd,id->j',w,G,gg)+np.einsum('l,ljab,ab->j',w,H,gh)+2e-4*alpha
        dw=np.einsum('lid,id->l',gi,gg)+np.einsum('lab,ab->l',hi,gh)
        grad_beta=2*beta/den*(dw-w@dw)+.05*np.sign(beta)+2e-4*beta
        return value,np.r_[grad_alpha,grad_beta]
    return value_grad

def recorded_minimize(fun,x0,*args,**kwargs):
    objective=analytic_objective(fun)
    f,j=objective(x0);f0=fun(x0)
    rng=np.random.default_rng(19);errs=[]
    for _ in range(4):
        d=rng.normal(size=len(x0));d/=np.linalg.norm(d);h=1e-9
        fd=(fun(x0+h*d)-fun(x0-h*d))/(2*h)
        errs.append(abs(fd-j@d)/max(1,abs(fd),abs(j@d)))
    checks.append(dict(value_abs_difference=float(abs(f-f0)),directional_gradient_relative_error=float(max(errs))))
    assert abs(f-f0)<1e-8*max(1,abs(f0)),checks[-1]
    assert max(errs)<2e-5,checks[-1]
    t=time.perf_counter()
    opt=minimize(objective,x0,jac=True,method='L-BFGS-B',options={'maxiter':5000,'maxfun':100000,'ftol':1e-13,'gtol':1e-8,'maxls':50,'maxcor':30})
    calls.append(dict(success=bool(opt.success),status=int(opt.status),message=str(opt.message),fun=float(opt.fun),nfev=int(opt.nfev),nit=int(opt.nit),gradient_max=float(max(abs(opt.jac))),seconds=time.perf_counter()-t))
    print(calls[-1],flush=True)
    return opt
ns['minimize']=recorded_minimize

def evaluate(x,xp,alpha,w):
    values=np.zeros(len(x));gradient=np.zeros_like(x)
    for lo in range(0,len(x),2000):
        xx=x[lo:lo+2000];t=1+xx@xp.T;delta=xx[:,None]-xp
        for l,weight in enumerate(w):
            if weight<=1e-8:continue
            if l<2:
                s=[.5,1.5][l];k=np.exp(-np.sum(delta**2,axis=2)/(2*s*s));dk=-delta*k[:,:,None]/s**2
            else:
                k=t**l;dk=l*t[:,:,None]**(l-1)*xp[None]
            values[lo:lo+len(xx)]+=weight*(k@alpha)
            gradient[lo:lo+len(xx)]+=weight*np.einsum('ijd,j->id',dk,alpha)
    return values,gradient

def metrics(v,g,x):
    truth=np.array([ns['V_true'](xx) for xx in x]);b=np.array([ns['b'](xx) for xx in x]);a=np.array([ns['a'](xx) for xx in x]);r=np.sum(b*g,axis=1)+.5*np.einsum('ia,iab,ib->i',g,a,g)
    return dict(rmse=float(np.sqrt(np.mean((v-truth)**2))),max_abs=float(max(abs(v-truth))),relative_l2=float(np.linalg.norm(v-truth)/np.linalg.norm(truth)),hj_mean_abs=float(np.mean(abs(r))),hj_max=float(max(abs(r))))

records=[]
with threadpool_limits(1):
 for n in [50,100,200,400,800]:
    calls.clear();checks.clear();start=time.perf_counter();print('BEGIN',n,flush=True)
    fun=ns['solve_method_4_for_N'](n,verbose=True);c=inspect.getclosurevars(fun).nonlocals
    xp,alpha,w=c['xp'],c['alpha_final'],c['beta_norm_final']
    points=ns['pts_eval'];values,g=evaluate(points,xp,alpha,w);anchor,anchor_gradient=evaluate(np.zeros((1,3)),xp,alpha,w);v=values-anchor[0]
    test=np.random.default_rng(20260923).uniform(-2,2,(10000,3));tv,tg=evaluate(test,xp,alpha,w);tv-=anchor[0]
    hess=sum(w[l]*sum(alpha[j]*ns['kernel_defs'][l]['hess'](np.zeros(3),z) for j,z in enumerate(xp)) for l in range(7))
    record=dict(n=n,seed=42,weights=w.tolist(),anchor_before=float(anchor[0]),anchor_after=0.,gradient_at_anchor=anchor_gradient[0].tolist(),hessian_error=float(np.linalg.norm(hess-ns['H_true'])),raw_metrics=metrics(values,g,points),normalized_metrics=metrics(v,g,points),independent_test_metrics=metrics(tv,tg,test),optimizers=list(calls),gradient_checks=list(checks),seconds=time.perf_counter()-start)
    records.append(record);print('COMPLETE',json.dumps(record),flush=True)
    np.savez_compressed(OUT/f'example3_converged_N{n}.npz',centers=xp,alpha=alpha,weights=w,points=points,raw_values=values,values=v,gradient=g,test_points=test,test_values=tv,test_gradient=tg)
    (OUT/'example3_converged.json').write_text(json.dumps(dict(protocol='Recovered objective; analytic gradient; maxiter=5000, maxfun=100000, ftol=1e-13, gtol=1e-8, maxls=50, maxcor=30; output anchored by V(x)-V(0); seed=42; no reference-based tuning',records=records),indent=2)+'\n')
