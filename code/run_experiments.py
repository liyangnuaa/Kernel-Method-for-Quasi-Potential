"""Reproducible revision experiments; no reference values enter training.

Run from the package directory: python code/run_experiments.py
Requires numpy, scipy, matplotlib, threadpoolctl. CPU, float64, one BLAS thread.
"""
import json
import os
import platform
import sys
from pathlib import Path
from time import perf_counter

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / '.mplcache'))
for thread_var in ['VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ[thread_var] = '1'
import numpy as np
import scipy
from scipy.linalg import solve_continuous_lyapunov
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits, threadpool_info
import matplotlib
matplotlib.use('Agg')

ROOT = Path(__file__).resolve().parents[1]
ETA = 1e-8
SEEDS = [230923, 230924, 230925]
BANDWIDTHS = [0.5, 1.0, 2.0]


def problem(name):
    if name == 'maier_stein':
        d, x0, lo, hi = 2, np.array([-1., 0.]), [-1.5, -.6], [-.5, .6]
        F = -2 * np.eye(2)
        def field(x):
            u, v = x.T
            return np.column_stack([u-u**3-u*v*v, -(1+u*u)*v])
        def exact(x):
            u, v = x.T
            return .5*(u*u-1)**2+(u*u+1)*v*v
        def exact_gradient(x):
            return -2*field(x)
        def diffusion(x):
            return np.ones_like(x)
    elif name == 'double_well':
        d, x0, lo, hi = 2, np.array([1., 0.]), [.5, -.6], [1.5, .6]
        F = np.diag([-2., -1.])
        def field(x):
            return np.column_stack([x[:, 0]-x[:, 0]**3, -x[:, 1]])
        def exact(x):
            return .5*(x[:, 0]**2-1)**2+x[:, 1]**2
        def exact_gradient(x):
            return -2*field(x)
        def diffusion(x):
            return np.ones_like(x)
    elif name == 'multiplicative':
        d, x0, lo, hi = 3, np.zeros(3), [-1]*3, [1]*3
        F = np.array([[-1., 0, 0], [0, -.5, -5], [0, 5, -.5]])
        def field(x):
            b = x @ F.T
            b[:, 0] -= x[:, 0]**3
            return b
        def exact(x):
            return x[:, 0]**2+.5*np.sum(x[:, 1:]**2, axis=1)
        def exact_gradient(x):
            return x*np.array([2., 1., 1.])
        def diffusion(x):
            a = np.ones_like(x)
            a[:, 0] += x[:, 0]**2
            return a
    elif name == 'rotational_8d':
        d, x0, lo, hi = 8, np.zeros(8), [-1]*8, [1]*8
        R = np.zeros((d, d))
        for j in range(d-1):
            R[j, j+1], R[j+1, j] = .25, -.25
        F = -np.eye(d)+2*R
        def exact_gradient(x):
            return 2*x+.4*x**3
        def field(x):
            g = exact_gradient(x)
            return -.5*g+g@R.T
        def exact(x):
            return np.sum(x*x+.1*x**4, axis=1)
        def diffusion(x):
            return np.ones_like(x)
    else:
        raise ValueError(name)
    C = solve_continuous_lyapunov(F, -np.eye(d))
    P = np.linalg.inv(C)
    return dict(name=name, d=d, x0=x0, lo=np.array(lo), hi=np.array(hi), F=F,
                P=P, b=field, a=diffusion, exact=exact, exact_gradient=exact_gradient)


def jet_basis(x, centers, x0, ell, order):
    """Gaussian sections with first/second Taylor jet at x0 subtracted."""
    z, z0, y = x[:, None, :]-centers[None, :, :], x0-centers, x-x0
    K = np.exp(-np.sum(z*z, axis=2)/(2*ell*ell))
    D = -z*K[:, :, None]/ell**2
    k0 = np.exp(-np.sum(z0*z0, axis=1)/(2*ell*ell))
    g0 = -z0*k0[:, None]/ell**2
    K -= k0[None, :]+y@g0.T
    D -= g0[None, :, :]
    if order == 2:
        h0 = k0[:, None, None]*(z0[:, :, None]*z0[:, None, :]/ell**4
                                -np.eye(x.shape[1])[None, :, :]/ell**2)
        K -= .5*np.einsum('ni,mij,nj->nm', y, h0, y)
        D -= np.einsum('mij,nj->nmi', h0, y)
    return K, D


def quadratic(p, x):
    y = x-p['x0']
    return .5*np.einsum('ni,ij,nj->n', y, p['P'], y), y@p['P']


def hj(p, x, gradient):
    return np.sum(p['b'](x)*gradient+.5*p['a'](x)*gradient**2, axis=1)


def fit(p, train, centers, ell, method):
    n, m, d = len(train), len(centers), p['d']
    order = 2 if method == 'direct' else 1
    K, D = jet_basis(train, centers, p['x0'], ell, order)
    b, a = p['b'](train), p['a'](train)
    if method == 'direct':
        _, gq = quadratic(p, train)
        def residual(c):
            g = gq+np.einsum('nmi,m->ni', D, c)
            return np.r_[np.sum(b*g+.5*a*g*g, axis=1)/np.sqrt(n), np.sqrt(ETA)*c]
        def jacobian(c):
            g = gq+np.einsum('nmi,m->ni', D, c)
            J = np.einsum('ni,nmi->nm', b+a*g, D)/np.sqrt(n)
            return np.vstack([J, np.sqrt(ETA)*np.eye(m)])
        opt = least_squares(residual, np.zeros(m), jac=jacobian, max_nfev=200,
                            ftol=1e-10, xtol=1e-10, gtol=1e-10)
        coef = opt.x
        info = dict(nfev=int(opt.nfev), status=int(opt.status), optimality=float(opt.optimality))
    else:
        # vec(Phi)=vec(y+K C), with row-major C (center, output).
        Db = np.einsum('nmi,ni->nm', D, b)
        L = (np.einsum('nm,ij->nimj', Db, np.eye(d))
             -np.einsum('nm,ij->nimj', K, p['F'])).reshape(n*d, m*d)
        rhs = -b+(train-p['x0'])@p['F'].T
        lhs = L.T@L/n+ETA*np.eye(m*d)
        coef = np.linalg.solve(lhs, L.T@rhs.reshape(-1)/n).reshape(m, d)
        info = dict(nfev=1, status=1, optimality=float(np.max(np.abs(lhs@coef.ravel()-L.T@rhs.ravel()/n))))
    return coef, info


def predict(p, x, centers, ell, method, coef):
    K, D = jet_basis(x, centers, p['x0'], ell, 2 if method == 'direct' else 1)
    if method == 'direct':
        v, g = quadratic(p, x)
        return v+K@coef, g+np.einsum('nmi,m->ni', D, coef)
    phi = x-p['x0']+K@coef
    dphi = np.eye(p['d'])[None, :, :]+np.einsum('nmi,mj->nji', D, coef)
    v = .5*np.einsum('ni,ij,nj->n', phi, p['P'], phi)
    return v, np.einsum('nji,nj->ni', dphi, phi@p['P'])


def metrics(p, test, v, g):
    truth, delta = p['exact'](test), v-p['exact'](test)
    h = hj(p, test, g)
    return dict(rmse=float(np.sqrt(np.mean(delta**2))), rel_l2=float(np.linalg.norm(delta)/np.linalg.norm(truth)),
                max_ae=float(np.max(np.abs(delta))), hj_rms=float(np.sqrt(np.mean(h*h))),
                hj_max=float(np.max(np.abs(h))), min_v=float(np.min(v)))


def check_implementation():
    """Check derivatives independently, including complex spectral blocks."""
    rng = np.random.default_rng(7)
    p = problem('multiplicative')
    x = rng.uniform(-.5, .5, (7, 3)); c = rng.uniform(-1, 1, (5, 3))
    eps = 1e-5
    for order in [1, 2]:
        _, D = jet_basis(x, c, p['x0'], 1.2, order)
        for j in range(3):
            e = np.eye(3)[j]*eps
            fd = (jet_basis(x+e,c,p['x0'],1.2,order)[0]-jet_basis(x-e,c,p['x0'],1.2,order)[0])/(2*eps)
            assert np.max(np.abs(fd-D[:,:,j])) < 2e-8
    for method in ['direct', 'koopman']:
        coef = rng.normal(size=5 if method == 'direct' else (5,3))*.1
        _, g = predict(p,x,c,1.2,method,coef)
        for j in range(3):
            e = np.eye(3)[j]*eps
            fd = (predict(p,x+e,c,1.2,method,coef)[0]-predict(p,x-e,c,1.2,method,coef)[0])/(2*eps)
            assert np.max(np.abs(fd-g[:,j])) < 2e-8
        v0,g0 = predict(p,p['x0'][None,:],c,1.2,method,coef)
        assert np.max(np.abs(v0))+np.max(np.abs(g0)) < 1e-12
        hess = np.column_stack([(predict(p,(p['x0']+np.eye(3)[j]*eps)[None,:],c,1.2,method,coef)[1][0]
                               -predict(p,(p['x0']-np.eye(3)[j]*eps)[None,:],c,1.2,method,coef)[1][0])/(2*eps) for j in range(3)])
        assert np.max(np.abs(hess-p['P'])) < 1e-8
    K,D=jet_basis(x,c,p['x0'],1.2,1)
    coef=rng.normal(size=(5,3))
    Db=np.einsum('nmi,ni->nm',D,p['b'](x))
    L=(np.einsum('nm,ij->nimj',Db,np.eye(3))-np.einsum('nm,ij->nimj',K,p['F'])).reshape(21,15)
    explicit=p['b'](x)+Db@coef-(x+K@coef)@p['F'].T
    assembled=(L@coef.ravel()).reshape(7,3)+p['b'](x)-x@p['F'].T
    assert np.max(np.abs(explicit-assembled)) < 1e-12
    for name in ['maier_stein','double_well','multiplicative','rotational_8d']:
        q = problem(name); x = rng.uniform(q['lo'],q['hi'],(100,q['d']))
        assert np.max(np.abs(hj(q,x,q['exact_gradient'](x)))) < 1e-12
        assert np.max(np.abs(q['F'].T@q['P']+q['P']@q['F']+q['P']@q['P'])) < 1e-12
    return 'PASS: analytic gradients, anchors, Hessians, exact HJ residuals, Riccati equations'


def main():
    (ROOT/'results').mkdir(exist_ok=True); (ROOT/'figures').mkdir(exist_ok=True)
    checks = check_implementation()
    records, arrays = [], {}
    for name in ['maier_stein','double_well','multiplicative','rotational_8d']:
        p = problem(name)
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            train = rng.uniform(p['lo'],p['hi'],(256,p['d']))
            valid = rng.uniform(p['lo'],p['hi'],(512,p['d']))
            test = rng.uniform(p['lo'],p['hi'],(4096,p['d']))
            centers = train[:32].copy()
            start = perf_counter(); v,g = quadratic(p,test); elapsed = perf_counter()-start
            records.append(dict(problem=name,seed=seed,method='riccati',train_seconds=0.,test_seconds=elapsed,
                                **metrics(p,test,v,g)))
            for method in ['direct','koopman']:
                start = perf_counter(); candidates=[]
                for factor in BANDWIDTHS:
                    ell = factor*np.sqrt(p['d'])
                    coef,info = fit(p,train,centers,ell,method)
                    _,gv = predict(p,valid,centers,ell,method,coef)
                    score = np.mean(hj(p,valid,gv)**2)
                    candidates.append((score,ell,coef,info))
                score,ell,coef,info = min(candidates,key=lambda z:z[0])
                elapsed = perf_counter()-start
                start=perf_counter(); v,g=predict(p,test,centers,ell,method,coef); test_time=perf_counter()-start
                row=dict(problem=name,seed=seed,method=method,bandwidth=ell,train_seconds=elapsed,
                         test_seconds=test_time,valid_hj_rms=float(np.sqrt(score)),**info,**metrics(p,test,v,g))
                row['candidates']=[dict(bandwidth=q[1],validation_hj_rms=float(np.sqrt(q[0])),**q[3]) for q in candidates]
                if method=='koopman':
                    kt,dt=jet_basis(test,centers,p['x0'],ell,1)
                    transport=p['b'](test)+np.einsum('nmi,ni->nm',dt,p['b'](test))@coef-(test-p['x0']+kt@coef)@p['F'].T
                    row['transport_rms']=float(np.sqrt(np.mean(np.sum(transport**2,axis=1))))
                records.append(row)
                arrays[f'{name}_{seed}_{method}_coef']=coef
                if seed==SEEDS[0] and p['d']==2:
                    xx,yy=np.meshgrid(np.linspace(p['lo'][0],p['hi'][0],90),np.linspace(p['lo'][1],p['hi'][1],90))
                    grid=np.column_stack([xx.ravel(),yy.ravel()]); value,_=predict(p,grid,centers,ell,method,coef)
                    arrays[f'{name}_{method}_grid_value']=value.reshape(xx.shape)
                    arrays[f'{name}_x'],arrays[f'{name}_y']=xx,yy
                print(json.dumps(row),flush=True)
            arrays[f'{name}_{seed}_centers']=centers
    # Dense assembly/factorization microbenchmark: fixed dimension versus N.
    costs=[]
    for d in [2,8]:
        for n in [128,256,512,1024]:
            rng=np.random.default_rng(230923+n+d); x=rng.normal(size=(n,d)); rhs=rng.normal(size=n)
            times=[]
            for rep in range(4):
                start=perf_counter()
                dist=np.maximum(np.sum(x*x,axis=1)[:,None]+np.sum(x*x,axis=1)[None,:]-2*x@x.T,0)
                G=np.exp(-dist/(2*d)); G.flat[::n+1]+=1e-6
                assembly=perf_counter()-start
                start=perf_counter(); chol=np.linalg.cholesky(G); factor=perf_counter()-start
                if rep: times.append([assembly,factor])
            assembly,factor=np.median(times,axis=0)
            costs.append(dict(d=d,N=n,assembly_seconds=float(assembly),factor_seconds=float(factor),
                              gram_MiB=8*n*n/2**20))
    meta=dict(date='2026-09-23',python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
              matplotlib=matplotlib.__version__,platform=platform.platform(),machine=platform.machine(),
              blas=threadpool_info(),checks=checks,N=256,M=32,validation=512,test=4096,
              seeds=SEEDS,bandwidth_factors=BANDWIDTHS,eta=ETA,
              thread_environment={key:os.environ[key] for key in ['VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']})
    (ROOT/'results/experiments.json').write_text(json.dumps(dict(metadata=meta,runs=records,dense_costs=costs),indent=2))
    np.savez_compressed(ROOT/'results/fitted_models.npz',**arrays)
    from generate_figures import make_figures
    make_figures(ROOT)
    print(checks,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
