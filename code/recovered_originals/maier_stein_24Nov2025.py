import numpy as np
from numpy.linalg import inv, norm, svd
from scipy.linalg import eig, solve_lyapunov
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 1. Define system and fixed point linearization
gamma = 1.0
def F_ms(x):
    x1, x2 = x[:,0], x[:,1]
    return np.column_stack([x1 - x1**3 - gamma*x1*x2**2,
                            -(1 + x1**2)*x2])

# Fixed point and Jacobian at xbar
eps = 1e-6
xbar = np.array([-1.0, 0.0])
J = np.zeros((2,2))
for i in range(2):
    dx = np.zeros(2); dx[i] = eps
    J[:,i] = (F_ms((xbar+dx)[None,:])[0] - F_ms((xbar-dx)[None,:])[0])/(2*eps)
w, v = eig(J)
evals = np.real(w)
evecR = np.real(v)

# 2. Precompute P_loc from Lyapunov
Z = solve_lyapunov(J, -np.eye(2))  # J Z + Z J^T = -I
P_loc = inv(Z)

# 3. Collocation grid
n = 10
xs = np.linspace(-4.0, 0.0, n)
ys = np.linspace(-3.0, 3.0, n)
dx = xs[1] - xs[0]
dy = ys[1] - ys[0]
Xg, Yg = np.meshgrid(xs, ys)
XY = np.column_stack([Xg.ravel(), Yg.ravel()]) + xbar
N = XY.shape[0]

# 4. Base kernels: RBFs + polynomials
degrees = [2,3,4,5,6]
sigmas = [0.5, 1.5]
base_kernels = []
def rbf(s,t,sigma): return np.exp(-np.dot(s-t,s-t)/(2*sigma**2))
def poly(s,t,d): return (1 + np.dot(s,t))**d
for sigma in sigmas:
    base_kernels.append(lambda s,t, sigma=sigma: rbf(s,t,sigma))
for d in degrees:
    base_kernels.append(lambda s,t, d=d: poly(s,t,d))
L = len(base_kernels)

# 5. Combined kernel function
def kernel_beta(s,t, beta): return sum(b * k(s,t) for b,k in zip(beta, base_kernels))

# 6. Build Koopman eigenfunction phi_i via collocation
def build_phi(beta, i):
    size = N + 3
    K = np.zeros((size, size)); Y = np.zeros(size)
    # anchor φ(x̄)=0
    K[0,0] = 1; Y[0] = 0
    # gradient anchors ∇φ(x̄)=e_i
    for j in range(2):
        for q in range(N):
            eps = 1e-5; d = np.zeros(2); d[j] = eps
            g = (kernel_beta(xbar + d, XY[q], beta) - kernel_beta(xbar - d, XY[q], beta))/(2*eps)
            K[1+j,3+q] = g
        Y[1+j] = 1.0 if j == i else 0.0
    # PDE collocation: ∇φ·F_ms = λ * linear term
    Fvals = F_ms(XY)
    for p in range(N):
        lam = evals[i]
        lin = np.dot(evecR[:,i], (XY[p] - xbar))
        for q in range(N):
            grad = np.zeros(2)
            for a in range(2):
                dd = np.zeros(2); dd[a] = eps
                grad[a] = (kernel_beta(XY[p] + dd, XY[q], beta) - kernel_beta(XY[p] - dd, XY[q], beta))/(2*eps)
            K[3+p,3+q] = np.dot(grad, Fvals[q])
        Y[3+p] = -lam * lin
    K += 1e-6*np.eye(size)
    h = np.linalg.solve(K, Y)
    alpha = h[3:]
    phi = alpha.reshape((n,n)) + ((XY - xbar) @ evecR[:,i]).reshape((n,n))
    return phi

# 7. Loss: HJ residual + curvature penalty + L1 on beta
def loss_with_penalty(beta, gamma=100.0, lam1=0.1):
    phi1 = build_phi(beta,0)
    phi2 = build_phi(beta,1)
    P_mid = evecR @ np.diag(-2*evals) @ inv(evecR)
    Phi = np.vstack([phi1.ravel(), phi2.ravel()]).T
    V_mid = 0.5 * np.einsum('ij,jk,ik->i', Phi, P_mid, Phi).reshape((n,n))
    Vx = np.gradient(V_mid, dx, axis=1)
    Vy = np.gradient(V_mid, dy, axis=0)
    Fgrid = F_ms(XY).reshape((n,n,2))
    HJ = Fgrid[:,:,0]*Vx + Fgrid[:,:,1]*Vy + 0.5*(Vx**2 + Vy**2)
    HJ_loss = norm(HJ)
    # curvature at xbar
    i0 = np.argmin(np.abs(xs + 1)); j0 = np.argmin(np.abs(ys))
    Vxx = (V_mid[j0,i0+1] - 2*V_mid[j0,i0] + V_mid[j0,i0-1])/dx**2
    Vyy = (V_mid[j0+1,i0] - 2*V_mid[j0,i0] + V_mid[j0-1,i0])/dy**2
    Vxy = (V_mid[j0+1,i0+1] - V_mid[j0+1,i0-1] - V_mid[j0-1,i0+1] + V_mid[j0-1,i0-1])/(4*dx*dy)
    H = np.array([[Vxx, Vxy],[Vxy, Vyy]])
    curv_loss = norm(H - P_loc)
    l1 = lam1 * np.sum(np.abs(beta))
    return HJ_loss + gamma*curv_loss + l1

# 8. Test multiple lambda1 values and compare results
lambda_list = [0.0, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0]
results = []
rmse_list = []
V_mid_list = []
beta_kept_list = []      # normalized nonzero coefficients
beta_full_list = []     # raw beta from optimizer
beta_zeroed_list = []   # boolean mask of kept vs pruned
cons = ({'type':'eq','fun':lambda b: np.sum(b)-1},
        {'type':'ineq','fun':lambda b: b})

for lam1 in lambda_list:
    beta0 = np.ones(L)/L
    res = minimize(loss_with_penalty, beta0, args=(10.0, lam1),
                   constraints=cons, method='SLSQP', options={'maxiter':200})
    beta_opt = res.x
    beta_full_list.append(beta_opt.copy())
    mask = beta_opt >= 1.0/L
    beta_zeroed_list.append(mask)
    beta_pruned = beta_opt[mask]
    beta_pruned /= beta_pruned.sum()
    beta_kept_list.append(beta_pruned)

    # compute rl, ru, rmse as before
    phi1_p = build_phi(beta_pruned, 0)
    phi2_p = build_phi(beta_pruned, 1)
    gx1 = np.gradient(phi1_p, dx, axis=1); gy1 = np.gradient(phi1_p, dy, axis=0)
    gx2 = np.gradient(phi2_p, dx, axis=1); gy2 = np.gradient(phi2_p, dy, axis=0)
    rl_vals, ru_vals = [], []
    for ii in range(n):
        for jj in range(n):
            svals = svd([[gx1[ii,jj],gy1[ii,jj]],[gx2[ii,jj],gy2[ii,jj]]], compute_uv=False)
            rl_vals.append(svals[-1]**2); ru_vals.append(svals[0]**2)
    approx_rl = min(rl_vals); approx_ru = max(ru_vals)

    P_mid = evecR @ np.diag(-2*evals) @ inv(evecR)
    Phi_p = np.vstack([phi1_p.ravel(), phi2_p.ravel()]).T
    V_mid_p = 0.5 * np.einsum('ij,jk,ik->i', Phi_p, P_mid, Phi_p).reshape((n,n))
    V_true = 0.5*((Xg+1)**2*(Xg-1)**2 + 2*(Xg**2+1)*Yg**2)
    s = np.dot(V_true.ravel(), V_mid_p.ravel())/np.dot(V_mid_p.ravel(), V_mid_p.ravel())
    V_mid_p *= s
    V_mid_list.append(V_mid_p)
    rmse = np.sqrt(np.mean((V_mid_p - V_true)**2))
    results.append((lam1, np.sum(mask), approx_rl, approx_ru, rmse))
    rmse_list.append(rmse)

# 9. Print comparison table
print("lam1 | #kernels | r_l    | r_u    | RMSE")
for lam1, nk, rl, ru, rmse in results:
    print(f"{lam1:6.2f} | {int(nk):8d} | {rl:.3e} | {ru:.3e} | {rmse:.3e}")

# 10. Plot and save kernel counts vs lam1
lams = [r[0] for r in results]
ks   = [r[1] for r in results]
plt.figure()
plt.plot(lams, ks, '-o')
plt.xlabel('lambda1'); plt.ylabel('Num Kernels'); plt.title('Pruned Kernels vs L1 Coef')
plt.savefig('pruned_kernels_vs_lambda1.png', dpi=300, bbox_inches='tight')
plt.show()

# 11. Plot and save RMSE vs lam1
plt.figure()
plt.plot(lams, rmse_list, '-o')
plt.xlabel('lambda1'); plt.ylabel('RMSE'); plt.title('RMSE vs L1 Coef')
plt.savefig('rmse_vs_lambda1.png', dpi=300, bbox_inches='tight')
plt.show()

# 12. Plot & save all V_mid surfaces plus true potential
num = len(lambda_list)
cols = min(3, num)
rows = int(np.ceil((num+1)/cols))
fig = plt.figure(figsize=(5*cols, 4*rows))
for idx, lam1 in enumerate(lambda_list):
    ax = fig.add_subplot(rows, cols, idx+1, projection='3d')
    ax.plot_surface(Xg, Yg, V_mid_list[idx], rstride=1, cstride=1, cmap='viridis')
    ax.set_title(f'V_mid (λ1={lam1})')
ax_true = fig.add_subplot(rows, cols, num+1, projection='3d')
ax_true.plot_surface(Xg, Yg, V_true, rstride=1, cstride=1, cmap='viridis')
ax_true.set_title('True Quasi-Potential')
plt.tight_layout()
fig.savefig('all_V_mid_surfaces.png', dpi=200, bbox_inches='tight')
plt.show()

# 13. Print full vs pruned β coefficients
def kernel_labels():
    return [f"RBF σ={σ}" for σ in sigmas] + [f"Poly deg={d}" for d in degrees]
labels = kernel_labels()
print("\nλ₁  |  β_full                                      | pruned?                   |  β_pruned")
print("-"*100)
for lam1, β_full, mask, β_pr in zip(lambda_list, beta_full_list, beta_zeroed_list, beta_kept_list):
    full_str = np.array2string(β_full, precision=3, separator=', ')
    mask_str = ','.join('✓' if m else '✗' for m in mask)
    pruned_str = np.array2string(β_pr, precision=3, separator=', ')
    print(f"{lam1:5.2f} | {full_str:<40} | {mask_str:<25} | {pruned_str}")

# 14. Save results for later inspection
to_save = {
    'lambda_list': lambda_list,
    'results': np.array(results, dtype=object),
    'betas_full': np.array(beta_full_list, dtype=object),
    'betas_pruned': np.array(beta_kept_list, dtype=object),
    'V_mid_surfaces': np.array(V_mid_list, dtype=object),
    'rmse_list': np.array(rmse_list, dtype=float)
}
np.savez('quasi_potential_results_with_pruning_details.npz', **to_save)
print("Results saved to 'quasi_potential_results_with_pruning_details.npz'.")

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Extract λ₁ and #kernels from your 'results' list
lams = [r[0] for r in results]
ks   = [r[1] for r in results]

fig, ax = plt.subplots()
ax.plot(lams, ks, '-o')

# Reviewer-compliant labels:
ax.set_xlabel(r'$\lambda_1$')
ax.set_ylabel('Number of active kernels')
ax.set_title(r'Number of active kernels vs sparsity parameter $\lambda_1$')

# Natural-number ticks on y-axis (reviewer request)
ax.yaxis.set_major_locator(mtick.MaxNLocator(integer=True))

fig.tight_layout()
fig.savefig('example1_num_kernels_vs_lambda1.pdf', dpi=300, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots()
ax.plot(lams, rmse_list, '-o')

ax.set_xlabel(r'$\lambda_1$')
ax.set_ylabel('RMSE')
ax.set_title(r'RMSE of reconstructed quasi-potential vs sparsity parameter $\lambda_1$')

fig.tight_layout()
fig.savefig('example1_rmse_vs_lambda1.pdf', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------------------------------------------
# 15. NEW: HJB residuals for local quadratic and kernel approximations
# -------------------------------------------------------------------

print("\n=== HJB residuals on the collocation grid ===")

# Local quadratic V_loc(x) = 0.5 (x-xbar)^T P_loc (x-xbar)
DX = XY - xbar                        # shape (N,2)
grad_loc_flat = (P_loc @ DX.T).T      # gradient at each point
grad_loc = grad_loc_flat.reshape(n, n, 2)

Fgrid = F_ms(XY).reshape(n, n, 2)
HJ_loc = (grad_loc[:, :, 0] * Fgrid[:, :, 0] +
          grad_loc[:, :, 1] * Fgrid[:, :, 1] +
          0.5 * (grad_loc[:, :, 0]**2 + grad_loc[:, :, 1]**2))

HJB_loc_L2 = np.sqrt(np.mean(HJ_loc**2))
HJB_loc_max = np.max(np.abs(HJ_loc))

print("\nLocal quadratic (Riccati) V_loc:")
print(f"  L2 residual  = {HJB_loc_L2:.3e}")
print(f"  max residual = {HJB_loc_max:.3e}")

# Kernel-based V_mid(λ1): use finite differences to approximate ∇V
def hjb_residual_from_grid(V_grid):
    Vx = np.gradient(V_grid, dx, axis=1)
    Vy = np.gradient(V_grid, dy, axis=0)
    HJ = (Fgrid[:, :, 0] * Vx +
          Fgrid[:, :, 1] * Vy +
          0.5 * (Vx**2 + Vy**2))
    L2 = np.sqrt(np.mean(HJ**2))
    mx = np.max(np.abs(HJ))
    return L2, mx

print("\nKernel-based V_mid(λ1):")
for lam1, V_mid in zip(lambda_list, V_mid_list):
    L2k, maxk = hjb_residual_from_grid(V_mid)
    print(f"  λ1 = {lam1:6.2f}:  L2 residual = {L2k:.3e},  max residual = {maxk:.3e}")
