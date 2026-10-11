import numpy as np
from numpy.linalg import inv, norm, svd
from scipy.linalg import eig, solve_lyapunov
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ============================================================
# 1. Define system and fixed point linearization
# ============================================================
gamma = 1.0

def F_ms(x):
    """
    Maier–Stein drift F(x).
    x : array (N,2)
    returns F(x) : array (N,2)
    """
    x1, x2 = x[:, 0], x[:, 1]
    return np.column_stack([
        x1 - x1**3 - gamma * x1 * x2**2,
        -(1 + x1**2) * x2
    ])

# Fixed point and Jacobian at xbar
eps = 1e-6
xbar = np.array([-1.0, 0.0])
J = np.zeros((2, 2))
for i in range(2):
    dx = np.zeros(2); dx[i] = eps
    J[:, i] = (F_ms((xbar + dx)[None, :])[0] - F_ms((xbar - dx)[None, :])[0]) / (2 * eps)

w, v = eig(J)
evals = np.real(w)
evecR = np.real(v)

# ============================================================
# 2. Precompute P_loc from Lyapunov: J Z + Z J^T = -I
# ============================================================
Z = solve_lyapunov(J, -np.eye(2))
P_loc = inv(Z)

# ============================================================
# 3. Collocation / evaluation grid
# ============================================================
n = 10
xs = np.linspace(-4.0, 0.0, n)
ys = np.linspace(-3.0, 3.0, n)
dx = xs[1] - xs[0]
dy = ys[1] - ys[0]
Xg, Yg = np.meshgrid(xs, ys)
XY = np.column_stack([Xg.ravel(), Yg.ravel()]) + xbar   # collocation points shifted by xbar
N = XY.shape[0]

# ============================================================
# 4. Base kernels: RBFs + polynomials
# ============================================================
degrees = [2, 3, 4, 5, 6]
sigmas  = [0.5, 1.5]
base_kernels = []

def rbf(s, t, sigma):
    return np.exp(-np.dot(s - t, s - t) / (2 * sigma**2))

def poly(s, t, d):
    return (1 + np.dot(s, t))**d

for sigma in sigmas:
    base_kernels.append(lambda s, t, sigma=sigma: rbf(s, t, sigma))
for d in degrees:
    base_kernels.append(lambda s, t, d=d: poly(s, t, d))

L = len(base_kernels)

# ------------------------------------------------------------
# Combined kernel function
# ------------------------------------------------------------
def kernel_beta(s, t, beta):
    return sum(b * k(s, t) for b, k in zip(beta, base_kernels))

# ============================================================
# 5. Build Koopman eigenfunction phi_i via collocation
# ============================================================
def build_phi(beta, i):
    """
    Construct Koopman eigenfunction φ_i on the n×n grid
    using kernel ansatz controlled by beta.
    """
    size = N + 3
    K = np.zeros((size, size))
    Y = np.zeros(size)

    # anchor φ(x̄) = 0
    K[0, 0] = 1.0
    Y[0]    = 0.0

    # gradient anchors ∇φ(x̄) = e_i
    for j in range(2):
        for q in range(N):
            eps_fd = 1e-5
            d = np.zeros(2); d[j] = eps_fd
            g = (kernel_beta(xbar + d, XY[q], beta)
                 - kernel_beta(xbar - d, XY[q], beta)) / (2 * eps_fd)
            K[1 + j, 3 + q] = g
        Y[1 + j] = 1.0 if j == i else 0.0

    # PDE collocation: ∇φ · F_ms = λ * linear term
    Fvals = F_ms(XY)
    for p in range(N):
        lam = evals[i]
        lin = np.dot(evecR[:, i], (XY[p] - xbar))
        for q in range(N):
            grad = np.zeros(2)
            for a in range(2):
                dd = np.zeros(2); dd[a] = eps
                grad[a] = (kernel_beta(XY[p] + dd, XY[q], beta)
                           - kernel_beta(XY[p] - dd, XY[q], beta)) / (2 * eps)
            K[3 + p, 3 + q] = np.dot(grad, Fvals[q])
        Y[3 + p] = -lam * lin

    K += 1e-6 * np.eye(size)
    h = np.linalg.solve(K, Y)
    alpha = h[3:]

    phi = alpha.reshape((n, n)) + ((XY - xbar) @ evecR[:, i]).reshape((n, n))
    return phi

# ============================================================
# 6. Loss: HJ residual + curvature penalty + L1 on beta
#      (implements Eq. (19) structure)
# ============================================================
def loss_with_penalty(beta, gamma_hjb=100.0, lam1=0.1):
    """
    L(β) = L_HJ(β) + γ ||∇²V_mid(x̄) - P_loc||_F² + λ1 ∑ |β_j|
    using V_mid built from Koopman eigenfunctions φ_1, φ_2.
    """
    phi1 = build_phi(beta, 0)
    phi2 = build_phi(beta, 1)

    # middle quadratic form P_mid from Koopman eigenvalues
    P_mid = evecR @ np.diag(-2 * evals) @ inv(evecR)

    Phi = np.vstack([phi1.ravel(), phi2.ravel()]).T   # (n², 2)
    V_mid = 0.5 * np.einsum('ij,jk,ik->i', Phi, P_mid, Phi).reshape((n, n))

    # HJB residual on the n×n grid
    Vx = np.gradient(V_mid, dx, axis=1)
    Vy = np.gradient(V_mid, dy, axis=0)
    Fgrid = F_ms(XY).reshape((n, n, 2))
    HJ = Fgrid[:, :, 0] * Vx + Fgrid[:, :, 1] * Vy + 0.5 * (Vx**2 + Vy**2)
    L_HJ = norm(HJ)

    # curvature at xbar (finite differences)
    i0 = np.argmin(np.abs(xs + 1.0))   # x1 = -1 relative to xbar=-1 → x1=0
    j0 = np.argmin(np.abs(ys - 0.0))   # x2 = 0
    Vxx = (V_mid[j0, i0 + 1] - 2 * V_mid[j0, i0] + V_mid[j0, i0 - 1]) / dx**2
    Vyy = (V_mid[j0 + 1, i0] - 2 * V_mid[j0, i0] + V_mid[j0 - 1, i0]) / dy**2
    Vxy = (V_mid[j0 + 1, i0 + 1] - V_mid[j0 + 1, i0 - 1]
           - V_mid[j0 - 1, i0 + 1] + V_mid[j0 - 1, i0 - 1]) / (4 * dx * dy)
    H = np.array([[Vxx, Vxy], [Vxy, Vyy]])
    L_curv = norm(H - P_loc)

    # L1 sparsity on raw beta (no sum=1 constraint)
    L_l1 = lam1 * np.sum(np.abs(beta))

    return L_HJ + gamma_hjb * L_curv + L_l1

# ============================================================
# 7. Sweep λ1, compute V_mid and RMSE against true V
# ============================================================
lambda_list = [0.0, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0]

results = []
rmse_list = []
V_mid_list = []
beta_kept_list = []   # normalized nonzero coefficients
beta_full_list = []   # raw beta from optimizer
beta_zeroed_list = [] # boolean mask of kept vs pruned

# NOTE: we now only enforce β_j ≥ 0 by bounds; no sum β = 1 constraint
bounds = [(0.0, None)] * L

for lam1 in lambda_list:
    beta0 = np.ones(L) / L
    res = minimize(
        loss_with_penalty, beta0,
        args=(100.0, lam1),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 200}
    )
    beta_opt = res.x
    beta_full_list.append(beta_opt.copy())

    # simple pruning rule from original code
    threshold = 1.0 / L
    mask = beta_opt >= threshold
    beta_zeroed_list.append(mask)

    beta_pruned = beta_opt[mask]
    beta_pruned /= beta_pruned.sum() if beta_pruned.sum() > 0 else 1.0
    beta_kept_list.append(beta_pruned)

    # recompute φ_i with pruned / renormalised β
    beta_for_V = np.zeros_like(beta_opt)
    beta_for_V[mask] = beta_pruned

    phi1_p = build_phi(beta_for_V, 0)
    phi2_p = build_phi(beta_for_V, 1)

    # compute rl, ru as before
    gx1 = np.gradient(phi1_p, dx, axis=1); gy1 = np.gradient(phi1_p, dy, axis=0)
    gx2 = np.gradient(phi2_p, dx, axis=1); gy2 = np.gradient(phi2_p, dy, axis=0)
    rl_vals, ru_vals = [], []
    for ii in range(n):
        for jj in range(n):
            svals = svd([[gx1[ii, jj], gy1[ii, jj]],
                         [gx2[ii, jj], gy2[ii, jj]]],
                        compute_uv=False)
            rl_vals.append(svals[-1]**2)
            ru_vals.append(svals[0]**2)
    approx_rl = min(rl_vals)
    approx_ru = max(ru_vals)

    P_mid = evecR @ np.diag(-2 * evals) @ inv(evecR)
    Phi_p = np.vstack([phi1_p.ravel(), phi2_p.ravel()]).T
    V_mid_p = 0.5 * np.einsum('ij,jk,ik->i', Phi_p, P_mid, Phi_p).reshape((n, n))

    # analytic Maier–Stein quasi-potential (up to constant factor)
    V_true = 0.5 * ((Xg + 1)**2 * (Xg - 1)**2 + 2 * (Xg**2 + 1) * Yg**2)

    # scale V_mid_p by least-squares factor s
    s = np.dot(V_true.ravel(), V_mid_p.ravel()) / np.dot(V_mid_p.ravel(), V_mid_p.ravel())
    V_mid_p *= s
    V_mid_list.append(V_mid_p)

    rmse = np.sqrt(np.mean((V_mid_p - V_true)**2))
    results.append((lam1, np.sum(mask), approx_rl, approx_ru, rmse))
    rmse_list.append(rmse)

# ============================================================
# 8. Print comparison table
# ============================================================
print("lam1 | #kernels | r_l    | r_u    | RMSE")
for lam1, nk, rl, ru, rmse in results:
    print(f"{lam1:6.2f} | {int(nk):8d} | {rl:.3e} | {ru:.3e} | {rmse:.3e}")

# ============================================================
# 9. Save plots: #kernels vs λ1 and RMSE vs λ1 (reviewer-style)
# ============================================================
lams = [r[0] for r in results]
ks   = [r[1] for r in results]

fig, ax = plt.subplots()
ax.plot(lams, ks, '-o')
ax.set_xlabel(r'$\lambda_1$')
ax.set_ylabel('Number of active kernels')
ax.set_title(r'Number of active kernels vs sparsity parameter $\lambda_1$')
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

# ============================================================
# 10. 3D surfaces for all V_mid and true potential
# ============================================================
num = len(lambda_list)
cols = min(3, num)
rows = int(np.ceil((num + 1) / cols))

fig = plt.figure(figsize=(5 * cols, 4 * rows))
for idx, lam1 in enumerate(lambda_list):
    ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
    ax.plot_surface(Xg, Yg, V_mid_list[idx], rstride=1, cstride=1, cmap='viridis')
    ax.set_title(f'V_mid (λ1={lam1})')
ax_true = fig.add_subplot(rows, cols, num + 1, projection='3d')
ax_true.plot_surface(Xg, Yg, V_true, rstride=1, cstride=1, cmap='viridis')
ax_true.set_title('True Quasi-Potential')
plt.tight_layout()
fig.savefig('all_V_mid_surfaces.png', dpi=200, bbox_inches='tight')
plt.show()

# ============================================================
# 11. Print full vs pruned β coefficients
# ============================================================
def kernel_labels():
    return [f"RBF σ={σ}" for σ in sigmas] + [f"Poly deg={d}" for d in degrees]

labels = kernel_labels()
print("\nλ₁  |  β_full                                      | pruned?                   |  β_pruned")
print("-" * 100)
for lam1, β_full, mask, β_pr in zip(lambda_list, beta_full_list, beta_zeroed_list, beta_kept_list):
    full_str = np.array2string(β_full, precision=3, separator=', ')
    mask_str = ','.join('✓' if m else '✗' for m in mask)
    pruned_str = np.array2string(β_pr, precision=3, separator=', ')
    print(f"{lam1:5.2f} | {full_str:<40} | {mask_str:<25} | {pruned_str}")

# ============================================================
# 12. Save raw data
# ============================================================
to_save = {
    'lambda_list': np.array(lambda_list, dtype=float),
    'results':     np.array(results, dtype=object),
    'betas_full':  np.array(beta_full_list, dtype=object),
    'betas_pruned':np.array(beta_kept_list, dtype=object),
    'V_mid_surfaces': np.array(V_mid_list, dtype=object),
    'rmse_list':   np.array(rmse_list, dtype=float)
}
np.savez('quasi_potential_results_with_pruning_details.npz', **to_save)
print("Results saved to 'quasi_potential_results_with_pruning_details.npz'.")

# ============================================================
# 13. Global HJB residuals: local quadratic vs kernel V_mid(λ1)
# ============================================================
def compute_HJB_residual_on_grid(V_grid):
    Vx = np.gradient(V_grid, dx, axis=1)
    Vy = np.gradient(V_grid, dy, axis=0)
    Fgrid = F_ms(np.column_stack([Xg.ravel(), Yg.ravel()])).reshape(n, n, 2)
    HJ = Fgrid[:, :, 0] * Vx + Fgrid[:, :, 1] * Vy + 0.5 * (Vx**2 + Vy**2)
    L2 = norm(HJ.ravel())
    maxres = np.max(np.abs(HJ))
    return L2, maxres

# local quadratic from P_loc
def V_local_grid():
    d1 = Xg + 1.0   # x1 - xbar_1 = x1 - (-1) = x1 + 1
    d2 = Yg - 0.0   # x2 - xbar_2
    D = np.stack([d1, d2], axis=-1)      # (n,n,2)
    # 0.5 * (x - xbar)^T P_loc (x - xbar)
    V_loc = 0.5 * (
        D[..., 0] * (P_loc[0, 0] * D[..., 0] + P_loc[0, 1] * D[..., 1]) +
        D[..., 1] * (P_loc[1, 0] * D[..., 0] + P_loc[1, 1] * D[..., 1])
    )
    return V_loc

V_loc_grid = V_local_grid()
L2_loc, max_loc = compute_HJB_residual_on_grid(V_loc_grid)

print("\n=== HJB residuals on the global grid ===\n")
print("Local quadratic (Riccati) V_loc:")
print(f"  L2 residual  = {L2_loc:.3e}")
print(f"  max residual = {max_loc:.3e}\n")

print("Kernel-based V_mid(λ1):")
for lam1, V_mid_p in zip(lambda_list, V_mid_list):
    L2_k, max_k = compute_HJB_residual_on_grid(V_mid_p)
    print(f"  λ1 = {lam1:6.2f}:  L2 residual = {L2_k:.3e},  max residual = {max_k:.3e}")
