import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig, solve_continuous_lyapunov
from scipy.spatial.distance import directed_hausdorff
from scipy.optimize import minimize
import time

# ==============================================================================
# 1. SYSTEM DYNAMICS (Example 3: 3D multiplicative noise system)
# ==============================================================================

rho = 5.0
epsilon = 1.0
xbar = np.array([0.0, 0.0, 0.0])
dim = 3

def V_true(x):
    """Analytic quasi-potential (used as ground truth)."""
    x1, x2, x3 = x
    return x1**2 + 0.5*x2**2 + 0.5*x3**2

def b(x):
    """Drift."""
    x1, x2, x3 = x
    return np.array([
        -x1 * (1 + x1**2),
        -0.5 * x2 - rho * x3,
        -0.5 * x3 + rho * x2
    ])

def a(x):
    """Diffusion matrix a(x) = sigma(x)sigma(x)^T."""
    x1 = x[0]
    return epsilon * np.diag([1 + x1**2, 1.0, 1.0])

# Linearization Jacobian at xbar
J = np.array([
    [-1.0, 0.0, 0.0],
    [0.0, -0.5, -rho],
    [0.0,  rho, -0.5]
])

# Target Hessian from V_true
H_true = np.diag([2.0, 1.0, 1.0])

# ==============================================================================
# 2. HYPERPARAMETERS, GRIDS, AND HELPER FUNCTIONS
# ==============================================================================

# Collocation / evaluation
N_col      = 200      # Collocation points
N_eval_1D  = 20       # 20^3 = 8000 evaluation points

gamma      = 10.0     # Curvature penalty
lambda1    = 100.0    # L1 sparsity penalty for hybrid kernel
eta        = 1e-4     # L2 regularization

def get_real_eigenbasis(M):
    """
    Compute a real basis for R^n from (possibly complex) eigenvectors of M.
    """
    eigvals, eigvecs = eig(M)
    real_basis = []
    processed = set()

    for i in range(len(eigvals)):
        if i in processed:
            continue
        if np.isreal(eigvals[i]):
            real_basis.append(eigvecs[:, i].real)
            processed.add(i)
        else:
            # Take real/imag pair
            real_basis.append(eigvecs[:, i].real)
            real_basis.append(eigvecs[:, i].imag)
            # mark conjugate
            for j in range(i+1, len(eigvals)):
                if np.isclose(eigvals[j], np.conj(eigvals[i])):
                    processed.add(j)
                    break
            processed.add(i)

    return np.array(real_basis).T  # shape (3,m)

V_real = get_real_eigenbasis(J)
m = V_real.shape[1]

def calculate_metrics(v_approx, v_true):
    """
    RMSE, max abs error, relative L2 error (no Hausdorff here).
    """
    diff = v_approx - v_true
    rmse   = np.sqrt(np.mean(diff**2))
    max_ae = np.max(np.abs(diff))
    rel_l2 = np.linalg.norm(diff) / (np.linalg.norm(v_true) + 1e-12)
    return rmse, max_ae, rel_l2

def levelset_hausdorff(
    V_true_vals,
    V_approx_vals,
    pts_eval,
    levels,
    band=0.05,
    verbose=True,
    label_true="true",
    label_approx="approx",
):
    """
    Hausdorff distance between level sets, as in Example 2.

    Use band { |V - c| <= band } to approximate level set { V = c }.
    """
    def hausdorff(A, B):
        d_ab = directed_hausdorff(A, B)[0]
        d_ba = directed_hausdorff(B, A)[0]
        return max(d_ab, d_ba)

    V_true_vals   = np.asarray(V_true_vals).ravel()
    V_approx_vals = np.asarray(V_approx_vals).ravel()
    pts_eval      = np.asarray(pts_eval)

    dists = []
    levels_used = []

    for c in levels:
        mask_true   = np.abs(V_true_vals   - c) <= band
        mask_approx = np.abs(V_approx_vals - c) <= band

        pts_t = pts_eval[mask_true]
        pts_a = pts_eval[mask_approx]

        if pts_t.shape[0] == 0 or pts_a.shape[0] == 0:
            continue

        dH = hausdorff(pts_t, pts_a)
        dists.append(dH)
        levels_used.append(c)

    dists = np.array(dists, dtype=float)
    finite_mask = np.isfinite(dists)
    avg_dist = np.nan
    if np.any(finite_mask):
        avg_dist = float(np.mean(dists[finite_mask]))

    if verbose:
        print(f"\nLevel-set Hausdorff distances ({label_true} vs {label_approx}):")
        for c, dH in zip(levels_used, dists):
            print(f"  Level {c:5.2f}:  d_H = {dH:.3e}")
        if np.any(finite_mask):
            print(f"\nAveraged over finite levels:  d_H = {avg_dist:.3e}")
        else:
            print("\nNo finite Hausdorff distances to average.")

    return levels_used, dists, avg_dist

# --- Evaluation grid ---
print(f"Setting up grids: N_col={N_col}, N_eval={N_eval_1D**3}")
x_coords = np.linspace(-2, 2, N_eval_1D)
y_coords = np.linspace(-2, 2, N_eval_1D)
z_coords = np.linspace(-2, 2, N_eval_1D)
X_grid, Y_grid, Z_grid = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
pts_eval = np.stack([X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()], axis=-1)

# --- Collocation points ---
xp  = np.random.uniform(-2, 2, (N_col, 3))
bxp = np.array([b(x) for x in xp])
axp = np.array([a(x) for x in xp])
yp  = (xp - xbar) @ V_real   # projected coordinates in eigenbasis

# ==============================================================================
# 3. METHOD 2a: RICCATI QUADRATIC
# ==============================================================================

def solve_method_2a():
    print("\n--- Running Method 2a: Riccati quadratic ---")
    A_eq = a(xbar)
    X = solve_continuous_lyapunov(J, -A_eq)
    Z = np.linalg.inv(X)

    print("Riccati Hessian Z =\n", Z)
    print("Target Hessian H_true =\n", H_true)
    if not np.allclose(Z, H_true):
        print("Warning: Riccati Hessian does not exactly match H_true.")

    def V_quad_riccati(x):
        d = x - xbar
        return 0.5 * d @ Z @ d

    return "V_quad (Riccati)", V_quad_riccati

# ==============================================================================
# 4. METHOD 2b: OPTIMIZED QUADRATIC (HJB + CURVATURE)
# ==============================================================================

def solve_method_2b():
    print("\n--- Running Method 2b: Optimized Quadratic ---")

    def grad_V_quad_opt(y, c):
        # y = V_real^T (x - xbar), grad_y V = c*y, grad_x V = V_real (c*y)
        return V_real @ (c * y)

    def hess_V_quad_opt(c):
        # Hessian = sum_i c_i v_i v_i^T
        return (V_real * c) @ V_real.T

    def objective_quad_opt(c):
        L_HJ = 0.0
        for j in range(N_col):
            grad_V = grad_V_quad_opt(yp[j], c)
            res    = bxp[j] @ grad_V + 0.5 * grad_V @ axp[j] @ grad_V
            L_HJ  += res**2

        H_approx = hess_V_quad_opt(c)
        L_curv   = gamma * np.linalg.norm(H_approx - H_true, 'fro')**2
        L_reg    = eta * np.dot(c, c)
        return (L_HJ / N_col) + L_curv + L_reg

    c0  = np.zeros(m)
    res = minimize(objective_quad_opt, c0, method="L-BFGS-B",
                   options={'disp': False, 'maxiter': 500})
    c_opt = res.x
    print("Optimized quadratic coefficients:", c_opt)

    def V_quad_opt(x):
        y = V_real.T @ (x - xbar)
        return 0.5 * np.sum(c_opt * y**2)

    return "V_quad (Optimized)", V_quad_opt

# ==============================================================================
# 5. METHOD 3: EXTENDED POLYNOMIAL ANSATZ (DEG 2–6)
# ==============================================================================

def solve_method_3():
    print("\n--- Running Method 3: Extended Polynomial (deg 2–6) ---")

    def grad_V_poly(y, c_flat):
        c2, c3, c4, c5, c6 = np.split(c_flat, 5)
        grad_y = (2*c2*y +
                  3*c3*y**2 +
                  4*c4*y**3 +
                  5*c5*y**4 +
                  6*c6*y**5)
        return V_real @ grad_y

    def hess_V_poly(c_flat):
        c2 = c_flat[:m]
        return (V_real * (2*c2)) @ V_real.T

    def objective_poly(c_flat):
        L_HJ = 0.0
        for j in range(N_col):
            grad_V = grad_V_poly(yp[j], c_flat)
            res    = bxp[j] @ grad_V + 0.5 * grad_V @ axp[j] @ grad_V
            L_HJ  += res**2
        H_approx = hess_V_poly(c_flat)
        L_curv   = gamma * np.linalg.norm(H_approx - H_true, 'fro')**2
        L_reg    = eta * np.dot(c_flat, c_flat)
        return (L_HJ / N_col) + L_curv + L_reg

    c0  = np.zeros(5*m)
    res = minimize(objective_poly, c0, method="L-BFGS-B",
                   options={'disp': False, 'maxiter': 1000})
    c_opt_flat = res.x

    def V_poly(x):
        y = V_real.T @ (x - xbar)
        c2, c3, c4, c5, c6 = np.split(c_opt_flat, 5)
        return np.sum(c2*y**2 + c3*y**3 + c4*y**4 + c5*y**5 + c6*y**6)

    return "V_poly (deg 2–6)", V_poly

# ==============================================================================
# 6. METHOD 4: HYBRID KERNEL WITH SPARSITY
# ==============================================================================

# --- Method 4: Hybrid Kernel with Warm Start + Sparsity Refinement ---

# (Re-use these kernel definitions if you don't already have them above)
def k_rbf(x, z, s): 
    return np.exp(-np.sum((x - z)**2) / (2 * s**2))

def grad_k_rbf(x, z, s):
    return k_rbf(x, z, s) * (-(x - z) / s**2)

def hess_k_rbf(x, z, s):
    k = k_rbf(x, z, s)
    diff = x - z
    return k * (np.outer(diff, diff) / (s**4) - np.eye(3) / (s**2))

def k_poly(x, z, r): 
    return (1 + np.dot(x, z))**r

def grad_k_poly(x, z, r):
    return r * ((1 + np.dot(x, z))**(r - 1)) * z

def hess_k_poly(x, z, r):
    if r < 2:
        return np.zeros((3, 3))
    return r * (r - 1) * ((1 + np.dot(x, z))**(r - 2)) * np.outer(z, z)

kernel_defs = [
    {'name': 'RBF(s=0.5)', 'func': lambda x, z: k_rbf(x, z, 0.5),
     'grad': lambda x, z: grad_k_rbf(x, z, 0.5),
     'hess': lambda x, z: hess_k_rbf(x, z, 0.5)},
    {'name': 'RBF(s=1.5)', 'func': lambda x, z: k_rbf(x, z, 1.5),
     'grad': lambda x, z: grad_k_rbf(x, z, 1.5),
     'hess': lambda x, z: hess_k_rbf(x, z, 1.5)},
    {'name': 'Poly(r=2)', 'func': lambda x, z: k_poly(x, z, 2),
     'grad': lambda x, z: grad_k_poly(x, z, 2),
     'hess': lambda x, z: hess_k_poly(x, z, 2)},
    {'name': 'Poly(r=3)', 'func': lambda x, z: k_poly(x, z, 3),
     'grad': lambda x, z: grad_k_poly(x, z, 3),
     'hess': lambda x, z: hess_k_poly(x, z, 3)},
    {'name': 'Poly(r=4)', 'func': lambda x, z: k_poly(x, z, 4),
     'grad': lambda x, z: grad_k_poly(x, z, 4),
     'hess': lambda x, z: hess_k_poly(x, z, 4)},
    {'name': 'Poly(r=5)', 'func': lambda x, z: k_poly(x, z, 5),
     'grad': lambda x, z: grad_k_poly(x, z, 5),
     'hess': lambda x, z: hess_k_poly(x, z, 5)},
    {'name': 'Poly(r=6)', 'func': lambda x, z: k_poly(x, z, 6),
     'grad': lambda x, z: grad_k_poly(x, z, 6),
     'hess': lambda x, z: hess_k_poly(x, z, 6)},
]
L_kernels = len(kernel_defs)


def solve_method_4():
    """
    Hybrid kernel surrogate for Example 3:
      - Warm start from Riccati quadratic
      - L1 sparsity on *unnormalized* beta parameters
      - Pruning + refit of active kernels
    Returns
    -------
    name : str
    V_hybrid : callable R^3 -> R
    """
    print("\n--- Running Method 4: Hybrid Kernel (Warm Start + Sparsity) ---")

    # Local settings (can override global lambda1 here if desired)
    N_col_hyb = 400
    gamma_hyb = 10.0
    lambda1_hyb = 0.05   # small → encourages discovery of Poly(r=2)
    eta_hyb = 1e-4

    # ------------------------------------------------------------
    # 1. Collocation data
    # ------------------------------------------------------------
    xp = np.random.uniform(-2, 2, (N_col_hyb, 3))
    bxp = np.array([b(x) for x in xp])
    axp = np.array([a(x) for x in xp])

    # ------------------------------------------------------------
    # 2. Precompute kernel matrices
    # ------------------------------------------------------------
    print("Pre-computing kernel matrices...")
    K_val = np.zeros((L_kernels, N_col_hyb, N_col_hyb))
    G_val = np.zeros((L_kernels, N_col_hyb, N_col_hyb, 3))
    H_val = np.zeros((L_kernels, N_col_hyb, 3, 3))

    for l in range(L_kernels):
        for j in range(N_col_hyb):
            H_val[l, j] = kernel_defs[l]['hess'](xbar, xp[j])
            for i in range(N_col_hyb):
                K_val[l, i, j] = kernel_defs[l]['func'](xp[i], xp[j])
                G_val[l, i, j] = kernel_defs[l]['grad'](xp[i], xp[j])

    # ------------------------------------------------------------
    # 3. Warm start from Riccati quadratic
    # ------------------------------------------------------------
    # Solve Riccati once locally (you can also call your solve_method_2a here)
    A_eq = a(xbar)
    X = solve_continuous_lyapunov(J, -A_eq)
    Z_ric = np.linalg.inv(X)

    V_target = np.array([0.5 * (x - xbar) @ Z_ric @ (x - xbar) for x in xp])

    # Heavily bias initial betas toward Poly(r=2)
    beta_init = np.ones(L_kernels) * 0.01
    beta_init[2] = 1.0  # Poly(r=2) is index 2
    beta_sq = beta_init**2
    beta_norm = beta_sq / np.sum(beta_sq)

    # Combined kernel for warm start
    K_comb = np.einsum('l,lij->ij', beta_norm, K_val)
    alpha_init = np.linalg.solve(K_comb + 1e-6*np.eye(N_col_hyb), V_target)

    params0 = np.concatenate([alpha_init, beta_init])

    # ------------------------------------------------------------
    # 4. Loss function (with optional active_mask for pruning)
    # ------------------------------------------------------------
    def loss(params, active_mask=None):
        if active_mask is None:
            alpha = params[:N_col_hyb]
            beta = params[N_col_hyb:]
            beta_full = beta
        else:
            alpha = params[:N_col_hyb]
            beta_sub = params[N_col_hyb:]
            beta_full = np.zeros(L_kernels)
            beta_full[active_mask] = beta_sub

        # Normalize β^2 → convex combination of kernels
        beta_sq = beta_full**2
        beta_n = beta_sq / (np.sum(beta_sq) + 1e-10)

        # --- HJB residual term ---
        grad_k = np.einsum('l,lijd->ijd', beta_n, G_val)        # (N,N,3)
        grad_V = np.einsum('j,ijd->id', alpha, grad_k)          # (N,3)

        term1 = np.einsum('ij,ij->i', bxp, grad_V)
        term2 = 0.5 * np.einsum('ij,ijk,ik->i', grad_V, axp, grad_V)
        L_HJ = np.mean((term1 + term2)**2)

        # --- curvature matching at xbar ---
        hess_k = np.einsum('l,ljmn->jmn', beta_n, H_val)        # (N,3,3)
        H_app = np.einsum('j,jmn->mn', alpha, hess_k)           # (3,3)
        L_curv = gamma_hyb * np.sum((H_app - H_true)**2)

        # --- L1 sparsity on *raw* betas + small L2 ---
        L_sparse = lambda1_hyb * np.sum(np.abs(beta_full))
        L_reg = eta_hyb * (np.dot(alpha, alpha) + np.dot(beta_full, beta_full))

        return L_HJ + L_curv + L_sparse + L_reg

    # ------------------------------------------------------------
    # 5. First optimization pass
    # ------------------------------------------------------------
    print("Optimizing Hybrid Kernel (pass 1)...")
    res1 = minimize(loss, params0, method='L-BFGS-B', options={'maxiter': 200})

    alpha_1 = res1.x[:N_col_hyb]
    beta_1 = res1.x[N_col_hyb:]
    beta_sq_1 = beta_1**2
    beta_norm_1 = beta_sq_1 / (np.sum(beta_sq_1) + 1e-10)

    # ------------------------------------------------------------
    # 6. Sparsity-based pruning
    # ------------------------------------------------------------
    threshold = 0.02
    active_mask = beta_norm_1 > threshold

    print(f"\nPruning kernels (threshold = {threshold}):")
    for i, kdef in enumerate(kernel_defs):
        status = "KEEP" if active_mask[i] else "DROP"
        print(f"  {kdef['name']:<12}: {beta_norm_1[i]:.4f} [{status}]")

    # ------------------------------------------------------------
    # 7. Second pass: refit only active kernels
    # ------------------------------------------------------------
    print("\nRefitting active kernels (pass 2)...")
    params_refit = np.concatenate([alpha_1, beta_1[active_mask]])

    def loss_active(p):
        return loss(p, active_mask)

    res2 = minimize(loss_active, params_refit, method='L-BFGS-B',
                    options={'maxiter': 200})

    alpha_final = res2.x[:N_col_hyb]
    beta_sub_final = res2.x[N_col_hyb:]
    beta_full_final = np.zeros(L_kernels)
    beta_full_final[active_mask] = beta_sub_final
    beta_sq_final = beta_full_final**2
    beta_norm_final = beta_sq_final / (np.sum(beta_sq_final) + 1e-10)

    print("\nFinal Hybrid Kernel weights (beta_l^2 normalized):")
    for i, kdef in enumerate(kernel_defs):
        print(f"  {kdef['name']:<12}: {beta_norm_final[i]:.6f}")

    # ------------------------------------------------------------
    # 8. Build the final surrogate V_hybrid(x)
    # ------------------------------------------------------------
    def V_hybrid(x):
        val = 0.0
        for j in range(N_col_hyb):
            k_sum = 0.0
            for l in range(L_kernels):
                if beta_norm_final[l] > 1e-8:
                    k_sum += beta_norm_final[l] * kernel_defs[l]['func'](x, xp[j])
            val += alpha_final[j] * k_sum
        return val

    return "V_hybrid (Kernel)", V_hybrid


# ==============================================================================
# 7. PLOTTING: 2D SLICES X3=0, X2=0, X1=0
# ==============================================================================

def plot_slices(eval_grid_shape, v_all_methods, filename="quasi_potential_slices_sparse.png"):
    print(f"\nGenerating slice plots -> {filename}")
    num_methods = len(v_all_methods)
    num_slices  = 3

    fig, axes = plt.subplots(num_methods, num_slices,
                             figsize=(num_slices*4, num_methods*3.5))
    if num_methods == 1:
        axes = axes.reshape(1, -1)

    N = eval_grid_shape[0]
    mid_idx = N // 2
    if N % 2 == 0:
        mid_idx = int(np.argmin(np.abs(x_coords - 0.0)))

    X_xy, Y_xy = np.meshgrid(x_coords, y_coords)
    X_xz, Z_xz = np.meshgrid(x_coords, z_coords)
    Y_yz, Z_yz = np.meshgrid(y_coords, z_coords)

    vmin = min(v.min() for v in v_all_methods.values())
    vmax = max(v.max() for v in v_all_methods.values())
    vmax = min(vmax, 50)
    levels = np.linspace(vmin, vmax, 10)

    for row, (mname, v_flat) in enumerate(v_all_methods.items()):
        v_grid = v_flat.reshape(eval_grid_shape)

        # Slice x3 = 0 (XY)
        ax = axes[row, 0]
        slice_data = v_grid[:, :, mid_idx].T
        cf = ax.contourf(X_xy, Y_xy, slice_data, levels=levels, vmin=vmin, vmax=vmax)
        ax.set_title(f"{mname}\n$x_3 = 0$")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_aspect("equal")

        # Slice x2 = 0 (XZ)
        ax = axes[row, 1]
        slice_data = v_grid[:, mid_idx, :].T
        cf = ax.contourf(X_xz, Z_xz, slice_data, levels=levels, vmin=vmin, vmax=vmax)
        ax.set_title(f"{mname}\n$x_2 = 0$")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_3$")
        ax.set_aspect("equal")

        # Slice x1 = 0 (YZ)
        ax = axes[row, 2]
        slice_data = v_grid[mid_idx, :, :].T
        cf = ax.contourf(Y_yz, Z_yz, slice_data, levels=levels, vmin=vmin, vmax=vmax)
        ax.set_title(f"{mname}\n$x_1 = 0$")
        ax.set_xlabel("$x_2$")
        ax.set_ylabel("$x_3$")
        ax.set_aspect("equal")

    fig.colorbar(cf, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    print("Saved:", filename)
    plt.close(fig)

# ==============================================================================
# 8. MAIN EXECUTION: SOLVE, EVALUATE, METRICS, HAUSDORFF
# ==============================================================================

if __name__ == "__main__":

    # True values on grid
    v_true_vals = np.array([V_true(x) for x in pts_eval])

    metrics_summary = []
    methods_for_plotting = {"V_true (Baseline)": v_true_vals}

    # Solver list
    solver_functions = [
        solve_method_2a,
        solve_method_2b,
        solve_method_3,
        solve_method_4,
    ]

    # Store approximations for Hausdorff and plotting
    approx_fields = {}

    for solve_func in solver_functions:
        start = time.time()
        name, V_fun = solve_func()
        v_vals = np.array([V_fun(x) for x in pts_eval])
        rmse, max_ae, rel_l2 = calculate_metrics(v_vals, v_true_vals)
        metrics_summary.append((name, rmse, max_ae, rel_l2))
        methods_for_plotting[name] = v_vals
        approx_fields[name] = v_vals
        print(f"Finished {name} in {time.time() - start:.2f}s "
              f"(RMSE={rmse:.3e}, MaxAE={max_ae:.3e}, RelL2={rel_l2:.3e})")

    # Print summary table
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY (Example 3)")
    print("="*80)
    print(f"{'Method':<25} | {'RMSE':<12} | {'MaxAE':<12} | {'Rel L2':<12}")
    print("-"*80)
    print(f"{'V_true (Baseline)':<25} | {0.0:<12.6f} | {0.0:<12.6f} | {0.0:<12.6f}")
    for name, rmse, max_ae, rel_l2 in metrics_summary:
        print(f"{name:<25} | {rmse:<12.6f} | {max_ae:<12.6f} | {rel_l2:<12.6f}")

    # Hausdorff distances (level-set based, same as Example 2)
    v_min = np.percentile(v_true_vals, 5)
    v_max = np.percentile(v_true_vals, 95)
    levels_3d = np.linspace(v_min, v_max, 6)

    print("\n" + "="*80)
    print("LEVEL-SET HAUSDORFF DISTANCES (Example 3)")
    print("="*80)

    for name, v_approx_vals in approx_fields.items():
        _, _, avg_dH = levelset_hausdorff(
            v_true_vals,
            v_approx_vals,
            pts_eval,
            levels_3d,
            band=0.05,
            verbose=True,
            label_true="true",
            label_approx=name
        )
        print(f"\nAverage d_H for {name}: {avg_dH:.3e}")

    # Slice plots
    plot_slices(
        eval_grid_shape=(N_eval_1D, N_eval_1D, N_eval_1D),
        v_all_methods=methods_for_plotting,
        filename="quasi_potential_slices_sparse.png"
    )
