import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig, solve_continuous_lyapunov
from scipy.spatial.distance import directed_hausdorff
from scipy.optimize import minimize
import time

# ==============================================================================
# 1. SYSTEM DYNAMICS (Example 3: 3D multiplicative noise)
# ==============================================================================

rho = 5.0
epsilon = 1.0
xbar = np.array([0.0, 0.0, 0.0])   # equilibrium
dim = 3

# True analytical quasi-potential (for validation)
def V_true(x):
    x1, x2, x3 = x
    return x1**2 + 0.5 * x2**2 + 0.5 * x3**2

# Drift
def b(x):
    x1, x2, x3 = x
    return np.array([
        -x1 * (1 + x1**2),
        -0.5 * x2 - rho * x3,
        -0.5 * x3 + rho * x2
    ])

# Diffusion matrix a(x) = sigma(x)sigma(x)^T
def a(x):
    x1, _, _ = x
    return epsilon * np.diag([1 + x1**2, 1.0, 1.0])

# Linearization at equilibrium
J = np.array([
    [-1.0,  0.0,  0.0],
    [ 0.0, -0.5, -rho],
    [ 0.0,  rho, -0.5]
])
A_eq = a(xbar)

# Target Hessian from V_true
H_true = np.diag([2.0, 1.0, 1.0])

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================

def get_real_eigenbasis(M):
    """Real basis from eigenvectors of a real matrix M (handles complex pairs)."""
    eigvals, eigvecs = eig(M)
    real_basis = []
    used = set()

    for i in range(len(eigvals)):
        if i in used:
            continue
        if np.isreal(eigvals[i]):
            real_basis.append(eigvecs[:, i].real)
            used.add(i)
        else:
            # complex pair: use Re and Im as basis
            real_basis.append(eigvecs[:, i].real)
            real_basis.append(eigvecs[:, i].imag)
            # mark conjugate as used
            for j in range(i + 1, len(eigvals)):
                if np.isclose(eigvals[j], np.conj(eigvals[i])):
                    used.add(j)
                    break
            used.add(i)

    return np.array(real_basis).T

V_real = get_real_eigenbasis(J)
m = V_real.shape[1]    # should be 3

def calculate_metrics(v_approx, v_true, grid_points, cloud_true):
    """RMSE, MaxAE, relative L2, Hausdorff distance (4D surfaces)."""
    rmse = np.sqrt(np.mean((v_approx - v_true) ** 2))
    max_ae = np.max(np.abs(v_approx - v_true))
    rel_l2 = np.linalg.norm(v_approx - v_true) / (np.linalg.norm(v_true) + 1e-10)

    cloud_approx = np.hstack([grid_points, v_approx.reshape(-1, 1)])
    hd = max(
        directed_hausdorff(cloud_approx, cloud_true)[0],
        directed_hausdorff(cloud_true, cloud_approx)[0]
    )
    return rmse, max_ae, rel_l2, hd

# ==============================================================================
# 3. GRIDS (collocation + evaluation)
# ==============================================================================

N_col = 200        # collocation points
N_eval_1D = 20     # 20^3 = 8000 evaluation points
gamma = 10.0       # curvature penalty
lambda1 = 0.1      # L1 sparsity (for Method 4)
eta = 1e-4         # L2 regularization (small)

print(f"Setting up grids: N_col={N_col}, N_eval={N_eval_1D**3}")

# Evaluation grid (cube [-2,2]^3)
x_vals = np.linspace(-2, 2, N_eval_1D)
Xg, Yg, Zg = np.meshgrid(x_vals, x_vals, x_vals, indexing='ij')
pts_eval = np.column_stack([Xg.ravel(), Yg.ravel(), Zg.ravel()])

# Collocation points for HJB
xp = np.random.uniform(-2, 2, (N_col, dim))
bxp = np.array([b(x) for x in xp])
axp = np.array([a(x) for x in xp])
yp = (xp - xbar) @ V_real   # eigenbasis projection

# ==============================================================================
# 4. METHODS
# ==============================================================================

# ---------------- Method 2a: Riccati quadratic ----------------

def solve_method_2a():
    print("\n--- Method 2a: Riccati quadratic ---")
    X = solve_continuous_lyapunov(J, -A_eq)
    Z = np.linalg.inv(X)
    print("Riccati Hessian Z:\n", Z)
    print("Target H_true:\n", H_true)
    if not np.allclose(Z, H_true, atol=1e-10):
        print("WARNING: Z != H_true within tolerance.")

    def V_quad_riccati(x):
        d = x - xbar
        return 0.5 * d @ Z @ d

    return "V_quad (Riccati)", V_quad_riccati

# ---------------- Method 2b: Optimized quadratic ----------------

def solve_method_2b():
    print("\n--- Method 2b: Optimized quadratic ---")

    def grad_V_quad(y, c):
        # grad_y V = c * y, grad_x V = V_real @ (c * y)
        return V_real @ (c * y)

    def hess_V_quad(c):
        return (V_real * c) @ V_real.T

    def objective(c):
        # HJB residual
        L_HJ = 0.0
        for j in range(N_col):
            gV = grad_V_quad(yp[j], c)
            res = bxp[j] @ gV + 0.5 * gV @ axp[j] @ gV
            L_HJ += res ** 2

        H_approx = hess_V_quad(c)
        L_curv = gamma * np.linalg.norm(H_approx - H_true, 'fro') ** 2
        L_reg = eta * np.dot(c, c)

        return (L_HJ / N_col) + L_curv + L_reg

    c0 = np.zeros(m)
    res = minimize(objective, c0, method='L-BFGS-B',
                   options={'disp': False, 'maxiter': 500})
    c_opt = res.x
    print("Optimized c:", c_opt)

    def V_quad_opt(x):
        y = V_real.T @ (x - xbar)
        return 0.5 * np.sum(c_opt * y ** 2)

    return "V_quad (Optimized)", V_quad_opt

# ---------------- Method 3: Extended polynomial (deg 2–6) ----------------

def solve_method_3():
    print("\n--- Method 3: Extended polynomial (deg 2–6) ---")

    def grad_V_poly(y, c_flat):
        c2, c3, c4, c5, c6 = np.split(c_flat, 5)
        grad_y = (
            2 * c2 * y +
            3 * c3 * y ** 2 +
            4 * c4 * y ** 3 +
            5 * c5 * y ** 4 +
            6 * c6 * y ** 5
        )
        return V_real @ grad_y

    def hess_V_poly(c_flat):
        c2 = c_flat[:m]
        return (V_real * (2 * c2)) @ V_real.T

    def objective(c_flat):
        L_HJ = 0.0
        for j in range(N_col):
            gV = grad_V_poly(yp[j], c_flat)
            res = bxp[j] @ gV + 0.5 * gV @ axp[j] @ gV
            L_HJ += res ** 2

        H_approx = hess_V_poly(c_flat)
        L_curv = gamma * np.linalg.norm(H_approx - H_true, 'fro') ** 2
        L_reg = eta * np.dot(c_flat, c_flat)

        return (L_HJ / N_col) + L_curv + L_reg

    c0 = np.zeros(5 * m)
    res = minimize(objective, c0, method='L-BFGS-B',
                   options={'disp': False, 'maxiter': 1000})
    c_opt = res.x

    def V_poly(x):
        y = V_real.T @ (x - xbar)
        c2, c3, c4, c5, c6 = np.split(c_opt, 5)
        return np.sum(
            c2 * y ** 2 +
            c3 * y ** 3 +
            c4 * y ** 4 +
            c5 * y ** 5 +
            c6 * y ** 6
        )

    return "V_poly (deg 2–6)", V_poly

# ---------------- Method 4: Hybrid kernel with sparsity ----------------

# base kernels
def k_rbf(x, z, s): return np.exp(-np.sum((x - z) ** 2) / (2 * s ** 2))
def grad_k_rbf(x, z, s): return k_rbf(x, z, s) * (-(x - z) / s ** 2)
def hess_k_rbf(x, z, s):
    k = k_rbf(x, z, s)
    diff = x - z
    return k * (np.outer(diff, diff) / s ** 4 - np.eye(dim) / s ** 2)

def k_poly(x, z, r): return (1 + np.dot(x, z)) ** r
def grad_k_poly(x, z, r): return r * (1 + np.dot(x, z)) ** (r - 1) * z
def hess_k_poly(x, z, r):
    if r < 2:
        return np.zeros((dim, dim))
    return r * (r - 1) * (1 + np.dot(x, z)) ** (r - 2) * np.outer(z, z)

kernel_defs = [
    {'name': 'RBF(s=0.5)', 'func': lambda x,z: k_rbf(x,z,0.5),
     'grad': lambda x,z: grad_k_rbf(x,z,0.5),
     'hess': lambda x,z: hess_k_rbf(x,z,0.5)},
    {'name': 'RBF(s=1.5)', 'func': lambda x,z: k_rbf(x,z,1.5),
     'grad': lambda x,z: grad_k_rbf(x,z,1.5),
     'hess': lambda x,z: hess_k_rbf(x,z,1.5)},
    {'name': 'Poly(r=2)',  'func': lambda x,z: k_poly(x,z,2),
     'grad': lambda x,z: grad_k_poly(x,z,2),
     'hess': lambda x,z: hess_k_poly(x,z,2)},
    {'name': 'Poly(r=3)',  'func': lambda x,z: k_poly(x,z,3),
     'grad': lambda x,z: grad_k_poly(x,z,3),
     'hess': lambda x,z: hess_k_poly(x,z,3)},
    {'name': 'Poly(r=4)',  'func': lambda x,z: k_poly(x,z,4),
     'grad': lambda x,z: grad_k_poly(x,z,4),
     'hess': lambda x,z: hess_k_poly(x,z,4)},
    {'name': 'Poly(r=5)',  'func': lambda x,z: k_poly(x,z,5),
     'grad': lambda x,z: grad_k_poly(x,z,5),
     'hess': lambda x,z: hess_k_poly(x,z,5)},
    {'name': 'Poly(r=6)',  'func': lambda x,z: k_poly(x,z,6),
     'grad': lambda x,z: grad_k_poly(x,z,6),
     'hess': lambda x,z: hess_k_poly(x,z,6)},
]
L_kernels = len(kernel_defs)

def solve_method_4():
    print("\n--- Method 4: Hybrid kernel (L1 sparsity) ---")
    start_pre = time.time()

    # Precompute G_all and H_all
    G_all = np.zeros((L_kernels, N_col, N_col, dim))
    H_all = np.zeros((L_kernels, N_col, dim, dim))

    for l in range(L_kernels):
        grad_f = kernel_defs[l]['grad']
        hess_f = kernel_defs[l]['hess']
        for j in range(N_col):
            H_all[l, j] = hess_f(xbar, xp[j])   # curvature at equilibrium
            for p in range(N_col):
                G_all[l, j, p] = grad_f(xp[j], xp[p])

    end_pre = time.time()
    print(f"Pre-computation finished in {end_pre - start_pre:.2f}s")

    def objective(params):
        alpha = params[:N_col]
        beta_params = params[N_col:]

        beta_sq = beta_params ** 2
        beta_norm = beta_sq / (np.sum(beta_sq) + 1e-10)

        # gradients
        grad_k_beta = np.einsum('l, ljpk -> jpk', beta_norm, G_all)
        grad_V = np.einsum('p, jpk -> jk', alpha, grad_k_beta)

        # HJB residual
        r_HJ = (
            np.einsum('jk, jk -> j', bxp, grad_V) +
            0.5 * np.einsum('jk, jkl, jl -> j', grad_V, axp, grad_V)
        )
        L_HJ = np.mean(r_HJ ** 2)

        # curvature at xbar
        hess_k_beta = np.einsum('l, lpik -> pik', beta_norm, H_all)
        H_approx = np.einsum('p, pik -> ik', alpha, hess_k_beta)
        L_curv = gamma * np.linalg.norm(H_approx - H_true, 'fro') ** 2

        # L1 on beta_params
        L1 = lambda1 * np.sum(np.abs(beta_params))
        L2 = eta * (np.dot(alpha, alpha) + np.dot(beta_params, beta_params))

        return L_HJ + L_curv + L1 + L2

    # initial params: alpha=0, beta=1
    params0 = np.concatenate([np.zeros(N_col), np.ones(L_kernels)])
    res = minimize(objective, params0, method='L-BFGS-B',
                   options={'disp': True, 'maxiter': 500})

    params_opt = res.x
    alpha_opt = params_opt[:N_col]
    beta_opt = params_opt[N_col:]
    beta_sq_opt = beta_opt ** 2
    beta_norm_opt = beta_sq_opt / (np.sum(beta_sq_opt) + 1e-10)

    print("\nOptimized kernel weights (beta^2 normalized):")
    for kdef, w in zip(kernel_defs, beta_norm_opt):
        print(f"  {kdef['name']:<12}: {w:.6f}")

    def V_hybrid(x):
        val = 0.0
        for p in range(N_col):
            k_mix = 0.0
            for l in range(L_kernels):
                if beta_norm_opt[l] > 1e-8:
                    k_mix += beta_norm_opt[l] * kernel_defs[l]['func'](x, xp[p])
            val += alpha_opt[p] * k_mix
        return val

    return "V_hybrid (Kernel)", V_hybrid

# ==============================================================================
# 5. PLOTTING: slices at x1=0, x2=0, x3=0
# ==============================================================================

def plot_slices(eval_grid_shape, v_all_methods, filename="quasi_potential_slices.png"):
    print(f"\nGenerating slice plots -> {filename}")
    num_methods = len(v_all_methods)
    num_slices = 3

    fig, axes = plt.subplots(num_methods, num_slices,
                             figsize=(num_slices * 4, num_methods * 3.5))

    if num_methods == 1:
        axes = axes.reshape(1, -1)

    N = eval_grid_shape[0]
    mid_idx = N // 2
    # safer index for x=0
    x_coords = np.linspace(-2, 2, N)
    mid_idx = int(np.argmin(np.abs(x_coords - 0.0)))

    X_xy, Y_xy = np.meshgrid(x_coords, x_coords)
    X_xz, Z_xz = np.meshgrid(x_coords, x_coords)
    Y_yz, Z_yz = np.meshgrid(x_coords, x_coords)

    vmin = min(v.min() for v in v_all_methods.values())
    vmax = min(max(v.max() for v in v_all_methods.values()), 50)
    levels = np.linspace(vmin, vmax, 10)

    for row, (name, v_flat) in enumerate(v_all_methods.items()):
        v_grid = v_flat.reshape(eval_grid_shape)

        # x3 = 0 (XY plane)
        ax = axes[row, 0]
        slice_xy = v_grid[:, :, mid_idx].T
        cf = ax.contourf(X_xy, Y_xy, slice_xy, levels=levels, vmin=vmin, vmax=vmax)
        ax.set_title(f"{name}\n$x_3=0$")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_aspect("equal")

        # x2 = 0 (XZ plane)
        ax = axes[row, 1]
        slice_xz = v_grid[:, mid_idx, :].T
        ax.contourf(X_xz, Z_xz, slice_xz, levels=levels, vmin=vmin, vmax=vmax)
        ax.set_title(f"{name}\n$x_2=0$")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_3$")
        ax.set_aspect("equal")

        # x1 = 0 (YZ plane)
        ax = axes[row, 2]
        slice_yz = v_grid[mid_idx, :, :].T
        ax.contourf(Y_yz, Z_yz, slice_yz, levels=levels, vmin=vmin, vmax=vmax)
        ax.set_title(f"{name}\n$x_1=0$")
        ax.set_xlabel("$x_2$")
        ax.set_ylabel("$x_3$")
        ax.set_aspect("equal")

    fig.colorbar(cf, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()
    print("Saved:", filename)

# ==============================================================================
# 6. MAIN
# ==============================================================================

if __name__ == "__main__":
    metrics_summary = []
    methods_for_plotting = {}

    # baseline
    v_true_vals = np.array([V_true(x) for x in pts_eval])
    cloud_true = np.hstack([pts_eval, v_true_vals.reshape(-1, 1)])
    metrics_summary.append(("V_true (baseline)", 0, 0, 0, 0))
    methods_for_plotting["V_true"] = v_true_vals

    solvers = [
        solve_method_2a,
        solve_method_2b,
        solve_method_3,
        solve_method_4,
    ]

    for solve_func in solvers:
        start = time.time()
        name, V_func = solve_func()
        v_vals = np.array([V_func(x) for x in pts_eval])
        rmse, max_ae, rel_l2, hd = calculate_metrics(v_vals, v_true_vals, pts_eval, cloud_true)
        metrics_summary.append((name, rmse, max_ae, rel_l2, hd))
        methods_for_plotting[name] = v_vals
        end = time.time()
        print(f"Finished {name} in {end - start:.2f}s")

    # Summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"{'Method':<25} | {'RMSE':<12} | {'MaxAE':<12} | {'Rel L2':<12} | {'Hausdorff':<12}")
    print("-"*80)
    for row in metrics_summary:
        name, rmse, max_ae, rel_l2, hd = row
        print(f"{name:<25} | {rmse:<12.6e} | {max_ae:<12.6e} | {rel_l2:<12.6e} | {hd:<12.6e}")

    # Plots
    plot_slices(
        eval_grid_shape=(N_eval_1D, N_eval_1D, N_eval_1D),
        v_all_methods=methods_for_plotting,
        filename="example3_quasi_potential_slices.png"
    )
