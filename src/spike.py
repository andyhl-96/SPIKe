import jax
import jax.numpy as jnp
from functools import partial
import yourdfpy
import pyroki as pk
import numpy as np

primes = jnp.array([
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
        53,
        59
        ]
    )

@jax.jit
def sample(primes, i):
    return jnp.mod(i[:, None] * jnp.sin(2 * jnp.pi * (1 / jnp.sqrt(primes)) + 1 / jnp.cbrt(primes)), 1)

@jax.jit
def scale_points(X, low, high):
    return low + (high - low) * X

def load_robot(urdf_path):
    urdf = yourdfpy.URDF.load(urdf_path)
    robot = pk.Robot.from_urdf(urdf)  
    robot_coll = pk.collision.RobotCollision.from_urdf(urdf)
    return urdf, robot, robot_coll

def compute_points(func, num):
    ts = jnp.linspace(0, 1, num)
    bat_func = jax.vmap(func, in_axes=(0))
    return bat_func(ts)

@partial(jax.jit, static_argnames=['f'])
def jacobi_proj(f, steps, q, robot):
    J = jax.jacobian(f, 0)
    def body(i, val):
        qk = val
        J_dag = jnp.linalg.pinv(jnp.array([J(qk, robot)]))
        dx = jnp.array([-f(qk, robot)])
        dq = J_dag @ dx
        return qk + dq
    initial = q
    q = jax.lax.fori_loop(0, steps, body, initial)
    return q

@jax.jit
def rbf(x0, x1, n):
    return jnp.exp(-jnp.dot(x0 - x1, x0 - x1) / n)

@partial(jax.jit, static_argnames=['f'])
def stein_proj(f, k, X, robot):
    n = len(X)
    # function to compute single xj in the sum
    def update(X_j, x):
        def logp(x):
            return jnp.log(jnp.exp(-jnp.dot(f(x, robot), f(x, robot))))
        rbf_batch = jax.vmap(rbf, in_axes=(0, None, None))
        logp_grad = jax.grad(logp)
        rbf_grad = jax.grad(rbf, argnums=0)
        rbf_grad_batch = jax.vmap(rbf_grad, in_axes=(0, None, None))
        logp_grad_batch = jax.vmap(logp_grad, in_axes=(0))
        r = jnp.linalg.matmul(rbf_batch(X_j, x, 1 / n).T, logp_grad_batch(X_j)) + jnp.linalg.matmul(jnp.ones(n), rbf_grad_batch(X_j, x, 1 / n))
        return r
    # X_k = X
    def loop1(i, X_k):
        update_batch = jax.vmap(update, in_axes=(None, 0))
        X_del = update_batch(X_k, X_k)
        return X_k + 1 / n * X_del
    initial = X
    X_k = jax.lax.fori_loop(0, k, loop1, initial)
    return X_k

@partial(jax.jit, static_argnames=['f'])
def jacobi_stein_proj(f, outer, inner, X, robot):
    jacobi_batch = jax.vmap(jacobi_proj, in_axes=(None, None, 0, None))
    def loop(i, proj):
        proj = stein_proj(f, inner, proj, robot)
        proj = jacobi_batch(f, inner, proj, robot)
        return proj
    X_k = jax.lax.fori_loop(0, outer, loop, X)
    return X_k

def find_best_sequence(layers):
    # T[i, j] minimum cost to get to layer i node j
    # T[i, j] = min over k {T[i - 1, k] + c(k, j)}
    def compute_candidates(i, j, T):
        cands = jnp.zeros(len(layers[0]))
        cands += T[i - 1, :] # + cost  TODO
        return cands
    T = jnp.ones((len(layers)), len(layers[0])) * jnp.inf
    T[0, :] = 0
    for i in range(1, len(layers)):
        for j in range(len(layers[0])):
            cands = compute_candidates(i, j, T)
            min = jnp.min(cands)
            T[i, j] = min

# number of boundary_conds should completely determine hermite poly of degree deg
def _monomial_deriv_coeff(k, r):
    """Coefficient for the r-th derivative of t^k: k*(k-1)*...*(k-r+1) (0 if k<r)."""
    if k < r:
        return 0.0
    c = 1.0
    for i in range(r):
        c *= (k - i)
    return c


def _build_constraint_rows(deg, specs, t0, t1):
    """Build constraint matrix rows for given (time, derivative_order) specs.

    specs: list of (time_selector, derivative_order) where time_selector is 0 for t0 and 1 for t1.
    Returns matrix A with shape (len(specs), deg+1).
    """
    rows = []
    for sel, r in specs:
        t = t0 if sel == 0 else t1
        row = [(_monomial_deriv_coeff(k, r) * (t ** (k - r) if k >= r else 0.0)) for k in range(deg + 1)]
        rows.append(row)
    return jnp.array(rows)


@jax.jit
def compute_hermite_poly4(bc, t0, t1):
    """Compute quartic (degree 4) Hermite interpolant (JIT-friendly).

    `bc` should be an iterable of 5 row-vectors (p0, p1, v0, v1, a0), each of
    shape (n_dof,) or a single 2D array of shape (5, n_dof). Returns ``coeffs``
    with shape (5, n_dof). Use ``eval_hermite_poly(coeffs, t)`` to evaluate the
    polynomial at scalar or array times ``t`` (this evaluator is JIT-compiled).
    """
    deg = 4

    # stack into (5, n_dof)
    B = jnp.asarray(bc)
    if B.ndim == 1:
        # single DOF and bc provided as flat vector
        B = B[:, None]
    # ensure shape
    if B.shape[0] != deg + 1:
        # maybe user passed shape (n_dof, 5)
        if B.shape[1] == deg + 1:
            B = B.T
        else:
            raise ValueError(f"compute_hermite_poly4 expects {deg+1} boundary conditions (p0,p1,v0,v1,a0); got shape {B.shape}")

    specs = [(0, 0),  # p(t0)
             (1, 0),  # p(t1)
             (0, 1),  # p'(t0)
             (1, 1),  # p'(t1)
             (0, 2)]  # p''(t0)

    A = _build_constraint_rows(deg, specs, t0, t1)  # shape (5,5)

    coeffs = jnp.linalg.solve(A, B)  # shape (5, n_dof)

    return coeffs


@jax.jit
def compute_hermite_poly5(bc, t0, t1):
    """Compute quintic (degree 5) Hermite interpolant (JIT-friendly).

    `bc` should be an iterable of 6 row-vectors (p0, p1, v0, v1, a0, a1), each of
    shape (n_dof,) or a single 2D array of shape (6, n_dof). Returns ``coeffs``
    with shape (6, n_dof). Use ``eval_hermite_poly(coeffs, t)`` to evaluate the
    polynomial at scalar or array times ``t`` (this evaluator is JIT-compiled).
    """
    deg = 5

    B = jnp.asarray(bc)
    if B.ndim == 1:
        B = B[:, None]
    if B.shape[0] != deg + 1:
        if B.shape[1] == deg + 1:
            B = B.T
        else:
            raise ValueError(f"compute_hermite_poly5 expects {deg+1} boundary conditions (p0,p1,v0,v1,a0,a1); got shape {B.shape}")

    specs = [(0, 0),  # p(t0)
             (1, 0),  # p(t1)
             (0, 1),  # p'(t0)
             (1, 1),  # p'(t1)
             (0, 2),  # p''(t0)
             (1, 2)]  # p''(t1)

    A = _build_constraint_rows(deg, specs, t0, t1)  # (6,6)

    coeffs = jnp.linalg.solve(A, B)  # (6, n_dof)

    return coeffs


@jax.jit
def eval_hermite_poly(coeffs, t):
    """Evaluate monomial Hermite polynomial(s) given ``coeffs`` at times ``t``.

    ``coeffs`` shape: (deg+1, n_dof). ``t`` may be scalar or 1D array. Returns
    an array with shape (nt, n_dof) (nt=1 for scalar).
    """
    coeffs = jnp.asarray(coeffs)
    t_a = jnp.atleast_1d(t)
    degp1 = coeffs.shape[0]
    # build powers (nt, deg+1)
    powers = jnp.stack([t_a ** k for k in range(degp1)], axis=1)
    return powers @ coeffs

# measure difference between computed path and true path
def compute_cost():

    pass