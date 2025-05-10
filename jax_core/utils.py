"""
utils.py

Author: Kristian Magnus Roen, Spencer M. Richards
Date:   2025-03-17

NOTE: two smat and Smat functions are defined in this file. 
"""
import jax
import numpy as np
import jax.numpy as jnp
from functools import partial
from jax.scipy.linalg import block_diag
from jax.flatten_util import ravel_pytree
# ---------------------------------------------------------------------------
# Helpers for angles and DOF conversions
# ---------------------------------------------------------------------------
def to_positive_angle(angle):
    """
    Force angle from *[-pi,pi) into [0, 2*pi).
    """
    return jnp.where(angle < 0, angle + 2*jnp.pi, angle)


def pipi(angle):
    """
    Force angle into [-pi, pi).
    """
    return jnp.mod(angle + jnp.pi, 2*jnp.pi) - jnp.pi

def Smat(v):
    """Skew-symmetric cross-product matrix (JAX)."""
    v = jnp.asarray(v)
    return jnp.array([[    0.0,  -v[2],  v[1]],
                      [  v[2],     0.0, -v[0]],
                      [ -v[1],   v[0],   0.0]])

def three2sixDOF(v):
    """
    Convert a 3DOF vector or matrix to 6DOF.
    
    - If `v` is a vector of shape (3,), returns an array of shape (6,) with:
      [v[0], v[1], 0, 0, 0, v[2]].
    
    - If `v` is a matrix of shape (3, 3), it flattens `v` and embeds it into a 6x6 matrix.
      The embedding is defined by:
          part1 = flat[0:2]
          part2 = zeros(3)
          part3 = flat[2:5]
          part4 = zeros(3)
          part5 = flat[5:6]
          part6 = zeros(18)
          part7 = flat[6:8]
          part8 = zeros(3)
          part9 = flat[8:9]
      The concatenated vector of length 36 is then reshaped to (6,6).
    """
    #If v is 6DOF, just print a warning and return
    if v.ndim == 1 and v.shape[0] == 6:
        print("Warning: Input is already 6DOF, returning as-is.")
        return v

    if v.ndim == 1:
        if v.shape[0] != 3:
            raise ValueError(f"Expected 1D array with 3 elements, got shape {v.shape}")
        return jnp.array([v[0], v[1], 0.0, 0.0, 0.0, v[2]], dtype=v.dtype)
    elif v.ndim == 2:
        if v.shape != (3, 3):
            raise ValueError(f"Expected 2D array of shape (3,3), got shape {v.shape}")
        flat = jnp.ravel(v)
        part1 = flat[0:2]
        part2 = jnp.zeros(3, dtype=v.dtype)
        part3 = flat[2:5]
        part4 = jnp.zeros(3, dtype=v.dtype)
        part5 = flat[5:6]
        part6 = jnp.zeros(18, dtype=v.dtype)
        part7 = flat[6:8]
        part8 = jnp.zeros(3, dtype=v.dtype)
        part9 = flat[8:9]
        out_flat = jnp.concatenate([part1, part2, part3, part4, part5, part6, part7, part8, part9])
        return jnp.reshape(out_flat, (6, 6))
    else:
        raise ValueError("Input v must be a 1D or 2D array.")


def six2threeDOF(v):
    """
    Convert a 6DOF vector or matrix to 3DOF.
    
    - If `v` is a vector of shape (6,), returns the elements at indices [0, 1, 5] → shape (3,).
    - If `v` is a matrix of shape (T,6) with T != 6, returns the columns [0, 1, 5] → shape (T,3).
    - If `v` is a 6x6 matrix, returns the submatrix with rows and columns [0, 1, 5] → shape (3,3).
    """
    #If v is 3DOF, just print a warning and return
    if v.ndim == 1 and v.shape[0] == 3:
        print("Warning: Input is already 3DOF, returning as-is.")
        return v
    
    if isinstance(v, np.ndarray):
        v = jnp.array(v)
        
    if v.ndim == 1:
        if v.shape[0] != 6:
            raise ValueError(f"Expected 1D array with 6 elements, got shape {v.shape}")
        return v[jnp.array([0, 1, 5])]
    elif v.ndim == 2:
        rows, cols = v.shape
        if rows == 6 and cols == 6:
            # Extract the submatrix with rows and cols [0,1,5]
            idx = jnp.array([0, 1, 5])
            return v[idx][:, idx]
        elif cols == 6:
            return v[:, jnp.array([0, 1, 5])]
        else:
            raise ValueError(f"Expected shape (T,6) or (6,6), got {v.shape} instead.")
    else:
        raise ValueError(f"Input must be a 1D or 2D array, but got array with {v.ndim} dimensions.")
# --------------------------------------------------------------------------
# Rotational matrix functions
# --------------------------------------------------------------------------
def Rx(phi):
    """Return the 3x3 rotation matrix about the x-axis by angle phi."""
    c = jnp.cos(phi)
    s = jnp.sin(phi)
    return jnp.array([[1,  0,  0],
                      [0,  c, -s],
                      [0,  s,  c]], dtype=jnp.float32)


def Ry(theta):
    """Return the 3x3 rotation matrix about the y-axis by angle theta."""
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    return jnp.array([[ c, 0, s],
                      [ 0, 1, 0],
                      [-s, 0, c]], dtype=jnp.float32)


def Rz(psi):
    """Return the 3x3 rotation matrix about the z-axis by angle psi."""
    c = jnp.cos(psi)
    s = jnp.sin(psi)
    return jnp.array([[c, -s, 0],
                      [s,  c, 0],
                      [0,  0, 1]])


def Rzyx(eta):
    """ Full roation matrix"""
    phi, theta, psi = eta[3], eta[4], eta[5]
    return Rz(psi) @ Ry(theta) @ Rx(phi)


def Tzyx(eta):
    phi, theta = eta[3], eta[4]
    return jnp.array([
        [1, jnp.sin(phi)*jnp.tan(theta), jnp.cos(phi)*jnp.tan(theta)],
        [0, jnp.cos(phi), -jnp.sin(phi)],
        [0, jnp.sin(phi)/jnp.cos(theta), jnp.cos(phi)/jnp.cos(theta)]
    ])


def J(eta):
    """6 DOF rotation matrix."""
    return jnp.block([
        [Rzyx(eta), jnp.zeros((3, 3))],
        [jnp.zeros((3, 3)), Tzyx(eta)]
    ])

# --------------------------------------------------------------------------
# ODE solver
# --------------------------------------------------------------------------
"""
Utility functions for integrating ODEs.

Author: Spencer M. Richards, Kristian Magnus Roen
"""
def rk4_step_impl(x, dt, f, *args):
    k1 = f(x, *args)
    k2 = f(x + 0.5 * dt * k1, *args)
    k3 = f(x + 0.5 * dt * k2, *args)
    k4 = f(x + dt * k3, *args)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
rk4_step = jax.jit(rk4_step_impl, static_argnums=(2,))


@partial(jax.jit, static_argnums=(0,))
def rk38_step(func, h, x, t, *args):
    """Do a single step of the RK-38 integration scheme."""
    # RK38 Butcher tableau
    s = 4
    A = jnp.array([
        [0,    0, 0, 0],
        [1/3,  0, 0, 0],
        [-1/3, 1, 0, 0],
        [1,   -1, 1, 0],
    ])
    b = jnp.array([1/8, 3/8, 3/8, 1/8])
    c = jnp.array([0,   1/3, 2/3, 1])

    def scan_fun(carry, cut):
        i, ai, bi, ci = cut
        x, t, h, K, *args = carry
        ti = t + h*ci
        xi = x + h*(K.T @ ai)
        ki = func(xi, ti, *args)
        K = K.at[i].set(ki)
        carry = (x, t, h, K, *args)
        return carry, ki

    init_carry = (x, t, h, jnp.squeeze(jnp.zeros((s, x.size))), *args)
    carry, K = jax.lax.scan(scan_fun, init_carry, (jnp.arange(s), A, b, c))
    xf = x + h*(K.T @ b)
    return xf


@partial(jax.jit, static_argnums=(0,))
def _odeint_ckpt(func, x0, ts, *args):

    def scan_fun(carry, t1):
        x0, t0, *args = carry
        x1 = rk38_step(func, t1 - t0, x0, t0, *args)
        carry = (x1, t1, *args)
        return carry, x1

    ts = jnp.atleast_1d(ts)
    init_carry = (x0, ts[0], *args)  # dummy state at same time as `t0`
    carry, xs = jax.lax.scan(scan_fun, init_carry, ts)
    return xs


@partial(jax.jit, static_argnums=(0,))
def odeint_ckpt(func, x0, ts, *args):
    """Integrate an ODE forward in time."""
    flat_x0, unravel = ravel_pytree(x0)

    def flat_func(flat_x, t, *args):
        x = unravel(flat_x)
        dx = func(x, t, *args)
        flat_dx, _ = ravel_pytree(dx)
        return flat_dx

    # Solve in flat form
    flat_xs = _odeint_ckpt(flat_func, flat_x0, ts, *args)
    xs = jax.vmap(unravel)(flat_xs)
    return xs


@partial(jax.jit, static_argnums=(0, 2, 3, 4))
def odeint_fixed_step(func, x0, t0, t1, step_size, *args):
    """Integrate an ODE forward in time with a fixed step and end time."""
    # Use `numpy` for purely static operations on static arguments
    # (see: https://github.com/google/jax/issues/5208)
    num_steps = int(np.maximum(np.abs((t1 - t0)/step_size), 1))

    ts = jnp.linspace(t0, t1, num_steps + 1)
    xs = odeint_ckpt(func, x0, ts, *args)
    return xs, ts

# --------------------------------------------------------------------------
# Params
# --------------------------------------------------------------------------
"""
Utility functions for handling data and parameters.

Author: Spencer M. Richards, Kristian Magnus Roen
"""
def epoch(key, data, batch_size, batch_axis=0, ragged=False):
    """Generate batches to form an epoch over a data set."""
    # Check for consistent dimensions along `batch_axis`
    flat_data, _ = jax.tree_util.tree_flatten(data)
    num_samples = jnp.array(jax.tree_util.tree_map(
        lambda x: jnp.shape(x)[batch_axis],
        flat_data
    ))
    if not jnp.all(num_samples == num_samples[0]):
        raise ValueError('Batch dimensions not equal!')
    num_samples = num_samples[0]

    # Compute the number of batches
    if ragged:
        num_batches = -(-num_samples // batch_size)  # ceiling division
    else:
        num_batches = num_samples // batch_size  # floor division

    # Loop through batches (with pre-shuffling)
    shuffled_idx = jax.random.permutation(key, num_samples)
    for i in range(num_batches):
        batch_idx = shuffled_idx[i*batch_size:(i+1)*batch_size]
        batch = jax.tree_util.tree_map(
            lambda x: jnp.take(x, batch_idx, batch_axis),
            data
        )
        yield batch


def mat_to_svec_dim(n):
    """Compute the number of unique entries in a symmetric matrix."""
    d = (n * (n + 1)) // 2
    return d


def svec_to_mat_dim(d):
    """Compute the symmetric matrix dimension with `d` unique elements."""
    n = (int(np.sqrt(8 * d + 1)) - 1) // 2
    if d != mat_to_svec_dim(n):
        raise ValueError('Invalid vector length `d = %d` for filling the '
                         'triangular of a symmetric matrix!' % d)
    return n


def svec_diag_indices(n):
    """Compute indices of `svec(A)` corresponding to diagonal elements.

    Example for `n = 3`:
    [ 0       ]
    [ 1  3    ]  => [0, 3, 5]
    [ 2  4  5 ]

    For general `n`, indices of `svec` corresponding to the diagonal are:
      [0, n, n + (n-1), ..., n*(n+1)/2 - 1]
      = n*(n+1)/2 - [n*(n+1)/2, (n-1)*n/2, ..., 1]
    """
    d = mat_to_svec_dim(n)
    idx = d - mat_to_svec_dim(np.arange(1, n+1)[::-1])
    return idx


def svec(X, scale=True):
    """Compute the symmetric vectorization of symmetric matrix `X`."""
    shape = jnp.shape(X)
    if len(shape) < 2:
        raise ValueError('Argument `X` must be at least 2D!')
    if shape[-2] != shape[-1]:
        raise ValueError('Last two dimensions of `X` must be equal!')
    n = shape[-1]

    if scale:
        # Scale elements corresponding to the off-diagonal, lower-triangular
        # part of `X` by `sqrt(2)` to preserve the inner product
        rows, cols = jnp.tril_indices(n, -1)
        X = X.at[..., rows, cols].mul(jnp.sqrt(2))

    # Vectorize the lower-triangular part of `X` in row-major order
    rows, cols = jnp.tril_indices(n)
    svec_X = X[..., rows, cols]
    return svec_X


def smat(svec_X, scale=True):
    """Compute the symmetric matrix `X` given `svec(X)`."""
    svec_X = jnp.atleast_1d(svec_X)
    d = svec_X.shape[-1]
    n = svec_to_mat_dim(d)  # corresponding symmetric matrix dimension

    # Fill the lower triangular of `X` in row-major order with the elements
    # of `svec_X`
    rows, cols = jnp.tril_indices(n)
    X = jnp.zeros((*svec_X.shape[:-1], n, n))
    X = X.at[..., rows, cols].set(svec_X)
    if scale:
        # Scale elements corresponding to the off-diagonal, lower-triangular
        # elements of `X` by `1 / sqrt(2)` to preserve the inner product
        rows, cols = jnp.tril_indices(n, -1)
        X = X.at[..., rows, cols].mul(1 / jnp.sqrt(2))

    # Make `X` symmetric
    rows, cols = jnp.triu_indices(n, 1)
    X = X.at[..., rows, cols].set(X[..., cols, rows])
    return X


def cholesky_to_params(L):
    """Uniquely parameterize a positive-definite Cholesky factor."""
    shape = jnp.shape(L)
    if len(shape) < 2:
        raise ValueError('Argument `L` must be at least 2D!')
    if shape[-2] != shape[-1]:
        raise ValueError('Last two dimensions of `L` must be equal!')
    n = shape[-1]
    rows, cols = jnp.diag_indices(n)
    log_L = L.at[..., rows, cols].set(jnp.log(L[..., rows, cols]))
    params = svec(log_L, scale=False)
    return params


def params_to_cholesky(params):
    """Map free parameters to a positive-definite Cholesky factor."""
    params = jnp.atleast_1d(params)
    d = params.shape[-1]
    n = svec_to_mat_dim(d)  # corresponding symmetric matrix dimension
    rows, cols = jnp.tril_indices(n)
    log_L = jnp.zeros((*params.shape[:-1], n, n)).at[...,
                                                     rows, cols].set(params)
    rows, cols = jnp.diag_indices(n)
    L = log_L.at[..., rows, cols].set(jnp.exp(log_L[..., rows, cols]))
    return L


def params_to_posdef(params):
    """Map free parameters to a positive-definite matrix."""
    L = params_to_cholesky(params)
    LT = jnp.swapaxes(L, -2, -1)
    X = L @ LT
    return X


def diag_chol_indices(n):
    """
    Return the indices in the length‐d Cholesky‐parameter vector
    corresponding to the diagonal of an n×n L.

    For n=3, this returns [0,2,5], since the param order is
        [L00, L10, L11, L20, L21, L22].
    """
    return jnp.array([i * (i+1)//2 + i for i in range(n)], dtype=int)


def vec_to_posdef_diag_cholesky(v):
    """
    Build a *purely diagonal* PD matrix X from an unconstrained vector v∈ℝⁿ
    by embedding it into the Cholesky‐parametrization and reusing params_to_posdef.

    Internally:
      • full = zeros(d) with d = n(n+1)/2
      • full[idxs] = v/2    # see note below
      • L = params_to_cholesky(full)  # exponentiates diag(full)
      • X = L @ L.T

    Because full[i*(i+1)/2 + i] = v_i/2, L_ii = exp(v_i/2), so X_ii = exp(v_i).

    Off‐diagonal slots of full remain zero ⇒ L_ij=0 for i≠j ⇒ X is exactly diagonal.
    """
    v = jnp.atleast_1d(v)
    n = v.shape[-1]
    d = mat_to_svec_dim(n)                # = n(n+1)/2
    idxs = diag_chol_indices(n)           # which param‑vector slots are diag
    full = jnp.zeros(v.shape[:-1] + (d,), dtype=v.dtype)
    # scatter half‐logs so that X_ii = exp(v_i)
    full = full.at[..., idxs].set(v/2)
    # now call your existing reparam → PD
    return params_to_posdef(full)


# --------------------------------------------------------------------------
# pytree
# --------------------------------------------------------------------------
"""
Some utilities for dealing with PyTrees of parameters.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""

def tree_scale(x_tree, a):
    """Scale the children of a PyTree by the scalar `a`."""
    return jax.tree_util.tree_map(lambda x: a * x, x_tree)


def tree_add(x_tree, y_tree):
    """Add pairwise the children of two PyTrees."""
    return jax.tree_util.tree_multimap(lambda x, y: x + y, x_tree, y_tree)


def tree_index(x_tree, i):
    """Index child arrays in PyTree."""
    return jax.tree_util.tree_map(lambda x: x[i], x_tree)


def tree_index_update(x_tree, i, y_tree):
    """Update indices of child arrays in PyTree with new values."""
    return jax.tree_util.tree_multimap(lambda x, y:
                                       jax.ops.index_update(x, i, y),
                                       x_tree, y_tree)


def tree_axpy(a, x_tree, y_tree):
    """Compute `a*x + y` for two PyTrees `(x, y)` and a scalar `a`."""
    ax = tree_scale(x_tree, a)
    axpy = jax.tree_util.tree_multimap(lambda x, y: x + y, ax, y_tree)
    return axpy


def tree_dot(x_tree, y_tree):
    """Compute the dot products between children of two PyTrees."""
    xy = jax.tree_util.tree_multimap(lambda x, y: jnp.sum(x*y), x_tree, y_tree)
    return xy


def tree_normsq(x_tree):
    """Compute sum of squared norms across a PyTree."""
    normsq = jax.tree_util.tree_reduce(lambda x, y: x + jnp.sum(y**2),
                                       x_tree, 0.)
    return normsq


def tree_anynan(tree):
    """Check if there are any NAN elements in the PyTree."""
    any_isnan_tree = jax.tree_util.tree_map(lambda a: jnp.any(jnp.isnan(a)),
                                            tree)
    any_isnan = jax.tree_util.tree_reduce(lambda x, y: jnp.logical_or(x, y),
                                          any_isnan_tree, False)
    return any_isnan

# --------------------------------------------------------------------------
# Trajectory generation
# --------------------------------------------------------------------------
"""
Utilities for trajectory generation, particularly via spline interpolation.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""

@partial(jax.jit, static_argnums=(2, 3))
def _scalar_smooth_trajectory(x_knots, t_knots, poly_order, deriv_order):
    """Construct a smooth trajectory through given points.

    References
    ----------
    .. [1] Charles Richter, Adam Bry, and Nicholas Roy,
           "Polynomial trajectory planning for aggressive quadrotor flight in
           dense indoor environments", ISRR 2013.
    .. [2] Daniel Mellinger and Vijay Kumar,
           "Minimum snap trajectory generation and control for quadrotors",
           ICRA 2011.
    .. [3] Declan Burke, Airlie Chapman, and Iman Shames,
           "Generating minimum-snap quadrotor trajectories really fast",
           IROS 2020.
    """
    num_coefs = poly_order + 1          # number of coefficients per polynomial
    num_knots = x_knots.size            # number of interpolating points
    num_polys = num_knots - 1           # number of polynomials
    primal_dim = num_coefs * num_polys  # number of unknown coefficients

    T = jnp.diff(t_knots)                # polynomial lengths in time
    powers = jnp.arange(poly_order + 1)  # exponents defining each monomial
    D = jnp.diag(powers[1:], -1)         # maps monomials to their derivatives

    c0 = jnp.zeros((deriv_order + 1, num_coefs)).at[0, 0].set(1.)
    c1 = jnp.zeros((deriv_order + 1, num_coefs)).at[0, :].set(1.)
    for n in range(1, deriv_order + 1):
        c0 = c0.at[n].set(D @ c0[n - 1])
        c1 = c1.at[n].set(D @ c1[n - 1])

    # Assemble constraints in the form `A @ x = b`, where `x` is the vector of
    # stacked polynomial coefficients

    # Knots
    b_knots = jnp.concatenate((x_knots[:-1], x_knots[1:]))
    A_knots = jnp.vstack([
        block_diag(*jnp.tile(c0[0], (num_polys, 1))),
        block_diag(*jnp.tile(c1[0], (num_polys, 1)))
    ])

    # Zero initial conditions (velocity, acceleration, jerk)
    b_init = jnp.zeros(deriv_order - 1)
    A_init = jnp.zeros((deriv_order - 1, primal_dim))
    A_init = A_init.at[:deriv_order - 1, :num_coefs].set(c0[1:deriv_order])

    # Zero final conditions (velocity, acceleration, jerk)
    b_fin = jnp.zeros(deriv_order - 1)
    A_fin = jnp.zeros((deriv_order - 1, primal_dim))
    A_fin = A_fin.at[:deriv_order - 1, -num_coefs:].set(c1[1:deriv_order])

    # Continuity (velocity, acceleration, jerk, snap)
    b_cont = jnp.zeros(deriv_order * (num_polys - 1))
    As = []
    zero_pad = jnp.zeros((num_polys - 1, num_coefs))
    Tn = jnp.ones_like(T)
    for n in range(1, deriv_order + 1):
        Tn = T * Tn
        diag_c0 = block_diag(*(c0[n] / Tn[1:].reshape([-1, 1])))
        diag_c1 = block_diag(*(c1[n] / Tn[:-1].reshape([-1, 1])))
        As.append(jnp.hstack((diag_c1, zero_pad))
                  - jnp.hstack((zero_pad, diag_c0)))
    A_cont = jnp.vstack(As)

    # Assemble
    A = jnp.vstack((A_knots, A_init, A_fin, A_cont))
    b = jnp.concatenate((b_knots, b_init, b_fin, b_cont))
    dual_dim = b.size

    # Compute the cost Hessian `Q(T)` as a function of the length `T` for each
    # polynomial, and stack them into the full block-diagonal Hessian
    ij_1 = powers.reshape([-1, 1]) + powers + 1
    D_snap = jnp.linalg.matrix_power(D, deriv_order)
    Q_snap = D_snap @ (1 / ij_1) @ D_snap.T
    Q_poly = lambda T: Q_snap / (T**(2*deriv_order - 1))  # noqa: E731
    Q = block_diag(*jax.vmap(Q_poly)(T))

    # Assemble KKT system and solve for coefficients
    K = jnp.block([
        [Q, A.T],
        [A, jnp.zeros((dual_dim, dual_dim))]
    ])
    soln = jnp.linalg.solve(K, jnp.concatenate((jnp.zeros(primal_dim), b)))
    primal, dual = soln[:primal_dim], soln[-dual_dim:]
    coefs = primal.reshape((num_polys, -1))
    r_primal = A@primal - b
    r_dual = Q@primal + A.T@dual
    return coefs, r_primal, r_dual


@partial(jax.jit, static_argnums=(2, 3))
def smooth_trajectory(x_knots, t_knots, poly_order, deriv_order):
    """Compute the coefficients of a smooth spline through the given knots."""
    num_knots = x_knots.shape[0]
    knot_shape = x_knots.shape[1:]
    flat_x_knots = jnp.reshape(x_knots, (num_knots, -1))
    in_axes = (1, None, None, None)
    out_axes = (2, 1, 1)
    flat_coefs, _, _ = jax.vmap(_scalar_smooth_trajectory,
                                in_axes, out_axes)(flat_x_knots, t_knots,
                                                   poly_order, deriv_order)
    num_polys = num_knots - 1
    coefs = jnp.reshape(flat_coefs, (num_polys, poly_order + 1, *knot_shape))
    return coefs


@jax.jit
def spline(t, t_knots, coefs):
    """Compute the value of a polynomial spline at time `t`."""
    num_polys = coefs.shape[0]
    poly_order = coefs.shape[1] - 1
    powers = jnp.arange(poly_order + 1)

    # Identify which polynomial segment to use for time `t`
    i = jnp.clip(jnp.searchsorted(t_knots, t, 'left') - 1, 0, num_polys - 1)

    # Evaluate the polynomial (with scaled time)
    tau = (t - t_knots[i]) / (t_knots[i+1] - t_knots[i])
    x = jnp.sum(coefs[i] * (tau**powers))
    return x


def spline_factory(t_knots, *coefs):
    """Define and return a polynomial spline function."""
    spline_func = (lambda t, t_knots=t_knots, coefs=coefs:
                   jnp.array([spline(t, t_knots, C) for C in coefs]))
    return spline_func


def uniform_random_walk(key, num_steps, shape=(), min_step=0., max_step=1.):
    """Sample a random walk of points.

    The step size is sampled uniformly from a closed interval.
    """
    minvals = jnp.broadcast_to(min_step, shape)
    maxvals = jnp.broadcast_to(max_step, shape)
    noise = minvals + (maxvals - minvals)*jax.random.uniform(key, (num_steps,
                                                                   *shape))
    points = jnp.concatenate((jnp.zeros((1, *shape)),
                              jnp.cumsum(noise, axis=0)), axis=0)
    return points


def random_spline(key, T_total, num_knots, poly_order, deriv_order,
                  shape=(), min_step=0., max_step=1.):
    """Sample a random walk and fit a spline to it."""
    knots = uniform_random_walk(key, num_knots - 1, shape, min_step, max_step)
    flat_knots = jnp.reshape(knots, (num_knots, -1))
    diffs = jnp.linalg.norm(jnp.diff(flat_knots, axis=0), axis=1)
    T = T_total * (diffs / jnp.sum(diffs))
    t_knots = jnp.concatenate((jnp.array([0., ]),
                               jnp.cumsum(T))).at[-1].set(T_total)
    coefs = smooth_trajectory(knots, t_knots, poly_order, deriv_order)
    return knots, t_knots, coefs


def random_ragged_spline(key, T_total, num_knots, poly_orders, deriv_orders,
                         min_step, max_step, min_knot=-jnp.inf,
                         max_knot=jnp.inf):
    """Sample a random walk and fit a spline to it.

    Different polynomial and smoothness orders can be used for each dimension.
    If this is done, the spline coefficient arrays will be of different shapes
    for each dimension, i.e., "ragged".
    """
    poly_orders = np.array(poly_orders).ravel().astype(int)
    deriv_orders = np.array(deriv_orders).ravel().astype(int)
    num_dims = poly_orders.size
    assert deriv_orders.size == num_dims
    shape = (num_dims,)
    knots = uniform_random_walk(key, num_knots - 1, shape, min_step, max_step)
    knots = jnp.clip(knots, min_knot, max_knot)
    flat_knots = jnp.reshape(knots, (num_knots, -1))
    diffs = jnp.linalg.norm(jnp.diff(flat_knots, axis=0), axis=1)
    T = T_total * (diffs / jnp.sum(diffs))
    t_knots = jnp.concatenate((jnp.array([0., ]),
                               jnp.cumsum(T))).at[-1].set(T_total)
    coefs = []
    for i, (p, d) in enumerate(zip(poly_orders, deriv_orders)):
        coefs.append(smooth_trajectory(knots[:, i], t_knots, p, d))
    coefs = tuple(coefs)
    knots = tuple(knots[:, i] for i in range(num_dims))
    return t_knots, knots, coefs
