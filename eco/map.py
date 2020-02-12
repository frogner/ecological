import numpy as np


def eco_map(u, v, C, which_prior, outer_iter=100, inner_iter=20,
            outer_tol=1e-4, inner_tol=1e-8, kappa0=1e0,
            max_norm_step=None, **param):
    """ MAP inference for models assuming perfectly-observed marginals.

    Args:
        u: first marginal, n x d_0 array of marginal data for n tables
            having first dimension d_0
        v: second marginal, n x d_1 array of marginal data for n tables
            having second dimension d_1
        C: prior parameters, d_0 x d1 array
        which_prior: one of {'entropic', 'tsallis', 'normal', 'dirichlet'}
        outer_iter: (optional) maximum number of outer Dykstra iterations
        inner_iter: (optional) maximum number of inner Newton steps
        outer_tol: (optional) convergence threshold for terminating
            outer iterations
        inner_tol: (optional) convergence threshold for terminating
            inner iterations
        kappa0: (optional) inner Newton step size
        max_norm_step: (optional) project large Newton steps onto a ball
            of this radius
        param: (optional) parameters specific to the prior distribution

    Returns:
        An array of shape n x d_0 x d_1, containing n inferred tables.
    """
    n = u.shape[0]
    dims = C.shape

    theta = np.zeros((n,) + dims)
    if which_prior == 'dirichlet':
        theta -= 1e-16

    theta = projbox(theta, C, which_prior, **param)
    for ii in range(outer_iter):
        theta_prev = theta
        theta = projmarg(1, theta, v, C, which_prior, 
            max_iter=inner_iter, tol=inner_tol, kappa0=kappa0,
            max_norm_step=max_norm_step, **param)
        theta = projbox(theta, C, which_prior, **param)
        theta = projmarg(0, theta, u, C, which_prior,
            max_iter=inner_iter, tol=inner_tol, kappa0=kappa0,
            max_norm_step=max_norm_step, **param)
        theta = projbox(theta, C, which_prior, **param)
        progress = \
            np.max(np.sqrt(np.sum(
                np.power((theta - theta_prev).reshape((n, -1)), 2), axis=1)))
        if progress < outer_tol:
            break

    return theta2T(theta, C, which_prior, **param)


def theta2T(theta, C, which_prior, **param):
    if which_prior == 'entropic':
        T = np.exp((1. / param['gamma']) * (theta - C) - 1.)
    elif which_prior == 'tsallis':
        q = param['q']
        gamma = param['gamma']
        T = np.power((1. / q) * (((q - 1) / gamma) * (theta - C) + 1.),
                     (1. / (q - 1.)))
    elif which_prior == 'normal':
        T = (param['sigma'] ** 2) * theta + C
    elif which_prior == 'dirichlet':
        T = -C / theta
    return T


def dTdtheta(theta, C, which_prior, **param):
    if which_prior == 'entropic':
        gamma = param['gamma']
        dtheta = (1. / gamma) * np.exp((1. / gamma) * (theta - C) - 1.)
    elif which_prior == 'tsallis':
        q = param['q']
        gamma = param['gamma']
        X = (1. / q) * (((q - 1) / gamma) * (theta - C) + 1.)
        dtheta = (1. / (q * gamma)) * np.power(X, (2. - q) / (q - 1.))
    elif which_prior == 'normal':
        dtheta = param['sigma'] ** 2
    elif which_prior == 'dirichlet':
        dtheta = C / np.power(theta, 2)
    return dtheta


def projbox(theta, C, which_prior, **param):
    if which_prior == 'tsallis':
        q = param['q']
        gamma = param['gamma']
        if (q > 1.):
            min_val = C - (gamma / (q - 1)) + 1e-8
        else:
            min_val = -np.inf
    elif which_prior == 'normal':
        min_val = (1 / (param['sigma'] ** 2)) * -C
    else:
        min_val = -np.inf
    return np.maximum(min_val, theta)


def projdomain(theta, C, which_prior, **param):
    if which_prior == 'dirichlet':
        return np.minimum(theta, -1e-32)
    else:
        return theta


def projmarg(which_marg, theta, obs, C, which_prior, max_iter=20, tol=1e-8,
             kappa0=1e0, max_norm_step=None, **param):
    n = theta.shape[0]
    n_obs = obs.shape[1]
    total_n_dim = len(C.shape)

    backtrack_factor = 4. / 5.

    if which_prior == 'entropic':
        gamma = param['gamma']
        if max_norm_step is None:
            max_norm_step = 1e1
    elif which_prior == 'tsallis':
        q = param['q']
        gamma = param['gamma']
        if max_norm_step is None:
            max_norm_step = 1e0
    elif which_prior == 'normal':
        sigma = param['sigma']
        if max_norm_step is None:
            max_norm_step = np.inf
    else:
        if max_norm_step is None:
            max_norm_step = np.inf

    norm_grad = np.nan * np.zeros((max_iter,))

    obs = obs.reshape((n,) + (1,) * which_marg 
                      + (n_obs,) + (1,) * (total_n_dim - which_marg - 1))

    for ii in range(max_iter):
        if which_prior == 'entropic':
            dQ = np.exp((1. / gamma) * (theta - C)  - 1.)
            d2Q = (1. / gamma) * dQ
        elif which_prior == 'tsallis':
            dQ = np.power((1. / q) * (((q - 1.) / gamma) * (theta - C) + 1.),
                          (1. / (q - 1.)))
            d2Q = (1. / (q * gamma)) \
                * np.power((1. / q) * (((q - 1.) / gamma) * (theta - C) + 1.),
                           (2. - q) / (q - 1.))
        elif which_prior == 'normal':
            dQ = (sigma ** 2) * theta + C
            d2Q = (sigma ** 2) * np.ones(theta.shape)
        elif which_prior == 'dirichlet':
            dQ = -C / theta
            d2Q = C / np.power(theta, 2)

        for dd in range(1, total_n_dim + 1):
            if dd != which_marg + 1:
                dQ = np.sum(dQ, axis=dd, keepdims=True)
                d2Q = np.sum(d2Q, axis=dd, keepdims=True)

        step = (obs - dQ) / (d2Q + 1e-32)
        is_too_big = np.abs(step) > max_norm_step
        step[is_too_big] = np.sign(step[is_too_big]) * max_norm_step
        kappa = kappa0 * np.ones((n,) + (1,) * total_n_dim)
        while True:
            theta_step = projdomain(theta + kappa * step,
                                    C, which_prior, **param)

            is_viol = np.zeros(kappa.shape).astype(np.bool)
            if which_prior == 'tsallis':
                if q < 1.:
                    is_viol = np.abs(((q - 1.) / gamma) 
                                     * (theta_step - C) + 1.) < 1e-16
                    for dd in range(1, total_n_dim + 1):
                        is_viol = np.any(is_viol, axis=dd, keepdims=True)
            elif which_prior == 'dirichlet':
                is_viol = theta_step >= 0.
                for dd in range(1, total_n_dim + 1):
                    is_viol = np.any(is_viol, axis=dd, keepdims=True)
            if np.any(is_viol):
                kappa[is_viol] = kappa[is_viol] * backtrack_factor
            else:
                theta = theta_step
                break

        norm_grad[ii] = np.sqrt(np.sum(np.power(obs - dQ, 2)))
        if norm_grad[ii] < tol:
            break

    return theta


def eco_noisy_map(u, v, C, which_prior, which_noise, outer_iter=100,
            inner_iter=20, outer_tol=1e-4, inner_tol=1e-8, kappa0=None,
            max_norm_step=None, **param):
    """ MAP inference for models assuming imperfectly-observed marginals.

    Args:
        u: first marginal, n x d_0 array of marginal data for n tables
            having first dimension d_0
        v: second marginal, n x d_1 array of marginal data for n tables
            having second dimension d_1
        C: prior parameters, d_0 x d1 array
        which_prior: one of {'entropic', 'tsallis', 'normal', 'dirichlet'}
        which_noise: marginal noise model, one of {None, 'multinomial'} --
            may specify different models for the two marginals, as a tuple,
            e.g. (None, 'multinomial')
        outer_iter: (optional) maximum number of outer Dykstra iterations
        inner_iter: (optional) maximum number of inner Newton steps
        outer_tol: (optional) convergence threshold for terminating
            outer iterations
        inner_tol: (optional) convergence threshold for terminating
            inner iterations
        kappa0: (optional) inner Newton step size
        max_norm_step: (optional) project large Newton steps onto a ball
            of this radius
        param: (optional) parameters specific to the prior distribution

    Returns:
        An array of shape n x d_0 x d_1, containing n inferred tables.
    """
    n = u.shape[0]
    dims = C.shape

    if not isinstance(which_noise, tuple) \
       and not isinstance(which_noise, list):
        which_noise = (which_noise, which_noise)
    if not isinstance(kappa0, tuple) \
       and not isinstance(kappa0, list):
        kappa0 = (kappa0, kappa0)
    if not isinstance(max_norm_step, tuple) \
       and not isinstance(max_norm_step, list):
        max_norm_step = (max_norm_step, max_norm_step)

    if (which_prior == 'tsallis') and (param['q'] > 1.):
        n_proj_simplex = 8
    else:
        n_proj_simplex = 1

    theta = np.zeros((n,) + dims)
    if which_prior == 'dirichlet':
        theta -= 1e-16

    Z = [np.zeros((n,) + dims), np.zeros((n,) + dims)]

    theta = projbox(theta, C, which_prior, **param)
    for ii in range(outer_iter):
        theta_prev = theta
        theta_double_prev = theta
        theta = projdomain(theta + Z[1], C, which_prior, **param)
        if which_noise[1] is None:
            theta = projmarg(1, theta, v, C, which_prior, max_iter=inner_iter,
             tol=inner_tol, kappa0=kappa0[1], max_norm_step=max_norm_step[1],
             **param)
            theta = projbox(theta, C, which_prior, **param)
        else:
            theta = projnoisymarg(1, theta, v, C, which_prior, which_noise[1],
                        max_iter=inner_iter, tol=inner_tol, kappa0=kappa0[1],
                        max_norm_step=max_norm_step[1], **param)
            for kk in range(n_proj_simplex):
                theta = projsimplex(theta, C, which_prior, **param)
        Z[1] = Z[1] + theta_prev - theta
        theta_prev = theta
        theta = projdomain(theta + Z[0], C, which_prior, **param)
        if which_noise[0] is None:
            theta = projmarg(0, theta, u, C, which_prior, max_iter=inner_iter,
             tol=inner_tol, kappa0=kappa0[0], max_norm_step=max_norm_step[0],
             **param)
            theta = projbox(theta, C, which_prior, **param)
        else:
            theta = projnoisymarg(0, theta, u, C, which_prior, which_noise[0],
                        max_iter=inner_iter, tol=inner_tol, kappa0=kappa0[0],
                        max_norm_step=max_norm_step[0], **param)
            for kk in range(n_proj_simplex):
                theta = projsimplex(theta, C, which_prior, **param)
        Z[0] = Z[0] + theta_prev - theta
        progress = \
            np.max(np.sqrt(np.sum(np.power(
                (theta - theta_double_prev).reshape((n, -1)), 2), axis=1)))
        if progress < outer_tol:
            break

    return theta2T(theta, C, which_prior, **param)


def projsimplex(theta, C, which_prior, **param):
    n = theta.shape[0]
    dims = C.shape

    bound_scale = 1e0
    bound_factor = 2.
    lamb_l = -bound_scale * np.ones((n,))
    lamb_h = bound_scale * np.ones((n,))

    is_converged = np.zeros((n,)).astype(np.bool)
    has_moved_l = np.zeros((n,)).astype(np.bool)
    has_moved_h = np.zeros((n,)).astype(np.bool)

    tol = 1e-8

    while True:
        not_converged_idx = np.nonzero(~is_converged)[0]
        lamb = (lamb_h[~is_converged] + lamb_l[~is_converged,]) / 2.
        theta_proj = projdomain(theta[~is_converged, :, :] 
                                - lamb[:, None, None], C, which_prior, **param)
        T = theta2T(theta_proj, C, which_prior, **param)
        dlamb = (1. - np.sum(np.sum(T, axis=2), axis=1))
        if which_prior == 'tsallis' and param['q'] < 1.:
            dlamb *= np.sum(dTdtheta(theta_proj, C, which_prior, **param)
                            .reshape((np.sum(~is_converged), -1)), axis=1)
        is_small_dlamb = np.abs(dlamb) < tol
        is_converged[~is_converged] = is_small_dlamb
        if np.all(is_converged):
            break
        is_h = ~is_small_dlamb & (dlamb > 0.)
        is_l = ~is_small_dlamb & (dlamb < 0.)
        if np.any(is_h):
            lamb_h[not_converged_idx[is_h]] = lamb[is_h]
            has_moved_h[not_converged_idx[is_h]] = True
        if np.any(is_l):
            lamb_l[not_converged_idx[is_l]] = lamb[is_l]
            has_moved_l[not_converged_idx[is_l]] = True
        maybe_converged = \
            (lamb_h[not_converged_idx] - lamb_l[not_converged_idx]) < tol
        is_stuck_l = \
            ~is_small_dlamb & maybe_converged & ~has_moved_l[not_converged_idx]
        if np.any(is_stuck_l):
            lamb_l[not_converged_idx[is_stuck_l]] -= bound_scale
            bound_scale *= bound_factor
        is_stuck_h = \
            ~is_small_dlamb & maybe_converged & ~has_moved_h[not_converged_idx]
        if np.any(is_stuck_h):
            lamb_h[not_converged_idx[is_stuck_h]] += bound_scale
            bound_scale *= bound_factor

    lamb = (lamb_h + lamb_l) / 2.

    return projbox(projdomain(theta - lamb[:, None, None],
        C, which_prior, **param), C, which_prior, **param)


def projnoisymarg(which_marg, theta, obs, C, which_prior, which_noise,
                  max_iter=20, tol=1e-8, kappa0=None,
                  max_norm_step=None, **param):
    n = theta.shape[0]
    n_obs = obs.shape[1]
    total_n_dim = len(C.shape)

    if total_n_dim != 2:
        raise NotImplementedError
    other_marg = (which_marg + 1) % 2

    kappa_tol = 1e-32
    backtrack_factor = 4. / 5.
    do_backtracking = False

    if which_prior == 'entropic':
        gamma = param['gamma']
        if kappa0 is None:
            kappa0 = 1e-2
        if max_norm_step is None:
            max_norm_step = 1e2
    elif which_prior == 'tsallis':
        q = param['q']
        gamma = param['gamma']
        if q > 1.:
            if kappa0 is None:
                kappa0 = 1e-2
            if max_norm_step is None:
                max_norm_step = 1e4
        else:
            if kappa0 is None:
                kappa0 = 1e-1
            if max_norm_step is None:
                max_norm_step = 1e4
    elif which_prior == 'normal':
        sigma = param['sigma']
        if kappa0 is None:
            kappa0 = 1e-1
        if max_norm_step is None:
            max_norm_step = 1e0
    elif which_prior == 'dirichlet':
        if kappa0 is None:
            kappa0 = 1e-1
        if max_norm_step is None:
            max_norm_step = 1e1
    else:
        if kappa0 is None:
            kappa0 = 1e-1
        if max_norm_step is None:
            max_norm_step = np.inf

    norm_grad = np.nan * np.zeros((max_iter,))

    obs = obs.reshape((n,) + (1,) * which_marg + (n_obs,) 
                        + (1,) * (total_n_dim - which_marg - 1))

    theta_orig = theta

    for ii in range(max_iter):
        if which_prior == 'entropic':
            dQ = np.exp((1. / gamma) * (theta - C)  - 1.)
            d2Q = (1. / gamma) * dQ
        elif which_prior == 'tsallis':
            dQ = np.power((1. / q) * (((q - 1.) / gamma) * (theta - C) + 1.),
                          (1. / (q - 1.)))
            d2Q = (1. / (q * gamma)) \
                * np.power((1. / q) * (((q - 1.) / gamma) * (theta - C) + 1.),
                           ((2. - q) / (q - 1.)))
        elif which_prior == 'normal':
            dQ = (sigma ** 2) * theta + C
            d2Q = (sigma ** 2) * np.ones(theta.shape)
        elif which_prior == 'dirichlet':
            dQ = -C / theta
            d2Q = C / np.power(theta, 2)

        T = theta2T(theta, C, which_prior, **param)
        if which_noise == 'multinomial':
            p_marg = np.sum(T, axis=other_marg + 1, keepdims=True)
            dnoise = -obs / (p_marg + 1e-32)
            d2noise = obs / (np.power(p_marg, 2) + 1e-32)

        for dd in range(1, total_n_dim + 1):
            if dd != which_marg + 1:
                dQ = np.sum(dQ, axis=dd, keepdims=True)
                d2Q = np.sum(d2Q, axis=dd, keepdims=True)

        step = (theta_orig - dQ - dnoise) / (d2Q + d2noise + 1e-32)
        norm_step = np.sqrt(np.sum(np.sum(
            np.power(step, 2), axis=2, keepdims=True), axis=1, keepdims=True))
        is_too_big = np.squeeze(norm_step) > max_norm_step
        if np.any(is_too_big):
            step[is_too_big, :, :] = max_norm_step * step[is_too_big, :, :] \
                / norm_step[is_too_big, :, :]

        kappa = kappa0 * np.ones((n,) + (1,) * total_n_dim)
        while True:
            if np.any(kappa < kappa_tol):
                kappa[kappa < kappa_tol] = 0.

            theta_step = projdomain(theta + kappa * step,
                                    C, which_prior, **param)

            is_viol = np.zeros(kappa.shape).astype(np.bool)
            if which_prior == 'tsallis':
                if q < 1.:
                    is_viol = np.abs(((q - 1.) / gamma)
                                      * (theta_step - C) + 1.) < 1e-16
                    for dd in range(1, total_n_dim + 1):
                        is_viol = np.any(is_viol, axis=dd, keepdims=True)
            elif which_prior == 'dirichlet':
                is_viol = theta_step >= 0.
                for dd in range(1, total_n_dim + 1):
                    is_viol = np.any(is_viol, axis=dd, keepdims=True)
            if np.any(is_viol):
                kappa[is_viol] = kappa[is_viol] * backtrack_factor
            else:
                theta = theta_step
                break

        norm_grad[ii] = np.sqrt(np.sum(np.power(theta_orig - dQ - dnoise, 2)))
        if norm_grad[ii] < tol:
            break

    return theta