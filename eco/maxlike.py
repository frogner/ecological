import numpy as np
import scipy.special as sp


def eco_max_like(Tobs, which_prior, max_iter=1000, tol=1e-8, lamb=1e-4,
                 **param):
    """ Estimate prior parameters by maximum likelihood, given some observed
        tables.

    Args:
        Tobs: n x dims array containing n tables of dimension dims
        which_prior: distribution whose parameters we're trying to estimate,
            one of {'entropic', 'tsallis', 'normal', 'dirichlet'}
        max_iter: maximum number of Newton steps
        tol: Newton convergence threshold
        lamb: regularization parameter
        param: parameters specific to the prior

    Returns:
        An array of dimension dims, containing the estimated prior parameters.
    """
    n = Tobs.shape[0]
    dims = Tobs.shape[1:]

    if which_prior == 'entropic':
        stepsize = 1e-1
        x_min = -np.inf
    elif which_prior == 'tsallis':
        if param['q'] < 1.:
            stepsize = 1e-1
        else:
            stepsize = 1e-2
        x_min = -np.inf
    elif which_prior == 'normal':
        stepsize = 1e-1
        x_min = -np.inf
    elif which_prior == 'dirichlet':
        stepsize = 1e-2
        x_min = 1e-8

    x = np.ones((np.prod(dims),))
    f = np.nan * np.zeros((max_iter,))
    norm_g = np.nan * np.zeros((max_iter,))

    for ii in range(max_iter):
        f[ii], g, H = ml_obj(x, Tobs, which_prior, lamb, **param)
        norm_g[ii] = np.sqrt(np.sum(np.power(g, 2)))

        if norm_g[ii] < tol:
            break

        x -= stepsize * np.linalg.lstsq(H, g, rcond=None)[0]
        x[x < x_min] = x_min

    return x.reshape(dims)


def ml_obj(x, Tobs, which_prior, lamb, **param):
    n = Tobs.shape[0]
    dims = Tobs.shape[1:]

    C = x.reshape(dims)

    if which_prior == 'entropic':
        gamma = param['gamma']
        Z = 1. / np.power(gamma + C, 2)
        f = np.sum(np.log(Z + 1e-32) 
                   + np.mean(Tobs * C + gamma * Tobs * np.log(Tobs), axis=0))
        g = -2. / (gamma + C) + np.mean(Tobs, axis=0)
        H = np.diag(2. / np.power(gamma + C.reshape((-1,)), 2))
    elif which_prior == 'tsallis':
        q = param['q']
        gamma = param['gamma']
        if q == 0.5:
            tgC = 2 * gamma + C
            eg2tgC = np.exp((gamma ** 2) / tgC)
            erfp1 = sp.erf(gamma / tgC) + 1.
            Z = np.sqrt(np.pi) * gamma * eg2tgC * erfp1 \
                    / np.power(tgC, 3. / 2.) \
                + 1 / tgC
            dlogZ = (-3. * np.sqrt(np.pi) * eg2tgC * erfp1 
                        / (2 * np.power(tgC, 5. / 2.))
                     - (gamma ** 2) / np.power(tgC, 3)
                     - np.sqrt(np.pi) * (gamma ** 3) * eg2tgC * erfp1 
                        / np.power(tgC, 7. / 2.) - 1. / np.power(tgC, 2)) \
                    / Z
        elif q == 2.:
            egC = np.exp(np.power(gamma - C, 2) / (4 * gamma))
            erfp1 = sp.erf((gamma - C) / (2 * np.sqrt(gamma))) + 1.
            Z = np.sqrt(np.pi) * egC * erfp1 / (2 * np.sqrt(gamma))
            dlogZ = 2 * np.sqrt(gamma) * (1. / egC) \
                        * (-np.sqrt(np.pi) * egC * (gamma - C) * erfp1 
                            / (4 * np.power(gamma, 3. / 2.))
                           - 1. / (2 * gamma)) \
                        / (np.sqrt(np.pi) * erfp1)
        else:
            raise NotImplementedError
        f = np.sum(np.log(Z + 1e-32)
                   + np.mean(Tobs * C - (gamma / (1. - q))
                             * (np.power(Tobs, q) - Tobs), axis=0))
        g = dlogZ + np.mean(Tobs, axis=0)
        H = np.eye(C.size)
    elif which_prior == 'normal':
        sigma = param['sigma']
        Z = np.sqrt(2 * np.pi * (sigma ** 2))
        f = np.log(Z) + (1 / (2 * (sigma ** 2))) \
            * np.mean(np.sum(np.sum(np.power(Tobs - C, 2), axis=2), axis=1))
        g = -(1 / (sigma ** 2)) * np.mean((Tobs - C), axis=0)
        H = (1 / (sigma ** 2)) * np.eye(C.size)
    elif which_prior == 'dirichlet':
        logZ = np.sum(sp.loggamma(C + 1.)) - sp.loggamma(np.sum(C + 1.))
        f = logZ - np.sum(C * np.mean(np.log(Tobs), axis=0))
        g = sp.digamma(C + 1.) - sp.digamma(np.sum(C + 1.)) \
            - np.mean(np.log(Tobs), axis=0)
        H = np.diag(sp.polygamma(1, C.reshape((-1,)) + 1.)) \
            - sp.polygamma(1, np.sum(C + 1.))

    f += (lamb / 2) * np.sum(np.power(C, 2))
    g += lamb * C
    H += lamb * np.eye(C.size)

    return f, g.reshape(x.shape), H