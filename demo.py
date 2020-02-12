def generate_tables(n_table, C):
    """ Sample tables from a Dirichlet distribution with parameters C. """
    T = np.concatenate([np.random.gamma(C, 1.).reshape((1,) + C.shape) 
            for ii in range(n_table)], axis=0)
    Tsum = np.sum(T.reshape((n_table, -1)), axis=1)
    return T / Tsum.reshape((-1,) + (1,) * C.ndim)


if __name__ == '__main__':
    import numpy as np

    from eco import eco_map, eco_noisy_map, eco_max_like

    ###
    ### Generate ground truth tables.
    ###

    n_table = 100   ### Number of tables to infer.
    dims = (4, 4)   ### Dimensions of each table.

    Cgen = np.random.gamma(2., 1., size=dims)       ### Generator parameters.
    Ttrue = generate_tables(n_table, Cgen)          ### Ground truth tables.
    Tavg = np.mean(Ttrue, axis=0, keepdims=True)    ### Population averages.

    ###
    ### Generate observed marginal data.
    ###

    n_samp = 100    ### Number of samples to draw from each marginal.
    u = np.zeros((n_table, dims[0]))    ### Observed marginal sample histograms.
    v = np.zeros((n_table, dims[1]))    ### Observed marginal sample histograms.
    for tt in range(n_table):
        u[tt, :] = np.random.multinomial(n_samp, np.sum(Ttrue[tt], axis=1))
        v[tt, :] = np.random.multinomial(n_samp, np.sum(Ttrue[tt], axis=0))
    u_norm = u / np.sum(u, axis=1, keepdims=True)   ### Normalized to sum to 1.
    v_norm = v / np.sum(v, axis=1, keepdims=True)   ### Normalized to sum to 1.

    ###
    ### Specify models to use.
    ###

    priors = ('entropic', 'tsallis', 'tsallis', 'normal', 'dirichlet')
    params = (
        {'gamma': 1.},
        {'gamma': 1., 'q': 0.5},
        {'gamma': 1., 'q': 2.},
        {'sigma': 0.1},
        {})
    noises = (None, 'multinomial')

    ###
    ### Run MAP inference for each model.
    ###

    print('------------------------------------------')
    print('MEDIAN ABSOLUTE ERROR (%d tables, %d x %d)'
          % (n_table, dims[0], dims[1]))
    print('------------------------------------------')

    for prior, param in zip(priors, params):
        Cml = eco_max_like(Tavg, prior, **param)    ### Estimate prior params.
        for noise in noises:
            ### Estimate tables from observed marginal histograms.
            if noise is None:
                ### Use a model that assumes perfect observations.
                Tinfer = eco_map(u_norm, v_norm, Cml, prior, **param)
            else:
                ### Use a model that assumes imperfect observations.
                Tinfer = eco_noisy_map(u, v, Cml, prior, noise, **param)
            abserr = np.median(np.abs(Tinfer - Ttrue))
            print('%s %s %s: %g' % (prior, str(param), noise, abserr))