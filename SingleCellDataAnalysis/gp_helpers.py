#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 14:29:14 2025

@author: user
"""

# gp_helpers.py

import GPy
from functools import reduce

def fit_exponential_gp(t, y, lengthscale_prior=(2.0, 3.0)):
    """Fit an Exponential GP model to the data."""
    kernel = GPy.kern.Exponential(input_dim=1)
    kernel.lengthscale.set_prior(GPy.priors.Gamma(*lengthscale_prior))
    model = GPy.models.GPRegression(t, y, kernel)
    model.optimize(messages=False)
    mu, _ = model.predict(t)
    return model, mu


def fit_periodic_gp(t, y, period_bounds=(2.0, 50.0)):
    """Fit a Periodic GP model to the detrended data."""
    kernel = GPy.kern.StdPeriodic(input_dim=1)
    kernel.lengthscale.set_prior(GPy.priors.Gamma(1.0, 1.0))
    kernel.period.constrain_bounded(*period_bounds)
    model = GPy.models.GPRegression(t, y, kernel)
    model.optimize(messages=False, max_iters=500)
    return model


def fit_linear_plus_periodic_gp(t, y, period_bounds=(2.0, 60.0), fix_noise=False):
    """Fit a GP with Linear + Periodic kernel and controlled noise."""
    k_lin = GPy.kern.Linear(input_dim=1)
    k_per = GPy.kern.StdPeriodic(input_dim=1)
    k_per.lengthscale.set_prior(GPy.priors.Gamma(1.0, 1.0))
    k_per.period.constrain_bounded(*period_bounds)

    kernel = k_lin + k_per
    model = GPy.models.GPRegression(t, y, kernel)

    if fix_noise:
        model.Gaussian_noise.constrain_fixed(1e-2)
    else:
        model.Gaussian_noise.set_prior(GPy.priors.Gamma(2.0, 0.1))

    model.optimize(messages=False, max_iters=500)
    return model


def fit_multi_periodic_gp(t, y, period_bounds_list, fix_noise=False):
    """Fit GP with Linear + multiple periodic kernels."""
    k_lin = GPy.kern.Linear(input_dim=1)
    periodic_kernels = []

    for bounds in period_bounds_list:
        k_per = GPy.kern.StdPeriodic(input_dim=1)
        k_per.lengthscale.set_prior(GPy.priors.Gamma(1.0, 1.0))
        k_per.period.constrain_bounded(*bounds)
        periodic_kernels.append(k_per)

    kernel = reduce(lambda x, y: x + y, [k_lin] + periodic_kernels)
    model = GPy.models.GPRegression(t, y, kernel)

    if fix_noise:
        model.Gaussian_noise.constrain_fixed(1e-2)
    else:
        model.Gaussian_noise.set_prior(GPy.priors.Gamma(2.0, 0.1))

    model.optimize(messages=False, max_iters=1000)
    return model
