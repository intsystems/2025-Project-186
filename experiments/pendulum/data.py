""" Data generation utilities
"""
from typing import Callable

import torch
from torchdiffeq import odeint_adjoint as odeint

from vf import PendulumVf


def generate_trajectories(
    g: Callable,
    L: float,
    z0: torch.FloatTensor,
    sigma: torch.FloatTensor,   # diagonal covariance for noise
    t: torch.FloatTensor,
    method="dopri5",
    rtol=1e-3
) -> torch.FloatTensor:
    vf = PendulumVf(g, L)
    trajectories = odeint(
         vf,
         z0,
         t,
         rtol=rtol,
         method=method
    )
    # place batch dimension first
    trajectories = torch.swapaxes(trajectories, 0, 1)
    # add noise
    trajectories += sigma * torch.randn_like(trajectories)

    return trajectories