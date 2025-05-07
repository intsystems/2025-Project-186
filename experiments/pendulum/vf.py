""" Vector field utilities
"""

from typing import Callable

import torch
from torch import nn
from torch import optim

import lightning as pl

from torchdiffeq import odeint_adjoint as odeint


class PendulumVf(nn.Module):
    """ Vector field with known g
    """
    def __init__(self, g: Callable, L: float):
        super().__init__()

        self.g = g
        self.L = L

    def forward(self, t, z):
        """ z = (phi, omega)
        """
        phi, omega = z[:, 0], z[:, 1]
        g = self.g(t)

        return torch.stack([omega, -(g / self.L) * torch.sin(phi)], dim=1)


class PendulumNodeVf(nn.Module):
    """ Vector field with g being approximated by its own vector field
    """
    def __init__(self, g_vf: nn.Module, L: float):
        """_summary_

        Args:
            g_vf (nn.Module): must depends only on t and g
            L (float): _description_
        """
        super().__init__()

        self.g_vf = g_vf
        self.L = L

    def forward(self, t, x):
        """z = (phi, omega, g)
        """
        phi, omega, g = x[:, 0], x[:, 1], x[:, 2]

        return torch.stack([omega, -(g / self.L) * torch.sin(phi), self.g_vf(t, g)], dim=1)


class ODELightningModule(pl.LightningModule):
    def __init__(self, vf: PendulumNodeVf, t: torch.Tensor, g_true: torch.FloatTensor, lr=1e-3):
        """_summary_

        Args:
            vf (nn.Module): _description_
            t (torch.Tensor): time grid, unified for all trajectories
            lr (float, optional): _description_. Defaults to 0.001.
            g_true (torch.FloatTensor): needed to validate predicted g(t)
        """
        super().__init__()
        self.vf = vf
        self.lr = lr
        self.register_buffer("g_true", g_true.clone())
        self.register_buffer('t', t.clone())
        self.criterion = nn.MSELoss()

    def forward(self, x0, t):
        return torch.swapaxes(odeint(self.vf, x0, t, rtol=1e-2), 0, 1)

    def training_step(self, batch, batch_idx):
        x0, x = batch
        pred_traj = self.forward(x0, self.t)

        # we compute loss without g(t)
        loss = self.criterion(pred_traj[..., :2], x[..., :2])

        self.log("train_loss", loss, prog_bar=True, on_step=True)
        return loss
    
    def on_train_epoch_end(self):
        # log uniform convergance norm of g functions
        g0 = self.g_true[0][None, ...]
        g_pred = odeint(self.vf.g_vf, g0, self.t, rtol=1e-2).squeeze()
        self.log("C_norm", torch.max(torch.abs(g_pred - self.g_true)), on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            ),
            'monitor': 'train_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return optimizer
