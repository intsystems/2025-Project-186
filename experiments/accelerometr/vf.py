import pandas as pd
import torch
from torch import nn, optim
import lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint_adjoint as odeint
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from datetime import datetime

class NodeField(nn.Module):
    def __init__(self, input_dim=3):
        super().__init__()
        self.dynamics = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, t, x):
        return self.dynamics(x)

class DynamicSystemLearner(pl.LightningModule):
    def __init__(self, dt=0.1, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.vf = NodeField()
        self.criterion = nn.L1Loss()
        
    def forward(self, x0, t):
        return odeint(self.vf, x0, t, rtol=1e-2)

    def training_step(self, batch, batch_idx):
        x_true = batch
        B, T, D = x_true.shape
        
        t = torch.linspace(0, (T-1)*self.hparams.dt, T, device=self.device)
        x0 = x_true[:, 0, :]
        pred_traj = odeint(self.vf, x0, t, rtol=1e-2).permute(1, 0, 2)

        
        loss = self.criterion(pred_traj, x_true)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def on_after_backward(self):
        if self.global_step % 50 == 0:
            total_norm = 0.0
            for p in self.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            self.log('grad_norm', total_norm, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return {
            "optimizer": optimizer
        }