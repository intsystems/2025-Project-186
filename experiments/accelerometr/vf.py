import pandas as pd
import torch
from torch import nn, optim
import lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint_adjoint as odeint

class NodeField(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.hidden_dynamics = nn.Sequential(
            nn.Linear(1, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.observed_dynamics = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, input_dim + hidden_dim),
            nn.ReLU(),
            nn.Linear(input_dim + hidden_dim, input_dim)
        )

    def forward(self, t, xw):
        x = xw[..., :self.input_dim]
        w = xw[..., self.input_dim:]

        t_ = t.view(1, -1, 1).expand(x.size(0), -1, -1)
        dw_dt = self.hidden_dynamics(t_)
        xw_concat = torch.cat([x, w], dim=-1)
        dx_dt = self.observed_dynamics(xw_concat)
        
        return torch.cat([dx_dt, dw_dt], dim=-1)


class DynamicSystemLearner(pl.LightningModule):
    def __init__(self, dt=0.1, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.vf = NodeField()
        self.criterion = nn.MSELoss()

    def forward(self, xw0, t):
        return odeint(self.vf, xw0, t, rtol=1e-3).squeeze(0)

    def training_step(self, batch, batch_idx):
        xw_true = batch
        
        num_steps = xw_true.size(0)
        t = torch.linspace(0, num_steps*self.hparams.dt, num_steps, device=self.device)
        
        xw0 = xw_true[0].unsqueeze(0)
        pred_traj = self(xw0, t).squeeze(1)
        loss = self.criterion(pred_traj, xw_true)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.hparams.lr)