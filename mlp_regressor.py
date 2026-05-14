"""
mlp_regressor.py
=================

A small feed-forward MLP wrapped in a numpy-friendly training / inference
API.  Used as the Q-network in the DQN agent.

The API has three pieces:
  • predict(X)            → batched forward pass, returns numpy
  • fit_step(X, Y, W)     → one masked-MSE SGD step
  • save_model / load_model + state_dict / load_state_dict for ckpt I/O

The "W" mask in fit_step is what makes this DQN-friendly.  In Q-learning
we only want to update the Q-value of the action that was actually taken
in each transition — all the other action columns of Y are irrelevant.
We express that by setting W[i, a_i] = 1 and W[i, other] = 0, so the
loss is

      L = sum_i sum_a  W[i, a] * ( Q(s_i)[a] - Y[i, a] )^2

normalised by the total weight count.
"""
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn


class MLPRegression:
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 learning_rate: float = 1e-3,
                 hidden_dim: Sequence[int] = (200, 500, 100)):
        # Build a [Linear, ReLU] stack ending in a bias-free output head.
        # Bias-free output keeps the early Q estimates near zero, which
        # is a friendlier starting point for bootstrapped targets than
        # the arbitrary bias that a Kaiming-init Linear would give us.
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dim:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim, bias=False))
        self.net = nn.Sequential(*layers)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def fit_step(self, X: np.ndarray, Y: np.ndarray, W: np.ndarray) -> None:
        """One masked-MSE gradient step.

        Shapes:
          X : (N, input_dim)
          Y : (N, output_dim)   target Q values
          W : (N, output_dim)   loss mask (0 ignored, 1 contributes)
        """
        self.net.train()
        X_t = torch.as_tensor(X, dtype=torch.float32)
        Y_t = torch.as_tensor(Y, dtype=torch.float32)
        W_t = torch.as_tensor(W, dtype=torch.float32)

        Q_pred = self.net(X_t)
        diff = (Q_pred - Y_t) * W_t
        # normalise by the number of active mask entries so the loss
        # magnitude doesn't depend on how many actions exist
        denom = max(float(W_t.sum().item()), 1.0)
        loss = (diff * diff).sum() / denom

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Forward pass, eval-mode, no grad.  Accepts (N, D) batches."""
        self.net.eval()
        X_t = torch.as_tensor(X, dtype=torch.float32)
        with torch.no_grad():
            out = self.net(X_t)
        return out.numpy()

    # ------------------------------------------------------------------
    # checkpoint I/O — must match the assignment-style call sites
    # ------------------------------------------------------------------
    def save_model(self, path: str) -> None:
        torch.save(self.net.state_dict(), path)

    def load_model(self, path: str) -> None:
        # weights_only=True matches the newer torch.load default and
        # avoids the "unsafe pickle" deprecation noise.
        self.net.load_state_dict(torch.load(path, weights_only=True))

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd) -> None:
        self.net.load_state_dict(sd)
