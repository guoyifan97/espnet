#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Optimizer module."""

import torch
import itertools


class NoamOpt(object):
    """Optim wrapper that implements rate."""

    def __init__(self, model_size, factor, warmup, optimizer):
        """Construct an NoamOpt object."""
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        """Update parameters and rate."""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Implement `lrate` above."""
        if step is None:
            step = self._step
        return (
            self.factor
            * self.model_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )

    def zero_grad(self):
        """Reset gradient."""
        self.optimizer.zero_grad()

    def state_dict(self):
        """Return state_dict."""
        return {
            "_step": self._step,
            "warmup": self.warmup,
            "factor": self.factor,
            "model_size": self.model_size,
            "_rate": self._rate,
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """Load state_dict."""
        for key, value in state_dict.items():
            if key == "optimizer":
                self.optimizer.load_state_dict(state_dict["optimizer"])
            else:
                setattr(self, key, value)


def get_std_opt(model_params, d_model, warmup, factor):
    """Get standard NoamOpt."""
    base = torch.optim.Adam(model_params, lr=0, betas=(0.9, 0.98), eps=1e-9)
    return NoamOpt(d_model, factor, warmup, base)


class FrontBackMixOpt(object):
    """Optim wrapper that implements rate."""

    def __init__(self, model_size, factor, warmup, frontend_optimizer, backend_optimizer):
        """Construct an NoamOpt object."""
        self.frontend_optimizer = frontend_optimizer
        self.backend_optimizer = backend_optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.frontend_optimizer.param_groups+self.backend_optimizer.param_groups

    def step(self):
        """Update parameters and rate."""
        self._step += 1
        rate = self.rate()
        for p in self.backend_optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.frontend_optimizer.step()
        self.backend_optimizer.step()

    def rate(self, step=None):
        """Implement `lrate` above."""
        if step is None:
            step = self._step
        return (
            self.factor
            * self.model_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )

    def zero_grad(self):
        """Reset gradient."""
        self.frontend_optimizer.zero_grad()
        self.backend_optimizer.zero_grad()

    def state_dict(self):
        """Return state_dict."""
        return {
            "_step": self._step,
            "warmup": self.warmup,
            "factor": self.factor,
            "model_size": self.model_size,
            "_rate": self._rate,
            "frontend_optimizer": self.frontend_optimizer.state_dict(),
            "backend_optimizer": self.backend_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """Load state_dict."""
        for key, value in state_dict.items():
            if key == "frontend_optimizer":
                self.frontend_optimizer.load_state_dict(state_dict["frontend_optimizer"])
            elif key == "backend_optimizer":
                self.backend_optimizer.load_state_dict(state_dict["backend_optimizer"])
            else:
                setattr(self, key, value)

def get_mix_opt(front_params, back_params, d_model, warmup, factor):
    """Get standard NoamOpt."""
    frontend_optimizer = torch.optim.Adadelta(
            front_params, rho=0.95, eps=1e-9, weight_decay=0.0
        )
    backend_optimizer = torch.optim.Adam(back_params, lr=0, betas=(0.9, 0.98), eps=1e-9)
    return FrontBackMixOpt(d_model, factor, warmup, frontend_optimizer, backend_optimizer)

