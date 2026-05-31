"""test_pcgrad.py — Unit tests for PCGrad gradient projection."""
import pytest
import torch
import torch.nn as nn
from src.training.pcgrad import PCGrad


class SimpleMLP(nn.Module):
    """Tiny shared network for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


class TestPCGrad:
    def _make_optimizer(self):
        model = SimpleMLP()
        adam = torch.optim.Adam(model.parameters(), lr=1e-3)
        pc = PCGrad(adam)
        return model, pc

    def test_step_without_error(self):
        model, optimizer = self._make_optimizer()
        x = torch.randn(8, 4)
        out = model(x)

        loss_a = out[:, 0].mean()
        loss_b = out[:, 1].mean()

        optimizer.zero_grad()
        optimizer.pc_backward([loss_a, loss_b])
        optimizer.step()   # should not raise

    def test_grad_written_to_params(self):
        model, optimizer = self._make_optimizer()
        x = torch.randn(8, 4)
        out = model(x)

        optimizer.zero_grad()
        optimizer.pc_backward([out[:, 0].mean(), out[:, 1].mean()])

        for p in model.parameters():
            assert p.grad is not None, "Gradient should be set after pc_backward"

    def test_conflict_rate_in_range(self):
        model, optimizer = self._make_optimizer()
        x = torch.randn(8, 4)
        out = model(x)

        optimizer.zero_grad()
        optimizer.pc_backward([out[:, 0].mean(), -out[:, 0].mean()])  # forced conflict

        assert 0.0 <= optimizer.conflict_rate <= 1.0

    def test_zero_conflict_rate_aligned_tasks(self):
        """Identical tasks should have zero conflict."""
        model, optimizer = self._make_optimizer()
        x = torch.randn(8, 4)
        out = model(x)

        optimizer.zero_grad()
        loss = out[:, 0].mean()
        # Using the same loss twice → identical gradients → no conflict
        optimizer.pc_backward([loss, loss])

        # Aligned or zero dot product → conflict_rate should be 0
        assert optimizer.conflict_rate == 0.0

    def test_param_groups_accessible(self):
        _, optimizer = self._make_optimizer()
        assert optimizer.param_groups is not None

    def test_state_dict_roundtrip(self):
        _, optimizer = self._make_optimizer()
        sd = optimizer.state_dict()
        optimizer.load_state_dict(sd)
