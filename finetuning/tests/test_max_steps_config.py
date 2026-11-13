"""Regression: standalone training must honor its configured optimizer budget."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import resolve_max_steps


class MaxStepsConfigTest(unittest.TestCase):
    def test_configured_budget_without_cli(self):
        self.assertEqual(resolve_max_steps(SimpleNamespace(max_steps=None, smoke=False),
                                          {"training": {"max_steps": 32}}), 32)

    def test_explicit_cli_has_precedence(self):
        self.assertEqual(resolve_max_steps(SimpleNamespace(max_steps=20, smoke=False),
                                          {"training": {"max_steps": 32}}), 20)

    def test_smoke_default_stays_short(self):
        self.assertEqual(resolve_max_steps(SimpleNamespace(max_steps=None, smoke=True),
                                          {"training": {"max_steps": 100}}), 5)

    def test_explicit_smoke_override(self):
        self.assertEqual(resolve_max_steps(SimpleNamespace(max_steps=2, smoke=True),
                                          {"training": {"max_steps": 100}}), 2)

    def test_missing_budget_retains_epoch_training(self):
        self.assertEqual(resolve_max_steps(SimpleNamespace(max_steps=None, smoke=False),
                                          {"training": {"epochs": 2}}), -1)

    def test_explicit_epoch_mode_overrides_config(self):
        self.assertEqual(resolve_max_steps(SimpleNamespace(max_steps=-1, smoke=False),
                                          {"training": {"max_steps": 32}}), -1)
