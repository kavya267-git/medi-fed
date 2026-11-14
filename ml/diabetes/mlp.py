"""A small neural network (multilayer perceptron) for diabetes prediction (PyTorch).

Eight inputs and a few hundred records per hospital suit a small network. On the real PIMA records split across
four hospitals, it scored a higher held-out ROC-AUC under FedAvg than the EHR TabNet did (about 0.80 against 0.77,
three seeds, 30 rounds; the comparison was made after seeing those held-out scores, so it is a design choice, not
a pre-registered result).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ml.common.tasks import DIABETES_TASK
from ml.common.torch_adapter import TorchModelAdapter
from ml.diabetes.features import NUM_DIABETES_FEATURES, standardize_diabetes

HIDDEN_UNITS = 16


class DiabetesMLP(nn.Module):
    def __init__(self, input_dim: int = NUM_DIABETES_FEATURES, hidden: int = HIDDEN_UNITS, num_classes: int = 2):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden)
        self.activation = nn.ReLU()
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.activation(self.hidden(x)))


class DiabetesMLPAdapter(TorchModelAdapter):
    modality = "diabetes"
    architecture = "MLP"
    summary = f"Multilayer perceptron ({NUM_DIABETES_FEATURES} -> {HIDDEN_UNITS} ReLU -> 2) on 8 diabetes measurements."

    def __init__(self):
        super().__init__(DIABETES_TASK)

    def build_model(self, for_init: bool = False) -> nn.Module:
        return DiabetesMLP(num_classes=self.task.num_classes)

    def to_input(self, X: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(standardize_diabetes(X)))
