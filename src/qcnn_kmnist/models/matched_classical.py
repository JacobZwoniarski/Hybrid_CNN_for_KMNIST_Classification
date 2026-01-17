from __future__ import annotations

import torch
import torch.nn as nn


class MatchedClassicalCNN(nn.Module):
    """
    Architecture matched to HybridQCNN:
      conv features -> pre_quantum (to n_qubits) -> classical_middle -> post_head -> logits

    This is the "baseline" for fair comparison: same shape flow as HybridQCNN,
    only replacing the quantum layer with a classical block.
    """

    def __init__(self, num_classes: int = 10, n_qubits: int = 6) -> None:
        super().__init__()
        self.n_qubits = n_qubits

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14 -> 7
        )

        self.pre_quantum = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, n_qubits),
        )

        # Classical replacement: (B, n_qubits) -> (B, n_qubits)
        self.classical_middle = nn.Sequential(
            nn.Linear(n_qubits, n_qubits),
            nn.Tanh(),
        )

        self.post_head = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pre_quantum(x)
        x = self.classical_middle(x)
        x = self.post_head(x)
        return x
