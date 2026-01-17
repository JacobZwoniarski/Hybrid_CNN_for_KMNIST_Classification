"""Quantum layer (PennyLane + TorchLayer)."""

from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn


def build_qnode_layer(n_qubits: int, n_layers: int) -> qml.qnn.TorchLayer:
    """
    Builds a PennyLane TorchLayer:
    input:  (batch, n_qubits)
    output: (batch, n_qubits)  -> expval(Z) for each qubit
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="best")
    def circuit(inputs, weights):
        # inputs: shape (n_qubits,)
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


class QuantumLayer(nn.Module):
    """
    Thin wrapper to:
    - constrain angles for stability (tanh * pi)
    - keep dtype consistent with the rest of the torch model
    """

    def __init__(self, n_qubits: int, n_layers: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.torch_layer = build_qnode_layer(n_qubits=n_qubits, n_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.n_qubits:
            raise ValueError(f"QuantumLayer expects shape (batch, {self.n_qubits}), got {tuple(x.shape)}")

        x_in = torch.tanh(x) * torch.pi  # constrain angles
        out = self.torch_layer(x_in)

        # PennyLane may return float64; cast to input dtype for safety
        return out.to(dtype=x.dtype)

