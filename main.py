import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from typing import Tuple


class XORNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return x


def train(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    learn_rate: float = 0.1,
    epochs: int = 10_000,
) -> None:
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learn_rate)

    for epoch in range(epochs):
        outputs = model.forward(x)
        loss = criterion.forward(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f"Epoch {epoch} | Loss {loss.item():.4f}")


def evaluate(model: nn.Module, X: Tensor) -> Tuple[Tensor, Tensor]:
    with torch.no_grad():
        outputs = model.forward(X)
        predictions = (outputs > 0.5).float()
        return outputs, predictions


def main() -> None:
    X = torch.tensor(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=torch.float32,
    )

    y = torch.tensor(
        [
            [0],
            [1],
            [1],
            [0],
        ],
        dtype=torch.float32,
    )

    model = XORNet()
    train(model, X, y)
    outputs, predictions = evaluate(model, X)

    print("\nFinal Raw Outputs:")
    print(outputs)
    print("Rounded Predictions:")
    print(predictions)


if __name__ == "__main__":
    main()
