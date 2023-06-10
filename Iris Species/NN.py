import torch
from torch import nn


class IrisModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_0 = nn.Linear(in_features=4, out_features=8*8, dtype=torch.float64)
        self.layer_1 = nn.Linear(in_features=8*8, out_features=8*8*8, dtype=torch.float64)
        self.layer_2 = nn.Linear(in_features=8*8*8, out_features=8*8*8*8, dtype=torch.float64)
        self.layer_3 = nn.Linear(in_features=8*8*8*8, out_features=3, dtype=torch.float64)
        self.act_fn = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_3(self.act_fn(self.layer_2(self.act_fn(self.layer_1(self.act_fn(self.layer_0(x)))))))


def train_model(model, optimizer, loss_fn, train_data, train_labels, epochs):
    model.train()
    for epoch in range(epochs):
        logits = model(train_data).squeeze()
        loss = loss_fn(logits, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
