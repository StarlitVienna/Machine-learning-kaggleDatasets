import torch
from torch import nn
import pandas as pd

df = pd.read_csv("./iris.csv")
features_array = [
        torch.tensor(df["SepalLengthCm"].values),
        torch.tensor(df["SepalWidthCm"].values),
        torch.tensor(df["PetalLengthCm"].values),
        torch.tensor(df["PetalWidthCm"].values)
        ]

#labels = torch.Tensor(df["Species"].values)
labels = torch.tensor(pd.Categorical(df["Species"]).codes)

for i in range(len(features_array)):
    features_array[i] = features_array[i].unsqueeze(dim=1)

features_tensor = torch.cat((features_array[0], features_array[1], features_array[2], features_array[3]), 1)
torch.save({"features": features_tensor, "labels": labels}, "./iris_data.pth")
