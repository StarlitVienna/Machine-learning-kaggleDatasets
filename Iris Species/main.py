import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from NN import *
# 0 = Iris-setosa
# 1 = Iris-versicolor
# 2 = Iris-virginica

seed = 8


data = torch.load("./iris_data.pth")
train_features, test_features, train_labels, test_labels = train_test_split(data["features"], data["labels"].long(), test_size = 0.1, random_state=seed)

torch.manual_seed(seed)
model = IrisModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()

epochs = 1000
#epochs = 1
train_model(model, optimizer, loss_fn, train_features, train_labels, epochs)

model.eval()
with torch.inference_mode():
    preds = torch.argmax(nn.Softmax(dim=1)(model(test_features)), dim=1)

print(f"Predss --> {preds}")
print(f"Labels --> {test_labels}")

test_accuracy = torch.sum(torch.eq(preds, test_labels))/len(preds)
print(f"Test accuracy --> {test_accuracy*100:.2f}%")

with torch.inference_mode():
    overral_preds = torch.argmax(nn.Softmax(dim=1)(model(data["features"])), dim=1)

overral_accuracy = torch.sum(torch.eq(overral_preds, data["labels"]))/len(overral_preds)
print(f"Overral accuracy --> {overral_accuracy*100:.2f}%")
