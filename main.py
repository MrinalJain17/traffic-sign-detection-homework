import pickle

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data import DATA_ROOT


class Trainer(object):
    def __init__(
        self, model, train_loader, val_loader, criterion=nn.CrossEntropyLoss(), lr=0.001
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.train_loss = list()
        self.test_loss = list()
        self.test_accuracy = list()
        self.criterion = criterion

    def run(self, epochs=50):
        for epoch in range(1, epochs + 1):
            losses = train(
                self.model,
                self.device,
                self.train_loader,
                self.optimizer,
                self.criterion,
                epoch,
            )  # Returns loss per batch
            self.train_loss.extend(losses)

            loss, accuracy = test(
                self.model, self.device, self.val_loader, self.criterion
            )  # Returns loss/accuracy per epoch
            self.test_loss.append(loss)
            self.test_accuracy.append(accuracy)


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    batch_loss = list()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
        if batch_idx % 100 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    batch_loss[-1],
                )
            )

    return batch_loss


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * len(
                data
            )  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * accuracy,
        )
    )

    return test_loss, accuracy


def make_predictions(model, device):
    model.eval()
    dataframe_dict = {"Filename": [], "ClassId": []}
    outfile = "gtsrb_kaggle.csv"

    test_data = torch.load(DATA_ROOT + "/testing/test.pt").to(device)
    file_ids = pickle.load(open(DATA_ROOT + "/testing/file_ids.pkl", "rb"))

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_data)):
            data = data.unsqueeze(0)

            output = model(data)
            pred = output.data.max(1, keepdim=True)[1].item()
            file_id = file_ids[i][0:5]
            dataframe_dict["Filename"].append(file_id)
            dataframe_dict["ClassId"].append(pred)

    df = pd.DataFrame(data=dataframe_dict)
    df.to_csv(outfile, index=False)
    print("Written to csv file {}".format(outfile))
