import pickle

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data import DATA_ROOT
from utils import AverageMeter, mixup_data, mixup_criterion


class Trainer(object):
    """An abstraction for the training/validation procedure, with added features.

    TODO
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion=nn.CrossEntropyLoss(),
        lr=0.001,
        early_stopping_tolerance=5,
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
        self.early_stopping_tolerance = early_stopping_tolerance

    def run(self, epochs, lr_decay=False, mixup=False):
        train_loss_meter = AverageMeter()
        min_val_loss = 1e8
        tolerance = self.early_stopping_tolerance

        self.scheduler_setup(lr_decay)
        for epoch in range(1, epochs + 1):
            losses = train(
                self.model,
                self.device,
                self.train_loader,
                self.optimizer,
                self.criterion,
                epoch,
                mixup,
                train_loss_meter,
            )  # Returns loss per batch
            self.train_loss.extend(losses)

            val_loss, val_accuracy = test(
                self.model, self.device, self.val_loader, self.criterion
            )  # Returns loss/accuracy per epoch
            self.test_loss.append(val_loss)
            self.test_accuracy.append(val_accuracy)

            self.scheduler_step(lr_decay, epoch)

            print(
                f"Epoch {epoch} \t"
                f"train_loss: {train_loss_meter.average:.6f}"
                f"\tval_loss: {val_loss:.4f}\tval_accuracy: {val_accuracy * 100:.2f}%"
            )
            train_loss_meter.reset()

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                tolerance = (
                    self.early_stopping_tolerance
                )  # Reset the tolerance because validation loss improved
            else:
                if epoch > 20:  # Early stopping doesn't start before 20 epochs
                    tolerance -= 1

            if tolerance == 0:
                # Early stopping the training process
                print(
                    f"\nEarly stopping. Val loss did not improve for "
                    f"{self.early_stopping_tolerance} consecutive epochs"
                )
                break

    def scheduler_setup(self, lr_decay):
        if lr_decay:
            self.warmup = optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=self.lambda_warmup
            )
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, [10, 15, 20], 0.1
            )
        else:
            pass

    def scheduler_step(self, lr_decay, epoch):
        if lr_decay:
            self.scheduler.step()
            if epoch <= 5:
                self.warmup.step()
        else:
            pass

    def lambda_warmup(self, epoch):
        return epoch / 5


def train(
    model,
    device,
    train_loader,
    optimizer,
    criterion,
    epoch,
    mixup=False,
    avg_meter=None,
):
    model.train()
    batch_loss = list()
    alpha = 0.2 if mixup else 0
    lam = None  # Required if doing mixup training

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data, target_a, target_b, lam = mixup_data(
            data, target, device, alpha
        )  # Targets here correspond to the pair of examples used to create the mix
        optimizer.zero_grad()
        output = model(data)
        loss = mixup_criterion(criterion, output, target_a, target_b, lam)
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
        if avg_meter is not None:
            avg_meter.update(batch_loss[-1], n=len(data))

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
