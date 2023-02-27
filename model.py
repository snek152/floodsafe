import gc
import pandas as pd
from torch import device
from torch.backends import mps
from tqdm import tqdm
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import sklearn.metrics as metrics
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import time
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch
from PIL import Image
from os import listdir
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


device = torch.device("cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu")
print(device.type)

class MyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, device):
        self.images = []
        self.weather_data = []
        self.labels = []
        self.device = device
        for i, row in df.iterrows():
            image = Image.open(row['image'])
            image = image.convert('RGB')
            self.images.append(image)
            self.weather_data.append(np.array(
                [row['temperature'], row["humidity"], row["dewpoint"], row["precipitation"]]))
            self.labels.append(row['ar'])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        tensor_image = transforms.ToTensor()(image)
        weather_data = self.weather_data[index]
        label = self.labels[index]
        return tensor_image.to(self.device), weather_data, label

    def collate_fn(self, batch):
        images = [item[0] for item in batch]
        weather_data = [item[1] for item in batch]
        target = [item[2] for item in batch]
        images = torch.stack(images, dim=0)
        weather_data = torch.tensor(
            np.array(weather_data), dtype=torch.float)
        target = torch.tensor(target, dtype=torch.float)
        return images, weather_data, target


my_dataset = MyDataset(pd.read_pickle("train_data.pkl"), device)
my_dataloader = DataLoader(my_dataset, batch_size=8,
                           shuffle=False, num_workers=0, collate_fn=my_dataset.collate_fn)

my_valid_dataset = MyDataset(pd.read_pickle("val_data.pkl"), device)
validloader = DataLoader(my_valid_dataset, batch_size=8,
                         shuffle=False, num_workers=0, collate_fn=my_valid_dataset.collate_fn)


class Model(nn.Module):
    def __init__(self, num_classes, num_weather_features):
        super(Model, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.fc1 = nn.Linear(1000 + num_weather_features, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x, weather_data):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, weather_data), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def train_model():
    cnn = Model(num_classes=2, num_weather_features=4).to(device)
    optimizer = torch.optim.SGD(cnn.parameters(), lr=0.01)
    loss_func = nn.CrossEntropyLoss()
    cnn.train()

    gc.collect()
    torch.cuda.empty_cache()

    best_acc = 0.0
    train_losses = []
    val_losses = []

    for epoch in range(30):
        gc.collect()
        torch.cuda.empty_cache()
        print("EPOCH # " + str(epoch+1))

        total = 0
        t_losses = []
        v_losses = []

        for batch in my_dataloader:
            gc.collect()
            torch.cuda.empty_cache()
            optimizer.zero_grad()

            inp, weather_data, labels = batch
            inp, weather_data, labels = inp.to(
                device), weather_data.to(device), labels.to(device)

            out = cnn(inp, weather_data)
            labels = labels.long()
            loss = loss_func(out, labels)

            loss.backward()
            optimizer.step()

            total += 1
            t_losses.append(loss.item())
            print('[%d, %5d] loss: %.3f' % (epoch + 1, total, loss.item()))

        train_losses.append(sum(t_losses)/len(t_losses))

        gc.collect()
        torch.cuda.empty_cache()
        cnn.eval()     # optional on non model-specific layers aka ours
        valid_correct = 0
        valid_total = 0

        with torch.no_grad():
            for data in validloader:
                gc.collect()
                torch.cuda.empty_cache()
                inp, weather_data, labels = data
                inp = inp.to(device)
                weather_data = weather_data.to(device)
                labels = labels.to(device)

                target = cnn(inp, weather_data)
                lo = loss_func(target, labels.long())
                v_losses.append(lo.item())

                _, predicted = torch.max(target.data, 1)
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()
        val_losses.append(sum(v_losses)/len(v_losses))
        cnn.train()
        gc.collect()
        torch.cuda.empty_cache()


        acc = 100 * valid_correct / valid_total
        print('Accuracy on the validation images: %d %%' % acc)

        if acc > best_acc:
            best_accuracy = acc
            torch.save(cnn.state_dict(), 'cnn.pt')

    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'g', label='Avg training loss')
    plt.plot(epochs, val_losses, 'b', label='Avg validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


@torch.no_grad()
def test_model():
    cnn = Model(num_classes=2, num_weather_features=4).to(device)
    cnn.load_state_dict(torch.load('best_cnn.pt'))
    cnn.eval()
    loss_func = nn.CrossEntropyLoss()

    test_dataset = MyDataset(pd.read_pickle("test_data.pkl"), device)
    loader = DataLoader(test_dataset, batch_size=1,
                        shuffle=False, num_workers=0, collate_fn=test_dataset.collate_fn)

    loss = 0
    correct = 0
    tp, tn, fp, fn = 0, 0, 0, 0

    for inp, weather_data, labels in loader:
        inp, weather_data, labels = inp.to(
            device), weather_data.to(device), labels.to(device)
        out = cnn(inp, weather_data)
        loss += loss_func(out, labels.long()).item()
        pred = out.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        tp += ((pred == 1) & (labels == 1)).sum().item()
        tn += ((pred == 0) & (labels == 0)).sum().item()
        fp += ((pred == 1) & (labels == 0)).sum().item()
        fn += ((pred == 0) & (labels == 1)).sum().item()


    loss /= len(test_dataset)
    accuracy = 100. * correct / len(test_dataset)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        loss, correct, len(test_dataset), accuracy))
    
    print('Precision: {:.4f}, Recall: {:.4f}, F1 score: {:.4f}'.format(
        precision, recall, f1_score))


if __name__ == "__main__":
    switcher = {"train": train_model, "test": test_model}
    print(switcher.keys())
    switcher[input("Enter command "
                   + str(list(switcher.keys()))
                   .replace("[", "(").replace("]", ")")+": ")]()
