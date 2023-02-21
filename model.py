import os
from os import listdir
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import time
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision
from tqdm import tqdm
from torch import device
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# image sizes 600 x 600
# img = Image.open('images/image_2003_1_8_18_27.5_-112.5.png')
# plt.imshow(np.array(img))
# plt.show()
# size = img.size
# print(size)

# "image", "lat", "long", "generationtime_ms", "utc_offset_seconds", "timezone", "elevation", "time", "temperature_2m", "ar", "hour"
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
            self.weather_data.append(row['temperature_2m'])
            self.labels.append(row['ar'])
        # for i, filename in enumerate(os.listdir(folder_path)):
        #     if filename.endswith('.png'):
        #         image_path = os.path.join(folder_path, filename)
        #         image = Image.open(image_path)

        #         image = image.convert('RGB')
        #         self.images.append(image)
        #         label = 0 if i < n else 1
        #         self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        tensor_image = transforms.ToTensor()(image)
        weather_data = self.weather_data[index]
        tensor_data = transforms.ToTensor()(weather_data)
        label = self.labels[index]
        return tensor_image.to(self.device), tensor_data.to(self.device), label

    def collate_fn(self, batch):
        data = [item[0] for item in batch]
        target = [item[1] for item in batch]
        data = torch.stack(data, dim=0)
        target = torch.tensor(target, dtype=torch.float)
        return data, target


my_dataset = MyDataset(pd.read_pickle("train_data.pkl"), device)
my_dataloader = DataLoader(my_dataset, batch_size=64,
                           shuffle=True, num_workers=0, collate_fn=my_dataset.collate_fn)

my_valid_dataset = MyDataset(pd.read_pickle("val_data.pkl"), device)
validloader = DataLoader(my_valid_dataset, batch_size=32,
                         shuffle=True, num_workers=0, collate_fn=my_valid_dataset.collate_fn)


# cnn = torchvision.models.resnet18(weights=None, num_classes=1)
# cnn.fc = nn.Linear(512, 100) #replaces the last fully connected layer with a small linear one cuz its taking up too much memory

class Model(nn.Module):
    def __init__(self, num_classes, num_weather_features):
        super(Model, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, num_classes)

        # Add an additional linear layer for the weather data
        self.weather_fc = nn.Linear(num_weather_features, 64)

    def forward(self, x, weather_data):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = F.avg_pool2d(x, x.shape[2])
        x = x.view(x.size(0), -1)

        # Concatenate the weather data to the output of the convolutional layers
        weather_data = F.relu(self.weather_fc(weather_data))
        x = torch.cat((x, weather_data), dim=1)

        x = self.resnet.fc(x)
        return x

# if torch.cuda.is_available():
#     cnn.cuda()
#     print("yes there is a cuda")
# else:
#     print("no cuda")

# print(cnn)


cnn = Model(num_classes=2, num_weather_features=1).to(device)
optimizer = torch.optim.SGD(cnn.parameters(), lr=0.01)
loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]))
# cnn.train()


best_acc = 0.0

for epoch in range(30):
    print("EPOCH # " + str(epoch))
    loss_sum = 0.0
    total = 0

    for batch in tqdm(my_dataloader):
        optimizer.zero_grad()

        # forward pass!
        inp, weather_data, labels = batch
        inp, weather_data, labels = inp.to(
            device), weather_data.to(device), labels.to(device)

        out = cnn(inp, weather_data)
        # labels = labels.view(-1, 1)  # reshapes labels to fit properly!
        loss = loss_func(out, labels)

        # backward pass!!
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        total += 1
        print('[%d, %5d] loss: %.3f' % (epoch + 1, total, loss_sum))

    # val using validation dataset
    # valid_loss = 0.0
    # cnn.eval()     # optional on non model-specific layers aka ours
    valid_correct = 0
    valid_total = 0
    with torch.no_grad():
        for data in validloader:
            inp, weather_data, labels = data
            inp = inp.to(device)
            weather_data = weather_data.to(device)
            labels = labels.to(device)

            target = cnn(inp, weather_data)
            # labels = labels.view(-1, 1)  # Reshape labels to [batch_size, 1]
            # lo = loss_func(target, labels.float())
            # valid_sum += lo
            # valid_total += 1
            _, predicted = torch.max(target.data, 1)
            valid_total += labels.size(0)
            valid_correct += (predicted == labels).sum().item()

    cnn.train()

    # print(loss_sum / total)
    # valid_loss_avg = valid_sum / valid_total
    # print(valid_loss_avg)
    # if (valid_loss_avg < min_avg_loss):
    #     min_avg_loss = valid_loss_avg
    #     torch.save(cnn, 'ay.pt')
    acc = 100 * valid_correct / valid_total
    print('Accuracy on the validation images: %d %%' % acc)

    if acc > best_acc:
        best_accuracy = acc
        torch.save(cnn.state_dict(), 'best_cnn.pt')
