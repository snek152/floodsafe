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
import sklearn.metrics as metrics
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
            self.weather_data.append(np.array([row['temperature_2m']]))
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
        images = [item[0] for item in batch]
        weather_data = [item[1] for item in batch]
        target = [item[2] for item in batch]
        images = torch.stack(images, dim=0)
        weather_data = torch.stack(weather_data, dim=0)
        target = torch.tensor(target, dtype=torch.float)
        return images, weather_data, target


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
        self.fc1 = nn.Linear(1000 + num_weather_features, 512)
        self.fc2 = nn.Linear(512, num_classes)
        # self.resnet.fc = nn.Linear(512, num_classes)

        # Add an additional linear layer for the weather data
        # self.weather_fc = nn.Linear(num_weather_features, 64)

    def forward(self, x, weather_data):
        x = self.resnet(x)
        # add two dimensions to match the ResNet output tensor
        # weather_data = weather_data.unsqueeze(-1)
        # repeat the weather data tensor to match the ResNet output tensor
        # weather_data = weather_data.repeat(1, 1)
        x = x.view(x.size(0), -1)
        # returning torch.Size([64, 1000]) torch.Size([64, 1]) PLS HELP
        print(x.shape, weather_data.shape)
        x = torch.cat((x, weather_data), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = self.resnet.bn1(x)
        # x = self.resnet.relu(x)
        # x = self.resnet.maxpool(x)

        # x = self.resnet.layer1(x)
        # x = self.resnet.layer2(x)
        # x = self.resnet.layer3(x)
        # x = self.resnet.layer4(x)

        # x = F.avg_pool2d(x, x.shape[2])
        # x = x.view(x.size(0), -1)

        # # Concatenate the weather data to the output of the convolutional layers
        # weather_data = F.relu(self.weather_fc(weather_data))
        # x = torch.cat((x, weather_data), dim=1)

        # x = self.resnet.fc(x)
        # return x

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




network_answers = []
true_answers = []
cnn = cnn.eval()
 
right = 0
total = 0
wrong = 0
 
net_pred = []
true_pred = []
for inp, weather_data, labels in my_dataloader:
 
 with torch.no_grad():
   out = cnn(inp, weather_data)
   
   sigmoid_layer = torch.nn.Sigmoid()
   predicted = sigmoid_layer(out).cpu().numpy() > 0.5
  #  predicted = torch.nn.Sigmoid(out, dim=-1).cpu().numpy() > 0.5
   actual = labels.cpu().numpy() > 0.5
   net_pred.append(predicted)
   true_pred.append(actual)

y_pred = (net_pred)[0]
y_true = (true_pred)[0]
acc = 0

correct = 0
total = 0
true_pos = 0
false_pos = 0
true_neg = 0
false_neg = 0
# print((y_true[0]))
# print(y_pred[0])
# print(len(y_true))
# print(data["test_labels"])

# y_true = 0 < data["test_labels"]
print(len(y_true))
print(len(y_pred))
for i in range(len(y_true)):
  pred = np.asarray(y_pred[i])
  true = np.asarray(y_true[i])
  for j in range(pred.shape[0]):
    if pred[j] == true[j] == True: 
        true_pos += 1
    elif pred[j] == true[j] == False: 
        true_neg += 1
    elif pred[j] == False and true[j] == True: 
        false_neg += 1
    elif pred[j] == True and true[j] == False: 
        false_pos += 1

print(true_pos, false_pos, true_neg, false_neg)
correct = (true_pos + true_neg) / (true_neg + false_neg + true_pos + false_pos)
print("Accuracy: " + str(correct))
precision = true_pos / (true_pos + false_pos)
print("precision: " + str(precision))
recall = true_pos / (true_pos + false_neg)
print("recall: " + str(recall))
f1_score = 2 * precision * recall / (precision + recall)
print("f1_score: " + str(f1_score))

labels = ['atmospheric river present']

print(metrics.classification_report(y_true, y_pred, target_names = labels))