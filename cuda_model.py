import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
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
import gc
# set PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# image sizes 600 x 600
# img = Image.open('images/image_2003_1_8_18_27.5_-112.5.png')
# plt.imshow(np.array(img))
# plt.show()
# size = img.size
# print(size)

class MyDataset(Dataset):
    def __init__(self, folder_path, n, device):
        self.folder_path = folder_path
        self.images = []
        self.labels = []
        self.device = device
        for i, filename in enumerate(os.listdir(folder_path)):
            if filename.endswith('.png'):
                image_path = os.path.join(folder_path, filename)
                image = Image.open(image_path)

                image = image.convert('RGB')
                self.images.append(image)
                label = 0 if i < n else 1
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        tensor = transforms.ToTensor()(image)
        label = self.labels[index]
        return tensor.to(self.device), label

    def collate_fn(self, batch):
        data = [item[0] for item in batch]
        target = [item[1] for item in batch]
        data = torch.stack(data, dim=0)
        target = torch.tensor(target, dtype=torch.float16)
        return data, target



my_dataset = MyDataset('/kaggle/input/water-vapor-images/dataset/train_images', 924, device)
my_dataloader = DataLoader(my_dataset, batch_size=8, shuffle=False, num_workers = 0, collate_fn=my_dataset.collate_fn)

my_valid_dataset = MyDataset('/kaggle/input/water-vapor-images/dataset/val_images', 116, device)
validloader = DataLoader(my_valid_dataset, batch_size=8, shuffle=False, num_workers = 0, collate_fn=my_valid_dataset.collate_fn)


cnn = torchvision.models.resnet18(weights= None, num_classes = 1).to(device)
# cnn.fc = nn.Linear(cnn.fc.in_features, 1).to(device)
# cnn.fc = nn.Linear(512, 100) #replaces the last fully connected layer with a small linear one cuz its taking up too much memory


gc.collect()
torch.cuda.empty_cache()

# print(cnn)



optimizer = torch.optim.SGD(cnn.parameters(), lr=0.01)
loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]))


cnn.train()

min_avg_loss = 1000000000000

for epoch in range(50):
    gc.collect()
    torch.cuda.empty_cache()
    print("EPOCH # " + str(epoch))
    loss_sum = 0
    total = 0
    valid_sum = 0
    valid_total = 0
    for batch in tqdm(my_dataloader):
        gc.collect()
        torch.cuda.empty_cache()
        optimizer.zero_grad()

        # forward pass!
        inp, labels = batch
        
        #(input, labels) = lazy(input, labels, batch=0)
        inp, labels = inp.to('cuda'), labels.to('cuda')
        cnn = cnn.to('cuda')
        
        
        out = cnn(inp).to('cuda')
        
        
        labels = labels.view(-1, 1)  # reshapes labels to fit properly!
        
        
        out = out.to('cpu')
        labels = labels.to('cpu')
        
        
        loss = loss_func(out, labels.float())
        

        # backward pass!!
       
        loss.backward()
        optimizer.step()

        
        loss_sum += loss.detach().item()
        total += 1

    #val using validation dataset
    valid_loss = 0.0
   
    gc.collect()
    torch.cuda.empty_cache()
    cnn.eval()     # optional on non model-specific layers aka ours
    valid_sum = 0
    valid_total = 0
    with torch.no_grad():
        for data, labels in validloader:
            gc.collect()
            torch.cuda.empty_cache()
        
            data.cuda()
            labels.cuda()
        
            gc.collect()
            torch.cuda.empty_cache()
        
            target = cnn(data)
            labels = labels.view(-1, 1).to(device)  # Reshape labels to [batch_size, 1]
            target = target.to('cpu')
            labels = labels.to('cpu')
            lo = loss_func(target, labels.float())
            valid_sum += lo
            valid_total += 1
    cnn.train()

    gc.collect()
    torch.cuda.empty_cache()
    print(loss_sum / total)
    valid_loss_avg = valid_sum / valid_total
    print(valid_loss_avg)
    if (valid_loss_avg < min_avg_loss):
        min_avg_loss = valid_loss_avg
        torch.save(cnn, 'ay.pt')