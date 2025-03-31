import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from glob import glob
from Gmodel import GenerateImage  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SuperResolutionDataset(Dataset):
    def __init__(self, lr_folder, hr_folder, transform=None):
        self.lr_paths = sorted(glob(os.path.join(lr_folder, '*.png')))  
        self.hr_paths = sorted(glob(os.path.join(hr_folder, '*.png')))
        self.transform = transform

    def __len__(self):
        return len(self.lr_paths)

    def __getitem__(self, idx):
        lr_img = cv2.imread(self.lr_paths[idx])
        hr_img = cv2.imread(self.hr_paths[idx])

        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
       
        lr_img = lr_img.astype(np.float32) / 255.0
        hr_img = hr_img.astype(np.float32) / 255.0
    
        lr_tensor = torch.from_numpy(np.transpose(lr_img, (2, 0, 1)))
        hr_tensor = torch.from_numpy(np.transpose(hr_img, (2, 0, 1)))
        return lr_tensor, hr_tensor


lr_folder = 'DIV2K_train_HR'
hr_folder = 'DIV2K_train_LR'


dataset = SuperResolutionDataset(lr_folder, hr_folder)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


model = GenerateImage().to(device)


criterion = nn.L1Loss()  
optimizer = optim.Adam(model.parameters(), lr=1e-4)

#Training 
num_epochs = 100
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for lr_imgs, hr_imgs in dataloader:
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)
        
        optimizer.zero_grad()
        sr_imgs = model(lr_imgs) 
        loss = criterion(sr_imgs, hr_imgs)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * lr_imgs.size(0)
    
    epoch_loss /= len(dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

torch.save(model.state_dict(), 'generator_model.pth')