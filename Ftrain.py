import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from glob import glob
from Fmodel import InterpNet  # Import the model

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("Using CPU:", device)

###############################################################################
# Dataset for Frame Interpolation Triplets
###############################################################################
class FrameTripletDataset(Dataset):
    """
    Expects a text file where each line has 3 image paths:
      Fₜ₋₁  Fₜ₊₁  Fₜ
    Returns:
      Input: A tensor of shape (7, H, W) where:
        - The first 6 channels are the concatenation of Fₜ₋₁ and Fₜ₊₁,
        - The 7th channel is a constant image filled with the interpolation factor (default 0.5).
      Target: A tensor of shape (3, H, W) corresponding to Fₜ.
    All images are resized to a fixed resolution (256×256) for consistency.
    """
    def __init__(self, triplets_txt, desired_size=(256, 256), interp_factor=0.5):
        super(FrameTripletDataset, self).__init__()
        with open(triplets_txt, 'r') as f:
            lines = f.read().strip().split('\n')
        # Filter out any empty lines
        self.samples = [line.split() for line in lines if line.strip()]
        self.desired_size = desired_size  # (width, height)
        self.interp_factor = interp_factor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path_tminus, path_tplus, path_t = self.samples[idx]

        # Read images in BGR
        img_tminus = cv2.imread(path_tminus)
        img_tplus  = cv2.imread(path_tplus)
        img_t      = cv2.imread(path_t)

        if img_tminus is None or img_tplus is None or img_t is None:
            raise ValueError(f"Error reading images at index {idx}: {self.samples[idx]}")

        # Resize images to the desired resolution (256x256)
        w, h = self.desired_size
        img_tminus = cv2.resize(img_tminus, (w, h), interpolation=cv2.INTER_LINEAR)
        img_tplus  = cv2.resize(img_tplus,  (w, h), interpolation=cv2.INTER_LINEAR)
        img_t      = cv2.resize(img_t,      (w, h), interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1]
        img_tminus = img_tminus.astype(np.float32) / 255.0
        img_tplus  = img_tplus.astype(np.float32)  / 255.0
        img_t      = img_t.astype(np.float32)      / 255.0

        # Convert from HWC to CHW
        tminus_tsr = torch.from_numpy(np.transpose(img_tminus, (2, 0, 1)))  # (3, H, W)
        tplus_tsr  = torch.from_numpy(np.transpose(img_tplus,  (2, 0, 1)))   # (3, H, W)
        t_tsr      = torch.from_numpy(np.transpose(img_t,      (2, 0, 1)))   # (3, H, W)

        # Concatenate Fₜ₋₁ and Fₜ₊₁ along channel dimension => shape (6, H, W)
        inp = torch.cat([tminus_tsr, tplus_tsr], dim=0)

        # Create an extra channel filled with the interpolation factor
        factor_channel = torch.full((1, h, w), self.interp_factor, dtype=torch.float32)

        # Concatenate the extra channel to form input of shape (7, H, W)
        inp = torch.cat([inp, factor_channel], dim=0)

        return inp, t_tsr

###############################################################################
# Training with Early Stopping and GPU Enhancements
###############################################################################
def main():
    # Hyperparameters
    triplets_txt = "trips.txt"  # Each line: Fₜ₋₁ Fₜ₊₁ Fₜ
    batch_size   = 32
    num_epochs   = 100
    lr           = 0.001
    desired_size = (256, 256)  # Resize images to 256x256

    # Early stopping parameters
    patience = 5
    best_loss = float('inf')
    epochs_without_improvement = 0

    # Create dataset and dataloader (enable pin_memory if using GPU)
    dataset = FrameTripletDataset(triplets_txt, desired_size=desired_size, interp_factor=0.5)
    total_samples = len(dataset)
    print("Total triplets found:", total_samples)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Instantiate the interpolation model (7 input channels)
    model = InterpNet().to(device)

    # Define loss and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Optional: Setup AMP for mixed precision training if using GPU
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # Training loop with early stopping
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        processed_samples = 0

        for i, (inp, tgt) in enumerate(dataloader):
            # Move data to GPU with non_blocking=True if possible
            inp, tgt = inp.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            if scaler:  # Using AMP for GPU
                with torch.cuda.amp.autocast():
                    pred = model(inp)  # Shape: (B, 3, H, W)
                    loss = criterion(pred, tgt)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(inp)
                loss = criterion(pred, tgt)
                loss.backward()
                optimizer.step()

            current_batch = inp.size(0)
            processed_samples += current_batch
            epoch_loss += loss.item() * current_batch

            if (i + 1) % 10 == 0 or (i + 1) == len(dataloader):
                print(f"Epoch [{epoch}/{num_epochs}], Step [{processed_samples}/{total_samples}], Loss: {loss.item():.4f}", flush=True)

        epoch_loss /= total_samples
        print(f"Epoch [{epoch}/{num_epochs}] - Avg Loss: {epoch_loss:.4f}", flush=True)

        # Early Stopping Check
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_without_improvement = 0
            # Optionally, save the best model weights
            torch.save(model.state_dict(), "best_interp_model.pth")
            print("Best model updated and saved.", flush=True)
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).", flush=True)

        if epochs_without_improvement >= patience:
            print("Early stopping triggered. Training halted.", flush=True)
            break

    # Save the final trained model weights
    torch.save(model.state_dict(), "interp_model.pth")
    print("Final model saved to interp_model.pth", flush=True)

if __name__ == "__main__":
    main()
