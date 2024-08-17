import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets, models
from lightning import LightningModule, LightningDataModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from Model import AcneClassifier

# Load the trained model
trained_model = AcneClassifier.load_from_checkpoint('path/to/your/checkpoint.ckpt')
trained_model.eval()

# Dummy inference
image = torch.randn(1, 3, 224, 224)  # Ganti dengan gambar input Anda
output = trained_model(image)
prediction = torch.argmax(output, dim=1)
print('Predicted class:', prediction.item())
