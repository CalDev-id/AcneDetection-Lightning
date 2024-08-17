import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets, models
from lightning import LightningModule, LightningDataModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint


data_module = AcneDataModule(data_dir='/datasets')
model = AcneClassifier(num_classes=2)

checkpoint_callback = ModelCheckpoint(
    monitor='val_acc',
    dirpath='checkpoints/',
    filename='acne-classifier-{epoch:02d}-{val_acc:.2f}',
    save_top_k=3,
    mode='max'
)

trainer = Trainer(max_epochs=10, devices=1, accelerator="gpu", callbacks=[checkpoint_callback])
trainer.fit(model, data_module)
