import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchmetrics import Accuracy, F1Score, CohenKappa
from SleepTransformerModel import SleepTransformer
from config import SleepTransformerConfig
from SleepDatasetSplit import SleepDatasetSplit
from torch.utils.data import DataLoader
import os

torch.set_float32_matmul_precision('high')

class SleepTransformerLightning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.config = SleepTransformerConfig()
        self.model = SleepTransformer(self.config)
        
        # Define ignored label
        self.ignore_label = 5
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_label)
        
        # Metrics for training
        self.train_acc = Accuracy(task="multiclass", num_classes=self.config.num_classes, ignore_index=self.ignore_label, average="micro")
        self.train_f1 = F1Score(task="multiclass", num_classes=self.config.num_classes, ignore_index=self.ignore_label, average="macro")
        self.train_kappa = CohenKappa(task="multiclass", num_classes=self.config.num_classes, ignore_index=self.ignore_label)
        
        # Metrics for validation
        self.val_acc = Accuracy(task="multiclass", num_classes=self.config.num_classes, ignore_index=self.ignore_label, average="micro")
        self.val_f1 = F1Score(task="multiclass", num_classes=self.config.num_classes, ignore_index=self.ignore_label, average="macro")
        self.val_kappa = CohenKappa(task="multiclass", num_classes=self.config.num_classes, ignore_index=self.ignore_label)

        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch):
        x, y = batch
        logits = self(x)
        
        if isinstance(logits, tuple):
            logits, _ = logits  # Ignore attention weights for now
        
        raw_loss = self.criterion(logits.view(-1, self.config.num_classes), y.view(-1))

        # Normalize the loss to [0,1] range
        loss = 1 - torch.exp(-raw_loss)  # This gives us a value between 0 and 1
        
        # Calculate metrics
        preds = logits.view(-1, self.config.num_classes).argmax(dim=1)
        true = y.view(-1)
        
        acc = self.train_acc(preds, true)
        f1 = self.train_f1(preds, true)
        kappa = self.train_kappa(preds, true)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_f1', f1, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_kappa', kappa, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        if isinstance(logits, tuple):
            logits, _ = logits
        
        raw_loss = self.criterion(logits.view(-1, self.config.num_classes), y.view(-1))

        # Normalize the loss to [0,1] range
        loss = 1 - torch.exp(-raw_loss)  # This gives us a value between 0 and 1
        
        # Calculate metrics
        preds = logits.view(-1, self.config.num_classes).argmax(dim=1)
        true = y.view(-1)
        
        acc = self.val_acc(preds, true)
        f1 = self.val_f1(preds, true)
        kappa = self.val_kappa(preds, true)
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        
        return loss
    
    def on_validation_epoch_end(self):
        # Compute metrics at the end of validation epoch
        self.log('val_acc', self.val_acc.compute(), prog_bar=True)
        self.log('val_f1', self.val_f1.compute(), prog_bar=True) 
        self.log('val_kappa', self.val_kappa.compute(), prog_bar=True)

    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999),
            eps=1e-7
        )

def train_on_real_data():
    model = SleepTransformerLightning()
    
    data_path = '/home/steff/SleepTransformer/v1.1.0/data'
    split_path = "splits/abc_split.json"

    # First check if split exists
    if os.path.exists(split_path):
        # Reuse existing split for both datasets

        print(f"Using existing split: {split_path}")
        train_dataset = SleepDatasetSplit(
            split_file=split_path,
            split_type="train",
            seq_length=21
        )
        val_dataset = SleepDatasetSplit(
            split_file=split_path,
            split_type="val",
            seq_length=21
        )
    else:
        # Create new split
        print("Creating new split")
        train_dataset = SleepDatasetSplit(
            data_path=data_path,
            split_type="train",
            seq_length=21
        )
        # Save it
        split_path = train_dataset.save_split("splits", "abc_split")
        # Use for validation
        val_dataset = SleepDatasetSplit(
            split_file=split_path,
            split_type="val",
            seq_length=21
        )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create DataLoaders
    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8)


    # Create a TensorBoard logger
    logger = pl.loggers.TensorBoardLogger("lightning_logs", name="sleep_transformer")
    
    # Train the model
    trainer = pl.Trainer(max_epochs=100, log_every_n_steps=1, logger=logger)
    trainer.fit(model, train_dataloader, val_dataloader)
    
     # After training, print the best metrics
    print(f"Best validation loss: {trainer.callback_metrics['val_loss']:.4f}")
    print(f"Best validation accuracy: {trainer.callback_metrics['val_acc']:.4f}")
    print(f"Best validation F1 score: {trainer.callback_metrics['val_f1']:.4f}")
    print(f"Best validation Cohen's kappa: {trainer.callback_metrics['val_kappa']:.4f}")

if __name__ == "__main__":
    train_on_real_data()


