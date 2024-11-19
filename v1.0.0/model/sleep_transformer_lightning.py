import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score, CohenKappa
from SleepTransformer import SleepTransformer
from config import SleepTransformerConfig
from dataclass import SleepDatasetHDF5
from torch.utils.data import DataLoader, random_split
import h5py

torch.set_float32_matmul_precision('high')

class SleepTransformerLightning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.config = SleepTransformerConfig()
        self.model = SleepTransformer(self.config)
        
        # Metrics for training
        self.train_acc = Accuracy(task="multiclass", num_classes=self.config.num_classes)
        self.train_f1 = F1Score(task="multiclass", num_classes=self.config.num_classes, average='macro')
        self.train_kappa = CohenKappa(task="multiclass", num_classes=self.config.num_classes)
        
        # Metrics for validation
        self.val_acc = Accuracy(task="multiclass", num_classes=self.config.num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=self.config.num_classes, average='macro')
        self.val_kappa = CohenKappa(task="multiclass", num_classes=self.config.num_classes)

        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        if isinstance(logits, tuple):
            logits, _ = logits  # Ignore attention weights for now
        
        raw_loss = F.cross_entropy(logits.view(-1, self.config.num_classes), y.view(-1))

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
        
        raw_loss = F.cross_entropy(logits.view(-1, self.config.num_classes), y.view(-1))

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
    config = SleepTransformerConfig()
    
    hdf5_path = '/home/steff/SleepTransformer/model/data/all_subjects.h5'
    
    # Get all subject IDs
    with h5py.File(hdf5_path, 'r') as hf:
        all_subjects = [key for key in hf.keys() if key != 'subject_ids']
    
    # Calculate sizes for train and validation sets
    total_subjects = len(all_subjects)
    train_size = int(0.8 * total_subjects)
    val_size = total_subjects - train_size

    train_subjects, val_subjects = random_split(all_subjects, [train_size, val_size])
    
    print(f"Total subjects: {len(all_subjects)}")
    print(f"Training subjects: {len(train_subjects)}")
    print(f"Validation subjects: {len(val_subjects)}")
    
    # Create datasets
    train_dataset = SleepDatasetHDF5(hdf5_path, subject_ids=train_subjects, config=config)
    val_dataset = SleepDatasetHDF5(hdf5_path, subject_ids=val_subjects, config=config)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create DataLoaders
    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8 )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8)

    batch_data, batch_labels = next(iter(train_dataloader))
    
    
    # Create a TensorBoard logger
    logger = pl.loggers.TensorBoardLogger("lightning_logs", name="sleep_transformer")
    
    # Train the model
    trainer = pl.Trainer(max_epochs=20, log_every_n_steps=1, logger=logger)
    trainer.fit(model, train_dataloader, val_dataloader)
    
     # After training, print the best metrics
    print(f"Best validation loss: {trainer.callback_metrics['val_loss']:.4f}")
    print(f"Best validation accuracy: {trainer.callback_metrics['val_acc']:.4f}")
    print(f"Best validation F1 score: {trainer.callback_metrics['val_f1']:.4f}")
    print(f"Best validation Cohen's kappa: {trainer.callback_metrics['val_kappa']:.4f}")

if __name__ == "__main__":
    train_on_real_data()


