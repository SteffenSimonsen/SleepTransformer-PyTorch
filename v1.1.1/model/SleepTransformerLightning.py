import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score, CohenKappa
from SleepTransformerModel import SleepTransformer
from config import SleepTransformerConfig
from SleepDatasetSplit import SleepDatasetSplit
from torch.utils.data import DataLoader
import os
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

torch.set_float32_matmul_precision('high')

class SleepTransformerLightning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # model configuration
        self.config = SleepTransformerConfig()
        # model definition
        self.model = SleepTransformer(self.config)

        self.model.count_parameters()

         # Save config as a hyperparameter
        self.save_hyperparameters({
            'learning_rate': self.config.learning_rate,
            'scheduler_patience': self.config.scheduler_patience,
            'scheduler_factor': self.config.scheduler_factor,
            'scheduler_min_lr': self.config.scheduler_min_lr,
            'dropout_rate': self.config.dropout_rate,
            'optimizer_beta1': self.config.optimizer_beta1,
            'optimizer_beta2': self.config.optimizer_beta2,
            'optimizer_eps': self.config.optimizer_eps,
            'num_classes': self.config.num_classes,
            'd_model': self.config.d_model,
            'epoch_num_heads': self.config.epoch_num_heads,
            'epoch_num_layers': self.config.epoch_num_layers,
            'seq_num_heads': self.config.seq_num_heads,
            'seq_num_layers': self.config.seq_num_layers
        })
        
        # ignored label (UNKNOWN)
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

        # Metrics for test
        self.test_acc = Accuracy(task="multiclass", num_classes=self.config.num_classes, ignore_index=self.ignore_label, average="micro")
        self.test_f1 = F1Score(task="multiclass", num_classes=self.config.num_classes, ignore_index=self.ignore_label, average="macro")
        self.test_kappa = CohenKappa(task="multiclass", num_classes=self.config.num_classes, ignore_index=self.ignore_label)

        #store predictions and labels
        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch):
        x, y = batch
        logits = self(x)
        
        if isinstance(logits, tuple):
            logits, _ = logits  # Ignore attention weights for now
        
        raw_loss = self.criterion(logits.view(-1, self.config.num_classes), y.view(-1))

        # normalize the loss to [0,1] range
        loss = 1 - torch.exp(-raw_loss)  # gives us a value between 0 and 1
        
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

        # normalize the loss to [0,1] range
        loss = 1 - torch.exp(-raw_loss)  #  gives us a value between 0 and 1
        
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
        # compute metrics at the end of validation epoch
        self.log('val_acc', self.val_acc.compute(), prog_bar=True)
        self.log('val_f1', self.val_f1.compute(), prog_bar=True) 
        self.log('val_kappa', self.val_kappa.compute(), prog_bar=True)

        # log learning rate
        if self.trainer is not None:
            current_lr = self.optimizers().param_groups[0]['lr']
            self.log('learning_rate', current_lr)


    def test_step(self, batch):
        (eeg_data, eog_data), y = batch

        batch_size = eeg_data.shape[0]
    
        votes = torch.zeros(batch_size, y.shape[1], self.config.num_classes).to(self.device)

        # aggregate votes from all EEG and EOG channels
        for eeg_idx in range(eeg_data.shape[1]):
            for eog_idx in range(eog_data.shape[1]):
                x_eeg = eeg_data[:, eeg_idx]
                x_eog = eog_data[:, eog_idx]
                
                x_eeg = x_eeg.unsqueeze(-1)
                x_eog = x_eog.unsqueeze(-1)
                
                x = torch.cat([x_eeg, x_eog], dim=-1)

                logits = self(x)
                
                if isinstance(logits, tuple):
                    logits, _ = logits
                
                votes += torch.nn.functional.softmax(logits, dim=-1)

        raw_loss = self.criterion(votes.view(-1, self.config.num_classes), y.view(-1))
        loss = 1 - torch.exp(-raw_loss)

        preds = votes.argmax(dim=-1)
        true = y

            
        self.test_step_outputs.append({
            'preds': preds.detach(),
            'labels': true.detach()
        })

        #  batch metrics 
        acc = self.test_acc(preds, true)
        f1 = self.test_f1(preds, true)
        
        # Log batch metrics
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        self.log('test_acc', acc, on_epoch=True, prog_bar=True)
        self.log('test_f1', f1, on_epoch=True, prog_bar=True)

        return loss

    def on_test_epoch_start(self):
        self.test_step_outputs = []


    def on_test_epoch_end(self):
        # cat all predictions and labels
        all_preds = torch.cat([x['preds'].view(-1) for x in self.test_step_outputs])
        all_labels = torch.cat([x['labels'].view(-1) for x in self.test_step_outputs])
        
        #  kappa on entire test set
        kappa = self.test_kappa(all_preds, all_labels)
        self.log('test_kappa', kappa)
        
    
    def configure_optimizers(self):
        # create optimizer
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.optimizer_beta1, self.config.optimizer_beta2),
            eps=self.config.optimizer_eps
        )
        # create scheduler
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max', # Monitor validation kappa
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience,
            min_lr=self.config.scheduler_min_lr,
            verbose=True,
            threshold=1e-4,
            threshold_mode='rel',
            cooldown=0,
            eps=1e-8
        ),
        'monitor': self.config.scheduler_monitor,
        'interval': 'epoch',
        'frequency': 1
    }
        return [optimizer], [scheduler]
    
def setup_callbacks(monitor_metrics):
    """
    Create callbacks for model checkpointing and early stopping
    """

    checkpoint_callbacks = {}
    
    # checkpoint callback for each metric
    for name, config in monitor_metrics.items():
        checkpoint_callbacks[name] = ModelCheckpoint(
            dirpath=f'checkpoints/{name}',
            filename=f'SleepTransformer-{name}-{{epoch:02d}}-{{{config["metric"]}:.3f}}',
            monitor=config['metric'],
            save_top_k=3,
            mode=config['mode'],
            verbose=True
        )
    
    # early stopping on validation kappa
    early_stopping = EarlyStopping(
        monitor='val_kappa',
        min_delta=0.001,
        patience=80,
        verbose=True,
        mode='max'
    )
    
    return checkpoint_callbacks, early_stopping

def setup_data():
    config = SleepTransformerConfig()

    # might define in config later
    data_path = '/home/steff/SleepTransformer/v1.1.0/data'
    split_path = "splits/abc_split.json"

    #  check if split exists
    if os.path.exists(split_path):
        # reuse existing split for datasets
        print(f"Using existing split: {split_path}")
        train_dataset = SleepDatasetSplit(
            split_file=split_path,
            split_type="train",
            seq_length=config.epoch_seq_len
        )
        val_dataset = SleepDatasetSplit(
            split_file=split_path,
            split_type="val", 
            seq_length=config.epoch_seq_len
        )
        test_dataset = SleepDatasetSplit(
            split_file=split_path,
            split_type="test",
            seq_length=config.epoch_seq_len
        )
    else:
        # new split
        print("Creating new split")
        train_dataset = SleepDatasetSplit(
            data_path=data_path,
            split_type="train",
            seq_length=config.epoch_seq_len
        )
        # save split
        split_path = train_dataset.save_split("splits", "abc_split")
        # use new validation and test
        val_dataset = SleepDatasetSplit(
            split_file=split_path,
            split_type="val",
            seq_length=config.epoch_seq_len
        )
        test_dataset = SleepDatasetSplit(
            split_file=split_path,
            split_type="test",
            seq_length=config.epoch_seq_len
        )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # DataLoaders
    batch_size = config.batch_size
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)

    return train_dataloader, val_dataloader, test_dataloader


def test_best_models(trainer, model, test_dataloader, checkpoint_callbacks):
    """
    Test the best models for each metric
    """
    results = {}
    
    for name, callback in checkpoint_callbacks.items():
        print(f"\nTesting best {name} model:")
        print(f"Model path: {callback.best_model_path}")
        best_model = SleepTransformerLightning.load_from_checkpoint(callback.best_model_path)
        test_results = trainer.test(best_model, test_dataloader)
        results[name] = test_results[0]
        
    return results

def train():
    model = SleepTransformerLightning()
    config = SleepTransformerConfig()
    
    train_dataloader, val_dataloader, test_dataloader = setup_data()


    checkpoint_callbacks, early_stopping = setup_callbacks(monitor_metrics=config.monitor_metrics)

    # TensorBoard logger
    logger = pl.loggers.TensorBoardLogger("lightning_logs", name="sleep_transformer")
    
    # train SleepTransformer model
    trainer = pl.Trainer(
        max_epochs=config.max_epochs, #set number of epochs for training
        callbacks=[early_stopping] + list(checkpoint_callbacks.values()),
        logger=logger
    )
    
    trainer.fit(model, train_dataloader, val_dataloader)
    
    # test best models
    results = test_best_models(trainer, model, test_dataloader, checkpoint_callbacks)
    
    print("\nTest Results Summary:")
    for metric_name, metric_results in results.items():
        print(f"\nBest {metric_name} model performance:")
        for metric, value in metric_results.items():
            print(f"{metric}: {value:.4f}")
    

if __name__ == "__main__":
    # run the model training
    train()


