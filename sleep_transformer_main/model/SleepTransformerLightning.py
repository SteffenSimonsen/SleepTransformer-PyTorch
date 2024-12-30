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
import neptune
from pytorch_lightning.loggers import NeptuneLogger
from dotenv import load_dotenv
from batch_processing import create_sliding_windows
import json
from datetime import datetime
from typing import List
from metrics import SampleMetadata, BasicMetrics, RecordMetrics, MetricsExporter, aggregate_metrics
import numpy as np
torch.set_float32_matmul_precision('high')

class SleepTransformerLightning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # model configuration
        self.config = SleepTransformerConfig()
        # model definition
        self.model = SleepTransformer(self.config)

        #self.model.count_parameters()

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

        self.per_record_metrics = []
        
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


    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch):
        x, y, metadata = batch

        logits = self(x)
        
        if isinstance(logits, tuple):
            logits, _ = logits  # Ignore attention weights for now
        
        raw_loss = self.criterion(logits.view(-1, self.config.num_classes), y.view(-1))

        # normalize the loss to [0,1] range
        loss = 1 - torch.exp(-raw_loss)  
        
        preds = logits.view(-1, self.config.num_classes).argmax(dim=1)
        true = y.view(-1)
        
        acc = self.train_acc(preds, true)
        f1 = self.train_f1(preds, true)
        kappa = self.train_kappa(preds, true)
        
        # Log metrics
        self.log('training_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('training_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('training_f1', f1, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('training_kappa', kappa, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, metadata = batch
        
        batched_record = create_sliding_windows(
            data=x,
            labels=y,
            window_size=21,
            step_size=21
        )
        
        
        all_preds = []
        all_labels = []
        record_losses = []
        
            
        num_windows = len(batched_record.data)
        batch_size = self.config.batch_size
        num_batches = (num_windows + batch_size - 1) // batch_size
        
        
        
        for batch_idx in range(num_batches):

            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_windows)
            
            window_data = batched_record.data[start_idx:end_idx]
            window_labels = batched_record.labels[start_idx:end_idx]
            
            
            logits = self(window_data)
            raw_loss = self.criterion(logits.view(-1, self.config.num_classes), window_labels.view(-1))
            loss = 1 - torch.exp(-raw_loss)
            record_losses.append(loss)
            
            
            preds = logits.view(-1, self.config.num_classes).argmax(dim=1)
            all_preds.append(preds)
            all_labels.append(window_labels.view(-1))
        
        
        if batched_record.has_remainder():
            remaining_data, remaining_labels = batched_record.get_remainder()
            
            logits = self(remaining_data.unsqueeze(0))
            raw_loss = self.criterion(logits.view(-1, self.config.num_classes), remaining_labels.view(-1))
            loss = 1 - torch.exp(-raw_loss)
            record_losses.append(loss)
            
            
            preds = logits.view(-1, self.config.num_classes).argmax(dim=1)
            all_preds.append(preds)
            all_labels.append(remaining_labels.view(-1))
        
        record_preds = torch.cat(all_preds)
        record_labels = torch.cat(all_labels)
        
        record_loss = torch.stack(record_losses).mean()
        
        record_kappa = self.val_kappa(record_preds, record_labels)
        record_acc = self.val_acc(record_preds, record_labels)
        record_f1 = self.val_f1(record_preds, record_labels)
        
        self.log('validation_loss', record_loss, on_epoch=True, sync_dist=True)
        
        return {
            "record_kappa": record_kappa,
            "record_acc": record_acc,
            "record_f1": record_f1,
            "record_loss": record_loss
        }
        
    def on_validation_epoch_end(self):
        
        kappas = self.all_gather(self.val_kappa.compute().clone().detach().to(self.device))
        accuracies = self.all_gather(self.val_acc.compute().clone().detach().to(self.device))
        f1s = self.all_gather(self.val_f1.compute().clone().detach().to(self.device))
        
        self.log('validation_kappa', kappas.mean(),  sync_dist=True)
        self.log('validation_accuracy', accuracies.mean(),  sync_dist=True)
        self.log('validation_f1', f1s.mean(),  sync_dist=True)
        
        self.val_kappa.reset()
        self.val_acc.reset()
        self.val_f1.reset()

        # log learning rate
        if self.trainer is not None:
            current_lr = self.optimizers().param_groups[0]['lr']
            self.log('training_learning_rate', current_lr, sync_dist=True)

    def test_step(self, batch):
        (eeg_data, eog_data), y, metadata = batch

        
        num_eeg = eeg_data.shape[1]
        num_eog = eog_data.shape[1]
        sequence_length = y.shape[1]
        
        # this tensor will store the accumulated softmax probabilities for each channel combination
        channel_votes = torch.zeros(
            num_eeg * num_eog,        # one set of votes per channel combination
            sequence_length,          # number of epochs in the sequence
            self.config.num_classes,  # probability for each sleep stage
            device=self.device
        )
        
        # loop each combination of EEG and EOG channels
        combination_idx = 0
        for eeg_idx in range(num_eeg):
            for eog_idx in range(num_eog):

                # prepare input channel combination
                x_eeg = eeg_data[:, eeg_idx]
                x_eog = eog_data[:, eog_idx]
                
                x_eeg = x_eeg.unsqueeze(-1)
                x_eog = x_eog.unsqueeze(-1)
                x = torch.cat([x_eeg, x_eog], dim=-1)
                
                # batch the record into sliding windows

                batched_record = create_sliding_windows(
                    data=x,
                    labels=y,
                    window_size=21,
                    step_size=1
                )
                
                # accumulator for this channel combination
                epoch_votes = torch.zeros(sequence_length, self.config.num_classes, device=self.device)
                
                # process complete windows in batches
                batch_size = self.config.batch_size
                num_complete_windows = len(batched_record.data)
                
                for i in range(0, num_complete_windows, batch_size):
                    # batch of windows
                    batch_end = min(i + batch_size, num_complete_windows)
                    batch_windows = batched_record.data[i:batch_end]
                    window_indices = batched_record.indices[i:batch_end]
                    
                    # predict
                    logits = self(batch_windows)
                    if isinstance(logits, tuple):
                        logits, _ = logits
                    
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    
                    # add softmax to each epoch that appears windows
                    for window_idx, window_probs in zip(window_indices, probs):
                        epoch_votes[window_idx:window_idx + 21] += window_probs
                
                # handle remaining window
                if batched_record.has_remainder():
                    remaining_data, _ = batched_record.get_remainder()
                    
                    # remaining window - note: add batch dimension with unsqueeze(0)
                    logits = self(remaining_data.unsqueeze(0))
                    if isinstance(logits, tuple):
                        logits, _ = logits
                    
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    
                    # starting index for remaining window
                    window_idx = batched_record.indices[-1] + 1
                    
                    # votes only for the epochs that exist in the remaining window
                    remaining_length = remaining_data.shape[0]  # < 21
                    epoch_votes[window_idx:window_idx + remaining_length] += probs[0, :remaining_length]
                
                #  accumulated votes for this channel combination
                channel_votes[combination_idx] = epoch_votes
                combination_idx += 1
        
        # for each channel combination, get its prediction by taking argmax of accumulated probabilities
        channel_predictions = channel_votes.argmax(dim=-1)  # Shape: [num_combinations, sequence_length]
        
        # majority voting across channel combinations
        preds, _ = torch.mode(channel_predictions, dim=0)  # Shape: [sequence_length]
        
        
        true_labels = y.squeeze()
        
    
        raw_loss = self.criterion(channel_votes.mean(dim=0), true_labels)
        
        # Normalize the loss to [0,1] 
        normalized_loss = 1 - torch.exp(-raw_loss)

        basic_metrics = BasicMetrics.from_tensors(
        loss=normalized_loss,
        accuracy=self.test_acc(preds, true_labels),
        f1=self.test_f1(preds, true_labels),
        kappa=self.test_kappa(preds, true_labels),
        num_epochs=len(true_labels)
        )
        
        record_metrics = RecordMetrics(
        metadata=SampleMetadata(
            dataset=metadata['dataset'][0],
            subject=metadata['subject'][0],
            record=metadata['record'][0]
        ),
        metrics=basic_metrics,
        predictions=preds.cpu().numpy(),
        ground_truth=true_labels.cpu().numpy()
        )
        
        self.test_acc.update(preds, true_labels)
        self.test_f1.update(preds, true_labels)
        self.test_kappa.update(preds, true_labels)

        self.per_record_metrics.append(record_metrics)
        
        self.log('test_loss', normalized_loss, on_epoch=True, prog_bar=True, sync_dist=True)

    
        return normalized_loss
   

    def on_test_epoch_start(self):
        self.test_acc.reset()
        self.test_f1.reset()
        self.test_kappa.reset()
        self.per_record_metrics: List[RecordMetrics] = [] 

    def on_test_epoch_end(self):
        if self.trainer.world_size > 1:
            gathered_metrics = self.all_gather(self.per_record_metrics)
            all_metrics = [item for sublist in gathered_metrics for item in sublist]
        else:
            all_metrics = self.per_record_metrics

        dataset_metrics = aggregate_metrics(all_metrics)
        
        # gather metrics 
        accuracies = self.all_gather(self.test_acc.compute().clone().detach().to(self.device))
        f1s = self.all_gather(self.test_f1.compute().clone().detach().to(self.device))
        kappas = self.all_gather(self.test_kappa.compute().clone().detach().to(self.device))
        
        # log final metrics
        self.log('test_accuracy', accuracies.mean(), sync_dist=True)
        self.log('test_f1', f1s.mean(), sync_dist=True)
        self.log('test_kappa', kappas.mean(), sync_dist=True)

        if self.trainer.is_global_zero:
            neptune_id = self.logger.experiment["sys/id"].fetch() 
            output_dir = os.path.join('metrics', f'run_neptune_{neptune_id}')
            os.makedirs(output_dir, exist_ok=True)
            
            csv_path = MetricsExporter.save_to_csv(all_metrics, output_dir)
            
            results = {
                "neptune_run_id": neptune_id,
                "per_record_metrics": [record.to_dict() for record in all_metrics],
                "dataset_metrics": {
                    dataset: metrics.to_dict() 
                    for dataset, metrics in dataset_metrics.items()
                },
                "overall_metrics": {
                    "accuracy": accuracies.mean().item(),
                    "f1": f1s.mean().item(), 
                    "kappa": kappas.mean().item(),
                    "loss": np.mean([record.metrics.loss for record in all_metrics]),
                    "num_records": len(all_metrics)
                }
            }

            json_path = os.path.join(output_dir, f'overall_metrics_neptune_{neptune_id}.json')
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=4)
                
            
    
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
            mode='max', 
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience,
            min_lr=self.config.scheduler_min_lr,
            threshold=1e-4,
            threshold_mode='rel',
            cooldown=0,
            eps=1e-8
        ),
        'monitor': self.config.scheduler_monitor, # val kappa
        'interval': 'epoch',
        'frequency': 1
    }
        return [optimizer], [scheduler]
    
def setup_callbacks(config: SleepTransformerConfig):
    """
    Create callbacks for model checkpointing and early stopping
    """
    experiment_name = config.experiment_name
    checkpoint_dir = os.path.join('experiments', experiment_name, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callbacks = {
        'kappa': ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f'SleepTransformer-{{epoch:02d}}-{{validation_kappa:.3f}}',
            monitor='validation_kappa',
            save_top_k=3,
            auto_insert_metric_name=True, 
            mode='max',
            verbose=True
        )
    }
    
    # early stopping on validation kappa
    early_stopping = EarlyStopping(
        monitor='validation_kappa',
        min_delta=0.001,
        patience=80,
        verbose=True,
        mode='max'
    )
    
    return checkpoint_callbacks, early_stopping

def setup_data(config: SleepTransformerConfig):

    data_path = config.data_path
    split_path = config.split_path

    #  check if split exists
    if os.path.exists(split_path):
        # reuse existing split for datasets
        print(f"Using existing split: {split_path}")
        train_dataset = SleepDatasetSplit(
            split_file=split_path,
            split_type="train",
            seq_length=config.epoch_seq_len,
            use_virtual_epochs=config.use_virtual_epochs,
            steps_per_epoch=config.steps_per_epoch
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
            seq_length=config.epoch_seq_len,
            use_virtual_epochs=config.use_virtual_epochs,
            steps_per_epoch=config.steps_per_epoch
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
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=6)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=6)

    return train_dataloader, val_dataloader, test_dataloader

def setup_neptune():

    load_dotenv()

    run = neptune.init_run(
        project=os.getenv("NEPTUNE_PROJECT"),       
        api_token=os.getenv("NEPTUNE_API_TOKEN"),       
        source_files=["*.py"],                      
        tags=["sleep-transformer", "test"]
    )
    
    neptune_logger = NeptuneLogger(run=run, prefix="SleepTransformer")
    
    return neptune_logger


def train():
    model = SleepTransformerLightning()
    config = SleepTransformerConfig()

    

    neptune_logger = setup_neptune()
    
    train_dataloader, val_dataloader, test_dataloader = setup_data(config)

    checkpoint_callbacks, early_stopping = setup_callbacks(config)

    # train SleepTransformer model
    trainer = pl.Trainer(
        max_epochs=config.max_epochs, #set number of epochs for training
        callbacks=[early_stopping] + list(checkpoint_callbacks.values()),
        logger=neptune_logger,
        log_every_n_steps=1,
        #multi gpu settings
        accelerator="gpu",
        devices=1,
        strategy="ddp"
    )
    
    trainer.fit(model, train_dataloader, val_dataloader)
    
    best_kappa_model_path = checkpoint_callbacks['kappa'].best_model_path
    best_model = SleepTransformerLightning.load_from_checkpoint(best_kappa_model_path)
    results = trainer.test(best_model, test_dataloader)

    # log test results
    test_metrics = results[0] 
    for metric_name, metric_value in test_metrics.items():
        neptune_logger.experiment[f"test_results/{metric_name}"] = metric_value
    

if __name__ == "__main__":
    # run the model training
    train()


