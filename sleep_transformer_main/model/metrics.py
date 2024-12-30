from dataclasses import dataclass
from typing import Dict, List
import torch
import numpy as np
from datetime import datetime
import pandas as pd
import os

@dataclass
class SampleMetadata:
    dataset: str
    subject: str
    record: str
    timestamp: datetime = datetime.now()

@dataclass
class BasicMetrics:
    loss: float
    accuracy: float
    f1_score: float
    kappa: float
    num_epochs: int

    @classmethod
    def from_tensors(cls, loss: torch.Tensor, accuracy: torch.Tensor, 
                    f1: torch.Tensor, kappa: torch.Tensor, num_epochs: int):
        return cls(
            loss=float(loss.item()),
            accuracy=float(accuracy.item()),
            f1_score=float(f1.item()),
            kappa=float(kappa.item()),
            num_epochs=num_epochs
        )

@dataclass
class RecordMetrics:
    metadata: SampleMetadata
    metrics: BasicMetrics
    predictions: np.ndarray  
    ground_truth: np.ndarray  

    def to_dict(self) -> Dict:
        return {
            'metadata': {
                'dataset': self.metadata.dataset,
                'subject': self.metadata.subject,
                'record': self.metadata.record,
                'timestamp': self.metadata.timestamp.isoformat()
            },
            'metrics': {
                'loss': self.metrics.loss,
                'accuracy': self.metrics.accuracy,
                'f1_score': self.metrics.f1_score,
                'kappa': self.metrics.kappa,
                'num_epochs': self.metrics.num_epochs
            },
            'predictions': self.predictions.tolist(),
            'ground_truth': self.ground_truth.tolist()
        }
    
    def to_row_dict(self) -> Dict:
        return {
            'dataset': self.metadata.dataset,
            'subject': self.metadata.subject,
            'record': self.metadata.record,
            'timestamp': self.metadata.timestamp,
            'loss': self.metrics.loss,
            'accuracy': self.metrics.accuracy,
            'f1_score': self.metrics.f1_score,
            'kappa': self.metrics.kappa,
            'num_epochs': self.metrics.num_epochs,
            'predictions': ','.join(map(str, self.predictions)),
            'ground_truth': ','.join(map(str, self.ground_truth))
        }

@dataclass
class AggregateMetrics:
    """Statistics aggregated over multiple records"""
    avg_loss: float
    avg_accuracy: float
    avg_f1: float
    avg_kappa: float

    @classmethod
    def from_record_list(cls, records: list[RecordMetrics]):
        # Extract all metrics into separate lists for computation
        losses = [r.metrics.loss for r in records]
        accuracies = [r.metrics.accuracy for r in records]
        f1_scores = [r.metrics.f1_score for r in records]
        kappas = [r.metrics.kappa for r in records]
        
        return cls(
            avg_loss=np.mean(losses),
            avg_accuracy=np.mean(accuracies),
            avg_f1=np.mean(f1_scores),
            avg_kappa=np.mean(kappas)
        )

    def to_dict(self) -> Dict:
        return {
            'averages': {
                'loss': self.avg_loss,
                'accuracy': self.avg_accuracy,
                'f1_score': self.avg_f1,
                'kappa': self.avg_kappa
            } 
        }

def aggregate_metrics(records: list[RecordMetrics]) -> Dict[str, AggregateMetrics]:
    """Group metrics by dataset and compute aggregate statistics"""
    dataset_groups = {}
    for record in records:
        dataset = record.metadata.dataset
        if dataset not in dataset_groups:
            dataset_groups[dataset] = []
        dataset_groups[dataset].append(record)
    
    return {
        dataset: AggregateMetrics.from_record_list(group)
        for dataset, group in dataset_groups.items()
    }


class MetricsExporter:
    """Handles exporting metrics data to different formats"""
    
    @staticmethod
    def save_to_csv(records: List[RecordMetrics], output_dir: str) -> str:

        rows = [record.to_row_dict() for record in records]
        df = pd.DataFrame(rows)
        
        df = df.sort_values(['dataset', 'subject', 'record'])
        
        os.makedirs(output_dir, exist_ok=True)
        

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'sleep_metrics_{timestamp}.csv'
        filepath = os.path.join(output_dir, filename)
        
        
        df.to_csv(filepath, index=False)
        
        return filepath


