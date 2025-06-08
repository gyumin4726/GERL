import torch
from torch.utils.data import DataLoader
from config import Config
from models.gerl import GERL
from data.mind_dataset import MINDDataset
from metrics import calculate_metrics
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import random
import os
import time
from datetime import datetime, timedelta
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def train_epoch(model, train_loader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    total_samples = 0
    
    start_time = time.time()
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass
        scores = model(batch)
        labels = batch['labels']
        
        # Calculate loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels.float())
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Update statistics
        total_loss += loss.item() * len(labels)
        total_samples += len(labels)
        
        # Update progress bar
        avg_loss = total_loss / total_samples
        elapsed = time.time() - start_time
        speed = (batch_idx + 1) * train_loader.batch_size / elapsed
        eta = timedelta(seconds=int((len(train_loader) - batch_idx - 1) / speed))
        
        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'speed': f'{speed:.1f} samples/s',
            'eta': str(eta)
        })
    
    return total_loss / total_samples

def evaluate(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            scores = model(batch)
            preds = torch.sigmoid(scores)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = calculate_metrics(all_labels, all_preds)
    return metrics

def main():
    # Setup
    setup_logging()
    config = Config()
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load datasets
    logging.info("\nLoading datasets...")
    train_dataset = MINDDataset(
        data_dir="MINDsmall_train",
        max_title_length=config.max_title_length,
        max_history_length=config.max_history_length,
        num_neighbors=config.max_neighbors
    )
    
    val_dataset = MINDDataset(
        data_dir="MINDsmall_dev",
        max_title_length=config.max_title_length,
        max_history_length=config.max_history_length,
        num_neighbors=config.max_neighbors,
        tokenizer=train_dataset.tokenizer  # Reuse tokenizer
    )
    
    # Update config with dataset statistics
    config.vocab_size = len(train_dataset.tokenizer)
    config.num_users = len(train_dataset.user2idx)
    config.num_news = len(train_dataset.news2idx)
    
    logging.info(f"\nDataset statistics:")
    logging.info(f"Train samples: {len(train_dataset)}")
    logging.info(f"Val samples: {len(val_dataset)}")
    
    # Create data loaders
    logging.info("\nCreating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    logging.info("\nInitializing model...")
    model = GERL(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    logging.info("\nStarting training...")
    best_metric = 0
    start_time = time.time()
    
    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch + 1)
        
        # Evaluate
        logging.info("\nEvaluating...")
        val_metrics = evaluate(model, val_loader, device)
        
        # Log results
        epoch_time = time.time() - epoch_start
        logging.info(f"\nEpoch {epoch+1}/{config.num_epochs} - Time: {timedelta(seconds=int(epoch_time))}")
        logging.info(f"Train Loss: {train_loss:.4f}")
        logging.info(f"Validation Metrics:")
        logging.info(f"AUC: {val_metrics['auc']:.4f}")
        logging.info(f"MRR: {val_metrics['mrr']:.4f}")
        logging.info(f"NDCG@5: {val_metrics['ndcg5']:.4f}")
        logging.info(f"NDCG@10: {val_metrics['ndcg10']:.4f}")
        
        # Save best model
        if val_metrics['auc'] > best_metric:
            best_metric = val_metrics['auc']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_metric': best_metric,
                'config': config.__dict__
            }, 'best_model.pth')
            logging.info("Saved new best model!")
        
        logging.info("-" * 50)
    
    total_time = time.time() - start_time
    logging.info(f"\nTraining completed in {timedelta(seconds=int(total_time))}")
    logging.info(f"Best AUC: {best_metric:.4f}")

if __name__ == "__main__":
    main() 