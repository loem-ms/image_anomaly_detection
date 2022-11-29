import os
import logging
import argparse

import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np

from utils import get_imagefiles, MVTecDataset
from model import ResNetVAE

def parse_args():
    parser = argparse.ArgumentParser(
        description="Training Settings"
    )
    
    parser.add_argument(
        "--epoch_num",
        type=int,
        help='Number of epochs for model training'
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        help='Learning rate'
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        help='Number of examples in mini batch'
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        help='Random seed'
    )
    
    parser.add_argument(
        "--encoded_dim",
        type=int,
        help='Dimension of laten space'
    )
    
    parser.add_argument(
        "--feedforward_hidden_dim",
        type=int,
        help='Dimension for hidden feed-forward layers'
    )
    
    parser.add_argument(
        "--dropout_rate",
        type=float,
        help='Dropout probability'
    )
    
    parser.add_argument(
        "--log_file",
        help='log filename'
    )
    
    parser.add_argument(
        "--dataset_dir",
        help='Path to dataset directory'
    )
    
    parser.add_argument(
        "--image_class",
        default=["pill"], nargs='+'
    )
    
    parser.add_argument(
        "--checkpoint_dir",
        help='Path to save checkpoints'
    )
    
    args = parser.parse_args()
    
    return args

def train_epoch(model, device, dataloader, loss_func, optimizer):
    model.train()

    train_loss = []
    for src in dataloader:
        src = src.to(device)
        res = model(src)
        loss = loss_func(res, src)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logging.info('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)

def main():
    args = parse_args()
    
    logging.basicConfig(
        filename=args.log_file,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    
    data_dir = args.dataset_dir #"./dataset" 
    image_classes = args.image_class #["pill"]
    data_files = get_imagefiles(image_classes, data_dir)
    train_data = MVTecDataset(data_files["train"]["input"])

    bsz = args.batch_size
    train_dataloader = DataLoader(train_data, batch_size=bsz, shuffle=True)
    
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoded_dim = args.encoded_dim
    ffn_hidden_dim = args.feedforward_hidden_dim
    dropout = args.dropout_rate

    model = ResNetVAE(encoded_dim, ffn_hidden_dim, ffn_hidden_dim, dropout).to(device)
    checkpoint_dir = args.checkpoint_dir
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'model_init.pth'))

    logging.info(model)

    loss_func = nn.MSELoss()
    lr = args.learning_rate

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-05)

    num_epochs = args.epoch_num
    loss_hist = {'train_loss': [], 'val_loss': []}

    model.to(device)
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch+1}/{num_epochs} : ")
        train_loss = train_epoch(model, device, train_dataloader, loss_func, optimizer)
        logging.info(f"Train Loss: {train_loss}")
        loss_hist['train_loss'].append(train_loss)
        if epoch > 0 and epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'model_{epoch}.pth'))

    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'model_last.pth'))
    
if __name__ == "__main__":
    main()