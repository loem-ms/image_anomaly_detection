import os
import logging
import argparse

import torch
from torch.utils.data import DataLoader
import numpy as np

from utils import get_imagefiles, MVTecDataset
from model import ResNetVAE


logging.basicConfig(
    filename='./checkpoints/encode.log',
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Settings for encoding image"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        help='Number of examples in mini batch'
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
        "--checkpoint",
        help='Path to checkpoints'
    )
    
    parser.add_argument(
        "--subset_to_encode",
        default=["train"], nargs='+',
        help='Subset (train/test) to encode'
    )
    
    parser.add_argument(
        "--output_file_pref",
        help='Path to save encoded vector (.csv)'
    )
    
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    
    logging.basicConfig(
        filename=args.log_file,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    
    data_dir = args.dataset_dir
    image_classes = args.image_class
    logging.info(f"Encode data from {data_dir} (class: {image_classes})")
    data_files = get_imagefiles(image_classes, data_dir)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoded_dim = args.encoded_dim
    ffn_hidden_dim = args.feedforward_hidden_dim

    model = ResNetVAE(encoded_dim, ffn_hidden_dim * 2, ffn_hidden_dim).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu')))
    model.eval()
    
    logging.info(f"Using checkpoint from {args.checkpoint}")
    
    for subset in args.subset_to_encode:
        logging.info(f"Encoding data in {subset} set")
        subset_data = MVTecDataset(data_files[subset]["input"])

        bsz = args.batch_size
        dataloader = DataLoader(subset_data, batch_size=bsz, shuffle=True)
        
        save_file = os.path.join(args.output_file_pref, f'{subset}.csv')
        with open(save_file,'w') as csvfile:
            for data in dataloader:
                encoded_data = model.encode(data)
                encoded_data = encoded_data.reshape(encoded_data.size(0), -1)
                np.savetxt(csvfile, encoded_data.detach().numpy(), delimiter=',', comments='')
        logging.info(f"Saved encoded vectors to {save_file}")
        
if __name__ == "__main__":
    main()