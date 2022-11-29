from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import os

def get_imagefiles(image_classes, base_dir):
    images = {
        "train": {"input": [], "output": []},
        "test": {"input": [], "output": []}
    }

    for img_class in image_classes:
        path_to_class = f"{base_dir}/{img_class}"
        for img_set in ["train", "test"]:
            path_to_set = f"{path_to_class}/{img_set}"
            subsets = sorted(os.listdir(path_to_set))
            for img_subset in subsets:
                path_to_subset = f"{path_to_set}/{img_subset}"
                if img_subset == "good":
                    img_files = sorted(os.listdir(path_to_subset))
                    images[img_set]["input"] += [f"{path_to_subset}/{file_id}" for file_id in img_files]
                    images[img_set]["output"] += [f"{path_to_subset}/{file_id}" for file_id in img_files]
                else:
                    img_files = sorted(os.listdir(path_to_subset))
                    images[img_set]["input"] += [f"{path_to_subset}/{file_id}" for file_id in img_files]
                    images[img_set]["output"] += [f"{path_to_class}/ground_truth/{img_subset}/{file_id}" for file_id in img_files]
    
    return images          

class MVTecDataset(Dataset):
    def __init__(self, data_files):
        self.data_files = data_files
        self.transforms = transforms.Compose([
            transforms.Resize(900),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        src_path = self.data_files[idx]
        img = Image.open(src_path).convert('RGB')
        return self.transforms(img)
    
