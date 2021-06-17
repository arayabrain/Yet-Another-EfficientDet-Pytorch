import os
import time
import glob

import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from efficientdet.dataset import CocoDataset, Resizer
from train import Params

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
compound_coef = 3
project_name = "fujiseal_stain"
data_path = "./datasets/"
train = True

params = Params(f"projects/{project_name}.yml")

if train:
    dataset_name = params.train_set
else:
    dataset_name = params.val_set

image_dataset = CocoDataset(
    root_dir=os.path.join(data_path, params.project_name),
    set=dataset_name,
    transform=transforms.Compose([
        Resizer(input_sizes[compound_coef])
    ])
)

image_loader = DataLoader(
    image_dataset,
    batch_size=1,
    shuffle=False,
    drop_last=False,
    num_workers=0
)

psum = torch.tensor([0.0, 0.0, 0.0])
psum_sq = torch.tensor([0.0, 0.0, 0.0])

start_time= time.time()
for inputs in image_loader:
    imgs = inputs['img']
    psum += imgs.sum(axis = [0, 1, 2])
    psum_sq += (imgs ** 2).sum(axis = [0, 1, 2])
end_time = time.time()

count = len(image_dataset) * (input_sizes[compound_coef] ** 2)
total_mean = psum / count
total_var = (psum_sq / count) - (total_mean ** 2)
total_std = torch.sqrt(total_var)

print('mean: ' + str(total_mean))
print('std: ' + str(total_std))
print(f"time: {end_time-start_time:0.2f}sec processed: {len(image_dataset)}files")
