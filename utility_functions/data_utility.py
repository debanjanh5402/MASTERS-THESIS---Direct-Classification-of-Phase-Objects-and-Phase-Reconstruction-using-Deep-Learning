import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import io


class HoloPhaseDataset(Dataset):
    def __init__(self, 
                 root_dir:str, 
                 split:str, 
                 img_shape:tuple=(256, 256),
                 transform=None):
        super().__init__()

        self.img_shape = img_shape
        self.transform = transform
        self.holo_dir = os.path.join(root_dir, split, "holo")
        self.phase_dir = os.path.join(root_dir, split, "phase")

        self.basenames = sorted([f.replace(".png", "") 
                                 for f in os.listdir(self.holo_dir)
                                 if f.endswith(".png")])
        assert len(self.basenames) > 0, "[INFO] NO HOLOGRAMS FOUND."

    def __len__(self):
        return len(self.basenames)
    
    def load_raw(self, path:str):
        phase = np.fromfile(path, dtype=np.float32).reshape(self.img_shape)
        return phase
    
    def __getitem__(self, index:int):
        basename = self.basenames[index]
        holopath = os.path.join(self.holo_dir, basename+".png")
        phasepath = os.path.join(self.phase_dir, basename+".raw")

        holo_tensor = io.read_image(holopath, mode=io.ImageReadMode.GRAY)
        holo_tensor = holo_tensor.div(255.0).to(torch.float32)

        phase_img = self.load_raw(phasepath)
        phase_tensor = torch.from_numpy(np.expand_dims(phase_img, axis=0))
        return holo_tensor, phase_tensor