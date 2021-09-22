import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io
from .base_dm import BaseDataModule

class ImageClassificationDataset(Dataset):
    def __init__(self, images, annotations, trans=None):
        super().__init__()
        self.images = images 
        self.annotations = annotations
        self.trans = trans

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        img = io.imread(self.images[ix])
        if self.trans:
            img = self.trans(image=img)['image']
        img_t = torch.from_numpy(img / 255).float()
        if img_t.ndim == 2: # hay algunas imágenes con un solo canal !
            img_t.unsqueeze_(2)
            img_t = img_t.repeat(1, 1, 3)
        # first label, bigger bbox
        label = torch.tensor(self.annotations[ix][0][1])
        return img_t, label


class ImageClassificationDataModule(BaseDataModule):
    def __init__(self, batch_size=32, path='data', num_workers=0, pin_memory=False, train_trans=None, val_trans=None, shuffle=True):
        super().__init__(batch_size, path, num_workers, pin_memory, train_trans, val_trans)
        self.shuffle = shuffle

    def setup(self, stage=None):
        super().setup(stage)
        # datasets
        self.train_ds = ImageClassificationDataset(self.data['train'].image_path.values, self.data['train'].annotations.values, trans = self.train_trans)
        self.val_ds = ImageClassificationDataset(self.data['val'].image_path.values, self.data['val'].annotations.values, trans = self.val_trans)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)
