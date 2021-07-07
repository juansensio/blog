import torch.nn.functional as F
import timm
import os
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import torch
from skimage import io
from torch.utils.data import DataLoader


class Dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        img = io.imread(self.images[ix])[..., (3, 2, 1)]
        img = torch.tensor(
            img / 4000, dtype=torch.float).clip(0, 1).permute(2, 0, 1)
        label = torch.tensor(self.labels[ix], dtype=torch.long)
        return img, label


class DataModule(pl.LightningDataModule):

    def __init__(self, path='./data', batch_size=1024, num_workers=20, test_size=0.2, random_state=42):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_size = test_size
        self.random_state = random_state

    def setup(self, stage=None):

        self.classes = sorted(os.listdir(self.path))

        print("Generating images and labels ...")
        images, encoded = [], []
        for ix, label in enumerate(self.classes):
            _images = os.listdir(f'{self.path}/{label}')
            images += [f'{self.path}/{label}/{img}' for img in _images]
            encoded += [ix]*len(_images)
        print(f'Number of images: {len(images)}')

        # train / val split
        print("Generating train / val splits ...")
        train_images, val_images, train_labels, val_labels = train_test_split(
            images,
            encoded,
            stratify=encoded,
            test_size=self.test_size,
            random_state=self.random_state
        )

        print("Training samples: ", len(train_labels))
        print("Validation samples: ", len(val_labels))

        self.train_ds = Dataset(train_images, train_labels)
        self.val_ds = Dataset(val_images, val_labels)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )


class Model(pl.LightningModule):

    def __init__(self, n_outputs=10, prof=None):
        super().__init__()
        self.model = timm.create_model(
            'tf_efficientnet_b5', pretrained=True, num_classes=n_outputs)
        self.prof = prof

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        self.log('loss', loss)
        self.log('acc', acc, prog_bar=True)
        if self.prof is not None:
            self.prof.step()
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def shared_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (torch.argmax(y_hat, axis=1) == y).sum().item() / y.size(0)
        return loss, acc

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


model = Model()
dm = DataModule()
trainer = pl.Trainer(gpus=2, accelerator='ddp', precision=16, max_epochs=3)
trainer.fit(model, dm)
