import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch.nn.functional as F
import timm
import os
from sklearn.model_selection import train_test_split
import torch
from skimage import io
from datetime import datetime


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


def setup(path='./data', test_size=0.2, random_state=42):

    classes = sorted(os.listdir(path))

    print("Generating images and labels ...")
    images, encoded = [], []
    for ix, label in enumerate(classes):
        _images = os.listdir(f'{path}/{label}')
        images += [f'{path}/{label}/{img}' for img in _images]
        encoded += [ix]*len(_images)
    print(f'Number of images: {len(images)}')

    # train / val split
    print("Generating train / val splits ...")
    train_images, val_images, train_labels, val_labels = train_test_split(
        images,
        encoded,
        stratify=encoded,
        test_size=test_size,
        random_state=random_state
    )

    print("Training samples: ", len(train_labels))
    print("Validation samples: ", len(val_labels))

    return classes, train_images, train_labels, val_images, val_labels


class Model(torch.nn.Module):

    def __init__(self, n_outputs=10, use_amp=True):
        super().__init__()
        self.model = timm.create_model(
            'tf_efficientnet_b5', pretrained=True, num_classes=n_outputs)
        self.use_amp = use_amp

    def forward(self, x, log=False):
        if log:
            print(x.shape)
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            return self.model(x)


def step(model, batch, rank):
    x, y = batch
    #x, y = x.to(device), y.to(device)
    y = y.to(rank)
    y_hat = model(x)
    loss = F.cross_entropy(y_hat, y)
    acc = (torch.argmax(y_hat, axis=1) == y).sum().item() / y.size(0)
    return loss, acc


def train_amp(model, dl, optimizer, rank, epochs=10, use_amp=True, prof=None, end=0):
    hist = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for e in range(1, epochs+1):
        start = datetime.now()
        # train
        model.train()
        l, a = [], []
        stop = False
        for batch_idx, batch in enumerate(dl['train']):
            optimizer.zero_grad()

            # AMP
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, acc = step(model, batch, rank)
            scaler.scale(loss).backward()
            # gradient clipping
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            scaler.step(optimizer)
            scaler.update()

            l.append(loss.item())
            a.append(acc)

            # profiling
            if prof:
                if batch_idx >= end:
                    stop = True
                    break
                prof.step()
        hist['loss'].append(np.mean(l))
        hist['acc'].append(np.mean(a))
        if stop:
            break
        # eval
        model.eval()
        l, a = [], []
        with torch.no_grad():
            for batch in dl['val']:
                loss, acc = step(model, batch, rank)
                l.append(loss.item())
                a.append(acc)
        hist['val_loss'].append(np.mean(l))
        hist['val_acc'].append(np.mean(a))
        # log
        if rank == 0:
            log = f'Epoch {e}/{epochs}'
            for k, v in hist.items():
                log += f' {k} {v[-1]:.4f}'
            print(log)
        if rank == 0:
            print("Epoch completed in: " + str(datetime.now() - start))
    return hist


def example(rank, world_size):

    # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # create local model
    model = Model(use_amp=False).to(rank)

    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])

    output = ddp_model(torch.randn(32, 3, 32, 32), log=True)
    print(output.size())

    dist.destroy_process_group()


def train(rank, world_size):

    # create default process group
    dist.init_process_group(
        backend='nccl',
                init_method='env://',
        world_size=world_size,
        rank=rank
    )

    use_amp = True

    # create local model
    model = Model(use_amp=use_amp).to(rank)

    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-3)

    classes, train_images, train_labels, val_images, val_labels = setup(
        './data')

    ds = {
        'train': Dataset(train_images, train_labels),
        'val': Dataset(val_images, val_labels)
    }

    sampler = {
        'train': torch.utils.data.distributed.DistributedSampler(
            ds['train'],
            num_replicas=world_size,
            rank=rank
        ),
        'val': torch.utils.data.distributed.DistributedSampler(
            ds['val'],
            num_replicas=world_size,
            rank=rank
        )
    }

    batch_size = 1024
    dl = {
        'train': torch.utils.data.DataLoader(ds['train'], batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, sampler=sampler['train']),
        'val': torch.utils.data.DataLoader(ds['val'], batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, sampler=sampler['val'])
    }

    hist = train_amp(ddp_model, dl, optimizer, rank, epochs=3, use_amp=use_amp)

    dist.destroy_process_group()


def main():
    world_size = 2
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    #fn = example
    fn = train
    mp.spawn(fn,
             args=(world_size,),
             nprocs=world_size)
    # join=True)


if __name__ == "__main__":
    main()
