import torch 

import os
from sklearn.model_selection import train_test_split

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

classes, train_images, train_labels, val_images, val_labels = setup('./data')

import torch
from skimage import io 

class Dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        img = io.imread(self.images[ix])[...,(3,2,1)]
        img = torch.tensor(img / 4000, dtype=torch.float).clip(0,1).permute(2,0,1)  
        label = torch.tensor(self.labels[ix], dtype=torch.long)        
        return img, label
    
ds = {
    'train': Dataset(train_images, train_labels),
    'val': Dataset(val_images, val_labels)
}

batch_size = 1024
dl = {
    'train': torch.utils.data.DataLoader(ds['train'], batch_size=batch_size, shuffle=True, num_workers=20, pin_memory=True),
    'val': torch.utils.data.DataLoader(ds['val'], batch_size=batch_size, shuffle=False, num_workers=20, pin_memory=True)
}

import torch.nn.functional as F
import timm

class Model(torch.nn.Module):

    def __init__(self, n_outputs=10, use_amp=True):
        super().__init__()
        self.model = timm.create_model('tf_efficientnet_b5', pretrained=True, num_classes=n_outputs)
        self.use_amp = use_amp

    def forward(self, x, log=False):
        if log:
            print(x.shape)
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            return self.model(x)

from tqdm import tqdm
import numpy as np

def step(model, batch, device):
    x, y = batch
    x, y = x.to(device), y.to(device)
    y_hat = model(x)
    loss = F.cross_entropy(y_hat, y)
    acc = (torch.argmax(y_hat, axis=1) == y).sum().item() / y.size(0)
    return loss, acc

def train_amp(model, dl, optimizer, epochs=10, device="cpu", use_amp = True, prof=None, end=0):
    model.to(device)
    hist = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for e in range(1, epochs+1):
        # train
        model.train()
        l, a = [], []
        bar = tqdm(dl['train'])
        stop=False
        for batch_idx, batch in enumerate(bar):
            optimizer.zero_grad()
            
            # AMP
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, acc = step(model, batch, device)
            scaler.scale(loss).backward()
            # gradient clipping 
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            scaler.step(optimizer)
            scaler.update()
            
            l.append(loss.item())
            a.append(acc)
            bar.set_description(f"training... loss {np.mean(l):.4f} acc {np.mean(a):.4f}")
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
        bar = tqdm(dl['val'])
        with torch.no_grad():
            for batch in bar:
                loss, acc = step(model, batch, device)
                l.append(loss.item())
                a.append(acc)
                bar.set_description(f"evluating... loss {np.mean(l):.4f} acc {np.mean(a):.4f}")
        hist['val_loss'].append(np.mean(l))
        hist['val_acc'].append(np.mean(a))
        # log
        log = f'Epoch {e}/{epochs}'
        for k, v in hist.items():
            log += f' {k} {v[-1]:.4f}'
        print(log)
        
    return hist

model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
hist = train_amp(model, dl, optimizer, epochs=3, device="cuda")