import tqdm
import numpy as np
import torch
import math
from unet import UNet
import os

EPOCHS = 30
BATCH_SIZE = 64


def noise_scheduler(start=1e-4, end=0.02, steps=1000):
    betas = torch.linspace(start, end, steps)
    alphas = 1. - betas
    alphas_hat = torch.cumprod(alphas, axis=0)
    return betas, alphas, alphas_hat


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, T=1000):
        self.X = X
        self.T = T
        self.betas, self.alphas, self.alphas_hat = noise_scheduler(steps=T)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ix):
        im = torch.from_numpy(self.X[ix])
        t = torch.randint(0, self.T, (1,))
        noise = torch.randn_like(im)
        x = noise * \
            torch.sqrt(1 - self.alphas_hat[t]) + \
            im * torch.sqrt(self.alphas_hat[t])
        return x.unsqueeze(0).float(), noise.unsqueeze(0).float(), t


class SinusoidalPositionEmbeddings(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        half_dim = dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim) * -embeddings)
        self.register_buffer('embeddings', embeddings)

    def forward(self, time):
        embeddings = time[:, None] * self.embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DiffusionModel(torch.nn.Module):
    def __init__(self, t_dim=32):
        super().__init__()
        self.time_embed = SinusoidalPositionEmbeddings(t_dim)
        self.unet = UNet(in_ch=1 + t_dim, n_classes=1)

    def forward(self, x, t):
        B, C, H, W = x.shape
        t = self.time_embed(t)
        t = t[:, :, None, None].repeat(1, 1, H, W)
        x = torch.cat((x, t), dim=1)
        return self.unet(x)


X = np.load("mnist.npz")["X"]
ds = Dataset(X)
dl = torch.utils.data.DataLoader(
    ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
model = DiffusionModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
ckpt_path = None

model.train()
for epoch in range(1, EPOCHS+1):
    losses = []
    pb = tqdm.tqdm(dl)
    for im, noise, t in pb:
        im, noise, t = im.cuda(), noise.cuda(), t.cuda().squeeze(-1)
        output = model(im, t)
        loss = torch.nn.functional.mse_loss(output, noise)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        pb.set_description(
            f'Epoch {epoch}/{EPOCHS} loss {np.mean(losses):.5f}')
    if ckpt_path is not None:
        os.remove(ckpt_path)
    ckpt_path = f"model_{epoch}.ckpt"
    torch.save(model.state_dict(), f"model_{epoch}.ckpt")
