import pytorch_lightning as pl
import torch 
import torch.nn.functional as F
from .perceiver import PerceiverIO

class Model(pl.LightningModule):
    def __init__(self, num_latents=256, latent_dim=512, num_blocks=6, num_classes=91, optimizer='Adam', lr=1e-3, scheduler=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.perceiver = PerceiverIO(num_classes=num_classes, max_freq=10, num_freq_bands=6, num_latents=num_latents, latent_dim=latent_dim, input_dim=3, num_blocks=num_blocks)


    def forward(self, x):
        return self.perceiver(x)

    def accuracy(self, y_hat, y):
        return (torch.argmax(y_hat, axis=1) == y).sum() / y.shape[0]

    def step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log('loss', loss)
        self.log('acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        val_loss, val_acc = self.step(batch)
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_acc', val_acc, prog_bar=True)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            y_hat = self(x.to(self.device))
            return torch.softmax(y_hat, axis=1)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams.optimizer)(
            self.parameters(), lr=self.hparams.lr)
        if self.hparams.scheduler:
            schedulers = [
                getattr(torch.optim.lr_scheduler, scheduler)(
                    optimizer, **params)
                for scheduler, params in self.hparams.scheduler.items()
            ]
            return [optimizer], schedulers
        return optimizer

