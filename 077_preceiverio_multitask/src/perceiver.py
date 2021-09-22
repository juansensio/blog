# https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
# https://github.com/esceptico/perceiver-io/blob/master/src/perceiver/attention.py

import math
import torch

class MultiHeadAttention(torch.nn.Module):

    def __init__(self, kv_dim, q_dim, n_heads=1, attn_pdrop=0., resid_pdrop=0.):
        super().__init__()
        self.n_embd = q_dim
        self.n_heads = n_heads
        assert self.n_embd % self.n_heads == 0
        # key, query, value projections
        self.key = torch.nn.Linear(kv_dim, self.n_embd)
        self.query = torch.nn.Linear(q_dim, self.n_embd)
        self.value = torch.nn.Linear(kv_dim, self.n_embd)
        # regularization
        self.attn_drop = torch.nn.Dropout(attn_pdrop)
        self.resid_drop = torch.nn.Dropout(resid_pdrop)
        # output projection
        self.proj = torch.nn.Linear(self.n_embd, q_dim)

    def forward(self, kv, q, mask = None): 
        B, M, C = kv.size()
        B, N, D = q.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(kv).view(B, M, self.n_heads, D // self.n_heads).transpose(1, 2) # (B, nh, M, hs)
        q = self.query(q).view(B, N, self.n_heads, D // self.n_heads).transpose(1, 2) # (B, nh, N, hs)
        v = self.value(kv).view(B, M, self.n_heads, D // self.n_heads).transpose(1, 2) # (B, nh, M, hs)
        # attention (B, nh, N, hs) x (B, nh, hs, M) -> (B, nh, N, M)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if mask is not None:
            att = att.masked_fill(self.mask[:,:,:N,:M] == 0, float('-inf')) 
        att = torch.nn.functional.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, N, M) x (B, nh, M, hs) -> (B, nh, N, hs)
        y = y.transpose(1, 2).contiguous().view(B, N, D) # re-assemble all head outputs side by side
        return self.resid_drop(self.proj(y)) # B, N, D

class Block(torch.nn.Module):
    def __init__(self, kv_dim, q_dim, n_heads=1, attn_pdrop=0., resid_pdrop=0.):
        super().__init__()
        self.ln1_kv = torch.nn.LayerNorm(kv_dim)
        self.ln1_q = torch.nn.LayerNorm(q_dim)
        self.ln2 = torch.nn.LayerNorm(q_dim)
        self.attn = MultiHeadAttention(kv_dim, q_dim, n_heads, attn_pdrop, resid_pdrop)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(q_dim, 4 * q_dim),
            torch.nn.GELU(),
            torch.nn.Linear(4 * q_dim, q_dim),
            torch.nn.Dropout(resid_pdrop),
        )

    def forward(self, kv, q, mask=None):
        x = q + self.attn(self.ln1_kv(kv), self.ln1_q(q), mask)
        x = x + self.mlp(self.ln2(x))
        return x

class PerceiverEncoder(torch.nn.Module):
    # ejemplo sin recurrencia para clasificación
    def __init__(self, num_latents, latent_dim, input_dim, num_blocks, n_heads=1, attn_pdrop=0., resid_pdrop=0.):
        # se podrían separar los params en función de si es cross o self ...
        super().__init__()
        self.num_blocks = num_blocks
        self.latents = torch.nn.Parameter(torch.randn(num_latents, latent_dim))
        # encode
        self.cross_attn = Block(
            kv_dim=input_dim, 
            q_dim=latent_dim, 
            n_heads=n_heads, 
            attn_pdrop=attn_pdrop, 
            resid_pdrop=resid_pdrop
        )
        # process
        self.self_attention_blocks = torch.nn.ModuleList([
            Block( # se podrían hacer bloques diferenciados para en el forward pasar solo x
                kv_dim=latent_dim,
                q_dim=latent_dim,
                n_heads=n_heads, 
                attn_pdrop=attn_pdrop, 
                resid_pdrop=resid_pdrop
            ) for _ in range(num_blocks)
        ])

    def forward(self, x, mask = None):
        B = x.size(0)
        if mask is not None:
            mask = mask[None, None, :, :] # esto no se si está bien :S
        x = self.cross_attn(
            kv=x,
            q=self.latents.repeat(B, 1, 1),
            mask=mask
        )
        for _ in range(self.num_blocks):
            for self_attn_layer in self.self_attention_blocks:
                x = self_attn_layer(x, x)
        return x

# https://github.com/esceptico/perceiver-io/blob/master/src/perceiver/decoders.py

class ClassificationDecoder(torch.nn.Module):
    def __init__(self, num_classes, latent_dim, n_heads=1, attn_pdrop=0., resid_pdrop=0.):
        super().__init__()
        self.task_ids = torch.nn.Parameter(torch.randn(1, num_classes))
        self.decoder = Block( 
            kv_dim=latent_dim,
            q_dim=num_classes,
            n_heads=n_heads, 
            attn_pdrop=attn_pdrop, 
            resid_pdrop=resid_pdrop
        )

    def forward(self, latents):
        b = latents.size(0)
        logits = self.decoder(
            kv=latents,
            q=self.task_ids.repeat(b, 1, 1)
        )
        return logits.squeeze(1)

from einops import rearrange, repeat
from math import pi, log

class FourierEncoder(torch.nn.Module):
    def __init__(self, max_freq, num_freq_bands, freq_base=2):
        super().__init__()
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.freq_base = freq_base

    def fourier_encode(self, x):
        x = x.unsqueeze(-1)
        device, dtype, orig_x = x.device, x.dtype, x
        scales = torch.logspace(0., log(self.max_freq / 2) / log(self.freq_base), self.num_freq_bands, base = self.freq_base, device = device, dtype = dtype)
        scales = scales[(*((None,) * (len(x.shape) - 1)), ...)]
        x = x * scales * pi
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        x = torch.cat((x, orig_x), dim = -1)
        return x

    def forward(self, x):
         # fourier encoding
        b, *axis, _, device = *x.shape, x.device
        axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps = size, device = device), axis))
        pos = torch.stack(torch.meshgrid(*axis_pos), dim = -1)
        enc_pos = self.fourier_encode(pos)
        enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
        enc_pos = repeat(enc_pos, '... -> b ...', b = b)
        x = torch.cat((x, enc_pos), dim = -1)
        x = rearrange(x, 'b ... d -> b (...) d')
        return x

class PerceiverIO(torch.nn.Module):
    def __init__(self, num_classes, max_freq ,num_freq_bands, num_latents, latent_dim, input_dim, num_blocks, freq_base = 2, n_heads=1, attn_pdrop=0., resid_pdrop=0.):
        super().__init__()
        fourier_channels = (2 * ((num_freq_bands * 2) + 1))
        input_dim += fourier_channels
        self.fourier_encoder = FourierEncoder(
            max_freq=max_freq,
            num_freq_bands=num_freq_bands,
            freq_base=freq_base
        )
        self.encoder = PerceiverEncoder(
            num_latents=num_latents, 
            latent_dim=latent_dim, 
            input_dim=input_dim,
            num_blocks=num_blocks,
            n_heads=n_heads, 
            attn_pdrop=attn_pdrop, 
            resid_pdrop=resid_pdrop
        )
        self.decoder = ClassificationDecoder(
            num_classes=num_classes,
            latent_dim=latent_dim,
            n_heads=n_heads,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop
        )

    def forward(self, x, mask = None):
        x = self.fourier_encoder(x)
        if mask is not None:
            mask = mask[None, None, :, :] # esto no se si está bien :S
        x = self.encoder(x, mask)
        return self.decoder(x)