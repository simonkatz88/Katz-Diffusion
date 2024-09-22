import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self, n_embd: int):
        super().__init()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear( 4 * n_embd, 4 * n_embd)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x: (1, 320)

        x = self.linear_1(x)

        x = F.silu(x)

        x = self.linear_2(x)

        # x: (1, 1280)
        return x

class UNET(nn.Module):
    
    def __init__(self):
        super().__init__(

        )

class Diffusion(nn.Module):
    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)
    
    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        # latent: (Batch_Size, 4, Height / 8, Width /8)
        # context (Batch,_Size, Seq_Len, Dim)
        # time (1, 320)
        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)

        # (Batch, 4, Height / 8, Width / 8) -> (Batch, 320, Height / 8, Width / 8)
        output = self.unet(latent, context, time)

        # (Batch, 320, Height / 8, Width / 8) -> (Batch, 4, Height / 8, Width / 8)
        output = self.final(output)

        return output
