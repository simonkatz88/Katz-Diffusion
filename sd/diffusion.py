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

class SwitchSequential(nn.Sequential):

    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ReidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
            return x


class UNET_ReidualBlock(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, n_time=1280):
        super().__init__()
        self.groupnorm_features = nn.GroupNorm(32 ,in_channels)
        self.conv_features = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged =nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0)

    def forward(self, feature, time):
        # feature: (Batch_Size, In_Channels, Height, Width)
        # time: (1, 1280)

        residue = feature 

        feature = self.groupnorm_features(feature)

        feature = F.silu()

        feature = self.linear_time(time)

        time = F.silu()

        time = self.linear_time(time)

        merged = feature + time.unsqueeze(-1).unsqueeze(-1)

        merged = self.groupnorm_merged(merged)

        merged = self.silu(merged)

        merged = self.conv_merged(merged)

        return merged + residue

class Upsample(nn.Module):
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * 2, Width * 2)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv





class UNET(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoders = nn.Module([
            # (Batch_Size, 4, Height / 8, Width /8)
            SwitchSequential(nn.Conv2d( 320, kernel_size=3, padding=1)),

            SwitchSequential(UNET_ReidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_ReidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 16, Width / 16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ReidualBlock(320, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ReidualBlock(640, 640), UNET_AttentionBlock(8, 80)),

            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ReidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ReidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),

            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ReidualBlock(1280, 1280)),

            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ReidualBlock(1280, 1280)),
        ])

        self.bottleneck = SwitchSequential(
            UNET_ReidualBlock(1280, 1280),

            UNET_AttentionBlock(8, 160),

            UNET_ReidualBlock(1280, 1280)
        )

        self.decoders = nn.ModuleList([

            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ReidualBlock(2560, 1280)),

            SwitchSequential(UNET_ReidualBlock(2560, 1280)),

            SwitchSequential(UNET_ReidualBlock(2560, 1280), UpSample(1280)),

            SwitchSequential(UNET_ReidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ReidualBlock(2560, 1280), UNET_AttentionBlock(8, 160), UpSample(1280)),
           
            SwitchSequential(UNET_ReidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ReidualBlock(1280, 640), UNET_AttentionBlock(8, 80), UpSample(640)),

            SwitchSequential(UNET_ReidualBlock(960, 640), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_ReidualBlock(640, 320), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ReidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            ])

class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        # (Batch, 320, Height / 8, Width / 8)

        x = self.groupnorm(x)

        x = F.silu(x)

        x = self.conv(x)

        # (Batch, 4, Height / 8, Width / 8)
        return x

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
