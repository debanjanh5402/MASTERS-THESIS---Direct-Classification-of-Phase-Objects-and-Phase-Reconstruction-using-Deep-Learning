import torch 
from torch import nn
from typing import List, Tuple


class EncoderBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, apply_norm:bool=True):
        super().__init__()

        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        if apply_norm: layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        layers.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor]:
        x = self.conv(x); p = self.pool(x)
        return x, p
    

class DecoderBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, apply_drop:bool=False):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        layers = [nn.Conv2d(out_channels*2, out_channels, kernel_size=3, padding=1),
                  nn.InstanceNorm2d(out_channels, affine=True),
                  nn.ReLU(inplace=True)]
        if apply_drop: layers.append(nn.Dropout(p=0.5))

        self.conv = nn.Sequential(*layers)

    def forward(self, x:torch.Tensor, skip:torch.Tensor) -> torch.Tensor:
        x = self.up(x); x = torch.cat([x, skip], dim=1); x = self.conv(x)
        return x
    

class Bottleneck(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()

        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                  nn.InstanceNorm2d(out_channels, affine=True),
                                  nn.ReLU(inplace=True))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.conv(x)
    

class FinalLayer(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetPhaseReconstruction_V1(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, features:List[int]=None):
        super().__init__()

        if features is None: features = [64, 128, 256, 512]

        self.enc1 = EncoderBlock(in_channels, features[0], apply_norm=False)
        self.enc2 = EncoderBlock(features[0], features[1])
        self.enc3 = EncoderBlock(features[1], features[2])
        self.enc4 = EncoderBlock(features[2], features[3])

        self.bn = Bottleneck(features[3], features[3]*2)

        self.dec4 = DecoderBlock(features[3]*2, features[3], apply_drop=True)
        self.dec3 = DecoderBlock(features[3], features[2], apply_drop=True)
        self.dec2 = DecoderBlock(features[2], features[1])
        self.dec1 = DecoderBlock(features[1], features[0])
        self.final = FinalLayer(features[0], out_channels)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        e1, p = self.enc1(x) # (bs, 001, 256, 256) -> (bs, 064, 256, 256), (bs, 064, 128, 128)
        e2, p = self.enc2(p) # (bs, 064, 128, 128) -> (bs, 128, 128, 128), (bs, 128, 064, 064)
        e3, p = self.enc3(p) # (bs, 128, 064, 064) -> (bs, 256, 064, 064), (bs, 256, 032, 032)
        e4, p = self.enc4(p) # (bs, 256, 032, 032) -> (bs, 512, 032, 032), (bs, 512, 016, 016)

        bn = self.bn(p) # (bs, 512, 016, 016) -> (bs, 1024, 016, 016)

        d4 = self.dec4(bn, e4) #(bs, 1024, 016, 016) -> (bs, 512, 032, 032) + (bs, 512, 032, 032) -> (bs, 512, 032, 032)
        d3 = self.dec3(d4, e3) # (bs, 512, 032, 032) -> (bs, 256, 064, 064) + (bs, 256, 064, 064) -> (bs, 256, 064, 064)
        d2 = self.dec2(d3, e2) # (bs, 256, 064, 064) -> (bs, 128, 128, 128) + (bs, 128, 128, 128) -> (bs, 128, 128, 128)
        d1 = self.dec1(d2, e1) # (bs, 128, 128, 128) -> (bs, 064, 256, 256) + (bs, 064, 256, 256) -> (bs, 064, 256, 256)

        output = self.final(d1) # (bs, 064, 256, 256) -> (bs, 001, 256, 256)
        return output
    

class UNetPhaseReconstruction_V2(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, features:List[int]=None):
        super().__init__()

        if features is None: features = [32, 64, 128, 256, 512]

        self.enc1 = EncoderBlock(in_channels, features[0], apply_norm=False)
        self.enc2 = EncoderBlock(features[0], features[1])
        self.enc3 = EncoderBlock(features[1], features[2])
        self.enc4 = EncoderBlock(features[2], features[3])
        self.enc5 = EncoderBlock(features[3], features[4])

        self.bn = Bottleneck(features[4], features[4]*2)

        self.dec5 = DecoderBlock(features[4]*2, features[4], apply_drop=True)
        self.dec4 = DecoderBlock(features[4], features[3], apply_drop=True)
        self.dec3 = DecoderBlock(features[3], features[2])
        self.dec2 = DecoderBlock(features[2], features[1])
        self.dec1 = DecoderBlock(features[1], features[0])
        self.final = FinalLayer(features[0], out_channels)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        e1, p = self.enc1(x) # (bs, 001, 256, 256) -> (bs, 032, 256, 256), (bs, 032, 128, 128)
        e2, p = self.enc2(p) # (bs, 032, 128, 128) -> (bs, 064, 128, 128), (bs, 064, 064, 064)
        e3, p = self.enc3(p) # (bs, 064, 064, 064) -> (bs, 128, 064, 064), (bs, 128, 032, 032)
        e4, p = self.enc4(p) # (bs, 128, 032, 032) -> (bs, 256, 032, 032), (bs, 256, 016, 016)
        e5, p = self.enc5(p) # (bs, 256, 016, 016) -> (bs, 512, 016, 016), (bs, 512, 008, 008)

        bn = self.bn(p) # (bs, 512, 008, 008) -> (bs, 1024, 008, 008)

        d5= self.dec5(bn, e5) # (bs, 1024, 008, 008) -> (bs, 512, 016, 016) + (bs, 512, 016, 016) -> (bs, 512, 016, 016) 
        d4 = self.dec4(d5, e4) # (bs, 512, 016, 016) -> (bs, 256, 032, 032) + (bs, 256, 032, 032) -> (bs, 256, 032, 032)
        d3 = self.dec3(d4, e3) # (bs, 256, 032, 032) -> (bs, 128, 064, 064) + (bs, 128, 064, 064) -> (bs, 128, 064, 064)
        d2 = self.dec2(d3, e2) # (bs, 128, 064, 064) -> (bs, 064, 128, 128) + (bs, 064, 128, 128) -> (bs, 064, 128, 128)
        d1 = self.dec1(d2, e1) # (bs, 064, 128, 128) -> (bs, 032, 256, 256) + (bs, 032, 256, 256) -> (bs, 032, 256, 256)

        output = self.final(d1) # (bs, 032, 256, 256) -> (bs, 001, 256, 256)
        return output