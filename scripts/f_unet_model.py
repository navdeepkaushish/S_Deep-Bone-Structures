# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 10:59:39 2023

@author: Navdeep Kumar
"""

import torch
import torch.nn as nn
from torchsummary import summary



class Double_conv(nn.Module):
    def __init__(self, in_channels, out_channels,mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding='same'),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding='same'),
                                         nn.ReLU(inplace=True),
                                         nn.BatchNorm2d(out_channels)
                                         )
            
    def forward(self, x):
        x = self.double_conv(x)
        
        return x
            
        

class Downsample_block(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, pooling_on=True):
        super().__init__()
        self.double_conv = Double_conv(in_channels, out_channels)
        self.pooling_on = pooling_on
        self.max_pool = nn.MaxPool2d(2, stride=2)
            
    def forward(self,x):
        x = self.double_conv(x)
        skip = x
        if self.pooling_on:
            x = self.max_pool(x)
        
        return x, skip
    
class Upsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.upsample = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                                      nn.ReLU(inplace=True),
                                      nn.BatchNorm2d(out_channels)
                                      
                                     )
        self.conv = Double_conv(in_channels, out_channels)
       
    def forward(self, x, skip):
        
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x  = self.conv(x)
        return x
        
        
        
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
    
class f_UNet(nn.Module):
    def __init__(self, in_channels, n_masks):
        
        super().__init__()
        #Downsample blocks
        self.down1 = Downsample_block(in_channels, 64)
        self.down2 = Downsample_block(64, 128)
        self.down3 = Downsample_block(128, 256)
        self.down4 = Downsample_block(256, 512)
        self.down5 = Downsample_block(512, 1024, pooling_on=False)
        #Upsample blocks
        self.up1 = Upsample_block(1024, 512)
        self.up2 = Upsample_block(512, 256)
        self.up3 = Upsample_block(256, 128)
        self.up4 = Upsample_block(128, 64)
        self.outconv = OutConv(64, n_masks)
        
    def forward(self,x):
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)
        x, _ = self.down5(x)
        #print(x.shape)
        x = self.up1(x,skip4)
        #print(x.shape, skip4.shape)
        x = self.up2(x,skip3)
        #print(x.shape, skip3.shape)
        x = self.up3(x,skip2)
        #print(x.shape, skip2.shape)
        x = self.up4(x,skip1)
        #print(x.shape, skip1.shape)
        out = self.outconv(x)
        #print(out.shape)
        return out
    
#x = torch.randn(1,3,256,256)
#model = UNet(3,25)
       
       
        