import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtraction(nn.Module):
    def __init__(self, feat_channels, grow=32, kernel_size=3):
        super(FeatureExtraction, self).__init__()
        padding = kernel_size// 2  # o preserve spatial dimensions
        
        self.conv1 = nn.Conv2d(feat_channels, grow, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(feat_channels + grow, grow, kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(feat_channels + 2 * grow, feat_channels, kernel_size, padding=padding)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), dim=1)))
        x3 = self.conv3(torch.cat((x, x1, x2), dim=1))
        return x3 * 0.2 + x

class StackFeatureExtraction(nn.Module):
    def __init__(self, feat_channels, grow=32, kernel_size=3, num_blocks=3):
        super(StackFeatureExtraction, self).__init__()
        #stack FeatureExtraction blocks.
        blocks = []
        for _ in range(num_blocks):
            blocks.append(FeatureExtraction(feat_channels, grow, kernel_size))
        self.blocks = nn.Sequential(*blocks)
        
    def forward(self, x):
        return self.blocks(x)

class GenerateImage(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, feat_channels=64, 
                 num_stacks=3, grow=32, kernel_size=3, upscale_factor=4):
        super(GenerateImage, self).__init__()
        self.convFeatMap = nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1)
        self.featExtract = StackFeatureExtraction(feat_channels, grow, kernel_size, num_blocks=num_stacks)
        self.convT = nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1)
        self.conv1Upscale = nn.Conv2d(feat_channels, feat_channels * 4, kernel_size=3, padding=1)
        self.shuffle1 = nn.PixelShuffle(2)
        self.conv2Upscale = nn.Conv2d(feat_channels, feat_channels * 4, kernel_size=3, padding=1)
        self.shuffle2 = nn.PixelShuffle(2)
        self.convLast = nn.Conv2d(feat_channels, out_channels, kernel_size=3, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
    def forward(self, x):
        fea = self.convFeatMap(x)
        refined = self.featExtract(fea)
        trunk = self.convT(refined)
        fea = fea + trunk
        fea = self.lrelu(self.shuffle1(self.conv1Upscale(fea)))
        fea = self.lrelu(self.shuffle2(self.conv2Upscale(fea)))
        out = self.convLast(fea)
        return out
