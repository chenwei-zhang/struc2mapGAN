import torch
from torch import nn


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(ResidualConvBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, stride=1, bias=True)
        self.norm1 = nn.InstanceNorm3d(mid_channels)
        self.prelu1 = nn.PReLU()
        
        self.conv2 = nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, stride=1,  bias=True)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.prelu2 = nn.PReLU()
        
        self.downsample = nn.Sequential()
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels,  kernel_size=3, padding=1, stride=1,  bias=True),
                nn.InstanceNorm3d(out_channels)
            )
            
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        residual = self.downsample(x) # if downsample true, otherwise residual = x
        out += residual
        out = self.prelu2(out)

        return out


class GeneratorNestedUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(GeneratorNestedUNet, self).__init__()
        
        # Nested UNet (UNet++), 4 levels
        num_channels = [32, 64, 128, 256, 512]

        # Depth 1
        self.conv0_0 = ResidualConvBlock(in_channels, num_channels[0], num_channels[0]) # input
        
        # Depth 2
        self.conv1_0 = ResidualConvBlock(num_channels[0], num_channels[1], num_channels[1]) # down 1
        self.conv0_1 = ResidualConvBlock(num_channels[0]+num_channels[1], num_channels[0], num_channels[0]) # up 1
        
        # Depth 3
        self.conv2_0 = ResidualConvBlock(num_channels[1], num_channels[2], num_channels[2]) # down 2
        self.conv1_1 = ResidualConvBlock(num_channels[1]+num_channels[2], num_channels[1], num_channels[1]) # up 1
        self.conv0_2 = ResidualConvBlock(num_channels[0]*2+num_channels[1], num_channels[0], num_channels[0]) # up 2

        # Depth 4
        self.conv3_0 = ResidualConvBlock(num_channels[2], num_channels[3], num_channels[3]) # down 3
        self.conv2_1 = ResidualConvBlock(num_channels[2]+num_channels[3], num_channels[2], num_channels[2]) # up 1
        self.conv1_2 = ResidualConvBlock(num_channels[1]*2+num_channels[2], num_channels[1], num_channels[1]) # up 2
        self.conv0_3 = ResidualConvBlock(num_channels[0]*3+num_channels[1], num_channels[0], num_channels[0]) # up 3
        
        self.final = nn.Conv3d(num_channels[0], out_channels, kernel_size=1, stride=1, padding=0, bias=True) # output
        
    def forward(self, x):
        up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        pool = nn.MaxPool3d(kernel_size=2, stride=2)
        # Downsampling
        x0_0 = self.conv0_0(x) 
        x1_0 = self.conv1_0(pool(x0_0)) # down 1
        x2_0 = self.conv2_0(pool(x1_0)) # down 2
        x3_0 = self.conv3_0(pool(x2_0)) # down 3
        # Upsampling
        x0_1 = self.conv0_1(torch.cat([x0_0, up(x1_0)], 1)) # up 1
        x1_1 = self.conv1_1(torch.cat([x1_0, up(x2_0)], 1)) # up 1
        x2_1 = self.conv2_1(torch.cat([x2_0, up(x3_0)], 1)) # up 1
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, up(x1_1)], 1)) # up 2
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, up(x2_1)], 1)) # up 2
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, up(x1_2)], 1)) # up 3
        out = self.final(x0_3)
        
        return out


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super(Discriminator, self).__init__()
        
        num_channels = [32, 64, 128, 256, 512]
        
        self.prelu = nn.PReLU()
        self.relu = nn.ReLU()
        
        self.conv1 = nn.Conv3d(in_channels, num_channels[0], kernel_size=3, padding=1, stride=1, bias=True)
        self.norm1 = nn.InstanceNorm3d(num_channels[0])
        
        self.conv2 = nn.Conv3d(num_channels[0], num_channels[1], kernel_size=3, padding=1, stride=1, bias=True)
        self.norm2 = nn.InstanceNorm3d(num_channels[1])

        self.conv3 = nn.Conv3d(num_channels[1], num_channels[2], kernel_size=3, padding=1, stride=1, bias=True)
        self.norm3 = nn.InstanceNorm3d(num_channels[2])
        
        self.conv4 = nn.Conv3d(num_channels[2], num_channels[3], kernel_size=3, padding=1, stride=1, bias=True)
        # self.norm4 = nn.InstanceNorm3d(num_channels[3])
        # self.conv5 = nn.Conv3d(num_channels[3], num_channels[4], kernel_size=1, padding=0, stride=1, bias=True)

        # Adaptive pooling and linear layers for classification
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(num_channels[3], 100)
        self.fc2 = nn.Linear(100, 25)
        self.fc3 = nn.Linear(25, num_classes)
        
    def forward(self, x):
        x = self.prelu(self.norm1(self.conv1(x)))
        x = self.prelu(self.norm2(self.conv2(x)))
        x = self.prelu(self.norm3(self.conv3(x)))
        # x = self.prelu(self.norm4(self.conv4(x)))
        x = self.conv4(x)
        # pooling and pass to fc layers
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x