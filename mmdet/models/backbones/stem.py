import torch
from .build_stem import STEM_MODELS
import torch.nn as nn
import wtorch.nn as wnn



@STEM_MODELS.register()
class MultiBranchStem12X(nn.Module):
    def __init__(self,in_channels,out_channels,activation_fn="LeakyReLU"):
        super().__init__()
        self.out_channels = out_channels
        branch_channels = out_channels//4
        self.branch0_0 = nn.Sequential(
            nn.Conv2d(in_channels,4,3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(num_features=4),
            wnn.get_activation(activation_fn,inplace=True),
            nn.Conv2d(4,branch_channels,3,stride=1,padding=1,bias=False),
            )
        self.branch0_1 = nn.ModuleList([nn.MaxPool2d(3,2,1),nn.MaxPool2d(5,2,2)])
        self.branch0_2 = nn.Sequential(
            wnn.get_activation(activation_fn,inplace=True),
            nn.Conv2d(branch_channels*2,branch_channels*2,3,3,padding=0,bias=False),
        )
        self.branch1 = nn.Conv2d(in_channels,branch_channels,7,stride=2,padding=3,bias=False)
        self.branch2 = nn.Conv2d(in_channels,branch_channels,12,12,0,bias=False)
        self.norm = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            wnn.get_activation(activation_fn,inplace=True),
        )

    def forward(self,x):
        downsampled = torch.nn.functional.interpolate(x,(x.shape[-2]//6,x.shape[-1]//6),mode='bilinear')
        x0 = self.branch0_0(x)
        x0_0 = self.branch0_1[0](x0)
        x0_1 = self.branch0_1[1](x0)
        x0 = torch.cat([x0_0,x0_1],dim=1)
        x0 = self.branch0_2(x0)
        x1 = self.branch1(downsampled)
        x2 = self.branch2(x)
        x = torch.cat([x0,x1,x2],dim=1)
        x = self.norm(x)
        return x

@STEM_MODELS.register()
class MultiBranchStemS12X(nn.Module):
    def __init__(self,in_channels,out_channels,activation_fn="LeakyReLU"):
        super().__init__()
        self.out_channels = out_channels
        branch_channels = out_channels//4
        self.branch0_0 = nn.Sequential(
            nn.Conv2d(in_channels,4,3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(num_features=4),
            wnn.get_activation(activation_fn,inplace=True),
            nn.Conv2d(4,branch_channels,3,stride=1,padding=1,bias=False))
        self.branch0_1 = nn.ModuleList([nn.MaxPool2d(3,2,1),nn.MaxPool2d(5,2,2)])
        self.branch0_2 = nn.Conv2d(branch_channels*2,branch_channels*2,3,3,padding=0,bias=False)
        self.branch1 = nn.Conv2d(in_channels,branch_channels*2,7,stride=2,padding=3,bias=False)
        self.norm = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            wnn.get_activation(activation_fn,inplace=True),
        )

    def forward(self,x):
        downsampled = torch.nn.functional.interpolate(x,(x.shape[-2]//6,x.shape[-1]//6),mode='bilinear')
        x0 = self.branch0_0(x)
        x0_0 = self.branch0_1[0](x0)
        x0_1 = self.branch0_1[1](x0)
        x0 = torch.cat([x0_0,x0_1],dim=1)
        x0 = self.branch0_2(x0)
        x1 = self.branch1(downsampled)
        x = torch.cat([x0,x1],dim=1)
        x = self.norm(x)
        return x

@STEM_MODELS.register()
class MultiBranchStemSA12X(nn.Module):
    def __init__(self,in_channels,out_channels,activation_fn="LeakyReLU"):
        super().__init__()
        self.out_channels = out_channels
        branch_channels = out_channels//4
        self.branch0_0 = nn.Sequential(
            nn.Conv2d(in_channels,4,3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(num_features=4),
            wnn.get_activation(activation_fn,inplace=True),
            nn.Conv2d(4,branch_channels,3,stride=1,padding=1,bias=False),
            )
        self.branch0_1 = nn.ModuleList([nn.MaxPool2d(3,2,1),nn.MaxPool2d(5,2,2)])
        self.branch0_2 = nn.Sequential(
            wnn.get_activation(activation_fn,inplace=True),
            nn.Conv2d(branch_channels*2,branch_channels*2,3,3,padding=0,bias=False),
        )
        self.branch1 = nn.Conv2d(in_channels,branch_channels*2,7,stride=2,padding=3,bias=False)
        self.norm = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            wnn.get_activation(activation_fn,inplace=True),
        )

    def forward(self,x):
        downsampled = torch.nn.functional.interpolate(x,(x.shape[-2]//6,x.shape[-1]//6),mode='bilinear')
        x0 = self.branch0_0(x)
        x0_0 = self.branch0_1[0](x0)
        x0_1 = self.branch0_1[1](x0)
        x0 = torch.cat([x0_0,x0_1],dim=1)
        x0 = self.branch0_2(x0)
        x1 = self.branch1(downsampled)
        x = torch.cat([x0,x1],dim=1)
        x = self.norm(x)
        return x

@STEM_MODELS.register()
class MultiBranchStemS4X(nn.Module):
    def __init__(self,in_channels,out_channels,activation_fn="LeakyReLU"):
        super().__init__()
        self.out_channels = out_channels
        branch_channels = out_channels//4
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channels,4,3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(num_features=4),
            wnn.get_activation(activation_fn,inplace=True),
            nn.Conv2d(4,branch_channels*2,3,stride=1,padding=1,bias=False))
        self.branch1 = nn.Conv2d(in_channels,branch_channels*2,7,stride=2,padding=3,bias=False)
        self.norm = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            wnn.get_activation(activation_fn,inplace=True),
        )

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x = torch.cat([x0,x1],dim=1)
        x = self.norm(x)
        return x

@STEM_MODELS.register()
class MultiBranchStemSA2X(nn.Module):
    def __init__(self,in_channels,out_channels,activation_fn="LeakyReLU"):
        super().__init__()
        self.out_channels = out_channels
        branch_channels = out_channels//4
        self.branch0_0 = nn.Sequential(
            nn.Conv2d(in_channels,branch_channels,3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(num_features=branch_channels),
            )
        self.branch0_1 = nn.ModuleList([nn.MaxPool2d(3,2,1),nn.MaxPool2d(5,2,2)])
        self.branch0_2 = nn.Sequential(
            wnn.get_activation(activation_fn,inplace=True),
            nn.Conv2d(branch_channels*2,branch_channels*2,3,1,padding=1,bias=False),
        )
        self.branch1 = nn.Conv2d(in_channels,branch_channels*2,7,stride=2,padding=3,bias=False)
        self.norm = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            wnn.get_activation(activation_fn,inplace=True),
        )

    def forward(self,x):
        x0 = self.branch0_0(x)
        x0_0 = self.branch0_1[0](x0)
        x0_1 = self.branch0_1[1](x0)
        x0 = torch.cat([x0_0,x0_1],dim=1)
        x0 = self.branch0_2(x0)
        x1 = self.branch1(x)
        x = torch.cat([x0,x1],dim=1)
        x = self.norm(x)
        return x

@STEM_MODELS.register()
class MultiBranchStemSL12X(nn.Module):
    def __init__(self,in_channels,out_channels,activation_fn="LeakyReLU"):
        super().__init__()
        self.out_channels = out_channels
        branch_channels = out_channels//4
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channels,4,3,stride=3,padding=0,bias=False),
            nn.BatchNorm2d(num_features=4),
            wnn.get_activation(activation_fn,inplace=True),
            nn.Conv2d(4,branch_channels,4,stride=4,padding=0,bias=False))
        self.branch1_0 = nn.Sequential(
            nn.Conv2d(in_channels,4,3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(num_features=4),
            wnn.get_activation(activation_fn,inplace=True),
            nn.Conv2d(4,8,3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(num_features=8),
            wnn.get_activation(activation_fn,inplace=True),
            )
        self.branch1_1 = nn.ModuleList([nn.MaxPool2d(3,2,1),nn.MaxPool2d(5,2,2)])
        self.branch1_2 = nn.Conv2d(branch_channels,branch_channels,3,3,padding=0,bias=False)
        self.branch2 = nn.Conv2d(in_channels,branch_channels*2,7,stride=2,padding=3,bias=False)
        self.norm = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            wnn.get_activation(activation_fn,inplace=True),
        )

    def forward(self,x):
        downsampled = torch.nn.functional.interpolate(x,(x.shape[-2]//6,x.shape[-1]//6),mode='bilinear')
        x0 = self.branch0(x)
        x1 = self.branch1_0(x)
        x1_0 = self.branch1_1[0](x1)
        x1_1 = self.branch1_1[1](x1)
        x1 = torch.cat([x1_0,x1_1],dim=1)
        x1 = self.branch1_2(x1)
        x2 = self.branch2(downsampled)
        x = torch.cat([x0,x1,x2],dim=1)
        x = self.norm(x)
        return x

@STEM_MODELS.register()
class MultiBranchStemM12X(nn.Module):
    def __init__(self,in_channels,out_channels,activation_fn="LeakyReLU"):
        super().__init__()
        self.out_channels = out_channels
        branch_channels = out_channels//4
        self.branch0_0 = nn.Sequential(
            nn.Conv2d(in_channels,4,3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(num_features=4),
            wnn.get_activation(activation_fn,inplace=True),
            nn.Conv2d(4,branch_channels,3,stride=1,padding=1,bias=True),
            )
        self.branch0_1 = nn.ModuleList([nn.MaxPool2d(3,2,1),nn.MaxPool2d(5,2,2)])
        self.branch0_2 = nn.Sequential(
            wnn.get_activation(activation_fn,inplace=True),
            nn.Conv2d(branch_channels*2,branch_channels*2,3,3,padding=0,bias=False),
        )
        self.branch1 = nn.Conv2d(in_channels,branch_channels,7,stride=2,padding=3,bias=False)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels,4,4,stride=4,padding=0,bias=True),
            #nn.BatchNorm2d(num_features=4),
            wnn.get_activation(activation_fn,inplace=True),
            nn.Conv2d(4,branch_channels,3,stride=3,padding=0,bias=False),
            )
        self.norm = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            wnn.get_activation(activation_fn,inplace=True),
            nn.Conv2d(out_channels,out_channels,3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            wnn.get_activation(activation_fn,inplace=True),
        )

    def forward(self,x):
        downsampled = torch.nn.functional.interpolate(x,(x.shape[-2]//6,x.shape[-1]//6),mode='bilinear')
        x0 = self.branch0_0(x)
        x0_0 = self.branch0_1[0](x0)
        x0_1 = self.branch0_1[1](x0)
        x0 = torch.cat([x0_0,x0_1],dim=1)
        x0 = self.branch0_2(x0)
        x1 = self.branch1(downsampled)
        x2 = self.branch2(x)
        x = torch.cat([x0,x1,x2],dim=1)
        x = self.norm(x)
        return x