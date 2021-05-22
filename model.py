import torch.nn as nn

def ConvBNReLU(inc, outc, kernel, stride, padding):
    return nn.Sequential(
        nn.Conv2d(inc, outc, kernel, stride=stride, padding=padding),
        nn.BatchNorm2d(outc),
        nn.ReLU()
    )

def SepConvBNReLU(inc, outc, kernel, stride, padding):
    return nn.Sequential(
        nn.Conv2d(inc, inc, kernel, stride=stride, padding=padding, groups=inc), # dw
        nn.Conv2d(inc, outc, 1, padding=0), # pw
        nn.BatchNorm2d(outc),
        nn.ReLU()
    )

class MNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNReLU(1, 8, 3, stride=2, padding=0),
            SepConvBNReLU(8, 16, 3, stride=2, padding=0),
            SepConvBNReLU(16, 32, 3, stride=2, padding=0),
            SepConvBNReLU(32, 64, 3, stride=2, padding=0),
            SepConvBNReLU(64, 64, 3, stride=2, padding=0),
            SepConvBNReLU(64, 128, 3, stride=2, padding=0),
        )
        self.gmp = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 1)
    def forward(self, x):
        x = self.features(x)
        x = self.gmp(x)
        x = x.view(-1, 128)
        x = self.fc(x)
        return x.view(-1)