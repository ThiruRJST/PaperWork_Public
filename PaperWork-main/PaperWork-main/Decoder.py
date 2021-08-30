import torch.nn as nn



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.up1 = nn.ConvTranspose2d(64,128,kernel_size=(5,5))
        self.up2 = nn.ConvTranspose2d(128,256,kernel_size=(5,5))
        self.up3 = nn.ConvTranspose2d(256,416,kernel_size=(5,5))
    def forward(self,x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)

        return x
