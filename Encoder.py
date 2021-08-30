import torch.nn as nn
from Utils.configs import params
from Utils.modules import MultiScaleA,MultiScaleB,Reduction,conv_mod
import json


#load Config.json files for parameters
files = open('configs.json')
hp = json.load(files)



class Multi_Block(nn.Module):
    def __init__(self,blocks):
        super(Multi_Block,self).__init__()
        self.BlockA_params = [i for i in blocks[0].values()]
        self.BlockB_params = [j for j in blocks[1].values()]
        self.BlockC_params = [k for k in blocks[2].values()]

        self.mulA = MultiScaleA(self.BlockA_params[0],self.BlockA_params[1],self.BlockA_params[2],self.BlockA_params[3],self.BlockA_params[4],self.BlockA_params[5])

        self.red = Reduction(self.BlockB_params[0],self.BlockB_params[1],self.BlockB_params[2],self.BlockB_params[3])

        self.mulB = MultiScaleB(self.BlockC_params[0],self.BlockC_params[1],self.BlockC_params[2],self.BlockC_params[3],self.BlockC_params[4],self.BlockC_params[5])

    def forward(self,x):

        x = self.mulA(x)
        x = self.red(x)
        x = self.mulB(x)

        return x

class Encoder(nn.Module):
    def __init__(self,params):
        super(Encoder,self).__init__()
        self.A = Multi_Block(params['BlockA'])
        self.B = Multi_Block(params['BlockB'])
        self.C = Multi_Block(params['BlockC'])
        self.sing_1 = conv_mod(416,256,kernel_size=(5,5))
        self.sing_2 = conv_mod(256,128,kernel_size=(5,5))
        self.sing_3 = conv_mod(128,64,kernel_size=(5,5))

    def forward(self,x):
        multi_x_1 = self.A(x)
        multi_x_2 = self.B(multi_x_1)
        multi_x_3 = self.C(multi_x_2)
        single_x_1 = self.sing_1(multi_x_3)
        single_x_2 = self.sing_2(single_x_1)
        single_x_3 = self.sing_3(single_x_2)

        return multi_x_1,multi_x_2,multi_x_3,single_x_1,single_x_2,single_x_3


#Model Declaration
enc_model = Encoder(hp)
print(enc_model)