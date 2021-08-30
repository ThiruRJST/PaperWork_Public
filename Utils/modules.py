import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class conv_mod(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=(1,1),stride=1,padding=0,activation='relu'):
        super(conv_mod,self).__init__()
        self.mod = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            

        )

    def forward(self,x):
        x = self.mod(x)
        return x



class MultiScaleA(nn.Module):
    def __init__(self,in_channels,stream_1_1x1_filters,stream_2_1x1_filters,stream_2_3x3_filters,stream_3_1x1_filters,stream_3_3x3_filters):
        super(MultiScaleA,self).__init__()
        
        self.stream1 = conv_mod(in_channels,stream_1_1x1_filters,kernel_size=(1,1)) #Stream ID=1
        self.stream2 = nn.Sequential(
            conv_mod(in_channels,stream_2_1x1_filters,kernel_size=(1,1)),
            conv_mod(stream_2_1x1_filters,stream_2_3x3_filters,kernel_size=(3,3),padding=1),     #Stream ID=2
        )
        self.stream3 = nn.Sequential(
            conv_mod(in_channels,stream_3_1x1_filters,kernel_size=(1,1)),
            conv_mod(stream_3_1x1_filters,stream_3_3x3_filters[0],kernel_size=(3,3),padding=1),
            conv_mod(stream_3_3x3_filters[0],stream_3_3x3_filters[1],kernel_size=(3,3),padding=1)
        )

    def forward(self,x):

        stream1 = self.stream1(x)
        stream2 = self.stream2(x)
        stream3 = self.stream3(x)

        concat = torch.cat([stream1,stream2,stream3],axis=1)

        return concat


class Reduction(nn.Module):
    def __init__(self,in_channels,red_stream_2_3x3_filters,red_stream_3_1x1_filters,red_stream_3_3x3_filters):

        super(Reduction,self).__init__()
        self.stream1_MF = nn.MaxPool2d(kernel_size=(3,3),stride=2)
        
        self.stream2_CF = conv_mod(in_channels,red_stream_2_3x3_filters,kernel_size=(3,3),stride=(2,2))
        
        self.stream3_CF = nn.Sequential(
            conv_mod(in_channels,red_stream_3_1x1_filters,kernel_size=(1,1)),
            conv_mod(red_stream_3_1x1_filters,red_stream_3_3x3_filters[0],kernel_size=(3,3)),
            conv_mod(red_stream_3_3x3_filters[0],red_stream_3_3x3_filters[1],kernel_size=(3,3),stride=(2,2),padding=1)
        )
        

    def forward(self,x):

        stream1_MF = self.stream1_MF(x)
        stream2_CF = self.stream2_CF(x)
        stream3_CF = self.stream3_CF(x)

        print(stream1_MF.shape,stream2_CF.shape,stream3_CF.shape)

        return torch.cat([stream1_MF,stream2_CF,stream3_CF],axis=1)



class MultiScaleB(nn.Module):
    def __init__(self,in_channels,Bstream_1_1x1,Bstream_2_1x1,Bstream_2_3x3,Bstream_3_1x1,Bstream_3_3x3):
        super(MultiScaleB,self).__init__()
        self.st1 = conv_mod(in_channels,Bstream_1_1x1,kernel_size=(1,1))

        self.st2 = nn.Sequential(
            conv_mod(in_channels,Bstream_2_1x1,kernel_size=(1,1)),
            conv_mod(Bstream_2_1x1,Bstream_2_3x3[0],kernel_size=(1,3)),
            conv_mod(Bstream_2_3x3[0],Bstream_2_3x3[1],kernel_size=(3,1),padding=(1,1))
            )

        self.st3 = nn.Sequential(
            conv_mod(in_channels,Bstream_3_1x1,kernel_size=(1,1)),
            conv_mod(Bstream_3_1x1,Bstream_3_3x3[0],kernel_size=(1,3)),
            conv_mod(Bstream_3_3x3[1],Bstream_3_3x3[2],kernel_size=(3,1)),
            conv_mod(Bstream_3_3x3[2],Bstream_3_3x3[3],kernel_size=(1,3)),
            conv_mod(Bstream_3_3x3[3],Bstream_3_3x3[4],kernel_size=(3,1),padding=(2,2))
        )


    def forward(self,x):

        st1 = self.st1(x)
        st2 = self.st2(x)
        st3 = self.st3(x)

        print(st1.shape,st2.shape,st3.shape)

        return torch.cat([st1,st2,st3],axis=1)

class self_attention(nn.Module):
    def __init__(self,feature):
        super(self_attention,self).__init__()
        self.f = nn.Conv2d(feature.shape[1],feature.shape[1],kernel_size=(1,1))
        self.g = nn.Conv2d(feature.shape[1],feature.shape[1],kernel_size=(1,1))
        self.h = nn.Conv2d(feature.shape[1],feature.shape[1],kernel_size=(1,1))
        self.soft = nn.Softmax()
    def forward(self,x):
        
        f = self.f(x)
        g = self.g(x)
        h = self.h(x)
        fg = torch.dot(f.T,g)
        fg_soft = self.soft(fg)
        return torch.dot(fg_soft,h)