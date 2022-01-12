import torch
import torch.nn as nn

class Gated_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size)->None:
        super(Gated_Conv,self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        A = self.conv1(x)
        B = self.conv2(x)
        B = self.sigmoid(B)
        
        out = A * B   # element-wise multiplication

        return out




class Wav2Letter2(nn.Module):
    def __init__(self) -> None:
        super(Wav2Letter2, self).__init__()
        self.conv1 = Gated_Conv(in_channels=40, out_channels=100, kernel_size=3)
        self.dropout1 = nn.Dropout(0.25)
        self.gated_conv_stack = nn.Sequential(*[Gated_Conv(in_channels=100, out_channels=100, kernel_size=4+i) for i in range(15)])
        # self.conv2 = Gated_Conv(in_channels=100, out_channels=100, kernel_size=4)
        # self.conv3 = Gated_Conv(in_channels=100, out_channels=100, kernel_size=5)
        # self.conv4 = Gated_Conv(in_channels=100, out_channels=100, kernel_size=6)
        # self.conv5 = Gated_Conv(in_channels=100, out_channels=100, kernel_size=7)
        # self.conv6 = Gated_Conv(in_channels=100, out_channels=100, kernel_size=8)
        # self.conv7 = Gated_Conv(in_channels=100, out_channels=100, kernel_size=9)
        # self.conv8 = Gated_Conv(in_channels=100, out_channels=100, kernel_size=10)
        # self.conv9 = Gated_Conv(in_channels=100, out_channels=100, kernel_size=11)
        # self.conv10 = Gated_Conv(in_channels=100, out_channels=100, kernel_size=12)
        # self.conv11 = Gated_Conv(in_channels=100, out_channels=100, kernel_size=13)
        # self.conv12 = Gated_Conv(in_channels=100, out_channels=100, kernel_size=14)
        # self.conv13 = Gated_Conv(in_channels=100, out_channels=100, kernel_size=15)
        # self.conv14 = Gated_Conv(in_channels=100, out_channels=100, kernel_size=16)
        # self.conv15 = Gated_Conv(in_channels=100, out_channels=100, kernel_size=17)
        # self.conv16 = Gated_Conv(in_channels=100, out_channels=100, kernel_size=18)
        self.conv17 = Gated_Conv(in_channels=100, out_channels=375, kernel_size=19)      
        self.dropout2 = nn.Dropout(0.25)

        self.linear1 = nn.Linear(in_features=375, out_features=1000)
        self.linear2 = nn.Linear(in_features=1000, out_features=29)
        



    def forward(self, x):
        # 'x' : (batch, n_mels, time)
        x = self.conv1(x)
        x = self.dropout1(x)
        x = self.gated_conv_stack(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)
        # x = self.conv6(x)
        # x = self.conv7(x)
        # x = self.conv8(x)
        # x = self.conv9(x)
        # x = self.conv10(x)
        # x = self.conv11(x)
        # x = self.conv12(x)
        # x = self.conv13(x)
        # x = self.conv14(x)
        # x = self.conv15(x)
        # x = self.conv16(x)
        x = self.conv17(x)
        x = self.dropout2(x)
        x = self.linear1(torch.transpose(x,1,2)) 
        out = self.linear2(x)

        # (batch, time, classes)
        
        return out

