import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import torch.nn as nn
import torch.nn.init as init
import sys

N = 1   # batch size
C = 8  # number of channels in
H = 24 # height in
W = 32 # width in
KH = 5 # kernel height
KW = 5 # kernel width
SH = 1 # vertical stride
SW = 1 # horizontal stride
PH = 2 # vertical padding
PW = 2 # horizontal padding
Co = 8 # number of channels out

class MyNet(nn.Module):
    def __init__(self, num_layers, inplace=True):
        super(MyNet, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            nn.Conv2d(C, Co, (KH, KW), (SH, SW), (PH, PW)),
            nn.ReLU(inplace=inplace)
            ])
        if num_layers > 1:
            for i in range(num_layers-1):
                self.layers.append(nn.Conv2d(C, Co, (KH, KW), (SH, SW), (PH, PW)))
                self.layers.append(nn.ReLU(inplace=inplace))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

if __name__ == "__main__":
    num_layers = 1
    if len(sys.argv) > 1:
        num_layers = int(sys.argv[1])

    num_mac = num_layers * Co * C * KH * KW * (H+PH-KH+1) * (W+PW-KW+1)
    model_name = "cnn"+str(num_mac)+".onnx"
    print("Tensor size:  "+str(N*C*H*W/1024)+"K")
    print("Estimated total compute:  "+str(num_mac/1000000)+" MMAC")

    # Create the model by using the above model definition.
    model = MyNet(num_layers)

    # set the model to inference mode
    model.eval()

    # Input to the model
    x = torch.randn((N,C,H,W), requires_grad=True)
    torch_out = model(x)

    # Export the model
    torch.onnx.export(model,       # model being run
        x,                         # model input (or a tuple for multiple inputs)
        model_name,                # where to save the model
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=11,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names = ['input'],   # the model's input names
        output_names = ['output'], # the model's output names
        )
