import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import torch.nn as nn
import torch.nn.init as init
import sys

n_inputs = 16
n_width = 16
n_outputs = 16

class MyNet(nn.Module):
    def __init__(self, num_layers, num_width, inplace=True):
        super(MyNet, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            nn.Linear(n_inputs, num_width),
            nn.Linear(num_width, n_outputs),
            nn.ReLU(inplace=inplace)
            ])
        if num_layers > 1:
            for i in range(num_layers-1):
                self.layers.append(nn.Linear(n_inputs, num_width))
                self.layers.append(nn.Linear(num_width, n_outputs))
                self.layers.append(nn.ReLU(inplace=inplace))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

if __name__ == "__main__":
    num_layers = 1
    num_width = n_width
    if len(sys.argv) > 1:
        num_layers = int(sys.argv[1])
    if len(sys.argv) > 2:
        num_width = int(sys.argv[2])

    num_mac = num_layers * (n_inputs * num_width + num_width * n_outputs)
    model_name = "fc"+str(num_mac)+".onnx"
    print("Tensor size:  "+str(num_width/1024)+"K")
    print("Estimated total compute:  "+str(num_mac/1000000)+" MMAC")

    # Create the model by using the above model definition.
    model = MyNet(num_layers, num_width)

    # set the model to inference mode
    model.eval()

    # Input to the model
    x = torch.randn(n_inputs, requires_grad=True)
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
