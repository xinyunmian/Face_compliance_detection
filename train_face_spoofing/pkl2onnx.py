import torch
from slim_net import Slim


myNet = Slim()
map_location = lambda storage, loc: storage
myNet.load_state_dict(torch.load("weights/ht_model.pth",map_location=map_location))

myNet.eval()
print('Finished loading model!')
print(myNet)
device = torch.device("cpu")
net = myNet.to(device)
output_onnx = 'weights/ht_model.onnx'
input_names = ["input0"]
output_names = ["output0"]
inputs = torch.randn(1, 3, 128, 128).to(device)
torch_out = torch.onnx._export(net, inputs, output_onnx, export_params=True, verbose=False,
                               input_names=input_names, output_names=output_names)
