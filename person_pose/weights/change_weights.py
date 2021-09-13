import torch
from mobilev3_face import mobilev3_retinaface, mobilev3_small, mobilev3Fpn_small
from config import cfg_mnet as cfg
checkpoint = '/home/peter/WorkSpace/Documents/xym/codes/retinaface_mobile/weights/mobilev3Fpn_0810_250.pth'

model = mobilev3Fpn_small(cfg=cfg)
model.load_state_dict(torch.load(checkpoint))
model.eval()

torch.save(model.state_dict(), "mobilev3Fpn_0222_250.pth", _use_new_zipfile_serialization=False)