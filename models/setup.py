import torch
from mono import Mono

# refd_sd = torch.load('../model_weights/s3fd.pth', map_location='cpu')
talk_sd = torch.load('../model_weights/talk.pth', map_location='cpu')

model = Mono()

# model.s3fd.load_state_dict(s3fd_sd)
model.talk.load_state_dict(talk_sd)

torch.save(model.state_dict(), '../model_weights/mono.pth')