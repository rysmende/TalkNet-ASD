import os
import subprocess

import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

from talkNet import talkNet

LINK = '1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea'
pretrained_model = 'pretrain_TalkSet.model'

if os.path.isfile(pretrained_model) == False: # Download the pretrained model
    cmd = "gdown %s -O %s"%(LINK, pretrained_model)
    subprocess.call(cmd, shell=True, stdout=None)

net = talkNet()
net.loadParameters(pretrained_model)
net.eval()

scripted_module = torch.jit.script(net)
optimized_scripted_module = optimize_for_mobile(scripted_module)

# Export full jit version model (not compatible with lite interpreter)
scripted_module.save("talknet_scripted.pt")
# Export lite interpreter version model (compatible with lite interpreter)
scripted_module._save_for_lite_interpreter("talknet_scripted.ptl")
# using optimized lite interpreter model makes inference about 60% faster than the non-optimized lite interpreter model, which is about 6% faster than the non-optimized full jit model
optimized_scripted_module._save_for_lite_interpreter("talknet_scripted_optimized.ptl")