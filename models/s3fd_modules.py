import torch
import torch.nn as nn
from itertools import product as product
from s3fd_utils import nms, decode

class L2Norm(nn.Module):

    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    # def reset_parameters(self):
    #     init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out
    

class Detect(object):

    def __init__(self, num_classes=2,
                    top_k=750, nms_thresh=0.3, conf_thresh=0.05,
                    variance=[0.1, 0.2], nms_top_k=5000):
        
        self.num_classes = num_classes
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        self.variance = variance
        self.nms_top_k = nms_top_k

    def forward(self, loc_data, conf_data, prior_data):

        num = loc_data.size(0)
        num_priors = prior_data.size(0)

        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)
        batch_priors = prior_data.view(-1, num_priors, 4).expand(num, num_priors, 4)
        batch_priors = batch_priors.contiguous().view(-1, 4)

        decoded_boxes = decode(loc_data.view(-1, 4), batch_priors, self.variance)
        decoded_boxes = decoded_boxes.view(num, num_priors, 4)

        output = torch.zeros(num, self.num_classes, self.top_k, 5)

        for i in range(num):
            boxes = decoded_boxes[i].clone()
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                
                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(boxes)
                boxes_ = boxes[l_mask].view(-1, 4)
                ids, count = nms(boxes_, scores, self.nms_thresh, self.nms_top_k)
                count = count if count < self.top_k else self.top_k

                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes_[ids[:count]]), 1)

        return output


class PriorBox(object):

    def __init__(self, input_size, feature_maps,
                    variance=[0.1, 0.2],
                    min_sizes=[16, 32, 64, 128, 256, 512],
                    steps=[4, 8, 16, 32, 64, 128],
                    clip=False):

        super(PriorBox, self).__init__()

        self.imh = input_size[0]
        self.imw = input_size[1]
        self.feature_maps = feature_maps

        self.variance = variance
        self.min_sizes = min_sizes
        self.steps = steps
        self.clip = clip

    def forward(self):
        mean = []
        for k, fmap in enumerate(self.feature_maps):
            feath = fmap[0]
            featw = fmap[1]
            for i, j in product(range(feath), range(featw)):
                f_kw = self.imw / self.steps[k]
                f_kh = self.imh / self.steps[k]

                cx = (j + 0.5) / f_kw
                cy = (i + 0.5) / f_kh

                s_kw = self.min_sizes[k] / self.imw
                s_kh = self.min_sizes[k] / self.imh

                mean += [cx, cy, s_kw, s_kh]

        output = torch.FloatTensor(mean).view(-1, 4)
        
        if self.clip:
            output.clamp_(max=1, min=0)
        
        return output

