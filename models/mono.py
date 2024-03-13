import math
import torch.nn as nn
import torch

from s3fd import S3FDNet
from talk import TalkNet

from mono_utils import postprocess_det, preprocess_talk

class Mono(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.talk = TalkNet()
        self.s3fd = S3FDNet()

    def forward(self, X, sizes, v_path, a_path, device):
        Y = []
        for x in X:
            x = x.unsqueeze(0).to(device)
            y = self.s3fd(x)
            Y.append(y)
        
        det_res = postprocess_det(Y, sizes, v_path, a_path)
        audio_X, video_X, length = preprocess_talk(det_res)
        duration_set = set(range(1, 7))
        all_scores = []
        for d in duration_set:
            batchSize = int(math.ceil(length / d))
            scores = []
            for i in range(batchSize):
                inputA = torch.FloatTensor(audio_X[
                        i * d * 100 : (i + 1) * d * 100, :
                    ]).unsqueeze(0).to(device)
                inputV = torch.FloatTensor(video_X[
                        i * d *  25 : (i + 1) * d *  25, :, :
                    ]).unsqueeze(0).to(device)
                with torch.no_grad():
                    score = self.talk(inputA, inputV)
                scores.extend(score)
            all_scores.append(scores)
        return all_scores
        