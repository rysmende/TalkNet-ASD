import io
import json
import base64
import cv2

import os
import numpy as np
from PIL import Image, ImageOps

import torch
from ts.torch_handler.base_handler import BaseHandler

# from mtcnn_utils import postprocess_face
from s3fd_utils import nms_
from fd_utils import track_shot

SCALE = 0.25
THRESHOLD = 0.9
FILE_OUTPUT = 'output.avi'

class S3FDHandler(BaseHandler):

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data and make it inference ready
        img_mean = np.array([104., 117., 123.])\
            [:, np.newaxis, np.newaxis].astype('float32')
        imgs  = []
        sizes = []
        for row in data:
            input_datas = row.get('data') or row.get('body')
        input_datas = json.loads(input_datas)
        video_path = input_datas['video_path']
        
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        n = 1
        while success:
            h, w, _ = image.shape
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            scaled_img = cv2.resize(
                    image, dsize=(0, 0), fx=SCALE, fy=SCALE, 
                    interpolation=cv2.INTER_LINEAR
                )
            
            scaled_img = np.swapaxes(scaled_img, 1, 2)
            scaled_img = np.swapaxes(scaled_img, 1, 0)
            scaled_img = scaled_img[[2, 1, 0], :, :]
            scaled_img = scaled_img.astype('float32')
            scaled_img -= img_mean
            scaled_img = scaled_img[[2, 1, 0], :, :]
            
            imgs.append(torch.from_numpy(scaled_img))
            sizes.append((w, h))

            if n == 50:
                break
            success, image = vidcap.read()
            n += 1

        return [imgs], [sizes]

    def inference(self, X):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        X = X[0]
        Y = []        
        self.model.eval()
        with torch.no_grad():
            for x in X:
                x = x.unsqueeze(0).to(self.device)
                y = self.model(x)
                Y.append(y)
        return [Y]

    def postprocess(self, Y, sizes):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # res = {'id': dict(), 'selfie': dict()}
        Y = Y[0]
        sizes = sizes[0]
        res_bboxes = []
        for frame_n, (y, (w, h)) in enumerate(zip(Y, sizes)):
            detections = y.data
            bboxes = np.empty(shape=(0, 5))

            scale = torch.Tensor([w, h, w, h])

            for i in range(detections.size(1)):
                j = 0
                while detections[0, i, j, 0] > THRESHOLD:
                    score = detections[0, i, j, 0]
                    pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                    bbox = (pt[0], pt[1], pt[2], pt[3], score)
                    bboxes = np.vstack((bboxes, bbox))
                    j += 1
            
            keep = nms_(bboxes, 0.1)
            bboxes = bboxes[keep]
            
            # TODO something with multiple faces
            bbox = bboxes[0]
            
            res_bboxes.append({'frame': frame_n, 'bbox': bbox[:-1]})
            print('!!!!!!!!!!!!!!!!!!!!res_bboxes: ', res_bboxes)
        tracks = track_shot(res_bboxes)
        return [tracks.tolist()]

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        X, sizes = self.preprocess(data)
        Y = self.inference(X)
        return self.postprocess(Y, sizes)