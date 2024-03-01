import io
import json
import base64
# import cv2

import numpy as np
from PIL import Image, ImageOps

import torch
from ts.torch_handler.base_handler import BaseHandler

# from mtcnn_utils import postprocess_face
from s3fd_utils import nms_

SCALE = 0.25
THRESHOLD = 0.8

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
            # input_datas = json.loads(input_datas)
            # print(input_datas)
            # for i in ['image']:
            full_image = ImageOps.exif_transpose(Image.open(io.BytesIO(
                    # base64.b64decode(
                            input_datas
                            # )
                        )
                ))
            w, h = full_image.size
            new_w, new_h = int(w * SCALE), int(h * SCALE)
            scaled_img = full_image.resize(
                    (new_w, new_h), 
                    resample=Image.Resampling.BILINEAR
                )
            # full_image = np.array(full_image)

            scaled_img = np.swapaxes(scaled_img, 1, 2)
            scaled_img = np.swapaxes(scaled_img, 1, 0)
            scaled_img = scaled_img[[2, 1, 0], :, :]
            scaled_img = scaled_img.astype('float32')
            scaled_img -= img_mean
            scaled_img = scaled_img[[2, 1, 0], :, :]
            
            imgs.append(torch.from_numpy(scaled_img))
            sizes.append((w, h))
        return imgs, sizes

    def inference(self, X):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        ys = []
        # print(self.device)
        self.model.eval().to(self.device)
        # print(self.model)
        with torch.no_grad():
            for x in X:
                x = x.unsqueeze(0).to(self.device)
                y = self.model(x)
                ys.append(y)
        return ys

    def postprocess(self, ys, sizes):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # res = {'id': dict(), 'selfie': dict()}
        y = ys[0]
        w, h = sizes[0]

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
        
        # for boxes_in_image, key in zip(boxes, ['id', 'selfie']):
        # # Get bounding boxes and probabilities.
        #     boxes, probs = postprocess_face(boxes_in_image)
        #     res[key]['boxes'] = boxes.tolist()
        #     res[key]['probs'] = probs.tolist()
        # return [res]
        return [bboxes.tolist()]

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        x, sizes = self.preprocess(data)
        y = self.inference(x)
        return self.postprocess(y, sizes)