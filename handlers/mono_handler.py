import io
import json
import base64
import cv2

import os
import numpy as np
from PIL import Image, ImageOps
import subprocess

import torch
from ts.torch_handler.base_handler import BaseHandler

# from mtcnn_utils import postprocess_face
# from s3fd_utils import nms_
# from fd_utils import track_shot, crop_video

SCALE = 0.25
VIDEO_TEMP   = 'temp.avi'
VIDEO_OUTPUT = 'output.avi'
AUDIO_OUTPUT = 'output.wav'

class MONOHandler(BaseHandler):

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        
        if data is None:
            return data

        for row in data:
            data = row.get('data') or row.get('body')
        data = json.loads(data)

        # Download file
        token = data['token']
        bucket_name = data['bucket_name']
        object_name = data['object_name']

        if os.path.isfile(VIDEO_TEMP):
            os.remove(VIDEO_TEMP)

        os.system(
            f'curl -X GET ' +
            f'-H "Authorization: Bearer {token}" -o {VIDEO_TEMP} '
            f'"https://storage.googleapis.com/storage/v1/b/{bucket_name}/o/{object_name}?alt=media"'
        )

        # with open(VIDEO_TEMP, 'wb') as out_file:
        #     out_file.write(data)
        
        command = f'ffmpeg -y -i {VIDEO_TEMP} -qscale:v 2 -threads 10 ' +\
            f'-async 1 -r 25 {VIDEO_OUTPUT} -loglevel panic'
        subprocess.call(command, shell=True, stdout=None)
        
        os.remove(VIDEO_TEMP)
    
        command = f'ffmpeg -y -i {VIDEO_OUTPUT} -qscale:a 0 -ac 1 -vn ' +\
            f'-threads 10 -ar 16000 {AUDIO_OUTPUT} -loglevel panic'
        subprocess.call(command, shell=True, stdout=None)

        # Base64 encode the image to avoid the framework throwing
        # non json encodable errors
        video_path = os.path.join(os.getcwd(), VIDEO_OUTPUT)
        audio_path = os.path.join(os.getcwd(), AUDIO_OUTPUT)
        
        # Take the input data and make it inference ready
        img_mean = np.array([104., 117., 123.])\
            [:, np.newaxis, np.newaxis].astype('float32')
        imgs  = []
        sizes = []
        
        vidcap = cv2.VideoCapture(video_path)
        ret, image = vidcap.read()
        n = 1
        while ret:
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

            # Restraint. Can be deleted
            if n == 25:
                break

            ret, image = vidcap.read()
            n += 1
        vidcap.release()

        return imgs, sizes, video_path, audio_path

    def inference(self, X, sizes, v_path, a_path):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        self.model.eval()
        with torch.no_grad():
            y = self.model(X, sizes, v_path, a_path, self.device)
        return y

    def postprocess(self, y):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        return np.round((np.mean(np.array(y), axis = 0)), 1).astype(float)

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        X, sizes, v_path, a_path = self.preprocess(data)
        Y = self.inference(X, sizes, v_path, a_path)
        res = self.postprocess(Y)
        return form_response(res)
    
def form_response(res):
    # Figure it out
    return [res.tolist()]

