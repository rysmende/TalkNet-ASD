# import io
import json
# import base64
import cv2
import time

import os
import numpy as np
# from PIL import Image, ImageOps
import subprocess

import torch
from ts.torch_handler.base_handler import BaseHandler
from torchvision.io import read_video
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

        if os.path.isfile(VIDEO_TEMP):
            os.remove(VIDEO_TEMP)

        for row in data:
            data = row.get('data') or row.get('body')

        is_dict = True
        if isinstance(data, dict):
            data = data['instances'][0]
        else:
            try:
                data = json.loads(data)
            except:
                is_dict = False
                with open(VIDEO_TEMP, 'wb') as out_file:
                    out_file.write(data)

        if is_dict:
            # Download file
            token = data['token']
            bucket_name = data['bucket_name']
            object_name = data['object_name']
            os.system(
                f'curl -X GET ' +
                f'-H "Authorization: Bearer {token}" -o {VIDEO_TEMP} '
                f'"https://storage.googleapis.com/storage/v1/b/{bucket_name}/o/{object_name}?alt=media"'
            )

        cur_time = time.time()
        command = f'ffmpeg -y -i {VIDEO_TEMP} -qscale:v 2 -threads 10 ' +\
            f'-async 1 -r 25 -vf scale="-2:640" {VIDEO_OUTPUT} -loglevel panic'
        subprocess.call(command, shell=True, stdout=None)
        
        os.remove(VIDEO_TEMP)
    
        command = f'ffmpeg -y -i {VIDEO_OUTPUT} -qscale:a 0 -ac 1 -vn ' +\
            f'-threads 10 -ar 16000 {AUDIO_OUTPUT} -loglevel panic'
        subprocess.call(command, shell=True, stdout=None)
        print('VP: ', time.time() - cur_time)
        # Base64 encode the image to avoid the framework throwing
        # non json encodable errors
        video_path = os.path.join(os.getcwd(), VIDEO_OUTPUT)
        audio_path = os.path.join(os.getcwd(), AUDIO_OUTPUT)
        cur_time = time.time()
        
        data_filename = os.path.abspath(video_path)
        video = read_video(data_filename, end_pts=10, pts_unit="sec")[0].numpy()
        print('IP:', time.time() - cur_time)
        return video, video_path, audio_path

    def inference(self, X, v_path, a_path):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        self.model.eval()
        with torch.no_grad():
            y = self.model(X, v_path, a_path, self.device)
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
        X, v_path, a_path = self.preprocess(data)
        Y = self.inference(X, v_path, a_path)
        if len(Y) == 0:
            return form_response(1)
        res = self.postprocess(Y)
        return form_response(0, res)
    
def form_response(code, res = None):
    # Figure it out
    if code == 1:
        return [{
            'code': code,
            'description': 'No face present',
            'result': 0.00,
        }]
    return [{
        'code': code,
        'description': 'Successful check',
        'result': round((res > 0).mean() * 100, 2)        
    }]

