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
VIDEO_TEMP   = 'input.mp4'
VIDEO_OUTPUT = 'output.mp4'
AUDIO_OUTPUT = 'output.mp4'

class MONOHandler(BaseHandler):

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        
        if data is None:
            return data
        cur_time = time.time()

        if os.path.isfile(VIDEO_TEMP):
            os.remove(VIDEO_TEMP)
        
        if os.path.isfile(VIDEO_OUTPUT):
            os.remove(VIDEO_OUTPUT)

        if os.path.isfile(AUDIO_OUTPUT):
            os.remove(AUDIO_OUTPUT)

        for row in data:
            data = row.get('data') or row.get('body')

        is_dict = True
        if isinstance(data, dict):
            data = data['instances'][0]
        else:
            try:
                data = json.loads(data)
            except:
                # USED WHEN BENCHMARKING (SENDING VIDEO DIRECTLY)
                is_dict = False
                result_object_name = VIDEO_TEMP
                with open(VIDEO_TEMP, 'wb') as out_file:
                    out_file.write(data)

        if is_dict:
            # Download file
            token = data['token']
            bucket_name = data['bucket_name']
            object_name = data['object_name']
            object_encoded_name = object_name.replace('/', '%2F')
            result_object_name = object_name.split('/')[-1]
            print(data)

            os.system(
                f'curl -X GET ' +
                f'-H "Authorization: Bearer {token}" -o {result_object_name} '
                f'"https://storage.googleapis.com/storage/v1/b/{bucket_name}/o/{object_encoded_name}?alt=media"'
            )
        print('DL: ', time.time() - cur_time)
        
        # cur_time = time.time()

        # if result_object_name[-5:] == '.webm':
        #     command = f'ffmpeg -y -fflags +genpts -i {result_object_name} -qscale:v 2 ' +\
        #         f'-max_muxing_queue_size 1024 -async 1 -r 25 -vf scale="-2:640" {VIDEO_OUTPUT}'
        #     os.system(command)
        # else:
        #     command = f'ffmpeg -y -i {result_object_name} -qscale:v 2 ' +\
        #         f'-async 1 -r 25 -vf scale="-2:640" {VIDEO_OUTPUT}'
        #     os.system(command)

        # command = f'ffmpeg -y -i {VIDEO_OUTPUT} -qscale:a 0 -ac 1 -vn ' +\
        #     f'-threads 10 -ar 16000 {VIDEO_OUTPUT} -loglevel panic'
        # os.system(command)

        cmd = ['ffmpeg', '-y']

        if result_object_name.lower().endswith('.webm'):
            cmd += ['-fflags', '+genpts']

        cmd += ['-i', result_object_name]

        cmd += [
            '-vf', 'scale=-2:640',
            '-qscale:v', '2',
            '-async', '1',
            '-r', '25'
        ]

        if result_object_name.lower().endswith('.webm'):
            cmd += ['-max_muxing_queue_size', '1024']

        cmd += [
            '-qscale:a', '0',
            '-ac', '1',
            '-threads', '10',
            '-ar', '16000'
        ]

        cmd += ['-loglevel', 'panic', VIDEO_OUTPUT]

        subprocess.run(cmd, check=True)

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
            return form_response(code=2)
        res = self.postprocess(Y)
        response = form_response(result=res)
        print(response)
        return response
    
def form_response(code = None, result = 0.0):
    # Figure it out
    descriptions = [
        'Successful check',
        'Person is not speaking',
        'No face present',
    ]
    if code != 2:
        result = round((result > 0).mean() * 100, 2)
        code = int(result < 4.0)

    if result < 10:
        score = round((result / 10) * 50, 2)
    else:
        score = round(50 + (result - 10) * 5 / 9, 2)

    return [{
        'code': code,
        'description': descriptions[code],
        'result': result,
        'score': score
    }]

