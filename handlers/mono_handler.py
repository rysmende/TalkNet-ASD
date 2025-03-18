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

# Error codes
SUCCESS = 0
ERROR_INVALID_INPUT = 1
ERROR_FILE_DOWNLOAD = 2
ERROR_FFMPEG = 3
ERROR_VIDEO_READ = 4
ERROR_MODEL_NOT_INITIALIZED = 5
ERROR_INFERENCE = 6
ERROR_POSTPROCESS = 7

class MONOHandler(BaseHandler):

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: tuple of (error_code, video, video_path, audio_path)
        """
        if data is None:
            return ERROR_INVALID_INPUT, None, None, None

        cur_time = time.time()

        # Clean up existing files
        for file in [VIDEO_TEMP, VIDEO_OUTPUT, AUDIO_OUTPUT]:
            if os.path.isfile(file):
                try:
                    os.remove(file)
                except:
                    pass  # Ignore file removal errors

        # Process input data
        for row in data:
            data = row.get('data') or row.get('body')
            if data is not None:
                break

        if data is None:
            return ERROR_INVALID_INPUT, None, None, None

        is_dict = True
        if isinstance(data, dict):
            data = data['instances'][0]
        else:
            try:
                data = json.loads(data)
            except:
                is_dict = False
                result_object_name = VIDEO_TEMP
                try:
                    with open(VIDEO_TEMP, 'wb') as out_file:
                        out_file.write(data)
                except:
                    return ERROR_INVALID_INPUT, None, None, None

        if is_dict:
            token = data.get('token')
            bucket_name = data.get('bucket_name')
            object_name = data.get('object_name')
            
            if not all([token, bucket_name, object_name]):
                return ERROR_INVALID_INPUT, None, None, None

            object_encoded_name = object_name.replace('/', '%2F')
            result_object_name = object_name.split('/')[-1]
            
            curl_cmd = f'curl -X GET -H "Authorization: Bearer {token}" -o {result_object_name} ' + \
                      f'"https://storage.googleapis.com/storage/v1/b/{bucket_name}/o/{object_encoded_name}?alt=media"'
            
            if os.system(curl_cmd) != 0:
                return ERROR_FILE_DOWNLOAD, None, None, None

        print('DL: ', time.time() - cur_time)

        # Process video with ffmpeg
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

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            return ERROR_FFMPEG, None, None, None

        print('VP: ', time.time() - cur_time)

        # Read video data
        video_path = os.path.join(os.getcwd(), VIDEO_OUTPUT)
        audio_path = os.path.join(os.getcwd(), AUDIO_OUTPUT)
        
        if not os.path.exists(video_path):
            return ERROR_VIDEO_READ, None, None, None

        cur_time = time.time()
        try:
            data_filename = os.path.abspath(video_path)
            video = read_video(data_filename, end_pts=10, pts_unit="sec")[0].numpy()
            print('IP:', time.time() - cur_time)
            return SUCCESS, video, video_path, audio_path
        except:
            return ERROR_VIDEO_READ, None, None, None

    def inference(self, X, v_path, a_path):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: tuple of (error_code, output)
        """
        if not self.initialized:
            return ERROR_MODEL_NOT_INITIALIZED, None

        if X is None or v_path is None or a_path is None:
            return ERROR_INVALID_INPUT, None

        try:
            self.model.eval()
            with torch.no_grad():
                y = self.model(X, v_path, a_path, self.device)
            return SUCCESS, y
        except:
            return ERROR_INFERENCE, None

    def postprocess(self, y):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: tuple of (error_code, result)
        """
        try:
            if not isinstance(y, (list, np.ndarray)) or len(y) == 0:
                return ERROR_POSTPROCESS, np.array([0.0])
            result = np.round((np.mean(np.array(y), axis=0)), 1).astype(float)
            return SUCCESS, result
        except:
            return ERROR_POSTPROCESS, np.array([0.0])

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        error_code, X, v_path, a_path = self.preprocess(data)
        if error_code != SUCCESS:
            return form_response(code=2, result=0.0, error_code=error_code)

        error_code, Y = self.inference(X, v_path, a_path)
        if error_code != SUCCESS:
            return form_response(code=2, result=0.0, error_code=error_code)

        if len(Y) == 0:
            return form_response(code=2, result=0.0, error_code=ERROR_INFERENCE)

        error_code, res = self.postprocess(Y)
        if error_code != SUCCESS:
            return form_response(code=2, result=0.0, error_code=error_code)

        response = form_response(result=res)
        print(response)
        return response

def form_response(code=None, result=0.0, error_code=SUCCESS):
    descriptions = [
        'Successful check',
        'Person is not speaking',
        'No face present',
    ]
    
    error_descriptions = {
        SUCCESS: "Success",
        ERROR_INVALID_INPUT: "Invalid input data",
        ERROR_FILE_DOWNLOAD: "Failed to download file",
        ERROR_FFMPEG: "Failed to process video",
        ERROR_VIDEO_READ: "Failed to read video",
        ERROR_MODEL_NOT_INITIALIZED: "Model not initialized",
        ERROR_INFERENCE: "Inference failed",
        ERROR_POSTPROCESS: "Post-processing failed"
    }

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
        'score': score,
        'error_code': error_code,
        'error_description': error_descriptions.get(error_code, "Unknown error")
    }]

