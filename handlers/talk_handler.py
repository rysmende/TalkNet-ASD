import io
import json
import base64
import cv2
import math
import python_speech_features

import numpy as np
from scipy.io import wavfile
from PIL import Image, ImageOps

import torch
from ts.torch_handler.base_handler import BaseHandler

# from mtcnn_utils import postprocess_face
# from s3fd_utils import nms_
import os

SCALE = 0.25
THRESHOLD = 0.8

class TalkHandler(BaseHandler):

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
        FILE_OUTPUT = 'output.avi'


        for row in data:
            input_datas = row.get('data') or row.get('body')
        input_datas = json.loads(input_datas)
        video_path = input_datas['video_path']
        audio_path = input_datas['audio_path']
        
        _, audio = wavfile.read(audio_path)
        audioFeature = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025, winstep = 0.010)
        
        videoFeature = []
        vidcap = cv2.VideoCapture(video_path)
        while vidcap.isOpened():
            ret, frames = vidcap.read()
            if not ret:
                break
            face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (224, 224))
            face = face[224 // 4 : 224 * 3 // 4, 224 // 4 : 224 * 3 // 4]
            videoFeature.append(face)
        vidcap.release()
        videoFeature = np.array(videoFeature)
        length = min(
                (audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100,
                videoFeature.shape[0] / 25
            )
        audioFeature = audioFeature[:int(round(length * 100)), :]
        videoFeature = videoFeature[:int(round(length * 25)), :, :]
        return audioFeature, videoFeature, length

    def inference(self, audioFeature, videoFeature, length):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        duration_set = set(range(1, 7))
        all_scores = []
        for d in duration_set:
            batchSize = int(math.ceil(length / d))
            scores = []
            for i in range(batchSize):
                inputA = torch.FloatTensor(audioFeature[
                        i * d * 100 : (i + 1) * d * 100, :
                    ]).unsqueeze(0).to(self.device)
                inputV = torch.FloatTensor(videoFeature[
                        i * d *  25 : (i + 1) * d *  25, :, :
                    ]).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    score = self.model(inputA, inputV)
                scores.extend(score)
            all_scores.append(scores)
        return all_scores
        
    def postprocess(self, y):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        all_score = np.round((np.mean(np.array(y), axis = 0)), 1).astype(float)
        return [all_score.tolist()]


    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        aX, vX, length = self.preprocess(data)
        y = self.inference(aX, vX, length)
        return self.postprocess(y)