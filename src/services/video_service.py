import os
import time
import subprocess
from typing import Tuple, Optional
from fastapi import UploadFile, HTTPException
import numpy as np
from torchvision.io import read_video

class VideoService:
    def __init__(self):
        self.video_temp = 'input.mp4'
        self.video_output = 'output.mp4'
        self.audio_output = 'output.mp4'

    def process_video(self, data: UploadFile | dict) -> Tuple[np.ndarray, str, str]:
        """
        Process video input and return processed video data
        """
        if data is None:
            raise HTTPException(status_code=400, detail="Input data is required")

        cur_time = time.time()
        self._cleanup_files()
        result_object_name = self._handle_input(data)
        print('DL: ', time.time() - cur_time)

        self._process_with_ffmpeg(result_object_name)
        print('VP: ', time.time() - cur_time)

        return self._read_video_data()

    def _cleanup_files(self):
        """Clean up existing temporary files"""
        for file in [self.video_temp, self.video_output, self.audio_output]:
            if os.path.isfile(file):
                try:
                    os.remove(file)
                except:
                    pass

    def _handle_input(self, data: UploadFile | dict) -> str:
        """Handle different types of input (file upload or GCS request)"""
        if isinstance(data, UploadFile):
            return self._handle_file_upload(data)
        else:
            return self._handle_gcs_request(data)

    def _handle_file_upload(self, file: UploadFile) -> str:
        """Handle file upload input"""
        with open(self.video_temp, 'wb') as out_file:
            content = file.file.read()
            out_file.write(content)
        return self.video_temp

    def _handle_gcs_request(self, data: dict) -> str:
        """Handle Google Cloud Storage request"""
        try:
            if not data.get('instances') or len(data['instances']) == 0:
                raise HTTPException(status_code=400, detail="No instances provided in GCS request")

            instance = data['instances'][0]
            if not all([instance.get('token'), instance.get('bucket_name'), instance.get('object_name')]):
                raise HTTPException(status_code=400, detail="Missing required GCS parameters")

            token = instance['token']
            bucket_name = instance['bucket_name']
            object_name = instance['object_name']

            object_encoded_name = object_name.replace('/', '%2F')
            result_object_name = object_name.split('/')[-1]
            
            curl_cmd = f'curl -X GET -H "Authorization: Bearer {token}" -o {result_object_name} ' + \
                      f'"https://storage.googleapis.com/storage/v1/b/{bucket_name}/o/{object_encoded_name}?alt=media"'
            
            if os.system(curl_cmd) != 0:
                raise HTTPException(status_code=404, detail="Failed to download file from GCS")

            return result_object_name
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid GCS request format: {str(e)}")

    def _process_with_ffmpeg(self, input_file: str):
        """Process video using FFmpeg"""
        cmd = ['ffmpeg', '-y']
        if input_file.lower().endswith('.webm'):
            cmd += ['-fflags', '+genpts']

        cmd += ['-i', input_file]
        cmd += [
            '-vf', 'scale=-2:640',
            '-qscale:v', '2',
            '-async', '1',
            '-r', '25'
        ]

        if input_file.lower().endswith('.webm'):
            cmd += ['-max_muxing_queue_size', '1024']

        cmd += [
            '-qscale:a', '0',
            '-ac', '1',
            '-threads', '10',
            '-ar', '16000'
        ]

        cmd += ['-loglevel', 'panic', self.video_output]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            raise HTTPException(status_code=500, detail="Failed to process video with FFmpeg")

    def _read_video_data(self) -> Tuple[np.ndarray, str, str]:
        """Read processed video data"""
        video_path = os.path.join(os.getcwd(), self.video_output)
        audio_path = os.path.join(os.getcwd(), self.audio_output)
        
        try:
            data_filename = os.path.abspath(video_path)
            video = read_video(data_filename, end_pts=10, pts_unit="sec")[0].numpy()
            return video, video_path, audio_path
        except:
            raise HTTPException(status_code=500, detail="Failed to read video data") 