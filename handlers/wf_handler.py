import os
import cv2
import base64
import subprocess

VIDEO_TEMP   = 'temp.avi'
VIDEO_OUTPUT = 'output.avi'
AUDIO_OUTPUT = 'output.wav'

def pre_processing(data, context):
    '''
    Empty node as a starting node since the DAG doesn't support multiple start nodes
    '''
    
    if data is None:
        return data
    
    b64_data = []

    for row in data:
        data = row.get('data')

        if os.path.isfile(VIDEO_TEMP):
            os.remove(VIDEO_TEMP)

        with open(VIDEO_TEMP, 'wb') as out_file:
            res = out_file.write(data)
        
        command = f'ffmpeg -y -i {VIDEO_TEMP} -qscale:v 2 -threads 10 ' +\
            f'-async 1 -r 25 {VIDEO_OUTPUT} -loglevel panic'
        subprocess.call(command, shell=True, stdout=None)
        
        os.remove(VIDEO_TEMP)
    

        command = f'ffmpeg -y -i {VIDEO_OUTPUT} -qscale:a 0 -ac 1 -vn ' +\
            f'-threads 10 -ar 16000 {AUDIO_OUTPUT} -loglevel panic'
        subprocess.call(command, shell=True, stdout=None)

        # Base64 encode the image to avoid the framework throwing
        # non json encodable errors
        
        b64_data.append({
            'video_path': os.path.join(os.getcwd(), VIDEO_OUTPUT),
            'audio_path': os.path.join(os.getcwd(), AUDIO_OUTPUT)
        })
    return b64_data
