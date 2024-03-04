from scipy import signal
from scipy.io import wavfile
import numpy as np
import os
import cv2
import subprocess
from scipy.interpolate import interp1d

# Minimum IOU between consecutive face detections
IOU_THRESHOLD = 0.5
MIN_TRACK = 10
MIN_FAILED_DET = 10
MIN_FACE_SIZE = 1
CROP_SCALE = 0.4
# TEMP_NAME = 'temp'

def track_shot(frameFaces: list):
        # CPU: Face tracking
        # tracks  = []
        while True:
            track = []
            for face in frameFaces[:]:
                if track == []:
                    track.append(face)
                    frameFaces.remove(face)
                elif face['frame'] - track[-1]['frame'] <= MIN_FAILED_DET:
                    iou = __bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                    if iou > IOU_THRESHOLD:
                        track.append(face)
                        frameFaces.remove(face)
            
            # if no tracks, then break
            if len(track) == 0:
                break
            # if length of tracks less than 11, continue loop
            if len(track) <= MIN_TRACK:
                continue
            
            frameNum = np.array([f['frame'] for f in track])
            bboxes   = np.array([np.array(f['bbox']) for f in track])
            frameI   = np.arange(frameNum[0], frameNum[-1] + 1)
            bboxesI  = []
            for ij in range(4):
                interpfn = interp1d(frameNum, bboxes[:,ij])
                bboxesI.append(interpfn(frameI))
            bboxesI  = np.stack(bboxesI, axis=1)
            if max(
                    np.mean(bboxesI[:,2] - bboxesI[:,0]), 
                    np.mean(bboxesI[:,3] - bboxesI[:,1])
                ) > MIN_FACE_SIZE:
                # only first track, can be modified
                return {'frame': frameI, 'bbox': bboxesI}
        # return tracks


def __bb_intersection_over_union(boxA, boxB):
    # CPU: IOU Function to calculate overlap between two image
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def crop_video(track, video_path, audio_path):
    # CPU: crop the face clips

    dets = {'x': [], 'y': [], 's': []}
    for det in track['bbox']: # Read the tracks
        dets['s'].append(max((det[3] - det[1]), (det[2] - det[0])) / 2) 
        dets['y'].append((det[1] + det[3]) / 2) # crop center x 
        dets['x'].append((det[0] + det[2]) / 2) # crop center y
    dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  # Smooth detections 
    dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
    dets['y'] = signal.medfilt(dets['y'], kernel_size=13)

    vidcap = cv2.VideoCapture(video_path)
    frames = []
    counter = 1
    ret, frame = vidcap.read()
    while ret:
        frames.append(frame)
        if counter == 25:
            break
        ret, frame = vidcap.read()
        counter += 1
    vidcap.release()
    
    vOut = cv2.VideoWriter(
            'temp.avi', 
            cv2.VideoWriter_fourcc(*'XVID'), 
            25, 
            (224, 224)
        )# Write video
    
    for fidx, frame in enumerate(track['frame']):
        bs  = dets['s'][fidx]   # Detection box size
        bsi = int(bs * (1 + 2 * CROP_SCALE))  # Pad videos by this amount 
        image = frames[frame]
        image = np.pad(
                image, 
                ((bsi, bsi), (bsi, bsi), (0, 0)),
                'constant', 
                constant_values=(110, 110)
            )
        my  = dets['y'][fidx] + bsi  # BBox center Y
        mx  = dets['x'][fidx] + bsi  # BBox center X
        face = image[
                int(my - bs)                    : int(my + bs * (1 + 2 * CROP_SCALE)),
                int(mx - bs * (1 + CROP_SCALE)) : int(mx + bs * (1 + CROP_SCALE))
            ]
        vOut.write(cv2.resize(face, (224, 224)))
    vOut.release()
    
    video_out  = 'sync.avi' 
    audio_out  = 'sync.wav'
    audioStart = (track['frame'][0]) / 25
    audioEnd   = (track['frame'][-1]+1) / 25
    
    command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads 10 -ss %.3f -to %.3f %s -loglevel panic" % \
              (audio_path, audioStart, audioEnd, audio_out)) 
    output = subprocess.call(command, shell=True, stdout=None) # Crop audio file

    command = ("ffmpeg -y -i temp.avi -i %s -threads 10 -c:v copy -c:a copy %s -loglevel panic" % \
              (audio_out, video_out)) # Combine audio and video file
    output = subprocess.call(command, shell=True, stdout=None)
    
    os.remove('temp.avi')
    video_path = os.path.join(os.getcwd(), video_out)
    audio_path = os.path.join(os.getcwd(), audio_out)
    return {
            # 'track': track, 'proc_track': dets, 
            'video_path': video_path, 'audio_path': audio_path
        }