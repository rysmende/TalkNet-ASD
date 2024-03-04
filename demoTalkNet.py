import sys, time, os, tqdm, torch, argparse, glob, subprocess, warnings, cv2, \
    pickle, math, python_speech_features

from scipy import signal
from scipy.io import wavfile
from scipy.interpolate import interp1d
from shutil import rmtree # can delete

import numpy as np

from models.s3fd import S3FDNet as S3FD
from talk_net import TalkNet

warnings.filterwarnings("ignore")

# LINK = '1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea'

parser = argparse.ArgumentParser(description = "TalkNet Demo or Columnbia ASD Evaluation")

parser.add_argument('--videoName',         type=str, default="001",  help='Demo video name')
parser.add_argument('--videoFolder',       type=str, default="demo", help='Path for inputs, tmps and outputs')
parser.add_argument('--pretrainModel',     type=str, default="pretrain_TalkSet.model",   help='Path for the pretrained TalkNet model')

parser.add_argument('--nDataLoaderThread', type=int,   default=10,   help='Number of workers')
parser.add_argument('--facedetScale',      type=float, default=0.25, help='Scale factor for face detection, the frames will be scale to 0.25 orig')
parser.add_argument('--minTrack',          type=int,   default=10,   help='Number of min frames for each shot')
parser.add_argument('--numFailedDet',      type=int,   default=10,   help='Number of missed detections allowed before tracking is stopped')
parser.add_argument('--minFaceSize',       type=int,   default=1,    help='Minimum face size in pixels')
parser.add_argument('--cropScale',         type=float, default=0.40, help='Scale bounding box')

parser.add_argument('--start',             type=int, default=0, help='The start time of the video')
parser.add_argument('--duration',          type=int, default=0, help='The duration of the video, when set as 0, will extract the whole video')
parser.add_argument('--colSavePath',       type=str, default="/data08/col", help='Path for inputs, tmps and outputs')
parser.add_argument('--device',            type=str, default='cuda', help='Device')

args = parser.parse_args()

device = torch.device(args.device)

# if os.path.isfile(args.pretrainModel) == False: # Download the pretrained model
#     cmd = "gdown %s -O %s"%(LINK, args.pretrainModel)
#     subprocess.call(cmd, shell=True, stdout=None)

args.videoPath = glob.glob(os.path.join(args.videoFolder, args.videoName + '.*'))[0]
args.savePath = os.path.join(args.videoFolder, args.videoName)

def get_biggest_bbox(bboxes):
    if len(bboxes) == 0:
        return []
    if len(bboxes) == 1:
        return bboxes[0]
    idx = bboxes[:, -1].argmax()
    return bboxes[idx]
    
def inference_video(args):
    # GPU: Face detection, output is the list contains the face location and score in this frame
    DET = S3FD(device=device)
    DET.load_state_dict('model_weights/s3fd.pth')
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
    flist.sort()
    dets = []
    for fidx, fname in enumerate(flist):
        image = cv2.imread(fname)
        imagenp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = DET(imagenp, conf_th=0.9, scales=[args.facedetScale])
        bbox = get_biggest_bbox(bboxes)
        # make sure only to take one face
        dets.append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]}) # dets has the frames info, bbox info, conf info
        sys.stderr.write('%s-%05d; %d dets\r' % (args.videoFilePath, fidx, 1))
    savePath = os.path.join(args.pyworkPath,'faces.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(dets, fil)
    return dets

def bb_intersection_over_union(boxA, boxB):
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

def track_shot(args, frameFaces: list):
    # CPU: Face tracking
    IOU_THR = 0.5     # Minimum IOU between consecutive face detections
    tracks  = []
    # print(frameFaces)
    removed = 0
    while True:
        track = []
        # print('while')
        for face in frameFaces[:]:
            # can be rewritten   
            if track == []:
                track.append(face)
                frameFaces.remove(face)
                removed += 1
            elif face['frame'] - track[-1]['frame'] <= args.numFailedDet:
                iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                if iou > IOU_THR:
                    track.append(face)
                    frameFaces.remove(face)
                    removed += 1
        # if no tracks, then break
        print(removed)
        if len(track) == 0:
            break
        # if length of tracks less than 11, continue loop
        if len(track) <= args.minTrack:
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
            ) > args.minFaceSize:
            tracks.append({'frame':frameI, 'bbox':bboxesI})
    return tracks

def crop_video(args, track, cropFile):
    # CPU: crop the face clips
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg')) # Read the frames
    flist.sort()
    vOut = cv2.VideoWriter(
            cropFile + 't.avi', 
            cv2.VideoWriter_fourcc(*'XVID'), 
            25, 
            (224, 224)
        )# Write video
    dets = {'x': [], 'y': [], 's': []}
    for det in track['bbox']: # Read the tracks
        dets['s'].append(max((det[3] - det[1]), (det[2] - det[0])) / 2) 
        dets['y'].append((det[1] + det[3]) / 2) # crop center x 
        dets['x'].append((det[0] + det[2]) / 2) # crop center y
    dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  # Smooth detections 
    dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
    dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
    for fidx, frame in enumerate(track['frame']):
        cs  = args.cropScale
        bs  = dets['s'][fidx]   # Detection box size
        bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount 
        image = cv2.imread(flist[frame])
        frame = np.pad(
                image, 
                ((bsi, bsi), (bsi, bsi), (0, 0)),
                'constant', 
                constant_values=(110, 110)
            )
        my  = dets['y'][fidx] + bsi  # BBox center Y
        mx  = dets['x'][fidx] + bsi  # BBox center X
        face = frame[
                int(my - bs)            : int(my + bs * (1 + 2 * cs)),
                int(mx - bs * (1 + cs)) : int(mx + bs * (1 + cs))
            ]
        vOut.write(cv2.resize(face, (224, 224)))
    audioTmp    = cropFile + '.wav'
    audioStart  = (track['frame'][0]) / 25
    audioEnd    = (track['frame'][-1]+1) / 25
    vOut.release()
    command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" % \
              (args.audioFilePath, args.nDataLoaderThread, audioStart, audioEnd, audioTmp)) 
    output = subprocess.call(command, shell=True, stdout=None) # Crop audio file
    _, audio = wavfile.read(audioTmp)
    command = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" % \
              (cropFile, audioTmp, args.nDataLoaderThread, cropFile)) # Combine audio and video file
    output = subprocess.call(command, shell=True, stdout=None)
    os.remove(cropFile + 't.avi')
    return {'track':track, 'proc_track':dets}

def evaluate_network(files, args):
    # GPU: active speaker detection by pretrained TalkNet
    s = TalkNet()
    s.load_state_dict(torch.load('talk_net.pt', map_location=device))
    s.to(device).eval()
    sys.stderr.write("Model %s loaded from previous state! \r\n"%args.pretrainModel)
    allScores = []
    durationSet = {1, 2, 3, 4, 5, 6} # Use this line can get more reliable result
    for file in tqdm.tqdm(files, total = len(files)):
        fileName = os.path.splitext(file.split('/')[-1])[0] # Load audio and video
        _, audio = wavfile.read(os.path.join(args.pycropPath, fileName + '.wav'))
        audioFeature = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025, winstep = 0.010)
        video = cv2.VideoCapture(os.path.join(args.pycropPath, fileName + '.avi'))
        videoFeature = []
        while video.isOpened():
            ret, frames = video.read()
            if not ret:
                break
            face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (224, 224))
            face = face[224 // 4 : 224 * 3 // 4, 224 // 4 : 224 * 3 // 4]
            videoFeature.append(face)
        video.release()
        videoFeature = np.array(videoFeature)
        length = min(
                (audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100,
                videoFeature.shape[0] / 25
            )
        audioFeature = audioFeature[:int(round(length * 100)), :]
        videoFeature = videoFeature[:int(round(length * 25)), :, :]
        allScore = [] # Evaluation use TalkNet
        for duration in durationSet:
            batchSize = int(math.ceil(length / duration))
            scores = []
            with torch.no_grad():
                for i in range(batchSize):
                    inputA = torch.FloatTensor(audioFeature[
                            i * duration * 100 : (i + 1) * duration * 100, :
                        ]).unsqueeze(0).to(device)
                    inputV = torch.FloatTensor(videoFeature[
                            i * duration *  25 : (i + 1) * duration *  25, :, :
                        ]).unsqueeze(0).to(device)
                    
                    # forward
                    score = s(inputA, inputV)
                    # embedA = s.model.forward_audio_frontend(inputA)
                    # embedV = s.model.forward_visual_frontend(inputV)    
                    # embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
                    # out = s.model.forward_audio_visual_backend(embedA, embedV)
                    # score = s.lossAV.forward(out)
                    
                    scores.extend(score)
            allScore.append(scores)
        allScore = np.round((np.mean(np.array(allScore), axis = 0)), 1).astype(float)
        allScores.append(allScore)    
    return allScores

def visualization(tracks, scores, args):
    # CPU: visulize the result for video format
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
    flist.sort()
    faces = [[] for _ in range(len(flist))]
    for tidx, track in enumerate(tracks):
        score = scores[tidx]
        for fidx, frame in enumerate(track['track']['frame'].tolist()):
            s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)] # average smoothing
            s = np.mean(s)
            faces[frame].append({'track':tidx, 'score':float(s),'s':track['proc_track']['s'][fidx], 'x':track['proc_track']['x'][fidx], 'y':track['proc_track']['y'][fidx]})
    firstImage = cv2.imread(flist[0])
    fw = firstImage.shape[1]
    fh = firstImage.shape[0]
    vOut = cv2.VideoWriter(os.path.join(args.pyaviPath, 'video_only.avi'), cv2.VideoWriter_fourcc(*'XVID'), 25, (fw,fh))
    colorDict = {0: 0, 1: 255}
    for fidx, fname in tqdm.tqdm(enumerate(flist), total = len(flist)):
        image = cv2.imread(fname)
        for face in faces[fidx]:
            clr = colorDict[int((face['score'] >= 0))]
            txt = round(face['score'], 1)
            cv2.rectangle(image, (int(face['x']-face['s']), int(face['y']-face['s'])), (int(face['x']+face['s']), int(face['y']+face['s'])),(0,clr,255-clr),10)
            cv2.putText(image,'%s'%(txt), (int(face['x']-face['s']), int(face['y']-face['s'])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,clr,255-clr),5)
        vOut.write(image)
    vOut.release()
    command = ("ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" % \
        (os.path.join(args.pyaviPath, 'video_only.avi'), os.path.join(args.pyaviPath, 'audio.wav'), \
        args.nDataLoaderThread, os.path.join(args.pyaviPath,'video_out.avi'))) 
    output = subprocess.call(command, shell=True, stdout=None)

# Main function
def main():
    # This preprocesstion is modified based on this [repository](https://github.com/joonson/syncnet_python).
    # ```
    # .
    # ├── pyavi
    # │   ├── audio.wav (Audio from input video)
    # │   ├── video.avi (Copy of the input video)
    # │   ├── video_only.avi (Output video without audio)
    # │   └── video_out.avi  (Output video with audio)
    # ├── pycrop (The detected face videos and audios)
    # │   ├── 000000.avi
    # │   ├── 000000.wav
    # │   ├── 000001.avi
    # │   ├── 000001.wav
    # │   └── ...
    # ├── pyframes (All the video frames in this video)
    # │   ├── 000001.jpg
    # │   ├── 000002.jpg
    # │   └── ...    
    # └── pywork
    #     ├── faces.pckl (face detection result)
    #     ├── scene.pckl (scene detection result)
    #     ├── scores.pckl (ASD result)
    #     └── tracks.pckl (face tracking result)
    # ```

    # Initialization 
    args.pyaviPath    = os.path.join(args.savePath, 'pyavi')
    args.pyframesPath = os.path.join(args.savePath, 'pyframes')
    args.pyworkPath   = os.path.join(args.savePath, 'pywork')
    args.pycropPath   = os.path.join(args.savePath, 'pycrop')

    if os.path.exists(args.savePath):
        rmtree(args.savePath)
    
    os.makedirs(args.pyaviPath,    exist_ok = True) # The path for the input video, input audio, output video
    os.makedirs(args.pyframesPath, exist_ok = True) # Save all the video frames
    os.makedirs(args.pyworkPath,   exist_ok = True) # Save the results in this process by the pckl method
    os.makedirs(args.pycropPath,   exist_ok = True) # Save the detected face clips (audio+video) in this process

    # Extract video
    args.videoFilePath = os.path.join(args.pyaviPath, 'video.avi')
    
    # Extract the whole video
    command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -async 1 -r 25 %s -loglevel panic" % \
        (args.videoPath, args.nDataLoaderThread, args.videoFilePath))
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + \
                     " Extract the video and save in %s \r\n" %(args.videoFilePath))
    
    # Extract audio
    args.audioFilePath = os.path.join(args.pyaviPath, 'audio.wav')
    command = ("ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic" % \
        (args.videoFilePath, args.nDataLoaderThread, args.audioFilePath))
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + \
                     " Extract the audio and save in %s \r\n" %(args.audioFilePath))

    # Extract the video frames
    command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -f image2 %s -loglevel panic" % \
        (args.videoFilePath, args.nDataLoaderThread, os.path.join(args.pyframesPath, '%06d.jpg'))) 
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + \
                     " Extract the frames and save in %s \r\n" %(args.pyframesPath))

    # Face detection for the video frames
    faces = inference_video(args)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + \
                     " Face detection and save in %s \r\n" %(args.pyworkPath))

    # Face tracking
    allTracks, vidTracks = [], []
    allTracks.extend(track_shot(args, faces)) # 'frames' to present this tracks' timestep, 'bbox' presents the location of the faces
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + \
                     " Face track and detected %d tracks \r\n" %len(allTracks))

    # Face clips cropping
    for ii, track in tqdm.tqdm(enumerate(allTracks), total = len(allTracks)):
        vidTracks.append(crop_video(args, track, os.path.join(args.pycropPath, '%05d'%ii)))
    
    # -----------------------------------------------------
    # This part only for visualization
    savePath = os.path.join(args.pyworkPath, 'tracks.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(vidTracks, fil)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + \
                     " Face Crop and saved in %s tracks \r\n" %args.pycropPath)
    fil = open(savePath, 'rb')
    vidTracks = pickle.load(fil)
    # -----------------------------------------------------

    # Active Speaker Detection by TalkNet
    files = glob.glob("%s/*.avi"%args.pycropPath)
    files.sort()
    scores = evaluate_network(files, args)
    savePath = os.path.join(args.pyworkPath, 'scores.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(scores, fil)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scores extracted and saved in %s \r\n" %args.pyworkPath)

    # Visualization, save the result as the new video    
    visualization(vidTracks, scores, args)    

if __name__ == '__main__':
    main()
