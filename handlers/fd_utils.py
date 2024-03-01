import numpy as np
from scipy.interpolate import interp1d

# Minimum IOU between consecutive face detections
IOU_THRESHOLD = 0.5
MIN_TRACK = 10
MIN_FAILED_DET = 10
MIN_FACE_SIZE = 1

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
                return bboxesI
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
