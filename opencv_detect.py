import numpy as np
import cv2
import time
import glob
import pathlib

pathlib.Path('./ref').mkdir(parents=True, exist_ok=True) 

CLASSES = [ "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", 
"cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep","sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)

threshold = 0.8
prototxt = './model/MobileNetSSD_deploy.prototxt'
model = './model/MobileNetSSD_deploy.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt, model)

srcvideo = 'video/test_dc_f20.265'
cap = cv2.VideoCapture(srcvideo)
if not cap.isOpened():
    print("ERROR: Cannot open input video file %s" % srcvideo)
    exit()

meta_data = []
frame_count, obj_count, obj_index = 0, 0, 0
encode_frame = 1

while True:
    ret, frame = cap.read()
    if ret == False:
        break;
    need_dump = False
    meta_data.append('frame# %d\n' % frame_count)
    (srch, srcw, _) = frame.shape
    scale = srcw/1280.0
    w2, h2 = (1280, int(srch/scale))
    frame2 = frame.copy() #cv2.resize(frame, (w2, h2), cv2.INTER_AREA)
    (h, w) = frame2.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame2, (300, 300)), 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()
    obj_count = 0
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > threshold:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            # draw mask
            cv2.rectangle(frame2, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame2, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, 2)
            meta_str = '    %s [%04d, %04d, %04d, %04d], prob = %.2f; \n' % (label, startX, startY, endX, endY, confidence * 100)
            # object classification
            crop_img = frame[startY:endY, startX:endX]
            #cv2.imwrite('./crop/roi_%02d.bmp' % (obj_index), crop_img)
            meta_data.append(meta_str)
            obj_count += 1
            obj_index += 1
    outimgfile = './ref/ref_' + str(frame_count).zfill(4) + '.png'
    frame_count += 1
    cv2.imshow("Frame", cv2.resize(frame2, (w2, h2), cv2.INTER_AREA))
    if need_dump:
        cv2.imwrite(outimgfile, frame2)
        #cv2.imwrite('encode/%04d.bmp'%encode_frame, frame)
        encode_frame += 1

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

meta_file = 'ref/opencv_meta_d.txt'
with open(meta_file, 'wt') as f:
    f.writelines(meta_data)
    print('%d objects found in %d frames, ref meta data is written to %s' % (obj_index, frame_count, meta_file))

cap.release()
cv2.destroyAllWindows()

print('done')