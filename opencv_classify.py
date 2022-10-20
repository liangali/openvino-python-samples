import numpy as np
import cv2
import time
import glob
import pathlib

pathlib.Path('./ref').mkdir(parents=True, exist_ok=True) 

COLOR_RED = (0, 0, 255)

labels = []
with open('model/resnet-50-label.txt', 'rt') as f:
    for line in f:
        strip_line = line.split(':')[1].strip().strip("',")
        labels.append(strip_line)

img_files = []
imgdir = './images'
for file in glob.glob(imgdir + "/*.jpg"):
    img_files.append(file)
for file in glob.glob(imgdir + "/*.png"):
    img_files.append(file)

class ClassifyOut():
    def __init__(self, topv, topi):
        self.topv = topv
        self.topi = topi
        self.strline = self.toString()
    def toString(self):
        out = ''
        topstr = ''
        for i in range(len(self.topv)):
            strv = "{0:.4f}".format(self.topv[i])
            topstr += strv + ': ' + str(topi[i]+1) + ': ' + labels[topi[i]] + '\n'
        out += topstr
        return out

prototxt = './model/resnet-50.prototxt'
model = './model/resnet-50.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt, model)

input_imgfolder = False
cap = cv2.VideoCapture('video/test_c_f21.265')
if not cap.isOpened():
    print("Cannot open VideoCapture, will load images from specificed folder")
    input_imgfolder = True

meta_data = []
frame_count = 0
while True:
    if input_imgfolder:
        if frame_count >= len(img_files):
            break
        imgfile = img_files[frame_count]
        frame = cv2.imread(imgfile)
    else:
        ret, frame = cap.read()
        if ret == False:
            break;
    scale_img = cv2.resize(frame, (224, 224))
    blob = cv2.dnn.blobFromImage(scale_img, 1.0, (224, 224), (104.0, 117.0, 123.0), False)
    net.setInput(blob)
    probs = net.forward()
    topv = sorted(probs[0], reverse=True)[0:1]
    topi = sorted(range(len(probs[0])), key=lambda k: probs[0][k], reverse=True)[0:1]
    result = ClassifyOut(topv, topi)
    if topv[0] > 0.8:
        meta_data.append(result.toString())
    
    #cv2.putText(frame, result.toString(), (32, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_RED)
    #cv2.imwrite('./ref/ref_%02d.png' % frame_count, frame)

    frame_count += 1
    key = cv2.waitKey(0)
    if key == ord("q"):
        break

meta_file = 'ref/opencv_meta_c.txt'
with open(meta_file, 'wt') as f:
    f.writelines(meta_data)
    print('%d frames processed, ref meta data is written to %s' % (len(meta_data), meta_file))

print('done')