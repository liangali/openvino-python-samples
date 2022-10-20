import sys
import os
import cv2
import csvref
import inferov

dump_image = True

video_file = 'video/test_c_f21.265'
model_xml = 'model/resnet_v1.5_50_i8.xml'
model_bin = 'model/resnet_v1.5_50_i8.bin'
classify = inferov.ClassifyOV(model_xml, model_bin)

cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    print("ERROR: failed to open video file")
    exit()

frame_count, meta_csv = 0, []
while True:
    ret, frame = cap.read()
    if ret == False:
        break
    top1 = classify.infer(frame)[0]
    cvsdata = csvref.MetaCSV(frame=frame_count, classid=top1[0], conf=top1[1])
    print(cvsdata.tostring())
    if dump_image:
        inferov.dump_mask_image(frame, [cvsdata], frame_count, prefix='./ref/ref_c')
    meta_csv.append(cvsdata)
    frame_count += 1

csvfile = 'ref/openvino_meta_c.csv'
csvref.dump2csv(meta_csv, csvfile)
print('%d frames processed, ref meta data is written to %s' % (frame_count, csvfile))

print('done')