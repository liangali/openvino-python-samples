import sys
import os
import cv2
import csvref
import inferov

dump_image = True

video_file = 'video/test_dc_f20.265'

model_xml = 'model/ssd_mobilenet_v2_coco_INT8.xml'
model_bin = 'model/ssd_mobilenet_v2_coco_INT8.bin'
detect = inferov.DetectOV(model_xml, model_bin)

model_xml = 'model/resnet_v1.5_50_i8.xml'
model_bin = 'model/resnet_v1.5_50_i8.bin'
classify = inferov.ClassifyOV(model_xml, model_bin)

cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    print("ERROR: failed to open video file")
    exit()

meta_csv = []
frame_count, obj_count = 0, 0
while True:
    ret, frame = cap.read()
    if ret == False:
        break
    result = detect.infer(frame)
    metalist = []
    for i, r in enumerate(result):
        xmin, ymin, xmax, ymax = r[2][0], r[2][1], r[2][2], r[2][3]
        crop_img = frame[ymin:ymax, xmin:xmax]
        top1 = classify.infer(crop_img)[0]
        csvdata = csvref.MetaCSV(frame=frame_count, obj=i, dclass=r[0], dconf=r[1], x1=xmin, y1=ymin, x2=xmax, y2=ymax, classid=top1[0], conf=top1[1])
        print(csvdata.tostring())
        meta_csv.append(csvdata)
        metalist.append(csvdata)
        obj_count += 1
    if dump_image:
        inferov.dump_mask_image(frame, metalist, frame_count, prefix='./ref/ref_dc')
    frame_count += 1

csvfile = 'ref/openvino_meta_dc.csv'
csvref.dump2csv(meta_csv, csvfile)
print('%d objects found in %d frames, ref meta data is written to %s' % (obj_count, frame_count, csvfile))

print('done')