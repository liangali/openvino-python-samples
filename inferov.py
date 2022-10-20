import sys
import os
import cv2
import numpy as np
from openvino.inference_engine import IECore
import csvref

GREEN = (255, 0, 0)
BLUE = (0, 255, 0)
RED = (0, 0, 255)

class OpenVinoBase():
    def __init__(self, model_xml, model_bin):
        self.device = 'CPU'
        self.xml = model_xml
        self.bin = model_bin
        self.ie = IECore()
        self.net = self.ie.read_network(model=model_xml, weights=model_bin)
        self.infer_req = self.ie.load_network(network=self.net, device_name=self.device)
        self.input_blob = next(iter(self.net.input_info))
        self.out_blob = next(iter(self.net.outputs))
        self.net.batch_size = 1
        n, c, h, w = self.net.input_info[self.input_blob].input_data.shape
        self.n, self.c, self.h, self.w = n, c, h, w
        self.images = np.ndarray(shape=(n, c, h, w))

class ClassifyOV(OpenVinoBase):
    def __init__(self, model_xml, model_bin):
        super().__init__(model_xml, model_bin)

    def infer(self, frame):
        frame2 = cv2.resize(frame, (self.w, self.h))
        frame2 = frame2.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        self.images[0] = frame2
        res = self.infer_req.infer(inputs={self.input_blob: self.images})
        res = res[self.out_blob]
        probs = np.squeeze(res)
        top10 = np.argsort(probs)[-10:][::-1]
        meta = []
        for i in range(10):
            meta.append([top10[i], probs[top10[i]]])
        return meta

class DetectOV(OpenVinoBase):
    def __init__(self, model_xml, model_bin):
        super().__init__(model_xml, model_bin)

    def infer(self, frame):
        meta = []
        ih, iw = frame.shape[:-1] # original frame size
        frame2 = cv2.resize(frame, (self.w, self.h))
        frame2 = frame2.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        self.images[0] = frame2
        res = self.infer_req.infer(inputs={self.input_blob: self.images})
        res = res[self.out_blob]
        data = res[0][0]
        for number, proposal in enumerate(data):
            if proposal[2] > 0.8:
                imid = np.int(proposal[0])
                idx = np.int(proposal[1])
                confidence = proposal[2]
                x1 = np.int(iw * proposal[3])
                y1 = np.int(ih * proposal[4])
                x2 = np.int(iw * proposal[5])
                y2 = np.int(ih * proposal[6])
                meta.append([idx, confidence, [x1, y1, x2, y2]])
        return meta

def dump_mask_image(frame, metalist, i, prefix='./ref'):
    new_frame = frame.copy()
    for m in metalist:
        d_meta = not (m.xmin == 0 and m.xmax == 0 and m.ymin == 0 and m.ymax == 0)
        c_meta = (m.classify_conf != 0)
        if d_meta and c_meta:
            cv2.rectangle(new_frame, (m.xmin, m.ymin), (m.xmax, m.ymax), GREEN, 2)
            y = m.ymin - 15 if m.ymin - 15 > 15 else m.ymin + 15
            draw_text = 'label=%d, conf=%0.4f' % (m.detect_class, m.detect_conf) 
            cv2.putText(new_frame, draw_text, (m.xmin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLUE, 2)
            draw_text = 'class=%d, prob=%0.4f' % (m.classify_id, m.classify_conf) 
            cv2.putText(new_frame, draw_text, (m.xmin, y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)
        elif d_meta:
            cv2.rectangle(new_frame, (m.xmin, m.ymin), (m.xmax, m.ymax), GREEN, 2)
            y = m.ymin - 15 if m.ymin - 15 > 15 else m.ymin + 15
            draw_text = 'label=%d, conf=%0.4f' % (m.detect_class, m.detect_conf) 
            cv2.putText(new_frame, draw_text, (m.xmin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLUE, 2)
        elif c_meta:
            draw_text = 'class=%d, prob=%0.4f' % (m.classify_id, m.classify_conf) 
            cv2.putText(new_frame, draw_text, (32, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)
    imgfile = '%s.%03d.png' % (prefix, i)
    cv2.imwrite(imgfile, new_frame)
    pass

def test_classify():
    video_file = 'video/test_c_f21.265'
    model_xml = 'model/resnet_v1.5_50_i8.xml'
    model_bin = 'model/resnet_v1.5_50_i8.bin'
    classify = ClassifyOV(model_xml, model_bin)

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("ERROR: failed to open video file")
        exit()

    meta_data = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        top1 = classify.infer(frame)[0]
        cvsdata = csvref.MetaCSV(frame=frame_count, classid=top1[0], conf=top1[1])
        print(cvsdata.tostring())
        dump_mask_image(frame, [cvsdata], frame_count, prefix='./ref/ref_c')
        frame_count += 1
    return frame_count

def test_detect():
    video_file = 'video/test_dc_f20.265'
    model_xml = 'model/ssd_mobilenet_v2_coco_INT8.xml'
    model_bin = 'model/ssd_mobilenet_v2_coco_INT8.bin'
    detect = DetectOV(model_xml, model_bin)

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("ERROR: failed to open video file")
        exit()

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        result = detect.infer(frame)
        metalist = []
        for i, r in enumerate(result):
            xmin, ymin, xmax, ymax = r[2][0], r[2][1], r[2][2], r[2][3]
            cvsdata = csvref.MetaCSV(frame=frame_count, obj=i, dclass=r[0], dconf=r[1], x1=xmin, y1=ymin, x2=xmax, y2=ymax)
            metalist.append(cvsdata)
            print(cvsdata.tostring())
        dump_mask_image(frame, metalist, frame_count, prefix='./ref/ref_d')
        frame_count += 1
    return frame_count

if __name__ == '__main__':
    test_classify()
    test_detect()
    print('done')