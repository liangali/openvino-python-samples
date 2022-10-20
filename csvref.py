import csv
import pathlib

class MetaCSV():
    def __init__(self, stream=0, frame=0, obj=0, dclass=0, dconf=0, x1=0.0, y1=0.0, x2=0.0, y2=0.0, classid=0, conf=0.0):
        self.stream = stream
        self.frame = frame
        self.object = obj
        self.xmin = x1
        self.ymin = y1
        self.xmax = x2
        self.ymax = y2
        self.detect_class = dclass
        self.detect_conf = dconf
        self.classify_id = classid
        self.classify_conf = conf
    def tolist(self):
        return [self.stream, self.frame, self.object, self.xmin, self.ymin, self.xmax, self.ymax, self.classify_id, self.classify_conf]
    def tostring(self):
        return 'stream_id=%03d, frame_id=%05d, obj_id=%02d, dclass=%02d, dconf =%0.4f, rect=[%04d, %04d, %04d, %04d], class=%03d, confidence=%0.4f' % \
             (self.stream, self.frame, self.object, self.detect_class, self.detect_conf, self.xmin, self.ymin, self.xmax, self.ymax, self.classify_id, self.classify_conf)

def dump2csv(metalist, filename):
    with open(filename, 'wt', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        for m in metalist:
            writer.writerow(m.tolist())
    pass

if __name__ == '__main__':
    pathlib.Path('./ref').mkdir(parents=True, exist_ok=True) 

    metalist = []
    for i in range(10):
        metalist.append(MetaCSV(frame=i, obj=3, classid=120, conf=0.9))

    dump2csv(metalist, './ref/tmp.csv')

    print('done')