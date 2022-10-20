import glob
import sys
import os

dst_w, dst_h = 1920, 1088

img_files = []
imgdir = './dog5'
for file in glob.glob(imgdir + "/*.png"):
    img_files.append(file)
for file in glob.glob(imgdir + "/*.jpg"):
    img_files.append(file)
for file in glob.glob(imgdir + "/*.bmp"):
    img_files.append(file)
print(len(img_files), 'images found')

outdir = './out'
os.makedirs(outdir, exist_ok=True)

vf_filter = "scale=%d:%d:force_original_aspect_ratio=decrease,pad=%d:%d:(ow-iw)/2:(oh-ih)/2" % (dst_w, dst_h, dst_w, dst_h)
vf_filter = ' "' + vf_filter + '" '
for i, file in enumerate(img_files):
    outfile = ' ' + outdir + '/' + str(i).zfill(4) + '.bmp'
    ffmpeg_cmdline = 'ffmpeg -y -i ' + file + ' -vf ' + vf_filter + outfile + ' 2>tmp.log'
    os.system(ffmpeg_cmdline)
    print(os.path.split(file)[1], 'converted')

os.system('ffmpeg -y -i ./out/%04d.bmp -pix_fmt yuv420p -vcodec libx264 -preset veryslow out.264')
#os.system('ffmpeg -y -i ./out/%04d.bmp -pix_fmt yuv420p -vcodec libx265 -preset veryslow out.265')

print('done')