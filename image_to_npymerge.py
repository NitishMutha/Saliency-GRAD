import numpy as np
import glob
import imageio as im
from skimage import io
import argparse
import sys
sys.path.insert(0, "/cs/student/projects4/ml/2016/nmutha/msc_project/code/Around360/brain")

from com.nitishmutha.around.constants import constant

VID = '2,3'


def parseArguments():
    parser = argparse.ArgumentParser(description='Setting for Around360')
    parser.add_argument("--video", action="store", dest="vid",
                        default=VID, help="Select video")

    return parser.parse_args()

args = parseArguments()
print(args)

folders = args.vid.split(',')
for id in folders:

    path = constant.FRAMES_DATA_FOLDER + '360/' + id + '/*.jpg'
    savepath = constant.FRAMES_DATA_FOLDER + '360/' + id + '/npy/frames_salient.npz'
    print('### Started Processing: ' + id)
    image_list = sorted(glob.glob(path))
    sal_images = []
    count = 0
    for i in image_list:
        #sal_images.append(im.imread(i))
        if count % 100 == 0:
            print('Processing ' + str(count))
        sal_images.append(io.imread(i, as_grey=True))
        count += 1
    np.savez_compressed(savepath, frames=sal_images)

    print('DONE! Processed: ' + id)
