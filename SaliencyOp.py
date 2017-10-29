import tensorflow as tf
import tensorflow.contrib.slim as slim
import constant
import resnet_v2 as resnet
import saliency
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from time import time
import gc

GPU = '0'
VID = '2,3'
START = '3'


def parseArguments():
    parser = argparse.ArgumentParser(description='Setting for Around360')
    parser.add_argument("--gpu", action="store", dest="gpu",
                        default=GPU, help="Select GPU for training")
    parser.add_argument("--video", action="store", dest="vid",
                        default=VID, help="Select video")
    parser.add_argument("--start", action="store", dest="start",
                        default=START, help="Select starting point")

    return parser.parse_args()


class Saliency():
    def __init__(self, height, width, channel):
        self.height = height
        self.width = width
        self.channel = channel

        print('Loading Saliency graph...')
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputs = tf.placeholder(shape=[None, self.height, self.width, self.channel],
                                         name=constant.FRAME_FEATURE_INPUTS,
                                         dtype=tf.float32)
            self.neuron_selector = tf.placeholder(tf.int32)

            with slim.arg_scope(resnet.resnet_arg_scope()):
                self.prediction, _ = resnet.resnet_v2_152(inputs=self.inputs,
                                                          num_classes=1001,
                                                          global_pool=True,
                                                          is_training=False)
                self.y = self.prediction[0][self.neuron_selector]
                self.pred = tf.argmax(self.prediction, axis=1)

            print('Resnet 152 model loaded.')
            self.saver = tf.train.Saver()

        self.saliency_sess = tf.Session(graph=self.graph)
        self.saver.restore(self.saliency_sess, constant.PRETRAINED_ROOT + constant.RESNET_152_CKPT)
        print('Resnet 152 checkpoints restored.\nSaliency graph load completed!\n')


    def close_session(self):
        self.saliency_sess.close()

    def getSaliency(self, images):
        # Images: [batch, height, width, channel]
        images = images / 127.5 - 1.0

        predicted_class = self.saliency_sess.run(self.pred, feed_dict={self.inputs: [images]})

        gradient_saliency = saliency.GradientSaliency(self.graph, self.saliency_sess, self.y, self.inputs)
        smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(images,
                                                               feed_dict={self.neuron_selector: predicted_class[0]})
        return saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)


# test example
if __name__ == '__main__':
    args = parseArguments()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    folders = args.vid.split(',')

    for vid in folders:
        path = constant.FRAMES_DATA_FOLDER + '360/' + vid + '/npy/frames.npz'
        print('Processing: ' + vid)
        video_data = np.load(path)
        video_frames = video_data['frames']
        frame_name = video_data['frame_names']

        with tf.name_scope(constant.RESNET152) as scope:

            batch = video_frames.shape[1]
            height = video_frames.shape[2]
            width = video_frames.shape[3]
            imSaliency = Saliency(batch, height, width)

            print('total ' + str(len(video_frames)))
            starttime = time()
            id = 0
            for img, fname in zip(video_frames[int(args.start):],frame_name[int(args.start):]):

                if(id%50 == 0 and id != 0):
                    gc.collect()
                    imSaliency = Saliency(batch, height, width)

                r1 = imSaliency.getSaliency(img)
                plt.imsave(constant.FRAMES_DATA_FOLDER + '360/' + vid + '/'+fname, r1, cmap=cm.gray, vmin=0, vmax=1)

                if(id%10==0):
                    endtime = time()
                    print('processed: ' + fname + ' time lapsed: ' + str(endtime - starttime)+ ' rate: ' + str((endtime - starttime)/10))
                    starttime = time()
                    gc.collect()
                id += 1

            print('Done processing : ' + vid)
            imSaliency.close_session()

