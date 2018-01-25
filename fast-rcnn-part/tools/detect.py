#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import division
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
#from fast_rcnn.nms_wrapper import nms
from utils.cython_nms import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
from generateXML import GenerateXml
import shutil
CLASSES=('__background__','aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}
global al
al = 0
global ss
ss = 0
def vis_detections(image_name, im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    # save the image
    fig = plt.gcf()
    fig.savefig("images/output_"+image_name)


def demo(net, image_name,obj_proposals,fs):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load pre-computed Selected Search object proposals
    # box_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo',
    #                         image_name + '_boxes.mat')
    # obj_proposals = sio.loadmat(box_file)['boxes']
    # # Load the demo image
    image_folder="data/VOCdevkit2007/VOC2007/JPEGImages/"
    xml_folder="data/VOCdevkit2007/VOC2007/Annotations/"
    im = cv2.imread(image_folder+image_name)
    height=im.shape[0]
    width=im.shape[1]

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im,obj_proposals)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    Label=[]
    BBox=[]
    Score=[]
    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
	    keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]

        if len(inds) == 0 :
            continue
        # vis_detections(image_name, im, cls, dets, thresh=CONF_THRESH)
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
            Label.append(cls_ind)
            Score.append(score)
            BBox.append(bbox[0])
            BBox.append(bbox[1])
            BBox.append(bbox[2])
            BBox.append(bbox[3])


    # xmlname=image_name.split(".")[0]+'.txt'
    # path='annos/'+xmlname
    # fs=open(path,'a')
#    for i in xrange(len(Label)):
#        fs.write(image_name.split(".")[0]+' '+str(Score[i])+' '+str(BBox[i])+' '+str(BBox[i+1])+' '+str(BBox[i+2])+' '+str(BBox[i+3]))
#        fs.write('\n')
    if len(BBox)<2:
       shutil.copy(xml_folder+image_name.split('.')[0]+'.xml','data/VOCdevkit2007/VOC2007/Predictions/')
       global al
       al +=1
    else:
	GenerateXml(image_name,width,height,Score,Label,BBox)
        global ss
        ss += 1

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Fast R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.ROOT_DIR, 'models/CaffeNet/test.prototxt')
    caffemodel = 'output/default/voc_2007_trainval/caffenet_fast_rcnn_gt_iter_40000.caffemodel'

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    fs=open('data/VOCdevkit20/VOC20/ImageSets/Main/val.txt','a')
    matfn='data/selective_search_data/voc_20_val.mat'
    all_boxes=sio.loadmat(matfn)
    for i in xrange(len(all_boxes['images'])):
        img_path = all_boxes['images'][i][0][0]+'.jpg'
        print img_path
        obj_proposals= all_boxes['boxes'][0][i]
        demo(net,img_path,obj_proposals,fs)
    fs.close()
    global al
    global ss
    print "Percentage of AL:",float(al/(al+ss))

