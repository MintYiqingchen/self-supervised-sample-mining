#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net, SolverWrapper, update_training_roidb
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import datasets.imdb
from utils.help import *
import caffe
import argparse
import pprint
import numpy as np
import sys, math, logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--unuse_al', help='do not use al process',
                        action='store_true')
    parser.add_argument('--unuse_ss', help='do not use ss process',
                        action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def combined_roidb(imdb_names):
    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        print 'Loaded dataset `{:s}` for training'.format(imdb.name)
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
        roidb = get_training_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        imdb = datasets.imdb.imdb(imdb_names)
    else:
        imdb = get_imdb(imdb_names)
    return imdb, roidb

def get_Imdbs(imdb_names):
    imdbs = [get_imdb(s) for s in imdb_names.split('+')]
    for im in imdbs:
        im.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
    return datasets.imdb.Imdbs(imdbs)

from bitmap import BitMap
if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)

    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    imdb = get_Imdbs(args.imdb_name)
    roidb = get_training_roidb(imdb)
    print '{:d} roidb entries'.format(len(roidb))

    output_dir = get_output_dir(imdb)
    print 'Output will be saved to `{:s}`'.format(output_dir)

    # some statistic to record
    alamount = 0; ssamount = 0
    # set bitmap for AL
    tableA = BitMap(imdb.num_images)
    # choose initiail samples:VOC2007
    sample_num = imdb.num_images
    train_num = len(imdb[imdb.item_name(0)].roidb)
    print 'All VOC2007 images use for initial train, image numbers:%d'%(train_num)
    for i in range(train_num):
        tableA.set(i)

    train_roidb = [roidb[i] for i in range(train_num)]
    pretrained_model_name = args.pretrained_model

    # get solver instance
    sw = SolverWrapper(args.solver, train_roidb, output_dir,
                        pretrained_model=pretrained_model_name)
    sw.train_model(70000)
'''
    while(True):
        # detact remaining samples
        remaining = list(set(range(imdb.num_images))-set(tableA.nonzero()))
        # load latest trained model
        pretrained_model_name = choose_model(output_dir)
        modelpath = os.path.join(output_dir, pretrained_model_name)
        protopath = os.path.join('models/pascal_voc/ResNet-50/rfcn_end2end',
                'test_agnostic.prototxt')
        print 'choose latest model:{}'.format(modelpath)
        model = load_model(protopath,modelpath)
        scoreMatrix, boxRecord,yVecs, eps = bulk_detect(model, remaining, imdb, mylambda)
        logging.debug('scoreMatrix:{}, boxRecord:{}, eps:{}, yVecs:{}'.format(scoreMatrix.shape,
            boxRecord.shape, eps, yVecs.shape))
        # use detect result updatable
        al_candidate = [] # record sample index in imdb
        ss_candidate = []
        ss_fake_gt = [] # record fake labels for ss
        cls_loss_sum = np.zeros((imdb.num_classes,))
        bulk_box_num = 0 # used for update mylambda
        for i in range(len(remaining)):
            img_boxes = []; cls=[]; # fake ground truth
            bulk_box_num += len(boxRecord[i])
            for j,box in enumerate(boxRecord[i]):
                boxscore = scoreMatrix[i][j]
                # fake label box
                y = yVecs[i][j]
                loss = -( (1+y)/2 * np.log(boxscore) + (1-y)/2 * np.log(1-boxscore+1e-30))
                cls_loss_sum += loss
                # choose u,v by loss
                u_star, v_star = judge_uv(loss, gamma, mylambda, eps)
                #logging.info('u_star is {}, v_star is {}'.format(u_star, v_star))
                # ss process
                if(u_star!=1):
                    if(np.sum(y==1)==1 and np.where(y==1)[0]!=0): # not background
                        img_boxes.append(box)
                        cls.append(np.where(y==1)[0])
                    elif(np.sum(y==1)!=0):
                         discard_num += 1
                else: # al process
                    #add image to al candidate
                    al_candidate.append(remaining[i])
                    img_boxes=[]; cls=[]
                    break
            # make fake ground truth for this img
            if len(img_boxes) != 0:
                ss_candidate.append(remaining[i])
                overlaps = np.zeros((len(img_boxes), imdb.num_classes), dtype=np.float32)
                for i in range(len(img_boxes)):
                    overlaps[i, cls[i]]=1.0

                overlaps = scipy.sparse.csr_matrix(overlaps)
                ss_fake_gt.append({'boxes':np.array(img_boxes),
                    'gt_classes':np.array(cls,dtype=np.int).flatten(),
                    'gt_overlaps':overlaps, 'flipped':False})

        if len(al_candidate)<=10 or iters_sum>args.max_iters:
            print 'all process finish at loop ',loopcounter
            print 'the net train for {} epoches'.format(iters_sum)
            break
        if not args.unuse_al:
            # control al proportion
            
            if alamount+len(al_candidate)>=checkpoint[0]:
                al_candidate = al_candidate[:int(checkpoint[0]-alamount)]
                tmp = checkpoint.pop(0)
                print 'checkpoint: {}%% samples for al'.format(tmp/imdb.num_images)
            
            print 'sample index chosen for al: ', al_candidate
        else:
            al_candidate = []
        if not args.unuse_ss:
            print 'sample chosen by ss: ',len(ss_candidate)
        else:
            ss_candidate=[]
            ss_fake_gt = []
        print 'sample discard:', discard_num
        alamount += len(al_candidate); ssamount += len(ss_candidate)
        logging.info('al amount:{}/{}, ss amount: {}'.format(alamount,imdb.num_images,ssamount))
        # generate training set for next loop
        for idx in al_candidate:
            tableA.set(idx)

        next_train_idx = tableA.nonzero(); next_train_idx.extend(ss_candidate)
        # cfg.TRAIN.USE_FLIPPED = False # dont need filp again
        roidb = update_training_roidb(imdb,ss_candidate,ss_fake_gt)
        train_roidb = [roidb[i] for i in next_train_idx]

        loopcounter += 1
        if iters_sum<=tao:
            mylambda = 0.9 * mylambda+0.1*cls_loss_sum/bulk_box_num
            cls_loss_sum = 0.0

        # use chosen samples finetune W
        train_iters = min(12000 ,len(train_roidb)*10-train_iters)
        iters_sum += train_iters
        sw.update_roidb(train_roidb)
        sw.train_model(iters_sum)
'''





