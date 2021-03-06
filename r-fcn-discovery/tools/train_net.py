#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""
from __future__ import division
import os
os.environ['GLOG_minloglevel'] = '2'
import _init_paths
from fast_rcnn.train import get_training_roidb, train_net, SolverWrapper, update_training_roidb,filter_roidb, ClassController, filter_blank_roidb
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import datasets.imdb
from utils.help import *
import caffe
import argparse
import pprint
import numpy as np
import sys, math, logging
import scipy

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
    parser.add_argument('--disable_al', help='do not use al process',
                        action='store_true')
    parser.add_argument('--disable_ss', help='do not use ss process',
                        action='store_true')
    parser.add_argument('--enable_icv', help='do not use image cross validation process',
                        action='store_true',default=True)
    parser.add_argument('--enable_mmv', help='do not use multi model validation process',
                        action='store_true',default=False)
    parser.add_argument('--class_schedule', help='list classes index for incremental learning',
                        default=[(0,81)], type=list)

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
    return datasets.imdb.Imdbs(imdbs,imdb_names)

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

    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)

    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    imdb = get_Imdbs(args.imdb_name)
    # construct classContoller
    classController = ClassController(args.class_schedule, imdb)

    # NOTICE: controlClass method should be invoked before get roidb
    classController.controlClass(nowepoch=0)

    roidb = get_training_roidb(imdb)
    print 'imdb classes: {}'.format(imdb.classes)
    print '{:d} roidb entries ,images:{}'.format(len(roidb), imdb.num_images)

    output_dir = get_output_dir(imdb)
    print 'Output will be saved to `{:s}`'.format(output_dir)

    # some statistic to record
    alamount = 0; ssamount = 0
    discardamount = 0
    # choose valid initiail samples
    train_roidb, valid_idx = filter_blank_roidb(roidb)

    sample_num = len(valid_idx)
    
    # set bitmap for AL
    bitmapImdb = BitMap(imdb.num_images)

    # initial image
    initial_num = int(len(valid_idx)*0.1)

    print ('The 10% coco of images for initial:{}'.format(initial_num))

    for i in range(initial_num):
        bitmapImdb.set(i)

    train_roidb = [train_roidb[i] for i in range(initial_num)]
    pretrained_model_name = args.pretrained_model

    # static parameters
    tao = args.max_iters
    # initial hypeparameters
    gamma = 0.3; clslambda = np.array([-np.log(0.9)]*imdb.num_classes)
    # train record
    loopcounter = 0; train_iters = 0; iters_sum = train_iters
    # control al proportion
    al_proportion_checkpoint = [int(x*sample_num) for x in np.linspace(0.1,23,12)]
    # control ss proportion with respect to al proportion
    ss_proportion_checkpoint = [int(x*sample_num) for x in np.linspace(0.1,23,12)]

    sw = SolverWrapper(args.solver, train_roidb, output_dir,
                        pretrained_model=pretrained_model_name)
    # with voc2007 to pretrained an initial model
    # sw.train_model(150000)

    while(True):
        # detact unlabeledidx samples
        unlabeledidx = list(set(range(imdb.num_images))-set(bitmapImdb.nonzero()))
        # detect labeledidx
        labeledidx = list(set(bitmapImdb.nonzero()))
        # load latest trained model
        trained_models = choose_model(output_dir)
        pretrained_model_name = trained_models[-1]
        modelpath = os.path.join(output_dir, pretrained_model_name)
        protopath = os.path.join('models/coco/ResNet-101/rfcn_end2end',
                'test_agnostic.prototxt')
        print 'choose latest model:{}'.format(modelpath)
        model = load_model(protopath,modelpath)

        # record some detect results for updatable
        al_candidate_idx = [] # record al samples index in imdb
        ss_candidate_idx = [] # record ss samples index in imdb
        ss_fake_gt = [] # record fake labels for ss
        cls_loss_sum = np.zeros((imdb.num_classes,)) # record loss for each cls
        count_box_num = 0 # used for update clslambda
        
        if not (args.disable_al or args.disable_ss):
           
            # return detect results of the unlabeledidx samples with the latest model
            scoreMatrix, boxRecord,yVecs, eps = bulk_detect(model, unlabeledidx, imdb, clslambda)
            logging.debug('scoreMatrix:{}, boxRecord:{}, eps:{}, yVecs:{}'.format(scoreMatrix.shape, boxRecord.shape, eps, yVecs.shape))
            
            for i in range(len(unlabeledidx)):
                img_boxes = []; cls=[]; # fake ground truth
                count_box_num += len(boxRecord[i])
                for j,box in enumerate(boxRecord[i]):
                    boxscore = scoreMatrix[i][j] # score of a box
                    # fake label box
                    y = yVecs[i][j]
                    # the fai function
                    loss = -((1+y)/2 * np.log(boxscore) + (1-y)/2 * np.log(1-boxscore+1e-30))

                    cls_loss_sum += loss
                    # choose u,v by loss
                    u_star, v_star = judge_uv(loss, gamma, clslambda, eps)
                    # ss process
                    if(u_star!=1):
                        if(np.sum(y==1)==1 and np.where(y==1)[0]!=0): # not background
                            # add fake gt
                            img_boxes.append(box)
                            cls.append(np.where(y==1)[0])
                        elif(np.sum(y==1)!=1):
                             discardamount += 1
                    else: # al process
                        #add image to al candidate
                        al_candidate_idx.append(unlabeledidx[i])
                        img_boxes=[]; cls=[]
                        break
                # replace the fake ground truth for the ss_candidate
                if len(img_boxes) != 0:
                    ss_candidate_idx.append(unlabeledidx[i])
                    overlaps = np.zeros((len(img_boxes), imdb.num_classes), dtype=np.float32)
                    for i in range(len(img_boxes)):
                        overlaps[i, cls[i]]=1.0

                    overlaps = scipy.sparse.csr_matrix(overlaps)
                    ss_fake_gt.append({'boxes':np.array(img_boxes),
                        'gt_classes':np.array(cls,dtype=np.int).flatten(),
                        'gt_overlaps':overlaps, 'flipped':False})

        if (not args.disable_al and len(al_candidate_idx)<=10) or iters_sum>args.max_iters:
            print ('all process finish at loop ',loopcounter)
            print ('the net train for {} epoches'.format(iters_sum))
            break

        if not args.disable_al:
            # 50% enter al
            r = np.random.rand(len(al_candidate_idx))
            al_candidate_idx = [x for i,x in enumerate(al_candidate_idx) if r[i]>0.5]

            # checkpoint information
            if alamount+len(al_candidate_idx)>=al_proportion_checkpoint[0]:
                al_candidate_idx = al_candidate_idx[:int(al_proportion_checkpoint[0]-alamount)]
                tmp = al_proportion_checkpoint.pop(0)
                print 'al_proportion_checkpoint: {:.3f}% samples for al'.format(tmp/initial_num*100)

            # log
            alamount += len(al_candidate_idx)
            al_factor = alamount/initial_num
            logging.info('last model name :{}, al amount:{}/{} ,al_factor:{}'.format(pretrained_model_name, alamount, initial_num, al_factor))
        else:
            al_candidate_idx = []

        if not args.disable_ss:
            # control ss proportion
            if ssamount+len(ss_candidate_idx)>=ss_proportion_checkpoint[0]:
                ss_candidate_idx = ss_candidate_idx[:int(ss_proportion_checkpoint[0]-ssamount)]
                ss_fake_gt = ss_fake_gt[:int(ss_proportion_checkpoint[0]-ssamount)]
                tmp = ss_proportion_checkpoint.pop(0)
                print 'ss_proportion_checkpoint: {}%% samples for ss, model name:{}'.format(tmp/initial_num,pretrained_model_name )

            ssamount += len(ss_candidate_idx) + discardamount
            ss_factor = float(ssamount/initial_num)
            logging.info('ss amount:{}/{}, ss factor:{}, discard amount:{}'.format(ssamount, initial_num, ss_factor, discardamount))
        else:
            ss_candidate_idx=[]
            ss_fake_gt = []
        
        # generate training set for next loop
        for idx in al_candidate_idx:
            bitmapImdb.set(idx)
        next_train_idx = bitmapImdb.nonzero(); next_train_idx.extend(ss_candidate_idx)
        
        # update the roidb with ss_fake_gt
    	control = classController.controlClass(iters_sum)
        print imdb.classes
        if control:
            print 'change class roidb'
        train_roidb = update_training_roidb(imdb,ss_candidate_idx,ss_fake_gt)
        train_roidb = blur_image(train_roidb)
        if not (args.disable_ss or args.disable_al):
            train_roidb = [train_roidb[i] for i in next_train_idx]

        print 'imdb all roidb:{}, imdb images:{}, train roidb:{}'.format(len(imdb.roidb), imdb.num_images, len(train_roidb))

        loopcounter += 1
        if iters_sum<=tao:
            clslambda = 0.9 * clslambda+0.1*cls_loss_sum/(count_box_num+1e-30)
            cls_loss_sum = 0.0

        # add the labeled samples to finetune W
        train_iters = 30000 # min(30000 ,len(train_roidb)*10-train_iters)
        iters_sum += train_iters
        sw.update_roidb(train_roidb)
        sw.train_model(iters_sum)

