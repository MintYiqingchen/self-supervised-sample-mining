python ./tools/train_net.py --gpu 1 --solver models/pascal_voc/ResNet-50/rfcn_end2end/solver_ohem.prototxt --iters 72000 --weights data/imagenet_models/ResNet-50-model.caffemodel --imdb voc_2007_trainval --cfg experiments/cfgs/rfcn_end2end_ohem.yml
