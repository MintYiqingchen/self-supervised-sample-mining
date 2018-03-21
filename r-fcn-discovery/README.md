# R-FCN diacovery new class
## design:
+ add *ClassController* in lib/fast_rcnn/train.py
+ before generate roidb, use ClassController to set valid class (a mask) and \_reload flag
+ lib/datasets/pascal_voc.py \_load_annotation method generate roidb
+ filter_roidb function will automatically filter invalid bbox