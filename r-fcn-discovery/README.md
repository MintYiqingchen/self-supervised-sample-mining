# R-FCN diacovery new class
## design:
+ add *ClassController* in lib/fast_rcnn/train.py
+ before generate roidb, use ClassController to set valid class (a mask) and \_reload flag
+ lib/datasets/pascal_voc.py \_load_annotation method generate roidb
+ filter_roidb function will automatically filter invalid bbox

## pipeline
1. controlClass
2. get_training_roidb
    + control: reload roidb than replace ground truth for ss
    + not control: replace ground truth for ss
3. filter blank roidb -> valid index
4. mark Bitmap
5. if enable ss or al: 
    + detect
    + choose ss, al candidate -> next train index
