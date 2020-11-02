#!/bin/bash
outputdir=outputs/$(date +%Y%m%d-%H%M%S)
echo $outputdir
mkdir -p $outputdir
LOADCOCOAPI=
if [ -e ./DSB_coco.pkl ]; then
	LOADCOCOAPI="--cocoapi ./DSB_coco.pkl"
fi
python train.py --dataset dsb --imageset stage1_train --data-path ../data-science-bowl-2018/  --model=maskrcnn_resnet50_fpn --output-dir $outputdir -j 8 --pretrained $LOADCOCOAPI #| tee $outputdir/train.log
#python train.py --dataset dsb --imageset stage1_train --data-path ../data-science-bowl-2018/  --model=maskrcnn_resnet50_fpn --output-dir $outputdir --resume outputs/20201027-223141/model_22.pth --start_epoch 22 -j 8 --pretrained $LOADCOCOAPI #| tee $outputdir/train.log


