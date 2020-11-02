#!/bin/bash
outputdir=outputs_retinanet/$(date +%Y%m%d-%H%M%S)
echo $outputdir
mkdir -p $outputdir
LOADCOCOAPI=
if [ -e ./DSB_coco.pkl ]; then
	LOADCOCOAPI="--cocoapi ./DSB_coco.pkl"
fi
python train.py --dataset dsb --device cuda:1 --imageset stage1_train --data-path ../data-science-bowl-2018/  --model=retinanet_resnet50_fpn --output-dir $outputdir -j 8 --pretrained --batch-size 8 --lr 0.005 $LOADCOCOAPI #| tee $outputdir/train.log



