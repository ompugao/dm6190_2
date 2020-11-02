#!/bin/bash
LOADCOCOAPI=
if [ -e ./DSB_coco.pkl ]; then
	LOADCOCOAPI="--cocoapi ./DSB_coco.pkl"
fi

#outputdir=outputs/$(date +%Y%m%d-%H%M%S)_base
outputdir=outputs/20201102-151143_base/
#echo python train.py --device cuda:0 --dataset dsb --imageset stage1_train --data-path ../data-science-bowl-2018/  --model=maskrcnn_resnet50_fpn --epochs 75 --start-epoch 26 --resume 'outputs/20201028-183245/model_25.pth' --output-dir $outputdir -j 8 --pretrained $LOADCOCOAPI
#echo python train.py --device cuda:1 --dataset dsb --imageset stage1_train --data-path ../data-science-bowl-2018/  --model=maskrcnn_resnet50_fpn --epochs 75 --start_epoch 27 --resume 'outputs/20201102-151143_base/model_26.pth' --output-dir $outputdir -j 8 --pretrained $LOADCOCOAPI
echo python train.py --device cuda:1 --dataset dsb --imageset stage1_train --data-path ../data-science-bowl-2018/  --model=maskrcnn_resnet50_fpn --epochs 75 --resume 'outputs/20201102-151143_base/model_27.pth' --output-dir $outputdir -j 6 --pretrained $LOADCOCOAPI
python train.py --device cuda:1 --dataset dsb --imageset stage1_train --data-path ../data-science-bowl-2018/  --model=maskrcnn_resnet50_fpn --epochs 75 --resume 'outputs/20201102-151143_base/model_27.pth' --output-dir $outputdir -j 6 --pretrained $LOADCOCOAPI
