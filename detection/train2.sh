#!/bin/bash
LOADCOCOAPI=
if [ -e ./DSB_coco.pkl ]; then
	LOADCOCOAPI="--cocoapi ./DSB_coco.pkl"
fi

#outputdir=outputs/$(date +%Y%m%d-%H%M%S)_prepro1_aug1
outputdir=outputs/20201102-145827_prepro1_aug1
#echo python train.py --device cuda:0 --dataset dsb --imageset stage1_train --data-path ../data-science-bowl-2018/  --model=maskrcnn_resnet50_fpn --epochs 75 --output-dir $outputdir -j 6 --pretrained $LOADCOCOAPI --image-preprocessing 1 --augmentation 1
#python train.py --device cuda:0 --dataset dsb --imageset stage1_train --data-path ../data-science-bowl-2018/  --model=maskrcnn_resnet50_fpn --epochs 75 --output-dir $outputdir -j 6 --pretrained $LOADCOCOAPI --image-preprocessing 1 --augmentation 1
echo python train.py --device cuda:0 --dataset dsb --imageset stage1_train --data-path ../data-science-bowl-2018/  --model=maskrcnn_resnet50_fpn --epochs 75 --output-dir $outputdir -j 6 --pretrained $LOADCOCOAPI --image-preprocessing 1 --augmentation 1 --resume outputs/20201102-145827_prepro1_aug1/model_4.pth
python train.py --device cuda:0 --dataset dsb --imageset stage1_train --data-path ../data-science-bowl-2018/  --model=maskrcnn_resnet50_fpn --epochs 75 --output-dir $outputdir -j 6 --pretrained $LOADCOCOAPI --image-preprocessing 1 --augmentation 1 --resume outputs/20201102-145827_prepro1_aug1/model_4.pth

outputdir=outputs/$(date +%Y%m%d-%H%M%S)_weightedsampling
echo python train.py --device cuda:0 --dataset dsb --imageset stage1_train --data-path ../data-science-bowl-2018/  --model=maskrcnn_resnet50_fpn --epochs 75 --output-dir $outputdir -j 6 --pretrained $LOADCOCOAPI --weighted-sampling
python train.py --device cuda:0 --dataset dsb --imageset stage1_train --data-path ../data-science-bowl-2018/  --model=maskrcnn_resnet50_fpn --epochs 75 --output-dir $outputdir -j 6 --pretrained $LOADCOCOAPI --weighted-sampling

outputdir=outputs/$(date +%Y%m%d-%H%M%S)_prepro1_aug1_weightedsampling
echo python train.py --device cuda:0 --dataset dsb --imageset stage1_train --data-path ../data-science-bowl-2018/  --model=maskrcnn_resnet50_fpn --epochs 75 --output-dir $outputdir -j 6 --pretrained $LOADCOCOAPI --image-preprocessing 1 --augmentation 1 --weighted-sampling
python train.py --device cuda:0 --dataset dsb --imageset stage1_train --data-path ../data-science-bowl-2018/  --model=maskrcnn_resnet50_fpn --epochs 75 --output-dir $outputdir -j 6 --pretrained $LOADCOCOAPI --image-preprocessing 1 --augmentation 1 --weighted-sampling

# outputdir=outputs/$(date +%Y%m%d-%H%M%S)_base
# echo python train.py --device cuda:0 --dataset dsb --imageset stage1_train --data-path ../data-science-bowl-2018/  --model=maskrcnn_resnet50_fpn --epochs 75 --start-epoch 26 --resume 'outputs/20201028-183245/model_25.pth' --output-dir $outputdir -j 8 --pretrained $LOADCOCOAPI
# python train.py --device cuda:0 --dataset dsb --imageset stage1_train --data-path ../data-science-bowl-2018/  --model=maskrcnn_resnet50_fpn --epochs 75 --start_epoch 26 --resume 'outputs/20201028-183245/model_25.pth' --output-dir $outputdir -j 8 --pretrained $LOADCOCOAPI
