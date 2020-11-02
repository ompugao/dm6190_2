r"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU

The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.

On top of that, for training Faster/Mask R-CNN, the default hyperparameters are
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3

Also, if you train Keypoint R-CNN, the default hyperparameters are
    --epochs 46 --lr-steps 36 43 --aspect-ratio-group-factor 3
Because the number of images is smaller in the person keypoint subset of COCO,
the number of epochs should be adapted so that we have the same number of iterations.
"""
import datetime
import os
import time
import pickle
import logging
import sys

import torch
import torch.utils.data
from torch import nn
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from coco_utils import get_coco, get_coco_kp

from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from engine import train_one_epoch, evaluate
from torch.utils.data import random_split

import utils
from utils import LogFile
import transforms as T
import albumentations
from albumentations.pytorch.transforms import ToTensorV2

from dsb_dataset import DSB
import preprocess
from albumentations_transforms import IfThen

torch.random.manual_seed(0)


def load_dsb_dataset(root_path, image_set, transforms, preprocess_ver, val_percent=0.15):
    preprocessfn = None
    if preprocess_ver == 1:
        preprocessfn = preprocess.ImagePreProcessor1()

    dataset = DSB(root_path, image_set, transforms, preprocessfn)

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    n_classes = 2 # Background, cell

    return train_set.dataset, val_set.dataset, n_classes

def get_dataset(name, image_set, transform, data_path):
    paths = {
        "coco": (data_path, get_coco, 91),
        "coco_kp": (data_path, get_coco_kp, 2)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


# def get_transform(train):
#     transforms = []
#     transforms.append(T.ToTensor())
#     if train:
#         transforms.append(T.RandomHorizontalFlip(0.5))
#     return T.Compose(transforms)

def get_transform(train, augmentation_ver=None):
    if augmentation_ver is None:
        data_transforms = albumentations.Compose([
            albumentations.Flip(),
            #albumentations.RandomBrightness(0.2),
            albumentations.ShiftScaleRotate(rotate_limit=90, scale_limit=0.10),
            #ToTensorV2()
            ], bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=['class_labels'])) # min_visibility=0.2
    elif augmentation_ver == 1:
        identifier = preproces.ImageClassIdentifier()
        def ifpred(**kwargs):
            img = kwargs['image']
            predicted_class = identifier.detect(np.array(img))
            if predicted_class == 2:
                return True
            return False

        data_transforms = albumentations.Compose([
            IfThen(ifpred, albumentations.Compose([
                albumentations.RandomSizedCrop((128, 1024), 1024, 1024)
                ])),
            albumentations.Flip(),
            albumentations.VerticalFlip(),
            albumentations.RandomRotate90(p=0.5),
            albumentations.ShiftScaleRotate(rotate_limit=15, scale_limit=0.10),
            #ToTensorV2()
            ], bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=['class_labels'])) # min_visibility=0.2
    return data_transforms

def load_coco_api(coco_api_path):
    with open(coco_api_path, "rb") as f:
        return pickle.load(f)

def create_model(modelname, pretrained, num_classes):
    model = torchvision.models.detection.__dict__[modelname](pretrained=pretrained)

    if modelname == 'maskrcnn_resnet50_fpn':
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                           hidden_layer,
                                                           num_classes)

    return model


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")

    if args.dataset != "dsb":
        dataset, num_classes = get_dataset(args.dataset, "train", get_transform(train=True, augmentation_ver=args.augmentation), args.data_path)
        dataset_test, _ = get_dataset(args.dataset, "val", get_transform(train=False, augmentation_ver=args.augmentation), args.data_path)
    else:
        dataset, dataset_test, num_classes = load_dsb_dataset(
                                    args.data_path,
                                    args.imageset,
                                    get_transform(train=True, augmentation_ver=arg.augmentation),
                                    args.image_preprocessing)

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    elif args.dataset == "dsb" and args.weighted_sampling:
        print("computing weights for training...")
        weights = dataset.compute_weights()
        train_sampler = torch.utils.data.WeightedRandomSampler(weights, len(dataset), replacement=True)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    print("Creating model")
    # model = torchvision.models.detection.__dict__[args.model](num_classes=num_classes,
                                                              # pretrained=args.pretrained)
    model = create_model(args.model, args.pretrained, num_classes)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                            T_0=args.epochs * (len(data_loader.dataset) // args.batch_size))

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    coco_api = load_coco_api(args.cocoapi) if args.cocoapi else None

    return
    if args.test_only:
        evaluate(model, data_loader_test, device=device, coco_api=coco_api)
        return

    print("Start training")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq)
        lr_scheduler.step()
        if args.output_dir:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'epoch': epoch},
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))

        # Evaluate after each epoch
        evaluate(model, data_loader_test, device=device, coco_api=coco_api)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--data-path', default='/datasets01/COCO/022719/', help='dataset')
    parser.add_argument('--dataset', default='coco', help='dataset')
    parser.add_argument('--imageset', default='stage1_train', help='imageset')
    parser.add_argument('--image-preprocessing', default=None, help='image preprocessing version', type=int)
    parser.add_argument('--augmentation', default=0, help='data augmentation version', type=int)
    parser.add_argument("--weighted-sampling", help="Use weighted sampler for training", action="store_true")
    parser.add_argument('--cocoapi', help='Path to pickled COCO API for test set')
    parser.add_argument('--model', default='maskrcnn_resnet50_fpn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--epochs', default=26, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', default=0.02, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                        'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='./runs', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    logging.basicConfig(filename=os.path.join(args.output_dir, "log.log"),
                        level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    sys.stdout = LogFile('stdout')
    sys.stderr = LogFile('stderr')
    main(args)
