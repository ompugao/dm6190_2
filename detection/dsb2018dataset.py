#!/usr/bin/env python3
__author__ = "Shohei Fujii"

import torch
from torchvision.datasets.vision import VisionDataset
import pathlib
import os
import numpy as np
from PIL import Image

class DSB2018TrainDataset(VisionDataset):
    def __init__(self, root, transforms=None, transform=None, target_transform=None):
        super(DSB2018TrainDataset, self).__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)

        self.root = pathlib.Path(self.root)
        self.datapaths = sorted(self.root.iterdir())

    def __getitem__(self, idx):
        parentpath = self.datapaths[idx]
        #image_id = int(parentpath.name, 16)
        image_id = idx
        imgpath = parentpath / 'images' / (parentpath.name + '.png')
        print(str(imgpath))
        img = np.array(Image.open(imgpath).convert("RGB"))

        masks = []
        for maskpath in (parentpath / 'masks').iterdir():
            maskid = maskpath.name[:-len(maskpath.suffix)]
            #masks.append((maskid, np.array(Image.open(maskpath))))
            masks.append(np.array(Image.open(maskpath)))
        masks = np.array(masks)

        # get bounding box coordinates for each mask
        num_objs = len(masks)
        boxes = []
        for i in range(num_objs):
            #pos = np.where(masks[i][1])
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = np.array(boxes, dtype=np.float64)

        # there is only one class
        #labels = torch.ones((num_objs,), dtype=torch.int64)
        labels = np.ones((num_objs,), dtype=np.int64)

        def compute_area(boxes):
            # TODO compute it based on mask
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            return area
        area = compute_area(boxes)
        # suppose all instances are not crowd
        # -- iscrowd (UInt8Tensor[N]): instances with iscrowd=True will be ignored during evaluation.
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        if self.transforms is not None:
            # see
            #  - https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
            #  - https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/#2-pass-class-labels-in-a-separate-argument-to-transform-the-preferred-way
            class_labels = ['nucleus'] * num_objs # dummy label for bboxes
            aug = self.transforms(image=img, masks=masks, bboxes=boxes, class_labels=class_labels)
            img = aug['image']
            img = img.float()
            masks = aug['masks']
            boxes = aug['bboxes']

            area = compute_area(boxes)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        target = {}
        target["boxes"] = boxes
        target["labels"] = torch.as_tensor(labels)
        target["masks"] = masks
        target["image_id"] = torch.tensor([image_id])
        #target["uuid"] = torch.tensor([])
        target["area"] = torch.as_tensor(area)
        target["iscrowd"] = iscrowd

        return img, target

    def __len__(self):
        return len(self.datapaths)

class DSB2018TestDataset(VisionDataset):
    def __init__(self, root, transform=None):
        super(DSB2018TestDataset, self).__init__(root, transform=transform)

        self.root = pathlib.Path(self.root)
        self.datapaths = list(self.root.iterdir())

    def __getitem__(self, idx):
        parentpath = self.datapaths[idx]
        image_id = parentpath.name
        imgpath = parentpath / 'images' / (image_id + '.png')
        img = np.array(Image.open(imgpath).convert("RGB"))

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.datapaths)

if __name__ == '__main__':
    import albumentations
    from albumentations.pytorch.transforms import ToTensorV2
    data_transforms = albumentations.Compose([
        #albumentations.Flip(),
        #albumentations.RandomBrightness(0.2),
        #albumentations.ShiftScaleRotate(rotate_limit=90, scale_limit=0.10),
        #albumentations.Normalize(),
        #albumentations.Resize(256, 256),
        ToTensorV2()
        ], bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    traindataset = DSB2018TrainDataset('../data-science-bowl-2018/stage1_train/', transforms=data_transforms)
    testdataset = DSB2018TestDataset('../data-science-bowl-2018/stage1_test/', transform = data_transforms)

    from IPython.terminal import embed; ipshell=embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())

