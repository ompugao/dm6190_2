from os import path, listdir
import math

import numpy as np
from PIL import Image

from skimage.measure import label, regionprops

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

class DSB(Dataset):

    def __init__(self, root_path="/home/jake/data/data-science-bowl-2018",
                 image_set="stage1_train", transforms=None, preprocessfn=None):
        super(DSB, self).__init__()
        self.root = root_path
        self.image_set = image_set

        self.path = path.join(self.root, self.image_set)

        self.img_ids = [f for f in sorted(listdir(self.path))]

        self.transforms = transforms
        self.preprocessfn = preprocessfn

    def get_height_and_width(self, index):
        image_id = self.img_ids[index]
        img = Image.open(path.join(self.path, image_id, "images", f"{image_id}.png")).convert("RGB")
        return img.height, img.width

    def generate_bbox_from_mask(self, mask):
        label_img = label(mask, connectivity=1)
        try:
            props = regionprops(label_img)[0]
        except Exception:
            print("Region not found")
            return
        return props.bbox

    def get_raw_image(self, index):
        image_id = self.img_ids[index]
        img = Image.open(path.join(self.path, image_id, "images", f"{image_id}.png")).convert("RGB")
        return img

    def compute_weights(self, image_class_identifier):
        klasses = np.zeros(len(self.img_ids))
        for i, image_id in enumerate(self.img_ids):
            img = np.array(Image.open(path.join(self.path, image_id, "images", f"{image_id}.png")).convert("RGB"))
            klasses[i] = image_class_identifier.detect(img)
        klasses = klasses.astype(np.int8)
        hist = []
        for i in np.unique(klasses):
            hist.append(len(klasses[klasses == i]))
        #np.histogram(klasses, bins=len(np.unique(klasses)))
        hist = np.array(hist)/np.sum(hist)
        weight = 1/hist
        weights = np.zeros(len(self.img_ids))
        for i, kls in enumerate(klasses):
            weights[i] = weight[kls]
        return weights

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        image_id = self.img_ids[index]

        img = Image.open(path.join(self.path, image_id, "images", f"{image_id}.png")).convert("RGB")
        img = np.array(img)

        if self.preprocessfn is not None:
            img = self.preprocessfn(img, image_id)

        boxes = []
        labels = []
        masks = []
        area = []
        is_crowd = []

        mask_path = path.join(self.path, image_id, "masks")

        for f in listdir(mask_path):
            # Load mask
            mask = np.array(Image.open(path.join(mask_path, f)))
            # Normalise mask
            mask = (mask - mask.min()) / (mask.max() - mask.min())
            box = self.generate_bbox_from_mask(mask)
            if box:
                box = (box[1], box[0], box[3], box[2])
                boxes.append(box)
                labels.append(1)
                masks.append(np.array(mask))
                area.append(np.count_nonzero(mask == 1))
                is_crowd.append(0)
            else:
                continue

        if self.transforms is not None:
            aug = self.transforms(image=np.array(img), masks=masks, bboxes=boxes, class_labels=labels)
            if type(aug['image']) in [np.ndarray, Image]:
                img = F.to_tensor(aug['image'])
            img = img.float()
            masks = aug['masks']
            boxes = aug['bboxes']
            labels = aug['class_labels']
            area = [np.count_nonzero(mask == 1) for mask in np.array(masks)]
        else:
            img = F.to_tensor(img).float()

        for box in boxes:
            xmin, ymin, xmax, ymax = box
            height = img.shape[1]
            width = img.shape[2]
            #print(xmin, ymin, xmax, ymax)
            if not (xmin >= 0 and xmax <= width and xmin < xmax and ymin >= 0 and ymax <= height and ymin < ymax):
                print(image_id)
                print(box, width, height)
                #from IPython.terminal import embed; ipshell=embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())

        target = {
            "boxes": torch.tensor(boxes),
            "labels": torch.tensor(labels),
            "masks": torch.tensor(masks, dtype=torch.uint8),
            "image_id": torch.tensor([index]),
            "area": torch.tensor(area),
            "iscrowd": torch.tensor(is_crowd)
        }

        # img, target = self.transforms(img, target)

        return img, target

if __name__ == '__main__':
    #import transforms as T
    #transforms = []
    #transforms.append(T.ToTensor())
    #transforms.append(T.RandomHorizontalFlip(0.5))
    #transforms = T.Compose(transforms)
    import albumentations
    from albumentations.pytorch.transforms import ToTensorV2
    data_transforms = albumentations.Compose([
        albumentations.Flip(),
        #albumentations.RandomBrightness(0.2),
        albumentations.ShiftScaleRotate(rotate_limit=90, scale_limit=0.10),
        #albumentations.Normalize(),
        #albumentations.Resize(512, 512),
        #ToTensorV2()
        ], bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=['class_labels'])) # min_visibility=0.2
    #self = DSB('/home/leus/3rdparty/github.com/ompugao/dm6190/2_segmentationreview/data-science-bowl-2018/', image_set="stage1_train", transforms=data_transforms)
    import preprocess
    preprocessfn = preprocess.ImagePreProcessor1('./kmeans.pkl')
    self = DSB('/home/leus/3rdparty/github.com/ompugao/dm6190/2_segmentationreview/data-science-bowl-2018/', image_set="stage1_train", transforms=data_transforms, preprocessfn=preprocessfn)
    from IPython.terminal import embed; ipshell=embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())

