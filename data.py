"""
Modified on the MUNIT `data.py`
"""
import cv2
import os.path
import torch.utils.data as data


def default_loader(path, is_gray=False):
    return cv2.imread(path, int(not is_gray))


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist


class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader, len_limit=None):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.loader = loader
        self.len_limit = len_limit

    def __getitem__(self, index):
        ll = self.len_limit if self.len_limit is not None and self.len_limit < len(self.imgs) else len(self.imgs)
        index = index % ll
        impath = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imlist)


class ImageLabelFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader, len_limit=None):
        self.root = root
        self.imlist = flist_reader(os.path.join(self.root, flist))
        self.transform = transform
        self.loader = loader
        self.classes = sorted(list(set([path.split('/')[0] for path in self.imlist])))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.imgs = [(impath, self.class_to_idx[impath.split('/')[0]]) for impath in self.imlist]
        self.len_limit = len_limit

    def __getitem__(self, index):
        ll = self.len_limit if self.len_limit is not None and self.len_limit < len(self.imgs) else len(self.imgs)
        index = index % ll
        impath, label = self.imgs[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data
import torch
import random
from PIL import Image
import os
import os.path
from torchvision import transforms
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def rotate_image(image, angle, mode=cv2.INTER_LINEAR):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=mode)
    return result


class ImageFolder(data.Dataset):

    def __init__(self, root, sem_root=None, return_paths=False,
                 loader=default_loader, len_limit=None, use_da=True, height=256, width=256):
        imgs = sorted(make_dataset(root))
        if sem_root is not None:
            sems = [x.replace(root, sem_root) for x in imgs]
            sems = [os.path.join(os.path.dirname(x), os.path.basename(x).split('.')[0] + '.png') for x in sems]
        else:
            sems = ['' for x in imgs]
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.sems = sems
        self.return_paths = return_paths
        self.loader = loader
        self.len_limit = len_limit
        self.use_da = use_da
        self.height = height
        self.width = width

    def img_transform(self, img):
        t_img = torch.from_numpy(np.transpose(img, axes=(2, 0, 1)) / 255. * 2 - 1).float()
        return t_img

    def data_argument(self, img, mode, random_pos, random_angle, random_flip):

        img = cv2.resize(img, (self.height, self.width), interpolation=mode)
        if random_flip > 0.5:
            img = np.fliplr(img)

        img = rotate_image(img, random_angle, mode=mode)
        if len(img.shape) > 2:
            img = img[random_pos[0]:random_pos[1], random_pos[2]:random_pos[3], :]
        else:
            img = img[random_pos[0]:random_pos[1], random_pos[2]:random_pos[3]]
        img = cv2.resize(img, (self.height, self.width), interpolation=mode)

        return img

    def data_convert(self, img, mode):
        img = cv2.resize(img, (self.height, self.width), interpolation=mode)
        return img

    def __getitem__(self, index):
        ll = self.len_limit if self.len_limit is not None and self.len_limit < len(self.imgs) else len(self.imgs)
        index = index % ll
        img_path = self.imgs[index]
        sem_path = self.sems[index]
        img = self.loader(img_path)[:, :, ::-1]
        sem = self.loader(sem_path, is_gray=True)

        if self.use_da:
            random_flip = random.random()
            random_angle = random.random() * 10 - 5
            random_start_y = random.randint(0, 9)
            random_start_x = random.randint(0, 9)

            random_pos = [random_start_y, random_start_y + self.height - 10, random_start_x,
                          random_start_x + self.width - 10]

            img = self.data_argument(img, cv2.INTER_LINEAR, random_pos, random_angle, random_flip)
            if sem is not None:
                sem = self.data_argument(sem, cv2.INTER_NEAREST, random_pos, random_angle, random_flip)
        else:
            img = self.data_convert(img, cv2.INTER_LINEAR)
            if sem is not None:
                sem = self.data_convert(sem, cv2.INTER_NEAREST)

        img = self.img_transform(np.float32(img))
        sem = torch.from_numpy(sem).long() if sem is not None else 0

        if self.return_paths:
            return img, sem, img_path
        else:
            return img, sem, ''

    def __len__(self):
        return len(self.imgs)


CAT_CLASS_IDS = [
                #id       name            color
                #─────────────────────────────────────
                (0,  'background',   (  0,   0,   0)),
                (1,  'face',         (255, 182, 193)),
                (2,  'left-eyebow',  ( 39, 105, 105)),
                (3,  'right-eyebow', (139,  54,  38)),
                (4,  'left-eye',     (135, 206, 255)),
                (5,  'right-eye',    (  0, 191, 255)),
                (6,  'nose',         (255, 215,   0)),
                (7,  'mouse',         (  0, 255, 127))
                ]

HUMAN_CLASS_IDS = [
                    #id       name            color
                    #─────────────────────────────────────
                    (0,  'background',   (  0,   0,   0)),
                    (1,  'face',         (255, 182, 193)),
                    (2,  'left-ear',     ( 39, 105, 105)),
                    (3,  'right-ear',    (139,  54,  38)),
                    (4,  'left-eye',     (135, 206, 255)),
                    (5,  'right-eye',    (  0, 191, 255)),
                    (6,  'nose',         (255, 215,   0)),
                    (7, 'mouse',        (  0, 255, 127))
                    ]


def mask2color(mask, name='human'):
    if name == 'human':
        class_id = HUMAN_CLASS_IDS
    elif name == 'cat':
        class_id = CAT_CLASS_IDS
    else:
        raise NotImplementedError

    c = np.zeros([mask.shape[0], mask.shape[1], 3], np.uint8)
    for i, cid in enumerate(class_id):
        if c[mask == cid[0]].shape[0] > 0:
            c[mask == cid[0]] = np.array(cid[2][::-1], np.uint8)
    return c


if __name__ == '__main__':
    dataset = ImageFolder('/home/zcq/ws/dataset/StyleI2I_bkp/cat2human1/trainA',
                          '/home/zcq/ws/dataset/StyleI2I_bkp/cat2human1_sem/trainA', use_da=True, len_limit=20,
                          height=256, width=256)
    for i in range(20):
        img, sem, _ = dataset[i]
        img = img.cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = np.uint8((img + 1) / 2 * 255)[:, :, ::-1]
        sem = sem.cpu().numpy()
        print(sem.max(), sem.min())
        sem = mask2color(sem)
        cv2.imwrite(f'results/test_image_{i}.jpg', img)
        cv2.imwrite(f'results/test_seman_{i}.jpg', sem)
