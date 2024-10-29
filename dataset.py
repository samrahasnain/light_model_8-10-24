import os
import cv2
import torch
from torch.utils import data
import numpy as np
import random
random.seed(10)
class ImageDataTrain(data.Dataset):
    def __init__(self, data_root, data_list,image_size):
        self.sal_root = data_root
        self.sal_source = data_list
        self.image_size = image_size

        with open(self.sal_source, 'r') as f:
            self.sal_list = [x.strip() for x in f.readlines()]

        self.sal_num = len(self.sal_list)

    def __getitem__(self, item):
        # sal data loading
        im_name = self.sal_list[item % self.sal_num].split()[0]
        de_name = self.sal_list[item % self.sal_num].split()[1]
        gt_name = self.sal_list[item % self.sal_num].split()[2]
        sal_image , im_size= load_image(os.path.join(self.sal_root, im_name), self.image_size)
        sal_depth, im_size = load_image(os.path.join(self.sal_root, de_name), self.image_size)
        sal_label,sal_edge = load_sal_label(os.path.join(self.sal_root, gt_name), self.image_size)

        sal_image, sal_depth, sal_label = cv_random_crop(sal_image, sal_depth, sal_label, self.image_size)
        edge_index, edge_attr = compute_edges_and_features(sal_label, sal_depth, sal_image,self.image_size)  # Pass sal_image

    
        sal_image = sal_image.transpose((2, 0, 1))
        sal_depth = sal_depth.transpose((2, 0, 1))
        sal_label = sal_label.transpose((2, 0, 1))
        sal_edge = sal_edge.transpose((2, 0, 1))

        sal_image = torch.Tensor(sal_image)
        sal_depth = torch.Tensor(sal_depth)
        sal_label = torch.Tensor(sal_label)
        sal_edge = torch.Tensor(sal_edge)

        return {
        'sal_image': sal_image,
        'sal_depth': sal_depth,
        'sal_label': sal_label,
        'edge_index': edge_index,
        'edge_attr': edge_attr    }
        

    def __len__(self):
        return self.sal_num

def compute_edges_and_features( sal_label, sal_depth,sal_image,image_size):
    h, w = image_size, image_size
    edge_index = []
    depth_attrs = []
    sal_image_attrs = []
    sal_label_attrs = []

    for i, j in product(range(h), range(w)):
        node_id = i * w + j
        for ni, nj in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
            if 0 <= ni < h and 0 <= nj < w:
                neighbor_id = ni * w + nj
                edge_index.append([node_id, neighbor_id])
                
                # Compute feature differences
                depth_diff = torch.abs(sal_depth[:, i, j] - sal_depth[:, ni, nj]).mean().item()
                sal_image_diff = torch.abs(sal_image[:, i, j] - sal_image[:, ni, nj]).mean().item()  # Assuming sal_image is passed
                sal_label_diff = torch.abs(sal_label[:, i, j] - sal_label[:, ni, nj]).mean().item()
                
                # Store individual attributes
                depth_attrs.append(depth_diff)
                sal_image_attrs.append(sal_image_diff)
                sal_label_attrs.append(sal_label_diff)

    edge_index = torch.LongTensor(edge_index).t().contiguous()

    # Combine the individual attributes
    combined_edge_attr = [(sal_depth, sal_image, sal_label) for sal_depth, sal_image, sal_label in zip(depth_attrs, sal_image_attrs, sal_label_attrs)]
    edge_attr = torch.Tensor(combined_edge_attr)  # Now it has three features per edge

    return edge_index, edge_attr



class ImageDataTest(data.Dataset):
    def __init__(self, data_root, data_list,image_size):
        self.data_root = data_root
        self.data_list = data_list
        self.image_size = image_size
        with open(self.data_list, 'r') as f:
            self.image_list = [x.strip() for x in f.readlines()]

        self.image_num = len(self.image_list)

    def __getitem__(self, item):
        image, im_size = load_image_test(os.path.join(self.data_root, self.image_list[item].split()[0]), self.image_size)
        depth, de_size = load_image_test(os.path.join(self.data_root, self.image_list[item].split()[1]), self.image_size)
        image = torch.Tensor(image)
        depth = torch.Tensor(depth)
        return {'image': image, 'name': self.image_list[item % self.image_num].split()[0].split('/')[1],
                'size': im_size, 'depth': depth}

    def __len__(self):
        return self.image_num


def get_loader(config, mode='train', pin=True):
    shuffle = False
    if mode == 'train':
        shuffle = True
        dataset = ImageDataTrain(config.train_root, config.train_list, config.image_size)
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=shuffle,
                                      num_workers=config.num_thread, pin_memory=pin)
    else:
        dataset = ImageDataTest(config.test_root, config.test_list, config.image_size)
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=shuffle,
                                      num_workers=config.num_thread, pin_memory=pin)
    return data_loader



def load_image(path,image_size):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path)
    in_ = np.array(im, dtype=np.float32)
    im_size = tuple(in_.shape[:2])
    in_ = cv2.resize(in_, (image_size, image_size))
    in_ = Normalization(in_)
    return in_,im_size



def load_image_test(path,image_size):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path)
    in_ = np.array(im, dtype=np.float32)
    im_size = tuple(in_.shape[:2])
    in_ = cv2.resize(in_, (image_size, image_size))
    in_ = Normalization(in_)
    in_ = in_.transpose((2, 0, 1))
    return in_, im_size


def load_sal_label(path,image_size):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    #gradient
    gX = cv2.Sobel(im, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    gY = cv2.Sobel(im, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
    gX = cv2.convertScaleAbs(gX)
    gY = cv2.convertScaleAbs(gY)
    combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
    combined = np.array(combined , dtype=np.float32)
    combined  = cv2.resize(combined , (image_size, image_size))
    combined  = combined  / 255.0
    combined  = combined [..., np.newaxis]

    label = np.array(im, dtype=np.float32)
    label = cv2.resize(label, (image_size, image_size))
    label = label / 255.0
    label = label[..., np.newaxis]
    return label,combined


def cv_random_crop(image, depth, label,image_size):
    crop_size = int(0.0625*image_size)
    croped = image_size - crop_size
    top = random.randint(0, crop_size)  #crop rate 0.0625
    left = random.randint(0, crop_size)

    image = image[top: top + croped, left: left + croped, :]
    depth = depth[top: top + croped, left: left + croped, :]
    label = label[top: top + croped, left: left + croped, :]
    image = cv2.resize(image, (image_size, image_size))
    depth = cv2.resize(depth, (image_size, image_size))
    label = cv2.resize(label, (image_size, image_size))
    label = label[..., np.newaxis]
    return image, depth, label

def Normalization(image):
    in_ = image[:, :, ::-1]
    in_ = in_ / 255.0
    in_ -= np.array((0.485, 0.456, 0.406))
    in_ /= np.array((0.229, 0.224, 0.225))
    return in_
