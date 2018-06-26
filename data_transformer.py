import numpy as np
import cv2
import os
import random
from random import shuffle
import math
import util
from skimage.transform import rotate


class BatchLoader(object):
    def __init__(self, params, result):
        self.result = result
        self.batch_size = params['batch_size']
        self.data_dir = params['data_dir']
        self.stride = params['stride']
        self.max_rotate_degree = params['max_rotate_degree']
        self.crop_x = params['crop_size_x']
        self.crop_y = params['crop_size_y']
        self.scale_prob = params['scale_prob']
        self.scale_min = params['scale_min']
        self.scale_max = params['scale_max']
        self.target_dist = params['target_dist']
        self.num_parts = params['num_parts']
        self.visualize_label = params['visualize_label']
        self.center_perterb_max = params['center_perterb_max']
        self.flip_prob = params['flip_prob']
        self.sigma = params['sigma']

        train_list = '{}/train.txt'.format(self.data_dir)
        self.indices = open(train_list, 'r').read().splitlines()
        self._cur = 0

    def load_next_image_label(self):
        if self._cur == len(self.indices):
            self._cur = 0
            shuffle(self.indices)

        im_name = self.indices[self._cur]
        im = cv2.imread('{}/images/{}'.format(self.data_dir, im_name))

        pre, _ = os.path.splitext(im_name)
        part = open('{}/parts/{}.txt'.format(self.data_dir, pre))
        joints = []
        vis = []
        for i, p in enumerate(part):
            joint = []
            is_visible, x ,y = p.split(' ')
            is_visible = int(is_visible)
            x = float(x)
            y = float(y)
            #joint.append(is_visible)
            vis.append(is_visible)
            joint.append(x)
            joint.append(y)
            joints.append(joint)

        #print joints, vis
        img_aug = np.zeros((self.crop_x, self.crop_y), np.uint8)

        #cv2.imwrite("org_img.jpg", im)

        img, joints, scale = self._augmentaion_scale(im, joints)
        #print("scale", scale)
        #cv2.imwrite("scale_img.jpg", img)
        #print joints

        img, joints, degree = self._augmentation_rotate(img, joints)
        #cv2.imwrite("rotate_img.jpg", img)
        #print joints

        img, joints = self._augmentation_croppad(img, joints)
        #cv2.imwrite("croppad_img.jpg", img)
        #print joints

        img, joints, vis, doflip = self._augmentation_flip(img, joints, vis)
        #cv2.imwrite("flip_img.jpg", img)
        #print joints

        labels = self._genreateLabelMap(img, joints, vis)
        #img = img[:, :, :] / 256.0 - 0.5
        self._cur += 1
        return img, labels


    def _augmentaion_scale(self, img, joints):
        dice = random.uniform(0, 1)
        #dice2 = random.uniform(1, 1.5)
        scale_multiplier = (self.scale_max - self.scale_min) * dice + self.scale_min
        #scale_multiplier *= dice2
        img = cv2.resize(img, (0, 0), fx=scale_multiplier, fy=scale_multiplier, interpolation=cv2.INTER_CUBIC)
        for i in range(self.num_parts):
            joints[i][0] *= scale_multiplier
            joints[i][1] *= scale_multiplier

        return img, joints, scale_multiplier

    def _augmentation_rotate(self, img, joints):
        dice = random.uniform(0, 1)
        degree = (dice - 0.5) * 2 * self.max_rotate_degree
        R = cv2.getRotationMatrix2D(center=(img.shape[1]/2, img.shape[0]/2), angle=degree, scale=1)


        angle = degree * cv2.cv.CV_PI / 180
        a = math.sin(angle)
        b = math.cos(angle)
        width_rotate = int(img.shape[0] * math.fabs(a) + img.shape[1] * math.fabs(b))
        height_rotate = int(img.shape[1] * math.fabs(a) + img.shape[0] * math.fabs(b))

        R[0, 2] += width_rotate/2.0 - img.shape[1]/2.0
        R[1, 2] += height_rotate/2.0 - img.shape[0]/2.0
        img = cv2.warpAffine(img, R, (width_rotate, height_rotate), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128))

        for i in range(self.num_parts):
            p = np.zeros((3, 1), np.double)
            p[0, 0] = joints[i][0]
            p[1, 0] = joints[i][1]
            p[2, 0] = 1
            new_p = np.dot(R, p)
            joints[i][0] = new_p[0, 0]
            joints[i][1] = new_p[1, 0]

        return img, joints, degree

    def _onPlane(self, x, y, w, h):
        if x < 0 or y < 0:
            return False
        if x >= w or y >= h:
            return False
        return True

    def _augmentation_croppad(self, img, joints):
        dice_x = random.uniform(0, 1)
        dice_y = random.uniform(0, 1)

        x_offset = int((dice_x - 0.5) * 2 * self.center_perterb_max)
        y_offset = int((dice_y - 0.5) * 2 * self.center_perterb_max)

        center_x = img.shape[1]/2.0 + x_offset
        center_y = img.shape[0]/2.0 + y_offset
        offset_left = -(center_x - (self.crop_x/2))
        offset_up = -(center_y - (self.crop_y/2))
        img_dst = np.zeros((self.crop_y, self.crop_x, 3), np.uint8) + (128, 128, 128)
        for i in range(self.crop_y):
            for j in range(self.crop_x):
                coord_x_on_img = int(center_x - self.crop_x/2 + j)
                coord_y_on_img = int(center_y - self.crop_y/2 + i)
                if self._onPlane(coord_x_on_img, coord_y_on_img, img.shape[1], img.shape[0]):
                    img_dst[i, j, :] = img[coord_y_on_img, coord_x_on_img, :]
        #cv2.imwrite("croppad_img.jpg", img_dst)
        for i in range(self.num_parts):
            joints[i][0] += offset_left
            joints[i][1] += offset_up

        return img_dst, joints

    def _swapLeftRight(self, joints, vis):
        if self.num_parts == 20:
            right = [2, 5, 15, 16, 19, 20]
            left = [1, 4, 13, 14, 17, 18]
            for i in range(6):
                ri = right[i] - 1
                li = left[i] - 1
                t_r = joints[ri][0]
                joints[ri][0] = joints[li][0]
                joints[li][0] = t_r
                t_r = joints[ri][1]
                joints[ri][1] = joints[li][1]
                joints[li][1] = t_r
                t_v = vis[ri]
                vis[ri] = vis[li]
                vis[li] = t_v
        return joints, vis

    def _augmentation_flip(self, img, joints, vis):
        dice = random.uniform(0, 1)
        doflip = (dice <= self.flip_prob)
        if doflip:
            img = cv2.flip(img, 1)
            w = img.shape[1]
            for i in range(self.num_parts):
                joints[i][0] = w - 1 - joints[i][0]

            joints, vis = self._swapLeftRight(joints, vis)

        return img, joints, vis, doflip

    def _putGaussianMaps(self, c_x, c_y, stride, sigma, grid_x, grid_y):
        gaussian_map = np.zeros((grid_y, grid_x))
        start = stride/2.0 - 0.5
        for g_y in range(grid_y):
            for g_x in range(grid_x):
                x = start + g_x * stride
                y = start + g_y * stride
                d2 = (x - c_x) * (x - c_x) + (y - c_y) * (y - c_y)
                exponent = d2 / 2.0 / sigma / sigma
                if exponent > 4.6052:
                    continue
                gaussian_map[g_y, g_x] = math.exp(-exponent)
                if gaussian_map[g_y, g_x] > 1:
                    gaussian_map = 1

        return gaussian_map

    def _genreateLabelMap(self, img, joints, vis):
        stride = self.stride
        grid_x = img.shape[1] / stride
        grid_y = img.shape[0] / stride
        label = np.zeros((grid_y, grid_x, self.num_parts+1), np.float32)
        for i in range(self.num_parts):
            if vis[i] == 1:
                label[:, :, i] = self._putGaussianMaps(joints[i][0], joints[i][1], self.stride,
                                                       self.sigma, grid_x, grid_y)

        for g_y in range(grid_y):
            for g_x in range(grid_x):
                maximum = 0.0
                for i in range(self.num_parts):
                    if maximum <= label[g_y, g_x, i]:
                        maximum = label[g_y, g_x, i]
                label[g_y, g_x, -1] = max(1.0 - maximum, 0.0)

        if self.visualize_label:
            for i in range(self.num_parts+1):
                part_map = label[:, :, i]
                part_map_resized = cv2.resize(part_map, (0, 0), fx=self.stride, fy=self.stride,
                                              interpolation=cv2.INTER_CUBIC)  # only for displaying
                part_map_color = util.colorize(part_map_resized)
                part_map_color_blend = part_map_color * 0.5 + img * 0.5
                cv2.imwrite("label_{}.jpg".format(i), part_map_color_blend)

        return label




if __name__ == "__main__":
    print("test")
    params = {}
    params['batch_size'] = 12
    params['stride'] = 8
    params['data_dir'] = '/home/xiaocc/Documents/caffe-cpm/release/dataset/redpanda_dataset'
    params['max_rotate_degree'] = 40
    params['crop_size_x'] = 368
    params['crop_size_y'] = 368
    params['scale_prob'] = 12
    params['scale_min'] = 0.9
    params['scale_max'] = 1.8
    params['target_dist'] = 12
    params['num_parts'] = 20
    params['visualize_label'] = True
    params['center_perterb_max'] = 0.0
    params['flip_prob'] = 0.5
    params['sigma'] = 7.0
    batch_loader = BatchLoader(params, None)
    batch_loader.load_next_image_label()
    batch_loader.load_next_image_label()
    batch_loader.load_next_image_label()
