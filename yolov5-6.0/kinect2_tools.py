# this module for convenient kinect usage
import time

from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import cv2
import numpy as np
import ctypes
from threading import Thread

# last_depthXs = np.zeros((1080, 1920))

# show single channel img better
import my_utils
from utils.augmentations import letterbox


def single_channel_image_show(img, title):
    # normalize depth for show
    ran = img.max() - img.min()
    depth_for_show = ((img - img.min() / ran * 255) + 0).astype(np.uint8)
    cv2.imshow(title, depth_for_show)


# we use this to do mapping
# depth(512*412) -> (1920, 1080)
# Map Depth Space to Color Space (Image)
def depth_2_color_space(kinect, depth_space_point, depth_frame_data):
    """

    :param kinect: kinect class
    :param depth_space_point: _DepthSpacePoint from PyKinectV2
    :param depth_frame_data: kinect._depth_frame_data
    :param show: shows the aligned image
    :return: return the mapped color frame to depth frame
    """
    # Map Color to Depth Space
    color2depth_points_type = depth_space_point * np.int64(1920 * 1080)  # replace np.int to np.int64
    color2depth_points = ctypes.cast(color2depth_points_type(), ctypes.POINTER(depth_space_point))
    kinect._mapper.MapColorFrameToDepthSpace(ctypes.c_uint(512 * 424), depth_frame_data, ctypes.c_uint(1920 * 1080), color2depth_points)
    depthXYs = np.copy(np.ctypeslib.as_array(color2depth_points, shape=(kinect.color_frame_desc.Height*kinect.color_frame_desc.Width,)))  # Convert ctype pointer to array
    depthXYs = depthXYs.view(np.float32).reshape(depthXYs.shape + (-1,))  # Convert struct array to regular numpy array https://stackoverflow.com/questions/5957380/convert-structured-array-to-regular-numpy-array
    # depthXYs += 0.5
    depthXYs += 0.5
    depthXYs = depthXYs.reshape(kinect.color_frame_desc.Height, kinect.color_frame_desc.Width, 2).astype(np.int64)
    depthXs = np.clip(depthXYs[:, :, 0], 0, kinect.depth_frame_desc.Width - 1)
    depthYs = np.clip(depthXYs[:, :, 1], 0, kinect.depth_frame_desc.Height - 1)
    # global last_depthXs
    # print((depthXs - last_depthXs).any())
    # print((depthXs - last_depthXs).sum())
    # last_depthXs = depthXs
    depth_frame = kinect.get_last_depth_frame()
    depth_img = depth_frame.reshape((kinect.depth_frame_desc.Height, kinect.depth_frame_desc.Width)).astype(np.uint16)
    return depth_img[depthYs, depthXs]
    # if show:
    #     # cv2.imshow('Aligned Image', cv2.resize(cv2.flip(align_depth_img, 1), (int(1920 / 2.0), int(1080 / 2.0))))
    #     cv2.imshow('Aligned Image', cv2.resize(cv2.flip(align_depth_img, 1), (960, 540)))
    #     # cv2.waitKey(3000)
    #     cv2.waitKey(5)
    # if return_aligned_image:
    #     return align_depth_img
    # return depthXs, depthYs


def get_color_image(kinect):
    while True:
        if kinect.has_new_color_frame():
            color_frame = kinect.get_last_color_frame()
            color_image = color_frame.reshape((kinect.color_frame_desc.Height, kinect.color_frame_desc.Width, 4)).astype(
                np.uint8)[:, :, :3]
            return cv2.flip(color_image, 1)


def get_depth_image(kinect):
    while True:
        if kinect.has_new_depth_frame():
            depth_frame = kinect.get_last_depth_frame()
            depth_image = depth_frame.reshape((kinect.depth_frame_desc.Height, kinect.depth_frame_desc.Width,)).astype(
                np.uint16)
            return cv2.flip(depth_image, 1)


def get_aligned_depth_image(kinect):
    while True:
        if kinect.has_new_depth_frame():
            return cv2.flip(depth_2_color_space(kinect, _DepthSpacePoint, kinect._depth_frame_data), 1)


def get_color_and_aligned_depth_image(kinect, is_test=False):
    if is_test:
        return [np.zeros((1920, 1080, 3), dtype=np.uint8), np.zeros((1920, 1080), dtype=np.uint16)]
    return [get_color_image(kinect),
            cv2.flip(depth_2_color_space(kinect, _DepthSpacePoint, kinect._depth_frame_data), 1)]  # for copy


class LoadStreamFromKinect2:
    """

    """
    def __init__(self, img_size=640, strides=(32, 32), auto=True):
        # create kinect2
        self.kinect2 = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth
                                                       | PyKinectV2.FrameSourceTypes_Color)
        self.mode = 'kinect2.0'
        self.img_size = img_size
        self.strides = strides
        self.auto = auto
        # # get w and h
        # w = int(self.kinect2.color_frame_desc.Width)
        # h = int(self.kinect2.color_frame_desc.Height)
        self.fps = 30
        self.is_test = False  # for test
        self.img = get_color_and_aligned_depth_image(self.kinect2, self.is_test)
        self.thread = Thread(target=self.update, daemon=True)
        print(f'success open kinect2.0')
        self.thread.start()
        print('')

    def update(self):
        while True:
            self.img = get_color_and_aligned_depth_image(self.kinect2, self.is_test)
            time.sleep(1 / self.fps)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not self.thread.is_alive() or cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            raise StopIteration

        # letterbox
        img0 = self.img.copy()
        img0[1] = my_utils.depth_to_pseudo_color(img0[1])  # do transfer
        img = [letterbox(x, self.img_size, stride=self.strides[i])[0] for i, x in enumerate(img0)]

        #add dim
        img = [np.expand_dims(x, 0) for x in img]
        img = np.stack(img, 0)

        # Convert
        # img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB and BHWC to BCHW
        # img = np.ascontiguousarray(x)
        img = img[:, :, :, :, ::-1].transpose(0, 1, 4, 2, 3)  # BGR to RGB and BHWC to BCHW
        img = np.ascontiguousarray(img)

        # img 和 img0 都是有一张rgb图一张深度图的ndarray, img = tensor[2, 1, 3, 960, 960], img0 = [tensor(960, 960, 3)] * 2
        return 'k', img, img0, None

    def __len__(self):
        return 1

if __name__ == '__main__':

    # test for alignment
    # I have found that some noise always shows nears the body and hands in depth image
    kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color)

    while True:
        if kinect.has_new_depth_frame():
            color_frame = kinect.get_last_color_frame()
            colorImage = color_frame.reshape((kinect.color_frame_desc.Height, kinect.color_frame_desc.Width, 4)).astype(
                np.uint8)[:, :, :3]
            colorImage = cv2.flip(colorImage, 1)
            cv2.imshow('Test Color View', cv2.resize(colorImage, (960, 540)))

            align_depth_img = depth_2_color_space(kinect, _DepthSpacePoint, kinect._depth_frame_data)
            single_channel_image_show(cv2.resize(cv2.flip(align_depth_img, 1), (960, 540)), 'Aligned Image')
            # single_channel_image_show(cv2.resize(align_depth_img, (960, 540)), 'Aligned Image')


        # Quit using q
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cv2.destroyAllWindows()
