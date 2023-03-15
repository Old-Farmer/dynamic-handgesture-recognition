import torch
import cv2 as cv
from timeit import default_timer as timer
import numpy as np
from utils.torch_utils import select_device
import utils.metrics


def weighted_boxes_fusion_simplified_cpu(preds, iou_thres=.3):
    """
    numpy version
    框融合，将对preds产生副作用
    preds: [[n1*(xyxy+confidence+class), n2* ...], [..., ...] ], xyxy不是归一化的 使用在gpu上的torch.tensor
    iou_thres: iou threshold， experimentally set as 0.55
    return: 预测框 np.ndarray
    """
    max_wh = 4096  # 最大照片长宽
    device = preds[0].device
    num_img_pair = len(preds) // 2
    preds = [t.to('cpu').numpy() for t in preds]  # 搬移到cpu并转换为numpy
    output = [np.empty((0, 6))] * num_img_pair
    for i in range(num_img_pair):
        c = i
        d = i + num_img_pair
        # 预测框偏移
        preds[c][:, :4] += preds[c][:, 5:6] * max_wh
        preds[d][:, :4] += preds[d][:, 5:6] * max_wh
        iou = box_iou(preds[c][:, :4], preds[d][:, :4])  # iou [preds[0].shape[0], preds[0].shape[1]]
        # return preds[0]
        row_index, column_index = np.where(iou > iou_thres)  # 取得iou满足大小的预测框索引
        p1 = preds[c][row_index]
        p2 = preds[d][column_index]
        pxyxy = (p1[:, :4] * p1[:, 4:5] + p2[:, :4] * p2[:, 4:5]) / (p1[:, 4:5] + p2[:, 4:5])
        pc = (p1[:, 4:5] + p2[:, 4:5]) / 2.0
        p_overlap = np.concatenate((pxyxy, pc, p1[:, 5:6]), axis=1)  # 重合框
        all1, all2 = np.arange(preds[c].shape[0], dtype=int), \
                     np.arange(preds[d].shape[0], dtype=int)
        diff1, diff2 = np.setdiff1d(all1, row_index), np.setdiff1d(all2, column_index)
        p = np.concatenate((p_overlap, preds[c][diff1], preds[d][diff2]))
        p[:, :4] -= p[:, 5:6] * max_wh
        output[i] = p
    return output

    # device = preds.device  # 设备
    # num_models = preds.shape[0]  # 模型数
    # max_wh = 4096  # 最大的宽高
    #
    # b = preds.flatten(0, 1)  # 模型间展开
    # cfs = b[:, 4]  # 置信度
    # b = b[cfs.argsort(descending=True)]  # 依据置信度降序排列
    # b[:, 0:4] += (b[:, 5] * max_wh).repeat(4).reshape(-1, 4)  # 防止对不同类别进行iou运算
    #
    # l, f = [], []
    # # 迭代b中每个预测框
    # for b_box in b:
    #     # 若f中无预测框
    #     if len(f) == 0:
    #         f.append(b_box.tolist())
    #         l.append([b_box.tolist()])
    #     # 否则
    #     else:
    #         iou = utils.metrics.box_iou(b_box[:4].reshape(1, 4), torch.tensor(f, device=device)[:, 0:4])  # iou.shape = [1, len(f)]
    #         for i, is_update in enumerate((iou > iou_thres).flatten()):
    #             flag = 0
    #             if is_update:
    #                 flag = 1
    #                 l[i].append(b_box.tolist())
    #                 temp = torch.tensor(l[i], device=device)
    #                 f[i] = torch.tensor([temp[:, 0:4] * temp[:, 4] / temp[:, 4].sum()])
    #         if flag == 0:
    #             f.append(b_box.tolist())
    #             l.append([b_box.tolist()])
    #
    # # 调整置信度
    # f = torch.tensor(f, device=device)  # f
    # t = torch.tensor([len(v) for v in l], device=device)  # 每个融合框，融合框的个数
    # f[:, 4] *= t / num_models
    # f[:, 0:4] -= (f[:, 5] * max_wh).repeat(4).reshape(-1, 4)  # 防止对不同类别进行iou运算
    # return f


def box_iou(box1, box2):
    """
    numpy version
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Array[N, 4])
        box2 (Array[M, 4])
    Returns:
        iou (Array[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (np.minimum(box1[:, None, 2:], box2[:, 2:]) - np.maximum(box1[:, None, :2], box2[:, :2])).clip(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def wbf_wrapper(pred_c, pred_d):
    boxes, scores, labels = ensemble_boxes.weighted_boxes_fusion([pred_c[:, :4].tolist(), pred_d[:, :4].tolist()],
                                                                 [pred_c[:, 4].tolist(), pred_d[:, 5].tolist()],
                                                                 [pred_c[:, 5].tolist(), pred_d[:, 5].tolist()])
    return torch.from_numpy(np.concatenate(boxes, scores, labels), dim=1).to(pred_c.device)


def depth_to_pseudo_color(depth_img):
    # color_img = np.zeros((depth_img.shape[0], depth_img.shape[1], 3), dtype=np.uint8)
    color_img = np.stack((depth_img >> 8, depth_img & 0xFF, np.zeros_like(depth_img)), axis=2).astype(np.uint8)
    # color_img[:] = [depth_img >> 8, depth_img & 0xFF, 0]
    # for i in range(color_img.shape[0]):
    #     for j in range(color_img.shape[1]):
    #         if depth_img[i, j] != 0:
    #             color_img[i, j] = [depth_img[i, j] >> 8, depth_img[i, j] & 0xFF, 0]
    return color_img


class LoadColorAndDepthImages:

    def __init__(self):
        pass


class Detector:
    """
    检测器，依据静态检测结果进行
    """

    def __init__(self):
        pass

    def inform(self):
        """
        通知其他组件，内容可能是一个检测结果
        """
        pass

    def update(self, pred):
        """
        更新状态
        """
        pass

    def do_detect(self):
        """
        进行检测
        """
        pass


def sub_img(img, xyxy):
    """
    获取xyxy框下 的子图
    """
    return img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]

if __name__ == '__main__':
    # fn = 'D:/yolo_and_data/HANDS/Subject1/images_depth/161.png'
    # img = cv.imread(fn, cv.IMREAD_UNCHANGED)
    # cv.imshow('color', depth_to_pseudo_color(img))
    # cv.imshow('depth', img)
    # while True:
    #     if cv.waitKey(0) == 27:
    #         break
    # cv.destroyAllWindows()
    device = select_device('')
    # device = torch.device('cuda:0')
    # preds = [np.array([[100, 100, 200, 200, 0.85, 5], [500, 500, 600, 600, 0.95, 7]]),
    #          np.array([[100, 99, 199, 201, 0.90, 5], [501, 501, 601, 601, 0.90, 7]])]
    # preds = [torch.tensor([[100, 100, 200, 200, 0.85, 5], [500, 500, 600, 600, 0.95, 7]]),
    #          torch.tensor([[100, 99, 199, 201, 0.90, 5], [501, 501, 601, 601, 0.90, 7]])]
    preds = torch.tensor([[[100, 100, 200, 200, 0.85, 5], [500, 500, 600, 600, 0.95, 7]],
                          [[100, 99, 199, 201, 0.90, 5], [501, 501, 601, 601, 0.90, 7]]], device=device)
    print(preds[0].dtype)
    t = 0
    for i in range(10):
        torch.cuda.synchronize()
        t1 = timer()
        weighted_boxes_fusion_simplified_cpu(preds)
        t2 = timer()
        t += t2 - t1
    # preds = wbf_wrapper(preds[0], preds[1])
    print(f'Average time = {t / 10}s')
    print(preds)

    # # 测试box_iou
    # box1 = torch.arange(4).reshape(-1, 4)
    # box2 = torch.arange(44).reshape(-1, 4)
    # iou = box_iou(box1, box2)
    # print(iou)
    # print(iou.shape)

    # boxes_list = [[
    #     [0.00, 0.51, 0.81, 0.91],
    #     [0.10, 0.31, 0.71, 0.61],
    #     [0.01, 0.32, 0.83, 0.93],
    #     [0.02, 0.53, 0.11, 0.94],
    #     [0.03, 0.24, 0.12, 0.35],
    # ], [
    #     [0.04, 0.56, 0.84, 0.92],
    #     [0.12, 0.33, 0.72, 0.64],
    #     [0.38, 0.66, 0.79, 0.95],
    #     [0.08, 0.49, 0.21, 0.89],
    # ]]
    # scores_list = [[0.9, 0.8, 0.2, 0.4, 0.7], [0.5, 0.8, 0.7, 0.3]]
    # labels_list = [[0, 1, 0, 1, 1], [1, 1, 1, 0]]
    # weights = [2, 1]
    #
    # iou_thr = 0.5
    # skip_box_thr = 0.0001
    # sigma = 0.1
    #
    # t1 = timer()
    # boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights,
    #                                               iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    # t2 = timer()
    # print(t2 - t1)
