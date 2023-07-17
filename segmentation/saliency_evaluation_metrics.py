import numpy as np
import torch
import torch.nn as nn
import math
import os
from PIL import Image
import torchvision.transforms as T

# Dice Loss
def SoftIoULoss( pred, target):
    # Old One
    pred = torch.sigmoid(pred)
    smooth = 1

    # print("pred.shape: ", pred.shape)
    # print("target.shape: ", target.shape)

    intersection = pred * target
    loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() -intersection.sum() + smooth)

    # loss = (intersection.sum(axis=(1, 2, 3)) + smooth) / \
    #        (pred.sum(axis=(1, 2, 3)) + target.sum(axis=(1, 2, 3))
    #         - intersection.sum(axis=(1, 2, 3)) + smooth)

    loss = 1 - loss.mean()
    # loss = (1 - loss).mean()

    return loss

# Focal loss
def FocalLoss(pred, gt, gamma, alpha):
    pred = torch.sigmoid(pred)
    loss = -1*gt*alpha*((1 - pred)**gamma)*torch.log(pred) - (1-gt)*(1-alpha)*(pred**gamma)*torch.log(1-pred)
    return loss.mean()

def CE_Dice(pred, gt):
    return nn.BCEWithLogitsLoss()(pred, gt)+SoftIoULoss(pred, gt)

def Focal_Dice(pred, gt, gamma, alpha):
    return FocalLoss(pred, gt, gamma, alpha)+SoftIoULoss(pred, gt)


# given pred and gt, return precision and recall for various thresholds
def eval_pr_curve(pred, gt):
    num = 255 # num of thresholds
    precision, recall = torch.zeros(num), torch.zeros(num)
    thlist = torch.linspace(0, 1-1e-10, num) # threshold 0.0 - 1.0

    for i in range(num):
        y_temp = (pred >= thlist[i]).float()
        tp = (y_temp*gt).sum()
        precision[i], recall[i] = tp/(y_temp.sum() + 1e-20), tp/(gt.sum() + 1e-20)
    return precision, recall

# given precision and recall, return AP
def eval_ap(precision, recall):
    # Ref:
    # https://github.com/facebookresearch/Detectron/blob/05d04d3a024f0991339de45872d02f2f50669b3d/lib/datasets/voc_eval.py#L54
    ap_r = np.concatenate(([0.], recall, [1.]))
    ap_p = np.concatenate(([0.], precision, [0.]))
    sorted_idx = np.argsort(ap_r)
    ap_r = ap_r[sorted_idx]
    ap_p = ap_p[sorted_idx]
    count = ap_r.shape[0]

    for i in range(count-1, 0, -1):
        ap_p[i-1] = max(ap_p[i], ap_p[i-1])
    
    i = np.where(ap_r[1:] != ap_r[:-1])[0]
    ap = np.sum((ap_r[i+1] - ap_r[i])*ap_p[i+1])
    return ap
    
# given pred and gt, return tpr(recall) and fpr for various thresholds
def eval_roc(pred, gt):
    num = 255 # num of thresholds
    tpr, fpr = torch.zeros(num), torch.zeros(num)
    thlist = torch.linspace(0, 1-1e-10, num) # threshold 0.0 - 1.0

    for i in range(num):
        y_temp = (pred >= thlist[i]).float()
        tp = (y_temp*gt).sum()
        fp = (y_temp*(1-gt)).sum()
        tn = ((1-y_temp)*(1-gt)).sum()
        fn = ((1-y_temp)*gt).sum()

        tpr[i], fpr[i] = tp/(tp+fn+1e-20), fp/(fp+tn+1e-20)
    return tpr, fpr

# given pred and gt, return E-measure score
def eval_e(pred, gt):
    num = 255
    score = torch.zeros(num)
    thlist = torch.linspace(0, 1 - 1e-10, num)

    for i in range(num):
        y_pred_th = (pred >= thlist[i]).float()
        fm = y_pred_th - y_pred_th.mean()
        y = gt - gt.mean()
        align_matrix = 2*y*fm/(y*y + fm*fm + 1e-20)
        enhanced = ((align_matrix + 1)*(align_matrix + 1))/4
        score[i] = torch.sum(enhanced) / (gt.numel() - 1 + 1e-20)
    return score

# given pred and gt, return S-measure
def eval_s(pred, gt):
    alpha = 0.5
    y = gt.mean()
    if y == 0:
        x = pred.mean()
        s_score = 1.0 - x
    elif y == 1:
        x = pred.mean()
        s_score = x
    else:
        gt[gt >= 0.5] = 1
        gt[gt < 0.5] = 0
        s_score = alpha*_s_object(pred, gt) + (1 - alpha)*_s_region(pred, gt)
        if s_score.item() < 0:
            s_score = torch.FloatTensor[0.0]
    return s_score

def _s_object(pred, gt):
    fg = torch.where(gt == 0, torch.zeros_like(pred), pred)
    bg = torch.where(gt == 1, torch.zeros_like(pred), 1 - pred)
    o_fg = _object(fg, gt)
    o_bg = _object(bg, 1-gt)
    u = gt.mean()
    score = u*o_fg + (1 - u)*o_bg
    return score

def _object(pred, gt):
    temp = pred[gt == 1]
    x = temp.mean()
    sigma_x = temp.std()
    score = 2.0 * x/(x*x + 1.0 +sigma_x + 1e-20)
    return score

def _s_region(pred, gt):
    x, y = _centroid(gt)
    gt1, gt2, gt3, gt4, w1, w2, w3, w4 = _divideGT(gt, x, y)
    p1, p2, p3, p4 = _dividePrediction(pred, x, y)
    Q1 = _ssim(p1, gt1)
    Q2 = _ssim(p2, gt2)
    Q3 = _ssim(p3, gt3)
    Q4 = _ssim(p4, gt4)
    Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
    return Q

def _centroid(gt):
    rows, cols = gt.size()[-2:]
    gt = gt.view(rows, cols)
    if gt.sum() == 0:
        x = torch.eye(1) * round(cols / 2)
        y = torch.eye(1) * round(rows / 2)
    else:
        total = gt.sum()
        i = torch.from_numpy(np.arange(0, cols)).float()
        j = torch.from_numpy(np.arange(0, rows)).float()
        x = torch.round((gt.sum(dim=0) * i).sum() / total + 1e-20)
        y = torch.round((gt.sum(dim=1) * j).sum() / total + 1e-20)
    return x.long(), y.long()

def _divideGT(gt, X, Y):
    h, w = gt.size()[-2:]
    area = h * w
    gt = gt.view(h, w)
    LT = gt[:Y, :X]
    RT = gt[:Y, X:w]
    LB = gt[Y:h, :X]
    RB = gt[Y:h, X:w]
    X = X.float()
    Y = Y.float()
    w1 = X * Y / area
    w2 = (w - X) * Y / area
    w3 = X * (h - Y) / area
    w4 = 1 - w1 - w2 - w3
    return LT, RT, LB, RB, w1, w2, w3, w4

def _dividePrediction(pred, X, Y):
    h, w = pred.size()[-2:]
    pred = pred.view(h, w)
    LT = pred[:Y, :X]
    RT = pred[:Y, X:w]
    LB = pred[Y:h, :X]
    RB = pred[Y:h, X:w]
    return LT, RT, LB, RB

def _ssim(pred, gt):
    gt = gt.float()
    h, w = pred.size()[-2:]
    N = h * w
    x = pred.mean()
    y = gt.mean()
    sigma_x2 = ((pred - x) * (pred - x)).sum() / (N - 1 + 1e-20)
    sigma_y2 = ((gt - y) * (gt - y)).sum() / (N - 1 + 1e-20)
    sigma_xy = ((pred - x) * (gt - y)).sum() / (N - 1 + 1e-20)

    aplha = 4 * x * y * sigma_xy
    beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

    if aplha != 0:
        Q = aplha / (beta + 1e-20)
    elif aplha == 0 and beta == 0:
        Q = 1.0
    else:
        Q = 0
    return Q

# load single image predicted saliency map and ground truth map
def load_pred_and_label(name, pred_dir, gt_dir):
    pred_path = os.path.join(pred_dir, name+'.png')
    gt_path = os.path.join(gt_dir, name+'.png')

    pred = Image.open(pred_path).convert('L')
    gt = Image.open(gt_path).convert('L')

    if pred.size != gt.size:
        pred = pred.resize(gt.size, Image.BILINEAR)

    trans = T.Compose([T.ToTensor()])

    pred = torch.squeeze(trans(pred), 0) # pred(h, w) from 0.0 to 1.0
    gt = torch.squeeze(trans(gt), 0) # gt(h, w) from 0.0 to 1.0
    return pred, gt