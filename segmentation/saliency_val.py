# evaluate the output saliency maps with ground truth maps

import torch
import os
import argparse
from PIL import Image
import numpy as np
import torchvision.transforms as T
from sklearn.metrics import mean_absolute_error

from segmentation.saliency_evaluation_metrics import *

def evaluate(args):

    root = '../datasets/'+args.data

    pred_dir = './results/'+args.data+'/'+args.mapname
    gt_dir = os.path.join(root, 'val/'+'labels')

    print('=================Evaluation Start!====================')
    avg_mae, avg_f, avg_p, avg_r = 0.0, 0.0, 0.0, 0.0
    avg_auc, avg_tpr, avg_fpr = 0.0, 0.0, 0.0
    avg_e, avg_s = 0.0, 0.0
    for img in os.listdir(gt_dir):
        name = img.split('.')[0]
        pred, gt = load_pred_and_label(name, pred_dir, gt_dir)

        # MAE
        mae = mean_absolute_error(gt, pred)
        avg_mae += mae

        # F-measure
        pred = (pred - torch.min(pred))/(torch.max(pred) - torch.min(pred))
        precision, recall = eval_pr_curve(pred, gt)
        f_score = 1.3*recall*precision/(recall+0.3*precision)
        f_score[f_score != f_score] = 0 # for Nan
        avg_f += f_score
        avg_p += precision
        avg_r += recall

        # AUC
        tpr, fpr = eval_roc(pred, gt)
        avg_tpr += tpr
        avg_fpr +=fpr

        # E-measure
        e_score = eval_e(pred, gt)
        avg_e += e_score

        # S-measure
        s_score = eval_s(pred, gt)
        avg_s += s_score


    # MAE
    avg_mae = avg_mae/len(os.listdir(gt_dir))

    # F-measure
    avg_f_score = avg_f/len(os.listdir(gt_dir))
    maxF = avg_f_score.max()
    meanF = avg_f_score.mean()
    avg_p = avg_p/len(os.listdir(gt_dir))
    avg_r = avg_r/len(os.listdir(gt_dir))

    # AP
    ap = eval_ap(avg_p, avg_r)

    # AUC
    avg_tpr, avg_fpr = avg_tpr/len(os.listdir(gt_dir)), avg_fpr/len(os.listdir(gt_dir))
    sorted_idx = torch.argsort(avg_fpr)
    avg_tpr = avg_tpr[sorted_idx]
    avg_fpr = avg_fpr[sorted_idx]
    avg_auc = torch.trapz(avg_tpr, avg_fpr)

    # E-measure
    avg_e = avg_e/len(os.listdir(gt_dir))
    maxE = avg_e.max()
    meanE = avg_e.mean()

    # S-measure
    avg_s = avg_s/len(os.listdir(gt_dir))

    print('=================Evaluation Finished!==================')
    print('MAE: {0:.4f} ||\t maxF: {1:.4f} ||\t aveF: {2:.4f} ||\t AP: {3:.4f} ||\t AUC: {4:.4f} ||\t maxE: {5:.4f} ||\t aveE: {6:.4f} ||\t S-measure: {7:.4f}'.format(avg_mae, maxF, meanF, ap, avg_auc, maxE, meanE, avg_s))




if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data', type=str, help='dataset for training', default='DUTS')

    args = argparser.parse_args()
    evaluate(args)