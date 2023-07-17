# use trained model to test images and output the saliency map
from time import time
import torch
import os
from sklearn.metrics import confusion_matrix
import torch.nn as nn
import argparse
import torchvision.transforms as T

from dataset.dsloader import Data
from torch.utils.data import DataLoader
import torch.nn.functional as F

# from model.EST_rgbm import EST_rgbm
# from model.VST.ImageDepthNet import ImageDepthNet
# from data.contour_pyramid_loader import CPData
from model.ssnet import SSNet, UNet
from model.swinNet import SST_RGB

from segmentation.saliency_val import evaluate

def predict(args):

    # load model
    device = torch.device(args.dev if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=3, n_classes=1, pretrain=False, bilinear=True)
    # model = SSNet(n_channels=3, n_classes=1, token_size=args.token_size, threshold=args.th,bilinear=True)
    # model = SST_RGB(num_classes=1, args=args)
    model.to(device)

    modelpath = './results/bdd10k05_07/checkpoints/bdd10k05_07256oriunet_final.pth'
    state_dict = torch.load(modelpath)
    model.load_state_dict(state_dict)

    # load data
    root = os.path.join('../datasets/', args.data)
    data_te = Data(root, args, 'val')
    test_loader = DataLoader(data_te, 1, shuffle=True)

    # predict saliency map and save
    print('==========================Predicting=======================')

    for xte, yte, h, w, name in test_loader:
        xte, yte = xte.to(device), yte.to(device)
        # pte, att = model(xte)
        pte = model(xte)
        # outputs_saliency, outputs_contour = model(xte)
        # _, _, _, pte = outputs_saliency
        smap = F.sigmoid(pte)
        smap = smap.data.cpu().squeeze(0)

        h, w = h[0], w[0]
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((h, w))
        ])
        smap = transform(smap)

        save_folder = os.path.join('./results/'+args.data, args.mapname)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        save_path = os.path.join(save_folder, name[0]+'.png')
        smap.save(save_path)

    print('==========================Done!=======================')

    evaluate(args)


    
if __name__=='__main__':
    argparser = argparse.ArgumentParser()
    # data
    argparser.add_argument('--data', type=str, help='dataset for training', default='bdd10k05_07')
    argparser.add_argument('--motion', action='store_true', help='train with motion map or not')
    argparser.add_argument('--gray', action='store_true', help='transform RGB to grayscale image or not')
    argparser.add_argument('--aug', action='store_true', help='data augmentation or not')
    argparser.add_argument('--norm', action='store_true', help='data normalization or not')
    argparser.add_argument('--imgsz', type=int, nargs='+', help='image size for training(could input one or two values(height/width))', default=256)
    argparser.add_argument('--img_size', type=int, nargs='+', help='image size for training(could input one or two values(height/width))', default=256)
    argparser.add_argument('--token_size', type=int, help='patch size for token generation(4*4, 7*7)', default=28)


    # model
    argparser.add_argument('--th', type=float, help='threshold for attention or not', default=0.05)
    argparser.add_argument('--pretrained_model', type=str, help='pretrained model path', default='./swin_tiny_patch4_window7_224.pth')
    argparser.add_argument('--batchsz', type=int, help='batch size', default=11)
    argparser.add_argument('--dev', type=str, help='cuda device', default='cuda:0')
    argparser.add_argument('--mapname', type=str, help='output saliency map folder name', default='map')
    args = argparser.parse_args()

    predict(args)