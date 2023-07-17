import torch.nn as nn
import torch

# train mots0_015, ce loss

import os
import argparse
from sklearn.metrics import confusion_matrix, precision_recall_curve, mean_absolute_error
import torch
from torch import nn, no_grad
from torch.utils.data import DataLoader
import numpy as np
from torch import optim

# from data.datagenerator import dataset
from dataset.dsloader import AttnData
from model.patch_attention import SAMAttn, patchAttn1, patchAttn2, SwinAttn, SwinAttn_pyramid, SwinAttn_pyramid_sc, SwinAttn_pyramid_token
from utils import wce_loss, DiceLoss

def adjust_learning_rate(optimizer, decay_rate=.1):
    update_lr_group = optimizer.param_groups
    for param_group in update_lr_group:
        print('before lr: ', param_group['lr'])
        param_group['lr'] = param_group['lr'] * decay_rate
        print('after lr: ', param_group['lr'])
    return optimizer

def main(args):
    print(args)

    # load data
    print('============================loading data============================')
    root = os.path.join('../datasets', args.data) # dataset path
    dataset_tr = AttnData(root, args, 'train')
    dataset_te = AttnData(root, args, 'val')
    train_loader = DataLoader(dataset_tr, args.batchsz, num_workers=4, shuffle=True)
    test_loader = DataLoader(dataset_te, args.batchsz, num_workers=4, shuffle=True)

    # check cuda
    device = torch.device(args.dev if torch.cuda.is_available() else 'cpu')
    print('training device:', device)

    # build model
    num_ch = 3 if args.channel == 'rgb' else 1
    num_cls = 2
    model = SwinAttn_pyramid_token(args, num_cls)
    model = model.to(device)
    # print(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    criterion = nn.CrossEntropyLoss(torch.Tensor([1, args.beta]).to(device=device)) # weighted cross entropy
    dice_loss = DiceLoss(num_cls)
    # results saving path
    resultfolder = os.path.join('./results', args.data)
    if not os.path.exists(resultfolder):
        os.mkdir(resultfolder)

    modelfolder = os.path.join(resultfolder, 'checkpoint')
    if not os.path.exists(modelfolder):
        os.mkdir(modelfolder)
    aug = 'aug' if args.aug else 'ori'
    netname = 'ulite' if args.lite else 'unet'
    lo = 'bce' if args.loss != 'wce' else 'wce'+str(args.beta)
    modelpath = os.path.join(modelfolder, str(args.beta)+'swinnet_pyramid_final.pth')
    best_modelpath = os.path.join(modelfolder, str(args.beta)+'swinnet_pyramid_best.pth')
    
    # train and validate
    print('============================Training============================')
    # model.load_state_dict(torch.load(modelpath))

    train_loss, test_loss = 0.0, 0.0
    tn, fp, fn, tp = 0.0, 0.0, 0.0, 0.0
    cm = np.zeros((2, 2))
    best_score = 0.0
    weights = [1, 1, 1]
    for epoch in range(args.epoch):
        if (epoch+1)%(args.epoch//3) == 0:
            optimizer = adjust_learning_rate(optimizer, decay_rate=0.1)
        # train model
        for xtr, ytr in train_loader:
            # print(xtr.shape, ytr.shape)
            # change learning rate
            a128tr = nn.MaxPool2d(kernel_size=128, stride=128)(ytr)
            a64tr = nn.MaxPool2d(kernel_size=64, stride=64)(ytr)
            a32tr = nn.MaxPool2d(kernel_size=32, stride=32)(ytr)

            xtr, ytr, a128tr = xtr.to(device), ytr.to(device), a128tr.to(device)
            a64tr, a32tr = a64tr.to(device), a32tr.to(device)
            optimizer.zero_grad()
            att, att64, att32 = model(xtr)

            if args.loss == 'wce':
                ltr = wce_loss(att, a128tr, args.beta, device)
            else:
                # atr = torch.squeeze(atr).long() if args.loss == 'ce' else atr
                ltr1 = criterion(att, torch.squeeze(a128tr).long() if args.loss == 'ce' else a128tr)
                ltr2 = criterion(att64, torch.squeeze(a64tr).long())
                ltr3 = criterion(att32, torch.squeeze(a32tr).long())
                ltr = weights[0]*ltr1+weights[1]*ltr2+weights[2]*ltr3
                # loss_dice = dice_loss(att, torch.squeeze(atr).long() if args.loss == 'ce' else atr, softmax=True)
                # ltr = 0.4 * ltr + 0.6 * loss_dice
            # ltr = criterion(ptr, ytr)
            ltr.backward()
            optimizer.step()

            train_loss += ltr.item()
        
        # evaluate model
        pred = np.zeros((0,1,args.token_size,args.token_size))
        gt = np.zeros((0,1,args.token_size,args.token_size))
        with torch.no_grad():
            for xte, yte, h, w, name in test_loader:
                ate = nn.MaxPool2d(kernel_size=128, stride=128)(yte)
                xte, yte, ate = xte.to(device), yte.to(device), ate.to(device)
                
                pte, p64te, p32te = model(xte)


                if args.loss == 'wce':
                    lte = wce_loss(pte, ate, args.beta, device)
                else:
                    # ate = torch.squeeze(ate).long() if args.loss == 'ce' else ate
                    lte = criterion(pte, torch.squeeze(ate).long() if args.loss == 'ce' else ate)
                    # loss_dice = dice_loss(pte, torch.squeeze(ate).long() if args.loss == 'ce' else ate, softmax=True)
                    # lte = 0.4 * lte + 0.6 * loss_dice
                test_loss += lte.item()
                
                # pte = nn.Upsample(scale_factor=4, mode='nearest')(pte)
                # p64te = nn.Upsample(scale_factor=2, mode='nearest')(p64te)
                p64te = nn.MaxPool2d(kernel_size=2, stride=2)(p64te)
                p32te = nn.MaxPool2d(kernel_size=4, stride=4)(p32te)

                # pte = pte+p64te+p32te
                pte = torch.maximum(pte, p64te)
                pte = torch.maximum(pte, p32te)

                if args.loss != 'bce':
                    pte = torch.unsqueeze(torch.argmax(pte, 1), 1)
                pte = pte.cpu().numpy()
                ate = ate.cpu().numpy()
                # print(pte.shape, yte.shape)
                # print(np.unique(ate))
                ate[ate>=0.5] = 1
                ate[ate<0.5] = 0

                pred = np.append(pred, pte, axis=0)
                gt = np.append(gt, ate, axis=0)
                # print(np.unique(pte))
                pte[pte>=0.001] = 1
                pte[pte<0.001] = 0
                cm += confusion_matrix(ate.astype(np.int32).flatten(), pte.flatten())

        pred = pred.flatten()
        gt = gt.flatten()
        precision, recall, threshold = precision_recall_curve(gt, pred)
        f_scores = 1.3*recall*precision/(recall+0.3*precision+ 1e-20)

        mae = mean_absolute_error(gt, pred)
        tn, fp, fn, tp = cm.ravel()
        # pr = tp/(tp+fp)
        # rc = tp/(tp+fn)

        print('epoch', epoch+1, '\ttrain loss:', "{:.4f}".format(train_loss/len(train_loader)), '\ttest loss', "{:.4f}".format(test_loss/len(test_loader)),'\ttp', "{:.4f}".format(tp/(tp+fn+tn+fp)), '\tfn', "{:.4f}".format(fn/(tp+fn+tn+fp)), '\tfp', "{:.4f}".format(fp/(tp+fn+tn+fp)),
              '\ttn', "{:.4f}".format(tn/(tp+fn+tn+fp)), '\tMAE:', "{:.4f}".format(mae), '\tmaxf:', "{:.4f}".format(np.max(f_scores)), '\tIoU:', "{:.4f}".format(tp/(tp+fn+fp+ 1e-20)), '\tTPR:', "{:.4f}".format(tp/(tp+fn+ 1e-20)))

        if tp/(tp+fn+ 1e-20) > best_score:
            best_score = tp/(tp+fn+ 1e-20)
            best_mae, best_maxf, best_IoU, best_tpr = mae, np.max(f_scores), tp/(tp+fn+fp), tp/(tp+fn)
            torch.save(model.state_dict(), best_modelpath)

        torch.save(model.state_dict(), modelpath)

        cm = np.zeros((2, 2))
        train_loss = 0.0
        test_loss = 0.0
    
    print('============================Training Done!============================')
    print('final result:\n', '\tMAE:', "{:.4f}".format(best_mae), '\tmaxf:', "{:.4f}".format(best_maxf),
        '\tIoU:', "{:.4f}".format(best_IoU), '\tTPR:', "{:.4f}".format(best_tpr))



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data', type=str, help='which dataset to use(mots/coco/DUTS/MSRA10K/bdd10k/bdd10ks)', default='mots')
    argparser.add_argument('--gray', action='store_true', help='transform RGB to grayscale image or not')
    argparser.add_argument('--channel', type=str, help='rgb/gray', default='rgb')
    argparser.add_argument('--imgsz', type=int, help='image size(for mots, need two values for height and width, eg. 320*180)', default=1024)
    argparser.add_argument('--padsz', type=int, help='(optional)if pad to some image size first and then downsample, then use', default=512) # then downsample ratio would be padsz/imgsz
    argparser.add_argument('--ds_method', type=str, help='downsample method(max/mean/bilinear)', default='max')
    argparser.add_argument('--aug', action='store_true', help='data augmentation or not(ori/aug)')
    argparser.add_argument('--norm', action='store_true', help='normalize or not')
    argparser.add_argument('--token_size', type=int, help='number of patches', default=8)
    argparser.add_argument('--pretrain', action='store_true', help='load pretrained model or not')
    argparser.add_argument('--reduce_ratio', type=float, help='how many tokens to remove', default='0.5')


    argparser.add_argument('--model', type=str, help='which model', default='unet')
    argparser.add_argument('--dim', type=int, help='attention embedding dimension', default=96)
    argparser.add_argument('--tokensz', type=int, help='tokensize for embedding', default=4)
    argparser.add_argument('--depth', type=int, help='number of attention layers in one block', default=1)
    argparser.add_argument('--th', type=float, help='threshold for attention or not', default=0.5)
    argparser.add_argument('--dev', type=str, help='cuda device', default='cuda:0')
    argparser.add_argument('--epoch', type=int, help='number of training epochs', default=50)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.00001)
    argparser.add_argument('--lr_slot', type=int, help='learning rate change point(related to batch size)', default=2000)
    argparser.add_argument('--batchsz', type=int, help='batch size(12/15/32/64/128)', default=16)
    argparser.add_argument('--loss', type=str, help='loss function(bce/ce/wce)', default='ce')
    argparser.add_argument('--beta', type=float, help='fn/fp ratio', default=1)
    argparser.add_argument('--w', type=int, help='weights for loss', default=0)
    # argparser.add_argument('--dir', type=str, help='model saving directory', default='mots320')
    argparser.add_argument('--lite', action='store_true', help='use simplified UNet or not')

    args = argparser.parse_args()
    main(args)
