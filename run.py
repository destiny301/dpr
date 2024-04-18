import torch.nn as nn
import torch
import os
import argparse
from sklearn.metrics import confusion_matrix, precision_recall_curve, mean_absolute_error
import torch
from torch import nn, no_grad
from torch.utils.data import DataLoader
import numpy as np
from torch import optim

from dataset.dsloader import Data
from model.patch_attention import PatchSelector
from PIL import Image
import cv2

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
    print('============================ loading data ============================')
    root = os.path.join('../datasets', args.data) # dataset path
    dataset_tr = Data(root, args, 'train')
    dataset_te = Data(root, args, 'val')
    train_loader = DataLoader(dataset_tr, args.batchsz, num_workers=4, shuffle=True)
    test_loader = DataLoader(dataset_te, args.batchsz, num_workers=4, shuffle=True)

    # check cuda
    device = torch.device(args.dev if torch.cuda.is_available() else 'cpu')
    print('training device:', device)

    # build model
    num_ch = 3
    num_cls = 1 if args.loss == 'bce' else 2
    model = PatchSelector(args, num_cls)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    criterion = nn.BCEWithLogitsLoss() if args.loss == 'bce' else nn.CrossEntropyLoss(torch.Tensor([1, args.beta]).to(device=device)) # weighted cross entropy

    # train and validate
    print('============================ Model Training ============================')
    train_loss, test_loss = 0.0, 0.0
    tn, fp, fn, tp = 0.0, 0.0, 0.0, 0.0
    cm = np.zeros((2, 2))
    weights = [1, 1, 1]
    for epoch in range(args.epoch):
        # adjust learning rate
        if (epoch+1)%(args.epoch//3) == 0:
            optimizer = adjust_learning_rate(optimizer, decay_rate=0.1)

        # train model
        for xtr, ytr in train_loader:
            # generate pyramid labels
            a3tr = nn.MaxPool2d(kernel_size=(args.imgsz//args.num_patch), stride=(args.imgsz//args.num_patch))(ytr)
            a2tr = nn.MaxPool2d(kernel_size=(args.imgsz//(args.num_patch*2)), stride=(args.imgsz//(args.num_patch*2)))(ytr)
            a1tr = nn.MaxPool2d(kernel_size=(args.imgsz//(args.num_patch*4)), stride=(args.imgsz//(args.num_patch*4)))(ytr)

            xtr, ytr = xtr.to(device), ytr.to(device)
            a3tr, a2tr, a1tr = a3tr.to(device), a2tr.to(device), a1tr.to(device)
            optimizer.zero_grad()
            p3tr, p2tr, p1tr = model(xtr)

            ltr1 = criterion(p3tr, torch.squeeze(a3tr, dim=1).long() if args.loss == 'ce' else a3tr)
            ltr2 = criterion(p2tr, torch.squeeze(a2tr, dim=1).long() if args.loss == 'ce' else a2tr)
            ltr3 = criterion(p1tr, torch.squeeze(a1tr, dim=1).long() if args.loss == 'ce' else a1tr)
            ltr = weights[0]*ltr1+weights[1]*ltr2+weights[2]*ltr3

            ltr.backward()
            optimizer.step()

            train_loss += ltr.item()
        
        # evaluate model
        pred = np.zeros((0,1,args.num_patch,args.num_patch))
        gt = np.zeros((0,1,args.num_patch,args.num_patch))
        with torch.no_grad():
            for xte, yte, name in test_loader:
                ate = nn.MaxPool2d(kernel_size=(args.imgsz//args.num_patch), stride=(args.imgsz//args.num_patch))(yte)
                xte, yte, ate = xte.to(device), yte.to(device), ate.to(device)
                
                pte, p64te, p32te = model(xte)

                # aggregate pyramid predictions
                p64te = nn.MaxPool2d(kernel_size=2, stride=2)(p64te)
                p32te = nn.MaxPool2d(kernel_size=4, stride=4)(p32te)
                pte = torch.maximum(pte, p64te)
                pte = torch.maximum(pte, p32te)
                
                lte = criterion(pte, torch.squeeze(ate, dim=1).long() if args.loss == 'ce' else ate)
                test_loss += lte.item()
                pte = nn.Sigmoid()(pte)
                if args.loss != 'bce':
                    pte = torch.unsqueeze(torch.argmax(pte, 1), 1)
                pte = pte.cpu().numpy()
                ate = ate.cpu().numpy()

                ate[ate>=0.5] = 1
                ate[ate<0.5] = 0

                
                pred = np.append(pred, pte, axis=0)
                gt = np.append(gt, ate, axis=0)
                pte[pte>=args.th] = 1 # adjustable threshold
                pte[pte<args.th] = 0
                cm += confusion_matrix(ate.astype(np.int32).flatten(), pte.flatten())

        pred = pred.flatten()
        gt = gt.flatten()
        precision, recall, threshold = precision_recall_curve(gt, pred)
        f_scores = 1.3*recall*precision/(recall+0.3*precision+ 1e-20)

        mae = mean_absolute_error(gt, pred)
        tn, fp, fn, tp = cm.ravel()

        print('epoch', epoch+1, '\ttrain loss:', "{:.4f}".format(train_loss/len(train_loader)), '\ttest loss', "{:.4f}".format(test_loss/len(test_loader)),'\ttp', "{:.4f}".format(tp/(tp+fn+tn+fp)), '\tfn', "{:.4f}".format(fn/(tp+fn+tn+fp)), '\tfp', "{:.4f}".format(fp/(tp+fn+tn+fp)),
              '\ttn', "{:.4f}".format(tn/(tp+fn+tn+fp)), '\tMAE:', "{:.4f}".format(mae), '\tmaxf:', "{:.4f}".format(np.max(f_scores)), '\tIoU:', "{:.4f}".format(tp/(tp+fn+fp+ 1e-20)), '\tTPR:', "{:.4f}".format(tp/(tp+fn+ 1e-20)))

        cm = np.zeros((2, 2))
        train_loss = 0.0
        test_loss = 0.0

    if args.patch_selection_and_save:
        # after training, patch selection and save selected patches
        print('============================ Patch Selection ============================')

        imgroot = os.path.join(root, 'images')
        selected_patchroot = os.path.join(root, 'selected_patches')
        if not os.path.exists(selected_patchroot):
            os.mkdir(selected_patchroot)
        if args.nonselection_save:
            nonselected_patchroot = os.path.join(root, 'nonselected_patches')
            if not os.path.exists(nonselected_patchroot):
                os.mkdir(nonselected_patchroot)

        id = np.arange(args.num_patch*args.num_patch)
        for dir in os.listdir(imgroot):
            for img in os.listdir(os.path.join(imgroot, dir)):
                name = img.split('.')[0]
                x = Image.open(os.path.join(imgroot, img))
                x = np.asarray(x)
                x = cv2.resize(x, [args.imgsz, args.imgsz], interpolation=cv2.INTER_LINEAR).astype(np.float32)
                x = torch.from_numpy(x)
                x = x.unsqueeze(0).permute(0, 3, 1, 2)

                pte, p64te, p32te = model(x)

                p64te = nn.MaxPool2d(kernel_size=2, stride=2)(p64te)
                p32te = nn.MaxPool2d(kernel_size=4, stride=4)(p32te)
                pte = torch.maximum(pte, p64te)
                pte = torch.maximum(pte, p32te)
                pte = nn.Sigmoid()(pte)
                pte[pte>=args.th] = 1 # adjustable threshold
                pte[pte<args.th] = 0
                pte = pte.flatten()

                selected_ids = id[pte == 1]
                x = x.permute(0, 2, 3, 1).squeeze(0)
                for i in range(args.num_patch*args.num_patch):
                    if i in selected_ids:
                        patch = x[i//args.num_patch*(args.imgsz//args.num_patch):(i//args.num_patch+1)*(args.imgsz//args.num_patch), i%args.num_patch*(args.imgsz//args.num_patch):(i%args.num_patch+1)*(args.imgsz//args.num_patch), :]
                        patch = Image.fromarray(patch.numpy().astype(np.uint8))
                        patch.save(os.path.join(selected_patchroot, name+'_'+str(i)+'.jpg'))
                    elif args.nonselection_save:
                        patch = x[i//args.num_patch*(args.imgsz//args.num_patch):(i//args.num_patch+1)*(args.imgsz//args.num_patch), i%args.num_patch*(args.imgsz//args.num_patch):(i%args.num_patch+1)*(args.imgsz//args.num_patch), :]
                        patch = Image.fromarray(patch.numpy().astype(np.uint8))
                        patch.save(os.path.join(nonselected_patchroot, name+'_'+str(i)+'.jpg'))
                print('select patches for image:', name)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data', type=str, help='which dataset to use(coco2017/MSRA10K/bdd100k)', default='bdd100k')
    argparser.add_argument('--imgsz', type=int, help='image size', default=1024)
    argparser.add_argument('--aug', action='store_true', help='data augmentation or not')
    argparser.add_argument('--norm', action='store_true', help='normalize data or not')
    argparser.add_argument('--num_patch', type=int, help='number of patches (eg. 4*4 or 8*8)', default=8)

    argparser.add_argument('--dim', type=int, help='attention embedding dimension for patch selection', default=96)
    argparser.add_argument('--tokensz', type=int, help='token size for image embedding', default=8)
    argparser.add_argument('--th', type=float, help='threshold for attention or not', default=0.5)
    argparser.add_argument('--dev', type=str, help='cuda device', default='cuda:0')
    argparser.add_argument('--epoch', type=int, help='number of training epochs', default=50)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.00001)
    argparser.add_argument('--batchsz', type=int, help='batch size', default=32)
    argparser.add_argument('--loss', type=str, help='loss function(bce/ce)', default='bce')
    argparser.add_argument('--beta', type=float, help='weighted cross entropy parameter', default=1)

    # patch selection
    argparser.add_argument('--patch_selection_and_save', action='store_true', help='patch selection or not')
    argparser.add_argument('--nonselection_save', action='store_true', help='save non-selected patches or not')
    args = argparser.parse_args()
    main(args)
