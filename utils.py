import os
from typing import Any
import cv2
import numpy as np
import torch
from torch import nn

from PIL import Image
from cleanfid import fid
# from ignite.metrics import SSIM, PSNR
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch.utils.data import DataLoader

from torch.utils.data import Dataset
from tqdm import tqdm
import shutil

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
    
def draw_bb():
    img_path = '../datasets/mots0_015/images/val/b1d59b1f-a38aec79-0000058.jpg'
    label_path = '../datasets/mots0_015/labels/val/b1d59b1f-a38aec79-0000058.txt'
    with open(label_path, 'r') as f:
        lines = f.readlines()
    # line = lines[0]
    img = cv2.imread(img_path)
    for line in lines:
        x, y, w, h = line.split(' ')[1], line.split(' ')[2], line.split(' ')[3], line.split(' ')[4].strip()
        x, y, w, h = float(x), float(y), float(w), float(h)
        # print(x, y, w, h)
        x, y, w, h = x*1280, y*720, w*1280, h*720
        x1, x2, y1, y2 = x-w/2, x+w/2, y-h/2, y+h/2
        start_point = (int(x1)-9, int(y1)-9)
        end_point = (int(x2)+9, int(y2)+9)
        
        mask = np.zeros((img.shape))
        cv2.rectangle(mask, start_point, end_point, color=(255,255,255), thickness=-1)
        # mask[int(y1):int(y2), int(x1):int(x2)] = 0
        # cv2.imwrite("mask.jpg", mask)
        img = img+0.1*mask
        cv2.imwrite("mask.jpg", img)
        cv2.rectangle(img, start_point, end_point, color=(255,51,51), thickness=5)
    cv2.imwrite("example_with_bounding_boxes_bg9.jpg", img)
    print(lines)

def compute_ratio():
    img_path = '../datasets/mots0_015/images/val/b1d59b1f-a38aec79-0000058.jpg'
    mask_path = '../datasets/mots0_015/masks/val/b1d59b1f-a38aec79-0000058.png'

    m = Image.open(mask_path)
    m = np.asarray(m)
    print(np.unique(m))
    fg = np.sum(m)/255.0
    total = 1280*720
    print(fg/total)

def add_circle():
    imgpath = './example_with_bounding_boxes_bg9.jpg'
    img = cv2.imread(imgpath)
    # img = cv2.resize(img, )

    mask = np.ones((img.shape))*255
    cv2.circle(mask, (88, 352), 50, color=(0,0,0), thickness=-1)
    cv2.circle(mask, (341, 339), 80, color=(0,0,0), thickness=-1)
    img = img+0.35*mask
    mask = np.zeros((img.shape))*255
    cv2.circle(mask, (88, 352), 50, color=(100,100,100), thickness=5)
    cv2.circle(mask, (341, 339), 80, color=(100,100,100), thickness=5)
    img = img+0.7*mask
    cv2.imwrite("circle4.jpg", img)

def draw_grid(img, grid_shape, color=(255, 255, 255), thickness=1):
    h, w, _ = img.shape
    rows, cols = grid_shape
    dy, dx = h / rows, w / cols

    # draw vertical lines
    for x in np.linspace(start=dx, stop=w-dx, num=cols-1):
        x = int(round(x))
        cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

    # draw horizontal lines
    for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
        y = int(round(y))
        cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

    return img

def compute_ave_object(dir):
    fg = 0.0
    total = 0.0

    for folder in os.listdir(dir):
        folderpath = os.path.join(dir, folder)
        for img in os.listdir(folderpath):
            maskpath = os.path.join(folderpath, img)
            m = Image.open(maskpath)
            m = np.asarray(m)
            # print(m.shape)
            fg += np.sum(m)
            total += 720*1280*255
            # print(img)
    print(fg, total, fg/total)

################# evaluate FID/SSIM... ################
#######################################################
def compute_FID(dir):
    ori_folder = os.path.join(dir, 'hr_128')
    bi_folder = os.path.join(dir, 'sr_32_128')
    dm_folder = os.path.join(dir, 'sr_128')

    fid1 = fid.compute_fid(ori_folder, bi_folder)
    print('FID score(Bilinear):', fid1)
    fid2 = fid.compute_fid(ori_folder, dm_folder)
    print('FID score(Diffusion):', fid2)

    kid1 = fid.compute_kid(ori_folder, bi_folder)
    print('KID score(Bilinear):', kid1)
    kid2 = fid.compute_kid(ori_folder, dm_folder)
    print('KID score(Diffusion):', kid2)

def compute_PSNR_SSIM(dir):
    # metric=SSIM(data_range=1.0)
    ori_folder = os.path.join(dir, 'hr_128')
    bi_folder = os.path.join(dir, 'sr_8_128')
    dm_folder = os.path.join(dir, 'sr_new_128')

    img0 = np.zeros((0, 128, 128, 3))
    img1 = np.zeros((0, 128, 128, 3))
    img2 = np.zeros((0, 128, 128, 3))
    for img in os.listdir(ori_folder):
        imgpath = os.path.join(ori_folder, img)
        x = Image.open(imgpath)
        x = np.asarray(x)
        img0 = np.append(img0, np.expand_dims(x, 0), axis=0)
    
    for img in os.listdir(bi_folder):
        imgpath = os.path.join(bi_folder, img)
        x = Image.open(imgpath)
        x = np.asarray(x)
        img1 = np.append(img1, np.expand_dims(x, 0), axis=0)

    for img in os.listdir(dm_folder):
        imgpath = os.path.join(dm_folder, img)
        x = Image.open(imgpath)
        x = np.asarray(x)
        img2= np.append(img2, np.expand_dims(x, 0), axis=0)

    s1 = psnr(img0, img1, data_range=255)
    print('PSNR score(Bilinear):', s1)
    s2 = psnr(img0, img2, data_range=255)
    print('PSNR score(Diffusion):', s2)

    s1 = ssim(img0, img1, channel_axis=3, data_range=255)
    print('SSIM score(Bilinear):', s1)
    s2 = ssim(img0, img2, channel_axis=3, data_range=255)
    print('SSIM score(Diffusion):', s2)

class Data(Dataset):
    def __init__(self, dir) -> None:
        super().__init__()
        
        self.ori_folder = os.path.join(dir, 'hr_128')
        self.bi_folder = os.path.join(dir, 'sr_32_128')
        self.dm_folder = os.path.join(dir, 'sr_128')

        self.ori_files = os.listdir(self.ori_folder)
        self.bi_files = os.listdir(self.bi_folder)
        self.dm_files = os.listdir(self.dm_folder)

    def __len__(self):
        return len(self.ori_files)
    
    def __getitem__(self, index: Any):
        ori_imgpath = os.path.join(self.ori_folder, self.ori_files[index])
        x = Image.open(ori_imgpath)
        ori = np.asarray(x)
        ori = np.transpose(ori, (2, 0, 1))

        bi_imgpath = os.path.join(self.bi_folder, self.bi_files[index])
        x = Image.open(bi_imgpath)
        bi = np.asarray(x)
        bi = np.transpose(bi, (2, 0, 1))

        dm_imgpath = os.path.join(self.dm_folder, self.dm_files[index])
        x = Image.open(dm_imgpath)
        dm = np.asarray(x)
        dm = np.transpose(dm, (2, 0, 1))
        return ori, bi, dm
    
def get_psnr_ssim(dir):
    dataset = Data(dir)
    dtloader = DataLoader(dataset, 256, num_workers=4, shuffle=True)
    psnr1, psnr2 = 0.0, 0.0
    ssim1, ssim2 = 0.0, 0.0
    num = 0.0
    for ori, bi, dm in tqdm(dtloader):
        n = ori.shape[0]
        # print(num)
        # print(ori.shape)
        p1 = psnr(ori.numpy(), bi.numpy(), data_range=255)
        # print('PSNR score(Bilinear):', s1)
        p2 = psnr(ori.numpy(), dm.numpy(), data_range=255)
        # print('PSNR score(Diffusion):', s2)

        s1 = ssim(ori.numpy(), bi.numpy(), channel_axis=1, data_range=255)
        # print('SSIM score(Bilinear):', s1)
        s2 = ssim(ori.numpy(), dm.numpy(), channel_axis=1, data_range=255)
        # print('SSIM score(Diffusion):', s2)
        psnr1 += p1*n
        psnr2 += p2*n

        ssim1 += s1*n
        ssim2 += s2*n
        
        num += n
    
    print('psnr_bi:', psnr1, psnr1/num)
    print('psnr_dm:', psnr2, psnr2/num)
    print('ssim_bi:', ssim1, ssim1/num)
    print('ssim_dm:', ssim2, ssim2/num)

################## re-gourp image #####################
#######################################################
from model.patch_attention import SwinAttn_pyramid_token

def selection(args, dir):
    device = torch.device(args.dev if torch.cuda.is_available() else 'cpu')
    modelpath = './results/mots015/checkpoint/0001swinnet_token_sum_final.pth'
    net = SwinAttn_pyramid_token(args, num_cls=2)
    net.load_state_dict(torch.load(modelpath))

    imgroot = '../datasets/mots0_015/images/'+dir
    # bi_patchroot = '../datasets/mots015_patches/images_bi/'+dir
    dm_patchroot = '../datasets/mots015_patches/images/'+dir
    out_patchroot = '../datasets/mots015_patches/selected_patches/'+dir
    if not os.path.exists(out_patchroot):
        os.mkdir(out_patchroot)

    id = np.arange(64)
    for img in os.listdir(imgroot):
        name = img.split('.')[0]
        x = Image.open(os.path.join(imgroot, img))
        x = np.asarray(x)
        x = cv2.resize(x, [1024, 1024], interpolation=cv2.INTER_LINEAR).astype(np.float32)
        x = torch.from_numpy(x)
        x = x.unsqueeze(0).permute(0, 3, 1, 2)
        # print(x.shape)

        pte, p64te, p32te = net(x)

        p64te = nn.MaxPool2d(kernel_size=2, stride=2)(p64te)
        p32te = nn.MaxPool2d(kernel_size=4, stride=4)(p32te)
        pte = torch.maximum(pte, p64te)
        pte = torch.maximum(pte, p32te)
        pte = torch.squeeze(torch.argmax(pte, 1), 0)
        # print(pte)
        pte = pte.flatten()
        # print(pte.shape, pte)
        ids = id[pte == 1]
        # print(idx)
        for i in range(len(ids)):
            shutil.copyfile(os.path.join(dm_patchroot, name+'_'+str(ids[i])+'.png'), os.path.join(out_patchroot, name+'_'+str(ids[i])+'.png'))
        print(img)

def recover(dir):
    patchroot = '../datasets/mots015_patches'
    bi_root = os.path.join(patchroot, 'images_bi/'+dir)
    dm_root = os.path.join(patchroot, 'images/'+dir)
    re_root = os.path.join(patchroot, 'images_selected/'+dir)

    outroot = '../datasets/mots015_recovered'
    out_bi_root = os.path.join(outroot, 'images_bi/'+dir)
    out_dm_root = os.path.join(outroot, 'images_dm/'+dir)
    out_re_root = os.path.join(outroot, 'images_selected/'+dir)
    if not os.path.exists(out_bi_root):
        os.mkdir(out_bi_root)
    if not os.path.exists(out_dm_root):
        os.mkdir(out_dm_root)
    if not os.path.exists(out_re_root):
        os.mkdir(out_re_root)

    imgroot = '../datasets/mots0_015/images/'+dir

    for img in os.listdir(imgroot):
        name = img.split('.')[0]

        bi = np.zeros((1024, 1024, 3))
        dm = np.zeros((1024, 1024, 3))
        re = np.zeros((1024, 1024, 3))
        for i in range(8):
            for j in range(8):
                id = 8*i+j
                bi_patch = Image.open(os.path.join(bi_root, name+'_'+str(id)+'.png'))
                dm_patch = Image.open(os.path.join(dm_root, name+'_'+str(id)+'.png'))
                re_patch = Image.open(os.path.join(re_root, name+'_'+str(id)+'.png'))

                bi_patch = np.asarray(bi_patch)
                dm_patch = np.asarray(dm_patch)
                re_patch = np.asarray(re_patch)

                bi[128*i:128*(i+1), 128*j:128*(j+1)] = bi_patch
                dm[128*i:128*(i+1), 128*j:128*(j+1)] = dm_patch
                re[128*i:128*(i+1), 128*j:128*(j+1)] = re_patch

        # Image.fromarray(bi.astype(np.uint8)).save(os.path.join(out_bi_root, img))
        Image.fromarray(dm.astype(np.uint8)).save(os.path.join(out_dm_root, img))
        Image.fromarray(re.astype(np.uint8)).save(os.path.join(out_re_root, img))
        print(img)

def recover_with_black_bg(dir):
    patchroot = '../datasets/mots015_patches'
    
    select_root = os.path.join(patchroot, 'images/'+dir)

    outroot = '../datasets/mots015_recovered'
    out_re_root = os.path.join(outroot, 'images_bgblack/'+dir)
    if not os.path.exists(out_re_root):
        os.mkdir(out_re_root)

    imgroot = '../datasets/mots0_015/images/'+dir

    for img in os.listdir(imgroot):
        name = img.split('.')[0]

        # bi = np.zeros((1024, 1024, 3))
        # dm = np.zeros((1024, 1024, 3))
        re = np.zeros((1024, 1024, 3))
        for i in range(8):
            for j in range(8):
                id = 8*i+j
                # bi_patch = Image.open(os.path.join(bi_root, name+'_'+str(id)+'.png'))
                # dm_patch = Image.open(os.path.join(dm_root, name+'_'+str(id)+'.png'))
                # re_patch = Image.open(os.path.join(re_root, name+'_'+str(id)+'.png'))
                patchpath = os.path.join(select_root, name+'_'+str(id)+'.png')
                if os.path.exists(patchpath):
                    patch = np.asarray(Image.open(patchpath))
                    re[128*i:128*(i+1), 128*j:128*(j+1)] = patch
        Image.fromarray(re.astype(np.uint8)).save(os.path.join(out_re_root, img))
        print(img)

def recover_with_near_bg(dir):
    patchroot = '../datasets/mots015_patches'
    ori_root = os.path.join(patchroot, 'images_ori/'+dir)
    select_root = os.path.join(patchroot, 'images/'+dir)

    outroot = '../datasets/mots015_recovered'
    out_re_root = os.path.join(outroot, 'images_nearblack/'+dir)
    if not os.path.exists(out_re_root):
        os.mkdir(out_re_root)

    imgroot = '../datasets/mots0_015/images/'+dir

    for img in os.listdir(imgroot):
        name = img.split('.')[0]

        # bi = np.zeros((1024, 1024, 3))
        # dm = np.zeros((1024, 1024, 3))
        re = np.zeros((1024, 1024, 3))
        for i in range(8):
            for j in range(8):
                id = 8*i+j
                patchpath = os.path.join(select_root, name+'_'+str(id)+'.png')
                if os.path.exists(patchpath):
                    patch = np.asarray(Image.open(patchpath))
                    re[128*i:128*(i+1), 128*j:128*(j+1)] = patch
                else:
                    patch = np.asarray(Image.open(os.path.join(ori_root, name+'_'+str(id)+'.jpg')))
                    patch = cv2.resize(patch, [16, 16], interpolation=cv2.INTER_LINEAR)
                    patch = cv2.resize(patch, [128, 128], interpolation=cv2.INTER_NEAREST)
                    re[128*i:128*(i+1), 128*j:128*(j+1)] = patch
        Image.fromarray(re.astype(np.uint8)).save(os.path.join(out_re_root, img))
        print(img)

def recover_with_cubic_bg(dir):
    patchroot = '../datasets/mots015_patches'
    ori_root = os.path.join(patchroot, 'images_ori/'+dir)
    select_root = os.path.join(patchroot, 'images/'+dir)

    outroot = '../datasets/mots015_recovered'
    out_re_root = os.path.join(outroot, 'images_cubblack/'+dir)
    if not os.path.exists(out_re_root):
        os.mkdir(out_re_root)

    imgroot = '../datasets/mots0_015/images/'+dir

    for img in os.listdir(imgroot):
        name = img.split('.')[0]

        # bi = np.zeros((1024, 1024, 3))
        # dm = np.zeros((1024, 1024, 3))
        re = np.zeros((1024, 1024, 3))
        for i in range(8):
            for j in range(8):
                id = 8*i+j
                patchpath = os.path.join(select_root, name+'_'+str(id)+'.png')
                if os.path.exists(patchpath):
                    patch = np.asarray(Image.open(patchpath))
                    re[128*i:128*(i+1), 128*j:128*(j+1)] = patch
                else:
                    patch = np.asarray(Image.open(os.path.join(ori_root, name+'_'+str(id)+'.jpg')))
                    patch = cv2.resize(patch, [16, 16], interpolation=cv2.INTER_LINEAR)
                    patch = cv2.resize(patch, [128, 128], interpolation=cv2.INTER_CUBIC)
                    re[128*i:128*(i+1), 128*j:128*(j+1)] = patch
        Image.fromarray(re.astype(np.uint8)).save(os.path.join(out_re_root, img))
        print(img)

def count_by_threshold(dir):
    patchroot = '../datasets/mots015_patches'
    bi_root = os.path.join(patchroot, 'images_bi/'+dir)
    dm_root = os.path.join(patchroot, 'images/'+dir)
    re_root = os.path.join(patchroot, 'images/'+dir)
    mask_root = os.path.join(patchroot, 'masks/'+dir)

    total = 0.0
    small = 0.0
    empty = 0.0
    for img in os.listdir(re_root):
        # name = img.split
        y = Image.open(os.path.join(mask_root, img))
        y = np.asarray(y)
        y = y/(y.max()+10e-8)
        # print(np.unique(y))
        if np.mean(y) == 0:
            empty += 1
        elif np.mean(y) < 0.15:
            small += 1
        total += 1
    print(total, small, empty, total-small-empty)


def group_by_threshold(dir):
    patchroot = '../datasets/mots015_patches'
    bi_root = os.path.join(patchroot, 'images_bi/'+dir)
    dm_root = os.path.join(patchroot, 'images/'+dir)
    select_root = os.path.join(patchroot, 'images/'+dir)
    mask_root = os.path.join(patchroot, 'masks/'+dir)

    outroot = '../datasets/mots015_recovered'
    out_group_root = os.path.join(outroot, 'images_group/'+dir)
    if not os.path.exists(out_group_root):
        os.mkdir(out_group_root)

    imgroot = '../datasets/mots0_015/images/'+dir

    for img in os.listdir(imgroot):
        name = img.split('.')[0]

        # bi = np.zeros((1024, 1024, 3))
        # dm = np.zeros((1024, 1024, 3))
        re = np.zeros((1024, 1024, 3))
        for i in range(8):
            for j in range(8):
                id = 8*i+j
                patchpath = os.path.join(select_root, name+'_'+str(id)+'.png')
                bipath = os.path.join(bi_root, name+'_'+str(id)+'.png')
                if os.path.exists(patchpath):
                    
                    y = Image.open(os.path.join(mask_root, name+'_'+str(id)+'.png'))
                    y = np.asarray(y)
                    y = y/(y.max()+10e-8)
                    # print(np.unique(y))
                    if np.mean(y) > 0 and np.mean(y) < 0.15:
                        patch = np.asarray(Image.open(patchpath))
                        re[128*i:128*(i+1), 128*j:128*(j+1)] = patch
                    else:
                        patch = np.asarray(Image.open(bipath))
                        re[128*i:128*(i+1), 128*j:128*(j+1)] = patch
                else:
                    patch = np.asarray(Image.open(bipath))
                    re[128*i:128*(i+1), 128*j:128*(j+1)] = patch
                # print(id, np.unique(patch))
        Image.fromarray(re.astype(np.uint8)).save(os.path.join(out_group_root, img))
        print(img)

def makeup_black(dir):
    patchroot = '../datasets/mots015_patches'
    bi_root = os.path.join(patchroot, 'images_bi/'+dir)
    dm_root = os.path.join(patchroot, 'images/'+dir)
    select_root = os.path.join(patchroot, 'selected_patches/'+dir)
    mask_root = os.path.join(patchroot, 'masks/'+dir)

    # outroot = '../datasets/mots015_recovered'
    out_root = os.path.join(patchroot, 'images_selected_bgblack/'+dir)
    if not os.path.exists(out_root):
        os.mkdir(out_root)

    imgroot = '../datasets/mots0_015/images/'+dir
    for img in os.listdir(imgroot):
        name = img.split('.')[0]

        makeup = np.zeros((128, 128, 3))
        for i in range(8):
            for j in range(8):
                id = 8*i+j
                patchpath = os.path.join(select_root, name+'_'+str(id)+'.png')

                if not os.path.exists(patchpath):
                    Image.fromarray(makeup.astype(np.uint8)).save(os.path.join(out_root, name+'_'+str(id)+'.png'))

        print(img)


if __name__=='__main__':
    # draw_bb()
    add_circle()
    # compute_ratio()
    # imgpath = './example_with_bounding_boxes_bg.jpg'
    # img = cv2.imread(imgpath)
    # img = draw_grid(img, (8, 8))
    # cv2.imwrite("example_with_bounding_boxes_bg_grid.jpg", img)
    # dir = '../datasets/mots23_/masks'
    # compute_ave_object(dir)
    # dir = '../datasets/mots015_32_128'
    # # compute_FID(dir)
    # get_psnr_ssim(dir)
    # # compute_FID(dir)