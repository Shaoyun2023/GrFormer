import os
import random
import numpy as np
import torch
import cv2
import scipy.linalg
from PIL import Image
from args_fusion import args
# from scipy.misc import imread, imsave, imresize
# from imageio import imread, imsave
import matplotlib as mpl
from torchvision import datasets, transforms
from os import listdir
from os.path import join
import matplotlib.pyplot as plt
import torch.nn as nn
import imageio
import torch.nn.functional as F
from numpy import linalg as la
from scipy.linalg import logm
# from skimage.transform import resize

conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)
conv = conv.to(args.device)

# Nest = NestFuse_autoencoder()

def scipy_misc_imresize(arr, size, interp='bilinear', mode=None):
   im = Image.fromarray(arr, mode=mode)
   ts = type(size)
   if np.issubdtype(ts, np.signedinteger):
      percent = size / 100.0
      size = tuple((np.array(im.size)*percent).astype(int))
   elif np.issubdtype(type(size), np.floating):
      size = tuple((np.array(im.size)*size).astype(int))
   else:
      size = (size[1], size[0])
   func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
   imnew = im.resize(size, resample=func[interp]) # 调用PIL库中的resize函数
   return np.array(imnew)

def get_patches(input, kernel_size, stride,padding):

    patches = F.unfold(input, kernel_size, stride=stride,padding=padding)

    # image = patches.squeeze().detach().cpu().numpy()
    #
    # image = (image - image.min()) / (image.max() - image.min())
    # plt.imshow(image, cmap='gray')
    # plt.axis('off')
    # plt.show()
    #

    patches = patches.transpose(1, 2).contiguous().view(patches.size(0)*patches.size(-1), -1)

    return patches

def fold_with_overlap_average(patches):
    patches = patches.contiguous()
    # 对 patches 做 fold 操作
    output = F.fold(patches.unsqueeze(0), (576,768), kernel_size=16, stride=8,padding=8)
    return output

def average_borders(input):
    up = F.avg_pool2d(input[..., :-1,:], kernel_size=(2, 1), stride=1)[:,:,1:,:]
    left = F.avg_pool2d(input[..., :-1], kernel_size=(1, 2), stride=1)[:,:,:,1:]
    return up, left


def list_images(directory):
    images = []
    names = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
        name1 = name.split('.')
        names.append(name1[0])
    return images


def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)

    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img


def tensor_save_rgbimage(tensor, filename, cuda=False):
    if cuda:
        # img = tensor.clone().cpu().clamp(0, 255).numpy()
        img = tensor.cpu().clamp(0, 255).data[0].numpy()
    else:
        # img = tensor.clone().clamp(0, 255).numpy()
        img = tensor.clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)


def tensor_save_bgrimage(tensor, filename, cuda=False):
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))
    tensor_save_rgbimage(tensor, filename, cuda)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    # pool = nn.MaxPool2d(4,4)
    # gram = pool(gram)
    # gram = torch.matmul(gram,y)
    #
    # image = torch.sum(gram, dim=1, keepdim=True)
    # image = image.squeeze().detach().cpu().numpy()
    #
    # image = (image - image.min()) / (image.max() - image.min())
    # plt.imshow(image, cmap='jet')
    # plt.axis('off')
    # plt.show()

    return gram


def matSqrt(x):
    U,D,V = torch.svd(x)
    return U * (D.pow(0.5).diag()) * V.t()



# load training images
def load_dataset(image_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(image_path)
    original_imgs_path = image_path[:num_imgs]
    # random
    random.shuffle(original_imgs_path)
    mod = num_imgs % BATCH_SIZE
    print('BATCH SIZE %d.' % BATCH_SIZE)
    print('Train images number %d.' % num_imgs)
    print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        original_imgs_path = original_imgs_path[:-mod]
    batches = int(len(original_imgs_path) // BATCH_SIZE)
    return original_imgs_path, batches


# def get_image(path, height=256, width=256, flag=False):
#     if flag is True:
#         image = imread(path, mode='RGB')
#     else:
#         image = imread(path, mode='L')
#
#     if height is not None and width is not None:
#         image = imresize(image, [height, width], interp='nearest')
#     return image

def get_image(path, height=256, width=256, flag=False):
    if flag is True:
        image = imageio.imread(path)
    else:
        image = imageio.imread(path)

    if height is not None and width is not None:
        image = scipy_misc_imresize(image, [height, width], interp='nearest')

    return image



# def get_image(path, height=256, width=256, mode='L'):
#     if mode == 'L':
#         image = imread(path, mode=mode)
#     elif mode == 'RGB':
#         image = Image.open(path).convert('RGB')
#
#     if height is not None and width is not None:
#         image = imresize(image, [height, width], interp='nearest')
#     return image

# def get_test_images(paths, height=None, width=None, mode='L'):
#     ImageToTensor = transforms.Compose([transforms.ToTensor()])
#     if isinstance(paths, str):
#         paths = [paths]
#     images = []
#     for path in paths:
#         image = get_image(path, height, width, mode=mode)
#         if mode == 'L':
#             image = np.reshape(image, [1,1, image.shape[0], image.shape[1]])
#         else:
#             # test = ImageToTensor(image).numpy()
#             # shape = ImageToTensor(image).size()
#             image = ImageToTensor(image).float().numpy()*255
#     images.append(image)
#     images = np.stack(images, axis=0)
#     images = torch.from_numpy(images).float()
#     return images

def get_test_image1(paths, height=None, width=None, flag=False):
    if isinstance(paths, str):
        paths = [paths]
    images = []

    for path in paths:
        image = get_image(path, height, width,  flag)
        if height is not None and width is not None:
            image = scipy_misc_imresize(image, [height, width], interp='nearest')
        base_size = 256
        # base_size = 256
        h = image.shape[0]

        w = image.shape[1]
        c = 0

        if 1 * base_size == h and 1 * base_size == w:
            c = 4
            images = get_img_parts11(image, h, w)
        if 1 * base_size < h <= 2 * base_size and 1 * base_size < w <= 2 * base_size:
            c = 4
            images = get_img_parts1(image, h, w)


        if h==576 and w==768:
            c = 9
            images = get_img_parts576768(image, h, w)

        if h==450 and w==620:
            c = 8
            images = get_img_parts450620(image, h, w)

        if 2 * base_size < h <= 3 * base_size and 2 * base_size < w <= 3 * base_size:
            c = 9
            images = get_img_parts2(image, h, w)
        if 2 * base_size < h < 3 * base_size and 3 * base_size < w < 4 * base_size:
            c = 12
            images = get_img_parts3(image, h, w)
        if 1 * base_size < h < 2 * base_size and 2 * base_size < w < 3 * base_size and h!=450 and w!=620:
            c = 6
            images = get_img_parts4(image, h, w)
        if 3 * base_size < h <= 4 * base_size+4 and 4 * base_size < w <= 5 * base_size+5:
            c = 20
            images = get_img_parts5(image, h, w)
        if 0 * base_size < h < 1 * base_size and 1 * base_size < w < 2 * base_size:
            c = 2
            images = get_img_parts6(image, h, w)
        if 0 * base_size < h < 1 * base_size and 2 * base_size < w < 3 * base_size:
            c = 3
            images = get_img_parts7(image, h, w)
        if h == 1 * base_size and 2 * base_size < w < 3 * base_size:
            c = 3
            images = get_img_parts8(image, h, w)

    return images, h, w, c

def save_images(path, data):
    # if isinstance(paths, str):
    #     paths = [paths]
    #
    # t1 = len(paths)
    # t2 = len(datas)
    # assert (len(paths) == len(datas))

    # if prefix is None:
    #     prefix = ''
    # if suffix is None:
    #     suffix = ''

    if data.shape[2] == 1:
        data = data.reshape([data.shape[0], data.shape[1]])
    imageio.imsave(path, data)

    # for i, path in enumerate(paths):
    #     data = datas[i]
    #     # print('data ==>>\n', data)
    #     if data.shape[2] == 1:
    #         data = data.reshape([data.shape[0], data.shape[1]])
    #     # print('data reshape==>>\n', data)
    #
    #     name, ext = splitext(path)
    #     name = name.split(sep)[-1]
    #
    #     path = join(save_path, prefix + suffix + ext)
    #     print('data path==>>', path)
    #
    #     # new_im = Image.fromarray(data)
    #     # new_im.show()
    #
    #     imsave(path, data)

# def get_test_image(paths, height=256, width=256, flag=False):
#     if isinstance(paths, str):
#         paths = [paths]
#     images = []
#     for path in paths:
#         image = imread(path, mode='L')
#         image1 = Image.fromarray(image)
#         # if height is not None and width is not None:
#         #     image = imresize(image, [height, width], interp='nearest')
#         # image = imresize(image, [height,width])
#         resize1 = transforms.Resize((height, width))
#         image = resize1(image1)
#         image = np.array(image)
#
#         # image = (image - image.min()) / (image.max() - image.min())
#         plt.imshow(image, cmap='gray')
#         plt.axis('off')
#         plt.show()
#
#         base_size = 512
#         h = image.shape[0]
#         w = image.shape[1]
#         c = 1
#
#
#         image = np.reshape(image, [1, image.shape[0], image.shape[1]])
#         images.append(image)
#         images = np.stack(images, axis=0)
#         images = torch.from_numpy(images).float()
#
#     # images = np.stack(images, axis=0)
#     # images = torch.from_numpy(images).float()
#     return images, h, w, c

def get_img_parts(image, h, w):
    images = []
    # h_cen = int(np.floor(h / 4))
    # w_cen = int(np.floor(w / 4))
    h_cen = 256
    w_cen = 256
    # img1 = image[:, 0:h_cen, 0: w_cen]
    # img1 = np.reshape(img1, [1, img1.shape[0], img1.shape[1], img1.shape[2]])
    # img2 = image[:, 0:h_cen, w_cen: w]
    # img2 = np.reshape(img2, [1, img2.shape[0], img2.shape[1], img2.shape[2]])
    # img3 = image[:, h_cen - 2:h, 0: w_cen + 3]
    # img3 = np.reshape(img3, [1, img3.shape[0], img3.shape[1], img3.shape[2]])
    # img4 = image[:, h_cen - 2:h, w_cen - 2: w]
    # img4 = np.reshape(img4, [1, img4.shape[0], img4.shape[1], img4.shape[2]])
    # images.append(torch.from_numpy(img1).float())
    # images.append(torch.from_numpy(img2).float())
    # images.append(torch.from_numpy(img3).float())
    # images.append(torch.from_numpy(img4).float())

    num_blocks_h = h // 256
    num_blocks_w = w // 256

    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            # 计算当前块的索引
            start_i = i * 256
            start_j = j * 256
            end_i = start_i + 256
            end_j = start_j + 256
            # 切分图像
            img_block = image[start_i:end_i, start_j:end_j]
            # 将图像块转换为PyTorch张量并添加到列表中
            images.append(torch.from_numpy(img_block).float().unsqueeze(0))
    return images

# def get_img_parts(image, h, w):
#     images = []
#     h_cen = int(np.floor(h / 2))
#     w_cen = int(np.floor(w / 2))
#     img1 = image[:, 0:h_cen + 3, 0: w_cen + 3]
#     img1 = np.reshape(img1, [1, img1.shape[0], img1.shape[1], img1.shape[2]])
#     img2 = image[:, 0:h_cen + 3, w_cen - 2: w]
#     img2 = np.reshape(img2, [1, img2.shape[0], img2.shape[1], img2.shape[2]])
#     img3 = image[:, h_cen - 2:h, 0: w_cen + 3]
#     img3 = np.reshape(img3, [1, img3.shape[0], img3.shape[1], img3.shape[2]])
#     img4 = image[:, h_cen - 2:h, w_cen - 2: w]
#     img4 = np.reshape(img4, [1, img4.shape[0], img4.shape[1], img4.shape[2]])
#     images.append(torch.from_numpy(img1).float())
#     images.append(torch.from_numpy(img2).float())
#     images.append(torch.from_numpy(img3).float())
#     images.append(torch.from_numpy(img4).float())
#     return images

def get_test_image(paths, height=None, width=None, flag=False):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = imageio.imread(path)
        if height is not None and width is not None:
            image = scipy_misc_imresize(image, [height, width], interp='nearest')

        base_size = 512
        h = image.shape[0]
        w = image.shape[1]
        c = 1
        if h > base_size or w > base_size:
            c = 4
            images = get_img_parts(image, h, w)
        else:
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
            images.append(image)
            images = np.stack(images, axis=0)
            images = torch.from_numpy(images).float()

    # images = np.stack(images, axis=0)
    # images = torch.from_numpy(images).float()
    return images, h, w, c

def get_test_image1(paths, height=None, width=None, flag=False):
    if isinstance(paths, str):
        paths = [paths]
    images = []

    for path in paths:
        image = get_image(path, height, width,  flag)
        if height is not None and width is not None:
            image = scipy_misc_imresize(image, [height, width], interp='nearest')
        base_size = 256
        if image.ndim == 3:
            image = np.sum(image, axis=2)
        image = image.astype(np.float32)
        # base_size = 256
        h = image.shape[0]

        w = image.shape[1]
        c = 0

        if 1 * base_size == h and 1 * base_size == w:
            c = 4
            images = get_img_parts11(image, h, w)
        if 1 * base_size < h <= 2 * base_size and 1 * base_size < w <= 2 * base_size:
            c = 4
            images = get_img_parts1(image, h, w)


        if h==576 and w==768:
            c = 9
            images = get_img_parts576768(image, h, w)

        if h==450 and w==620:
            c = 8
            images = get_img_parts450620(image, h, w)

        if 2 * base_size < h <= 3 * base_size and 2 * base_size < w <= 3 * base_size:
            c = 9
            images = get_img_parts2(image, h, w)
        if 2 * base_size < h < 3 * base_size and 3 * base_size < w < 4 * base_size:
            c = 12
            images = get_img_parts3(image, h, w)
        if 1 * base_size < h < 2 * base_size and 2 * base_size < w < 3 * base_size and h!=450 and w!=620:
            c = 6
            images = get_img_parts4(image, h, w)
        if 3 * base_size < h <= 4 * base_size+4 and 4 * base_size < w <= 5 * base_size+5:
            c = 20
            images = get_img_parts5(image, h, w)
        if 0 * base_size < h < 1 * base_size and 1 * base_size < w < 2 * base_size:
            c = 2
            images = get_img_parts6(image, h, w)
        if 0 * base_size < h < 1 * base_size and 2 * base_size < w < 3 * base_size:
            c = 3
            images = get_img_parts7(image, h, w)
        if h == 1 * base_size and 2 * base_size < w < 3 * base_size:
            c = 3
            images = get_img_parts8(image, h, w)

    return images, h, w, c

def get_img_parts1(image, h, w):
    pad = nn.ConstantPad2d(padding=(0, 512-w, 0, 512-h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    print(image.shape)
    images = []
    img1 = image[0:256, 0: 256]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:256, w-256: w]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[h-256:h, 0: 256]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    img4 = image[h-256:h, w-256: w]
    img4 = torch.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    images.append(img4.float())
    # image = torch.from_numpy(image).unsqueeze(0)
    # images = []
    # h_cen = int(np.floor(h / 2))
    # w_cen = int(np.floor(w / 2))
    # img1 = image[:, 0:h_cen + 3, 0: w_cen + 3]
    # img1 = np.reshape(img1, [1, img1.shape[0], img1.shape[1], img1.shape[2]])
    # img2 = image[:, 0:h_cen + 3, w_cen - 2: w]
    # img2 = np.reshape(img2, [1, img2.shape[0], img2.shape[1], img2.shape[2]])
    # img3 = image[:, h_cen - 2:h, 0: w_cen + 3]
    # img3 = np.reshape(img3, [1, img3.shape[0], img3.shape[1], img3.shape[2]])
    # img4 = image[:, h_cen - 2:h, w_cen - 2: w]
    # img4 = np.reshape(img4, [1, img4.shape[0], img4.shape[1], img4.shape[2]])
    # images.append(img1.float())
    # images.append(img2.float())
    # images.append(img3.float())
    # images.append(img4.float())

    return images

def get_img_parts11(image, h, w):
    pad = nn.ConstantPad2d(padding=(0, 512-w, 0, 512-h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    print(image.shape)
    images = []
    img1 = image[0:256, 0: 256]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:256, 256: 512]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[256:512, 0: 256]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    img4 = image[256:512, 256: 512]
    img4 = torch.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    images.append(img4.float())
    # image = torch.from_numpy(image).unsqueeze(0)
    # images = []
    # h_cen = int(np.floor(h / 2))
    # w_cen = int(np.floor(w / 2))
    # img1 = image[:, 0:h_cen + 3, 0: w_cen + 3]
    # img1 = np.reshape(img1, [1, img1.shape[0], img1.shape[1], img1.shape[2]])
    # img2 = image[:, 0:h_cen + 3, w_cen - 2: w]
    # img2 = np.reshape(img2, [1, img2.shape[0], img2.shape[1], img2.shape[2]])
    # img3 = image[:, h_cen - 2:h, 0: w_cen + 3]
    # img3 = np.reshape(img3, [1, img3.shape[0], img3.shape[1], img3.shape[2]])
    # img4 = image[:, h_cen - 2:h, w_cen - 2: w]
    # img4 = np.reshape(img4, [1, img4.shape[0], img4.shape[1], img4.shape[2]])
    # images.append(img1.float())
    # images.append(img2.float())
    # images.append(img3.float())
    # images.append(img4.float())

    return images


# def get_img_parts2(image, h, w):
#
#     pad = nn.ConstantPad2d(padding=(0, 768-w, 0, 768-h), value=0)
#     image = torch.from_numpy(image)
#     image = pad(image)
#     images = []
#
#     img1 = image[0:256+3, 0: 256+3]
#     img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
#     img2 = image[0:256+3, 256-2: 512+3]
#     img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
#     img3 = image[0:256+3, 512-2: 768]
#     img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
#     img4 = image[256-2:512+3, 0: 256+3]
#     img4 = torch.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
#     img5 = image[256-2:512+3, 256-2: 512+3]
#     img5 = torch.reshape(img5, [1, 1, img5.shape[0], img5.shape[1]])
#     img6 = image[256-2:512+3, 512-2: 768]
#     img6 = torch.reshape(img6, [1, 1, img6.shape[0], img6.shape[1]])
#     img7 = image[512-2:768, 0: 256+3]
#     img7 = torch.reshape(img7, [1, 1, img7.shape[0], img7.shape[1]])
#     img8 = image[512-2:768, 256-2: 512+3]
#     img8 = torch.reshape(img8, [1, 1, img8.shape[0], img8.shape[1]])
#     img9 = image[512-2:768, 512-2: 768]
#     img9 = torch.reshape(img9, [1, 1, img9.shape[0], img9.shape[1]])
#     images.append(img1.float())
#     images.append(img2.float())
#     images.append(img3.float())
#     images.append(img4.float())
#     images.append(img5.float())
#     images.append(img6.float())
#     images.append(img7.float())
#     images.append(img8.float())
#     images.append(img9.float())
#     return images

def get_img_parts576768(image, h, w):

    pad = nn.ConstantPad2d(padding=(0, 768-w, 0, 768-h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []

    img1 = image[0:256 , 0: 256 ]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:256 , 256 : 512 ]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[0:256 , w-256 : w]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    img4 = image[256 :512 , 0: 256 ]
    img4 = torch.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
    img5 = image[256 :512 , 256 : 512 ]
    img5 = torch.reshape(img5, [1, 1, img5.shape[0], img5.shape[1]])
    img6 = image[256 :512 , w-256 : w]
    img6 = torch.reshape(img6, [1, 1, img6.shape[0], img6.shape[1]])
    img7 = image[h - 256 :h, 0: 256 ]
    img7 = torch.reshape(img7, [1, 1, img7.shape[0], img7.shape[1]])
    img8 = image[h - 256 :h, 256 : 512 ]
    img8 = torch.reshape(img8, [1, 1, img8.shape[0], img8.shape[1]])
    img9 = image[h - 256 :h, w-256 : w]
    img9 = torch.reshape(img9, [1, 1, img9.shape[0], img9.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    images.append(img4.float())
    images.append(img5.float())
    images.append(img6.float())
    images.append(img7.float())
    images.append(img8.float())
    images.append(img9.float())
    return images

# def get_img_parts450620(image, h, w):
#
#     pad = nn.ConstantPad2d(padding=(0, 620-w, 0, 620-h), value=0)
#     image = torch.from_numpy(image)
#     image = pad(image)
#     images = []
#
#     img1 = image[0:256 , 0: 256 ]
#     img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
#     img2 = image[0:256 , 256 : 512 ]
#     img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
#     img3 = image[0:256 , w-256 : w]
#     img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
#     img4 = image[h-256 :h , 0: 256 ]
#     img4 = torch.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
#     img5 = image[h-256 :h , 256 : 512 ]
#     img5 = torch.reshape(img5, [1, 1, img5.shape[0], img5.shape[1]])
#     img6 = image[h-256 :h , w-256 : w]
#     img6 = torch.reshape(img6, [1, 1, img6.shape[0], img6.shape[1]])
#
#     images.append(img1.float())
#     images.append(img2.float())
#     images.append(img3.float())
#     images.append(img4.float())
#     images.append(img5.float())
#     images.append(img6.float())
#     return images

def get_img_parts450620(image, h, w):

    pad = nn.ConstantPad2d(padding=(0, 620-w, 0, 620-h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []

    img1 = image[0:256 , 0: 256 ]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:256 , 128 : 384 ]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[0:256 , 256 : 512]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    img4 = image[0:256, w-256: w]
    img4 = torch.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
    img5 = image[h-256 :h , 0: 256 ]
    img5 = torch.reshape(img5, [1, 1, img5.shape[0], img5.shape[1]])
    img6 = image[h-256 :h , 128: 384]
    img6 = torch.reshape(img6, [1, 1, img6.shape[0], img6.shape[1]])
    img7 = image[h-256 :h , 256 : 512]
    img7 = torch.reshape(img7, [1, 1, img7.shape[0], img7.shape[1]])
    img8 = image[h - 256:h, w-256: w]
    img8 = torch.reshape(img8, [1, 1, img8.shape[0], img8.shape[1]])

    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    images.append(img4.float())
    images.append(img5.float())
    images.append(img6.float())
    images.append(img7.float())
    images.append(img8.float())
    return images

def get_img_parts2(image, h, w):

    pad = nn.ConstantPad2d(padding=(0, 768-w, 0, 768-h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []

    # img1 = image[0:256+3, 0: 256+3]
    # img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    # img2 = image[0:256+3, 256-2: 512+3]
    # img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    # img3 = image[0:256+3, 512-2: w]
    # img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    # img4 = image[256-2:512+3, 0: 256+3]
    # img4 = torch.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
    # img5 = image[256-2:512+3, 256-2: 512+3]
    # img5 = torch.reshape(img5, [1, 1, img5.shape[0], img5.shape[1]])
    # img6 = image[256-2:512+3, 512-2: w]
    # img6 = torch.reshape(img6, [1, 1, img6.shape[0], img6.shape[1]])
    # img7 = image[576-256-2:h, 0: 256+3]
    # img7 = torch.reshape(img7, [1, 1, img7.shape[0], img7.shape[1]])
    # img8 = image[576-256-2:h, 256-2: 512+3]
    # img8 = torch.reshape(img8, [1, 1, img8.shape[0], img8.shape[1]])
    # img9 = image[576-256-2:h, 512-2: w]
    # img9 = torch.reshape(img9, [1, 1, img9.shape[0], img9.shape[1]])

    img1 = image[0:256 , 0: 256 ]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:256 , 256 : 512 ]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[0:256 , w-256 : w]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    img4 = image[256 :512 , 0: 256 ]
    img4 = torch.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
    img5 = image[256 :512 , 256 : 512 ]
    img5 = torch.reshape(img5, [1, 1, img5.shape[0], img5.shape[1]])
    img6 = image[256 :512 , w-256 : w]
    img6 = torch.reshape(img6, [1, 1, img6.shape[0], img6.shape[1]])
    img7 = image[h - 256 :h, 0: 256 ]
    img7 = torch.reshape(img7, [1, 1, img7.shape[0], img7.shape[1]])
    img8 = image[h - 256 :h, 256 : 512 ]
    img8 = torch.reshape(img8, [1, 1, img8.shape[0], img8.shape[1]])
    img9 = image[h - 256 :h, w-256 : w]
    img9 = torch.reshape(img9, [1, 1, img9.shape[0], img9.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    images.append(img4.float())
    images.append(img5.float())
    images.append(img6.float())
    images.append(img7.float())
    images.append(img8.float())
    images.append(img9.float())
    return images

def get_img_parts3(image, h, w):
    pad = nn.ConstantPad2d(padding=(0, 1024-w, 0, 768-h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[0:256, 0: 256]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:256, 256: 512]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[0:256, 512: 768]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    img4 = image[0:256, w-256: w]
    img4 = torch.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
    img5 = image[256:512, 0: 256]
    img5 = torch.reshape(img5, [1, 1, img5.shape[0], img5.shape[1]])
    img6 = image[256:512, 256: 512]
    img6 = torch.reshape(img6, [1, 1, img6.shape[0], img6.shape[1]])
    img7 = image[256:512, 512: 768]
    img7 = torch.reshape(img7, [1, 1, img7.shape[0], img7.shape[1]])
    img8 = image[256:512, w-256: w]
    img8 = torch.reshape(img8, [1, 1, img8.shape[0], img8.shape[1]])
    img9 = image[h-256:h, 0: 256]
    img9 = torch.reshape(img9, [1, 1, img9.shape[0], img9.shape[1]])
    img10 = image[h-256:h, 256: 512]
    img10 = torch.reshape(img10, [1, 1, img10.shape[0], img10.shape[1]])
    img11 = image[h-256:h, 512: 768]
    img11 = torch.reshape(img11, [1, 1, img11.shape[0], img11.shape[1]])
    img12 = image[h-256:h, w-256: w]
    img12 = torch.reshape(img12, [1, 1, img12.shape[0], img12.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    images.append(img4.float())
    images.append(img5.float())
    images.append(img6.float())
    images.append(img7.float())
    images.append(img8.float())
    images.append(img9.float())
    images.append(img10.float())
    images.append(img11.float())
    images.append(img12.float())
    return images


def get_img_parts4(image, h, w):
    pad = nn.ConstantPad2d(padding=(0, 768-w, 0, 512-h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[0:256, 0: 256]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:256, 256: 512]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[0:256, w-256: w]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    img4 = image[h-256:h, 0: 256]
    img4 = torch.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
    img5 = image[h-256:h, 256: 512]
    img5 = torch.reshape(img5, [1, 1, img5.shape[0], img5.shape[1]])
    img6 = image[h-256:h, w-256: w]
    img6 = torch.reshape(img6, [1, 1, img6.shape[0], img6.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    images.append(img4.float())
    images.append(img5.float())
    images.append(img6.float())
    return images


def get_img_parts5(image, h, w):
    pad = nn.ConstantPad2d(padding=(0, 1280-w, 0, 1024-h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[0:256, 0: 256]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:256, 256: 512]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[0:256, 512: 768]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    img4 = image[0:256, 768: 1024]
    img4 = torch.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
    img5 = image[0:256, w-256: w]
    img5 = torch.reshape(img5, [1, 1, img5.shape[0], img5.shape[1]])
    img6 = image[256:512, 0: 256]
    img6 = torch.reshape(img6, [1, 1, img6.shape[0], img6.shape[1]])
    img7 = image[256:512, 256: 512]
    img7 = torch.reshape(img7, [1, 1, img7.shape[0], img7.shape[1]])
    img8 = image[256:512, 512: 768]
    img8 = torch.reshape(img8, [1, 1, img8.shape[0], img8.shape[1]])
    img9 = image[256:512, 768: 1024]
    img9 = torch.reshape(img9, [1, 1, img9.shape[0], img9.shape[1]])
    img10 = image[256:512, w-256: w]
    img10 = torch.reshape(img10, [1, 1, img10.shape[0], img10.shape[1]])
    img11 = image[512:768, 0: 256]
    img11 = torch.reshape(img11, [1, 1, img11.shape[0], img11.shape[1]])
    img12 = image[512:768, 256: 512]
    img12 = torch.reshape(img12, [1, 1, img12.shape[0], img12.shape[1]])
    img13 = image[512:768, 512: 768]
    img13 = torch.reshape(img13, [1, 1, img13.shape[0], img13.shape[1]])
    img14 = image[512:768, 768: 1024]
    img14 = torch.reshape(img14, [1, 1, img14.shape[0], img14.shape[1]])
    img15 = image[512:768, w-256: w]
    img15 = torch.reshape(img15, [1, 1, img15.shape[0], img15.shape[1]])
    img16 = image[h-256:h, 0: 256]
    img16 = torch.reshape(img16, [1, 1, img16.shape[0], img16.shape[1]])
    img17 = image[h-256:h, 256: 512]
    img17 = torch.reshape(img17, [1, 1, img17.shape[0], img17.shape[1]])
    img18 = image[h-256:h, 512: 768]
    img18 = torch.reshape(img18, [1, 1, img18.shape[0], img18.shape[1]])
    img19 = image[h-256:h, 768: 1024]
    img19 = torch.reshape(img19, [1, 1, img19.shape[0], img19.shape[1]])
    img20 = image[h-256:h, w-256: w]
    img20 = torch.reshape(img20, [1, 1, img20.shape[0], img20.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    images.append(img4.float())
    images.append(img5.float())
    images.append(img6.float())
    images.append(img7.float())
    images.append(img8.float())
    images.append(img9.float())
    images.append(img10.float())
    images.append(img11.float())
    images.append(img12.float())
    images.append(img13.float())
    images.append(img14.float())
    images.append(img15.float())
    images.append(img16.float())
    images.append(img17.float())
    images.append(img18.float())
    images.append(img19.float())
    images.append(img20.float())

    return images


def get_img_parts6(image, h, w):
    pad = nn.ConstantPad2d(padding=(0, 512-w, 0, 256-h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[h-256:h, 0: 256]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[h-256:h, w-256: w]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    return images


def get_img_parts7(image, h, w):
    pad = nn.ConstantPad2d(padding=(0, 768-w, 0, 256-h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[h-256:h, 0: 256]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[h-256:h, 256: 512]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[h-256:h, w-256: w]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    return images


def get_img_parts8(image, h, w):
    pad = nn.ConstantPad2d(padding=(0, 768-w, 0, 256), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[h - 256:h, 0: 256]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[h - 256:h, 256: 512]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[h - 256:h, w - 256: w]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    return images

def recons_fusion_images11(img_lists, h, w):
    img_f_list = []
    ones_temp = torch.ones(1, h, w).to(args.device)
    for i in range(len(img_lists[0])):

        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]

        img_f = torch.zeros(1, h, w).to(args.device)
        count = torch.zeros(1, h, w).to(args.device)
        print(img_f.size())

        img_f[:, 0:256, 0: 256] += img1
        count[:, 0:256, 0: 256 ] += ones_temp[:, 0:256 , 0: 256 ]
        img_f[:, 0:256, 256: w] += img2[:, 0:256, 0:w-256]
        img_f[:, 256:h, 0: 256] += img3[:, 0:h-256, 0:256]
        img_f[:, 256:h, 256: w] += img4[:, 0:h-256, 0:w-256]

        img_f = img_f / count

        img_f_list.append(img_f)
    # img_f_list = []
    # h_cen = int(np.floor(h / 2))
    # w_cen = int(np.floor(w / 2))
    # c = img_lists[0][0].shape[1]
    # ones_temp = torch.ones(1, c, h, w).cuda()
    # for i in range(len(img_lists[0])):
    #     # img1, img2, img3, img4
    #     img1 = img_lists[0][i]
    #     img2 = img_lists[1][i]
    #     img3 = img_lists[2][i]
    #     img4 = img_lists[3][i]
    #
    #     img_f = torch.zeros(1, c, h, w).cuda()
    #     count = torch.zeros(1, c, h, w).cuda()
    #
    #     img_f[:, :, 0:h_cen + 3, 0: w_cen + 3] += img1
    #     count[:, :, 0:h_cen + 3, 0: w_cen + 3] += ones_temp[:, :, 0:h_cen + 3, 0: w_cen + 3]
    #     img_f[:, :, 0:h_cen + 3, w_cen - 2: w] += img2
    #     count[:, :, 0:h_cen + 3, w_cen - 2: w] += ones_temp[:, :, 0:h_cen + 3, w_cen - 2: w]
    #     img_f[:, :, h_cen - 2:h, 0: w_cen + 3] += img3
    #     count[:, :, h_cen - 2:h, 0: w_cen + 3] += ones_temp[:, :, h_cen - 2:h, 0: w_cen + 3]
    #     img_f[:, :, h_cen - 2:h, w_cen - 2: w] += img4
    #     count[:, :, h_cen - 2:h, w_cen - 2: w] += ones_temp[:, :, h_cen - 2:h, w_cen - 2: w]
    #     img_f = img_f / count
    #     img_f_list.append(img_f)
    return img_f_list

def recons_fusion_images1(img_lists, h, w):
    img_f_list = []
    ones_temp = torch.ones(1, h, w).to(args.device)
    for i in range(len(img_lists[0])):

        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]

        img_f = torch.zeros(1, h, w).to(args.device)
        count = torch.zeros(1, h, w).to(args.device)
        print(img_f.size())

        img_f[:, 0:256, 0: 256] += img1
        count[:, 0:256, 0: 256] += ones_temp[:, 0:256, 0: 256]
        img_f[:, 0:256, w-256: w] += img2
        count[:, 0:256, w-256: w] += ones_temp[:, 0:256, w-256:w]
        img_f[:, h-256:h, 0: 256] += img3
        count[:, h-256:h, 0: 256] += ones_temp[:, h-256:h, 0: 256]
        img_f[:, h-256:h, w-256: w] += img4
        count[:, h-256:h, w-256: w] += ones_temp[:, h-256:h, w-256: w]
        # img_g = img_f
        # for i in range(6):
        #     img_f[:, 256 + i, 0:w] = img_g[:, 256 - i, 0:w] = (img_g[:, 256 + i, 0:w] + img_g[:, 256 - i, 0:w]) / 2
        #     img_f[:, 0:h, 256 + i] = img_f[:, 0:h, 256 - i] = (img_g[:, 0:h, 256 + i] + img_g[:, 0:h, 256 - i]) / 2

        img_f = img_f / count
        img_f_list.append(img_f)
    # img_f_list = []
    # h_cen = int(np.floor(h / 2))
    # w_cen = int(np.floor(w / 2))
    # c = img_lists[0][0].shape[1]
    # ones_temp = torch.ones(1, c, h, w).cuda()
    # for i in range(len(img_lists[0])):
    #     # img1, img2, img3, img4
    #     img1 = img_lists[0][i]
    #     img2 = img_lists[1][i]
    #     img3 = img_lists[2][i]
    #     img4 = img_lists[3][i]
    #
    #     img_f = torch.zeros(1, c, h, w).cuda()
    #     count = torch.zeros(1, c, h, w).cuda()
    #
    #     img_f[:, :, 0:h_cen + 3, 0: w_cen + 3] += img1
    #     count[:, :, 0:h_cen + 3, 0: w_cen + 3] += ones_temp[:, :, 0:h_cen + 3, 0: w_cen + 3]
    #     img_f[:, :, 0:h_cen + 3, w_cen - 2: w] += img2
    #     count[:, :, 0:h_cen + 3, w_cen - 2: w] += ones_temp[:, :, 0:h_cen + 3, w_cen - 2: w]
    #     img_f[:, :, h_cen - 2:h, 0: w_cen + 3] += img3
    #     count[:, :, h_cen - 2:h, 0: w_cen + 3] += ones_temp[:, :, h_cen - 2:h, 0: w_cen + 3]
    #     img_f[:, :, h_cen - 2:h, w_cen - 2: w] += img4
    #     count[:, :, h_cen - 2:h, w_cen - 2: w] += ones_temp[:, :, h_cen - 2:h, w_cen - 2: w]

    #     img_f_list.append(img_f)
    return img_f_list


def recons_fusion_images2(img_lists, h, w):
    img_f_list = []
    ones_temp = torch.ones(1, h, w).to(args.device)

    for i in range(len(img_lists[0])):

        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]
        img5 = img_lists[4][i]
        img6 = img_lists[5][i]
        img7 = img_lists[6][i]
        img8 = img_lists[7][i]
        img9 = img_lists[8][i]
        img_f = torch.zeros(1, h, w).to(args.device)
        count = torch.zeros(1, h, w).to(args.device)
        sum = torch.zeros(1,24,w).to(args.device)
        print(w)

        # img_f[:, 0:256, 0: 256] += img1
        # img_f[:, 0:256, 256: 512] += img2
        # img_f[:, 0:256, 512: w] += img3[:, 0:256, 0:w - 512]
        # img_f[:, 256:512, 0: 256] += img4
        # img_f[:, 256:512, 256: 512] += img5
        # img_f[:, 256:512, 512: w] += img6[:, 0:256, 0:w - 512]
        # img_f[:, 512:h, 0: 256] += img7[:, 0:h - 512, 0:256]
        # img_f[:, 512:h, 256: 512] += img8[:, 0:h - 512, 0:256]
        # img_f[:, 512:h, 512: w] += img9[:, 0:h - 512, 0:w - 512]



        # img_f[:, 0:256+3, 0: 256+3] += img1
        # count[:, 0:256 + 3, 0: 256 + 3] += ones_temp[:, 0:256 + 3, 0: 256 + 3]
        # img_f[:, 0:256+3, 256-2: 512+3] += img2
        # count[:, 0:256 + 3, 256-2: 512 + 3] += ones_temp[:, 0:256 +3, 256-2: 512 + 3]
        # img_f[:, 0:256+3, 512-2: w] += img3
        # count[:, 0:256 + 3, 512-2: w] += ones_temp[:, 0:256 + 3, 512-2: w]
        # img_f[:, 256-2:512+3, 0: 256+3] += img4
        # count[:, 256-2:512 + 3, 0: 256 + 3] += ones_temp[:, 256-2:512 + 3, 0: 256 + 3]
        # img_f[:, 256-2:512+3, 256-2: 512+3] += img5
        # count[:, 256-2:512 + 3, 256-2: 512 + 3] += ones_temp[:, 256-2:512 + 3, 256-2: 512 + 3]
        # img_f[:, 256-2:512+3, 512-2: w] += img6
        # count[:, 256-2:512 + 3, 512-2: w] += ones_temp[:, 256-2:512+3, 512-2: w]
        # img_f[:, 576-256-2:h, 0: 256+3] += img7
        # count[:, 576-256-2:h, 0: 256 + 3] += ones_temp[:, 576-256-2, 0: 256 + 3]
        # img_f[:, 576-256-2:h, 256-2: 512+3] += img8
        # count[:, 576-256-2:h, 256-2: 512 + 3] += ones_temp[:, 576-256-2, 256-2: 512+3]
        # img_f[:, 576-256-2:h:, 512-2: w] += img9
        # count[:, 576-256-2:h, 512-2: w] += ones_temp[:, 576-256-2:h, 512-2: w]

        img_f[:, 0:256, 0: 256] += img1
        count[:, 0:256 , 0: 256 ] += ones_temp[:, 0:256 , 0: 256 ]
        img_f[:, 0:256, 256: 512] += img2
        count[:, 0:256 , 256: 512 ] += ones_temp[:, 0:256, 256: 512 ]
        img_f[:, 0:256, w-256: w] += img3
        count[:, 0:256 , w-256: w] += ones_temp[:, 0:256 , w-256: w]
        img_f[:, 256:512, 0: 256] += img4
        count[:, 256:512 , 0: 256 ] += ones_temp[:, 256:512 , 0: 256 ]
        img_f[:, 256:512, 256: 512] += img5
        count[:, 256:512 , 256: 512 ] += ones_temp[:, 256:512 , 256: 512 ]
        img_f[:, 256:512, w-256: w] += img6
        count[:, 256:512 , w-256: w] += ones_temp[:, 256:512, w-256: w]
        img_f[:, h-256:h, 0: 256] += img7
        count[:, h-256:h, 0: 256 ] += ones_temp[:, h-256, 0: 256 ]
        img_f[:, h-256:h, 256: 512] += img8
        count[:, h-256:h, 256: 512 ] += ones_temp[:, h-256, 256: 512]
        img_f[:, h-256:h:, w-256: w] += img9
        count[:, h-256:h, w-256: w] += ones_temp[:, h-256:h, w-256: w]

        # img_f[:, 255:257,0:w]=(img_g[:, 255,0:w]+img_g[:,256,0:w])/2
        # img_f[:, 255:257, 0:w] = (img_g[:, 255, 0:w] + img_g[:, 256, 0:w]) / 2

        img_g = img_f
        # for i in range(6):
        #     img_f[:, 256 + i, 0:w] = img_g[:, 256 - i, 0:w] = (img_g[:, 256 + i, 0:w] + img_g[:, 256 - i, 0:w]) / 2
        #     img_f[:,512+i,0:w] = img_g[:,512-i,0:w]=(img_g[:,512+i,0:w]+img_g[:,512-i,0:w])/2
        #     img_f[:, 0:h, 256+i] =img_f[:, 0:h, 256-i]= (img_g[:, 0:h, 256+i] + img_g[:, 0:h, 256-i]) / 2
        #     img_f[:, 0:h, 512 + i] = img_f[:, 0:h, 512 - i] = (img_g[:, 0:h, 512 + i] + img_g[:, 0:h, 512 - i]) / 2
        # img_f[:, 0:h, 256] = (img_g[:, 0:h, 255] + img_g[:, 0:h, 256]) / 2
        # img_f[:, 0:h, 511] = (img_g[:, 0:h, 511] + img_g[:, 0:h, 512]) / 2

        # img_f[:, 0:h, 511:513] = (img_g[:, 0:h, 511] + img_g[:, 0:h, 512]) / 2
        # for i in range(24):
        #     sum += img_f[:500+i,0:w]

        # img_f[:, 500:524, 0:w] = sum/24


        img_f = img_f / count

        # img_f[:,254:259,254:259] = img_f[:,254:259,254:259]/2
        # img_f[:, 510:515, 510:515] = img_f[:, 510:515, 510:515] / 2
        # img_f[:, 254:259, 510:515] = img_f[:, 254:259, 510:515] / 2
        # img_f[:, 510:515, 254:259] = img_f[:, 510:515, 254:259] / 2
        img_f_list.append(img_f)
    return img_f_list
def recons_fusion_images576768(img_lists, h, w):
    img_f_list = []
    ones_temp = torch.ones(1, h, w).to(args.device)

    for i in range(len(img_lists[0])):
        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]
        img5 = img_lists[4][i]
        img6 = img_lists[5][i]
        img7 = img_lists[6][i]
        img8 = img_lists[7][i]
        img9 = img_lists[8][i]
        img_f = torch.zeros(1, h, w).to(args.device)
        count = torch.zeros(1, h, w).to(args.device)
        sum = torch.zeros(1, 24, w).to(args.device)
        print(w)


        img_f[:, 0:256, 0: 256] += img1
        count[:, 0:256, 0: 256] += ones_temp[:, 0:256, 0: 256]
        img_f[:, 0:256, 256: 512] += img2
        count[:, 0:256, 256: 512] += ones_temp[:, 0:256, 256: 512]
        img_f[:, 0:256, 512: w] += img3
        count[:, 0:256, 512: w] += ones_temp[:, 0:256, 512: w]
        img_f[:, 256:512, 0: 256] += img4
        count[:, 256:512, 0: 256] += ones_temp[:, 256:512, 0: 256]
        img_f[:, 256:512, 256: 512] += img5
        count[:, 256:512, 256: 512] += ones_temp[:, 256:512, 256: 512]
        img_f[:, 256:512, 512: w] += img6
        count[:, 256:512, 512: w] += ones_temp[:, 256:512, 512: w]
        img_f[:, 576 - 256:h, 0: 256] += img7
        count[:, 576 - 256:h, 0: 256] += ones_temp[:, 576 - 256:h, 0: 256]
        img_f[:, 576 - 256:h, 256: 512] += img8
        count[:, 576 - 256:h, 256: 512] += ones_temp[:, 576 - 256:h, 256: 512]
        img_f[:, 576 - 256:h:, 512: w] += img9
        count[:, 576 - 256:h, 512: w] += ones_temp[:, 576 - 256:h, 512: w]

        img_f = img_f / count


        img_f_list.append(img_f)

    return img_f_list

def recons_fusion_images450620(img_lists, h, w):
    img_f_list = []
    ones_temp = torch.ones(1, h, w).to(args.device)

    for i in range(len(img_lists[0])):
        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]
        img5 = img_lists[4][i]
        img6 = img_lists[5][i]
        img7 = img_lists[6][i]
        img8 = img_lists[7][i]

        img_f = torch.zeros(1, h, w).to(args.device)
        count = torch.zeros(1, h, w).to(args.device)
        sum = torch.zeros(1, 24, w).to(args.device)
        print(w)


        img_f[:, 0:256, 0: 256] += img1
        count[:, 0:256, 0: 256] += ones_temp[:, 0:256, 0: 256]
        img_f[:, 0:256, 128: 384] += img2
        count[:, 0:256, 128: 384] += ones_temp[:, 0:256, 128: 384]
        img_f[:, 0:256, 256: 512] += img3
        count[:, 0:256, 256: 512] += ones_temp[:, 0:256, 256: 512]
        img_f[:, 0:256, w-256: w] += img4
        count[:, 0:256, w-256: w] += ones_temp[:, 0:256, w-256: w]
        img_f[:, h-256:h, 0: 256] += img5
        count[:, h-256:h, 0: 256] += ones_temp[:, h-256:h, 0: 256]
        img_f[:, h-256:h, 128: 384] += img6
        count[:, h-256:h, 128: 384] += ones_temp[:, h-256:h, 128: 384]
        img_f[:, h - 256:h, 256: 512] += img7
        count[:, h - 256:h, 256: 512] += ones_temp[:, h - 256:h, 256: 512]
        img_f[:, h - 256:h, w-256: w] += img8
        count[:, h - 256:h, w-256: w] += ones_temp[:, h - 256:h, w-256: w]

        img_f = img_f / count


        img_f_list.append(img_f)

    return img_f_list

# def recons_fusion_images450620(img_lists, h, w):
#     img_f_list = []
#     ones_temp = torch.ones(1, h, w).to(args.device)
#
#     for i in range(len(img_lists[0])):
#         img1 = img_lists[0][i]
#         img2 = img_lists[1][i]
#         img3 = img_lists[2][i]
#         img4 = img_lists[3][i]
#         img5 = img_lists[4][i]
#         img6 = img_lists[5][i]
#
#         img_f = torch.zeros(1, h, w).to(args.device)
#         count = torch.zeros(1, h, w).to(args.device)
#         sum = torch.zeros(1, 24, w).to(args.device)
#         print(w)
#
#
#         img_f[:, 0:256, 0: 256] += img1
#         count[:, 0:256, 0: 256] += ones_temp[:, 0:256, 0: 256]
#         img_f[:, 0:256, 256: 512] += img2
#         count[:, 0:256, 256: 512] += ones_temp[:, 0:256, 256: 512]
#         img_f[:, 0:256, w-256: w] += img3
#         count[:, 0:256, w-256: w] += ones_temp[:, 0:256, w-256: w]
#         img_f[:, h-256:h, 0: 256] += img4
#         count[:, h-256:h, 0: 256] += ones_temp[:, h-256:h, 0: 256]
#         img_f[:, h-256:h, 256: 512] += img5
#         count[:, h-256:h, 256: 512] += ones_temp[:, h-256:h, 256: 512]
#         img_f[:, h-256:h, w-256: w] += img6
#         count[:, h-256:h, w-256: w] += ones_temp[:, h-256:h, w-256: w]
#
#         img_f = img_f / count
#
#
#         img_f_list.append(img_f)
#
#     return img_f_list

# def recons_fusion_images2(img_lists, h, w):
#     img_f_list = []
#     for i in range(len(img_lists[0])):
#
#         img1 = img_lists[0][i]
#         img2 = img_lists[1][i]
#         img3 = img_lists[2][i]
#         img4 = img_lists[3][i]
#         img5 = img_lists[4][i]
#         img6 = img_lists[5][i]
#         img7 = img_lists[6][i]
#         img8 = img_lists[7][i]
#         img9 = img_lists[8][i]
#         img_f = torch.zeros(1, h, w).to(args.device)
#         print(w)
#         img_f[:, 0:256, 0: 256] += img1
#         img_f[:, 0:256, 256: 512] += img2
#         img_f[:, 0:256, 512: w] += img3[:, 0:256, 0:w-512]
#         img_f[:, 256:512, 0: 256] += img4
#         img_f[:, 256:512, 256: 512] += img5
#         img_f[:, 256:512, 512: w] += img6[:, 0:256, 0:w-512]
#         img_f[:, 512:h, 0: 256] += img7[:, 0:h-512, 0:256]
#         img_f[:, 512:h, 256: 512] += img8[:, 0:h-512, 0:256]
#         img_f[:, 512:h, 512: w] += img9[:, 0:h-512, 0:w - 512]
#
#         # patch = F.unfold(img_f.unsqueeze(0), kernel_size=256, stride=128, padding=249)
#         #
#         # patch = F.fold(patch, (576, 768), kernel_size=256, stride=128, padding=249)
#
#         # img_f_list.append(patch.squeeze(0))
#
#
#         img_f_list.append(img_f)
#
#     return img_f_list


def recons_fusion_images3(img_lists, h, w):
    img_f_list = []
    ones_temp = torch.ones(1, h, w).to(args.device)

    for i in range(len(img_lists[0])):

        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]
        img5 = img_lists[4][i]
        img6 = img_lists[5][i]
        img7 = img_lists[6][i]
        img8 = img_lists[7][i]
        img9 = img_lists[8][i]
        img10 = img_lists[9][i]
        img11 = img_lists[10][i]
        img12 = img_lists[11][i]
        img_f = torch.zeros(1, h, w).to(args.device)
        count = torch.zeros(1, h, w).to(args.device)

        img_f[:, 0:256, 0: 256] += img1
        count[:, 0:256, 0: 256] += ones_temp[:, 0:256, 0: 256]
        img_f[:, 0:256, 256: 512] += img2
        count[:, 0:256, 256: 512] += ones_temp[:, 0:256, 256: 512]
        img_f[:, 0:256, 512: 768] += img3
        count[:, 0:256, 512: 768] += ones_temp[:, 0:256, 512: 768]
        img_f[:, 0:256, w-256: w] += img4
        count[:, 0:256, w-256: w] += ones_temp[:, 0:256, w-256: w]
        img_f[:, 256:512, 0: 256] += img5
        count[:, 256:512, 0: 256] += ones_temp[:, 256:512, 0: 256]
        img_f[:, 256:512, 256: 512] += img6
        count[:, 256:512, 256: 512] += ones_temp[:, 256:512, 256: 512]
        img_f[:, 256:512, 512: 768] += img7
        count[:, 256:512, 512: 768] += ones_temp[:, 256:512, 512: 768]
        img_f[:, 256:512, w-256: w] += img8
        count[:, 256:512, w-256: w] += ones_temp[:, 256:512, w-256: w]
        img_f[:, h-256:h, 0: 256] += img9
        count[:, h-256:h, 0: 256] += ones_temp[:, h-256:h, 0: 256]
        img_f[:, h-256:h, 256: 512] += img10
        count[:, h-256:h, 256: 512] += ones_temp[:, h-256:h, 256: 512]
        img_f[:, h-256:h, 512: 768] += img11
        count[:, h-256:h, 512: 768] += ones_temp[:, h-256:h, 512: 768]
        img_f[:, h-256:h, w-256: w] += img12
        count[:, h-256:h, w-256: w] += ones_temp[:, h-256:h, w-256: w]
        # img_g = img_f
        # for i in range(6):
        #     img_f[:, 256 + i, 0:w] = img_g[:, 256 - i, 0:w] = (img_g[:, 256 + i, 0:w] + img_g[:, 256 - i, 0:w]) / 2
        #     img_f[:,512+i,0:w] = img_g[:,512-i,0:w]=(img_g[:,512+i,0:w]+img_g[:,512-i,0:w])/2
        #
        #     img_f[:, 0:h, 256+i] =img_f[:, 0:h, 256-i]= (img_g[:, 0:h, 256+i] + img_g[:, 0:h, 256-i]) / 2
        #     img_f[:, 0:h, 512 + i] = img_f[:, 0:h, 512 - i] = (img_g[:, 0:h, 512 + i] + img_g[:, 0:h, 512 - i]) / 2
        #     img_f[:, 0:h, 768 + i] = img_f[:, 0:h, 768 - i] = (img_g[:, 0:h, 768 + i] + img_g[:, 0:h, 768 - i]) / 2
        img_f = img_f / count
        img_f_list.append(img_f)
    return img_f_list


def recons_fusion_images4(img_lists, h, w):
    img_f_list = []
    ones_temp = torch.ones(1, h, w).to(args.device)
    for i in range(len(img_lists[0])):

        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]
        img5 = img_lists[4][i]
        img6 = img_lists[5][i]

        img_f = torch.zeros(1, h, w).to(args.device)
        count = torch.zeros(1, h, w).to(args.device)
        img_f[:, 0:256, 0: 256] += img1
        count[:, 0:256, 0: 256] += ones_temp[:, 0:256, 0: 256]
        img_f[:, 0:256, 256: 512] += img2
        count[:, 0:256, 256: 512] += ones_temp[:, 0:256, 256: 512]
        img_f[:, 0:256, w-256: w] += img3
        count[:, 0:256, w-256: w] += ones_temp[:, 0:256, w-256: w]
        img_f[:, h-256:h, 0: 256] += img4
        count[:, h-256:h, 0: 256] += ones_temp[:, h-256:h, 0: 256]
        img_f[:, h-256:h, 256: 512] += img5
        count[:, h-256:h, 256: 512] += ones_temp[:, h-256:h, 256: 512]
        img_f[:, h-256:h, w-256: w] += img6
        count[:, h-256:h, w-256: w] += ones_temp[:, h-256:h, w-256: w]
        # img_g = img_f
        # for i in range(6):
        #     img_f[:, 256 + i, 0:w] = img_g[:, 256 - i, 0:w] = (img_g[:, 256 + i, 0:w] + img_g[:, 256 - i, 0:w]) / 2
        #     img_f[:, 0:h, 256 + i] = img_f[:, 0:h, 256 - i] = (img_g[:, 0:h, 256 + i] + img_g[:, 0:h, 256 - i]) / 2
        #     img_f[:, 0:h, 512 + i] = img_f[:, 0:h, 512 - i] = (img_g[:, 0:h, 512 + i] + img_g[:, 0:h, 512 - i]) / 2
        img_f = img_f/count

        img_f_list.append(img_f)
    return img_f_list


def recons_fusion_images5(img_lists, h, w):
    img_f_list = []
    ones_temp = torch.ones(1, h, w).to(args.device)
    for i in range(len(img_lists[0])):

        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]
        img5 = img_lists[4][i]
        img6 = img_lists[5][i]
        img7 = img_lists[6][i]
        img8 = img_lists[7][i]
        img9 = img_lists[8][i]
        img10 = img_lists[9][i]
        img11 = img_lists[10][i]
        img12 = img_lists[11][i]
        img13 = img_lists[12][i]
        img14 = img_lists[13][i]
        img15 = img_lists[14][i]
        img16 = img_lists[15][i]
        img17 = img_lists[16][i]
        img18 = img_lists[17][i]
        img19 = img_lists[18][i]
        img20 = img_lists[19][i]
        img_f = torch.zeros(1, h, w).to(args.device)
        count = torch.zeros(1, h, w).to(args.device)

        img_f[:, 0:256, 0: 256] += img1
        count[:, 0:256, 0: 256] += ones_temp[:, 0:256, 0: 256]
        img_f[:, 0:256, 256: 512] += img2
        count[:, 0:256, 256: 512] += ones_temp[:, 0:256, 256: 512]
        img_f[:, 0:256, 512: 768] += img3
        count[:, 0:256, 512: 768] += ones_temp[:, 0:256, 512: 768]
        img_f[:, 0:256, 768: 1024] += img4
        count[:, 0:256, 768: 1024] += ones_temp[:, 0:256, 768: 1024]
        img_f[:, 0:256, w-256: w] += img5
        count[:, 0:256, w-256: w] += ones_temp[:, 0:256, w-256: w]
        img_f[:, 256:512, 0: 256] += img6
        count[:, 256:512, 0: 256] += ones_temp[:, 256:512, 0: 256]
        img_f[:, 256:512, 256: 512] += img7
        count[:, 256:512, 256: 512] += ones_temp[:, 256:512, 256: 512]
        img_f[:, 256:512, 512: 768] += img8
        count[:, 256:512, 512: 768] += ones_temp[:, 256:512, 512: 768]
        img_f[:, 256:512, 768: 1024] += img9
        count[:, 256:512, 768: 1024] += ones_temp[:, 256:512, 768: 1024]
        img_f[:, 256:512, w-256: w] += img10
        count[:, 256:512, w-256: w] += ones_temp[:, 256:512, w-256: w]
        img_f[:, 512:768, 0: 256] += img11
        count[:, 512:768, 0: 256] += ones_temp[:, 512:768, 0: 256]
        img_f[:, 512:768, 256: 512] += img12
        count[:, 512:768, 256: 512] += ones_temp[:, 512:768, 256: 512]
        img_f[:, 512:768, 512: 768] += img13
        count[:, 512:768, 512: 768] += ones_temp[:, 512:768, 512: 768]
        img_f[:, 512:768, 768: 1024] += img14
        count[:, 512:768, 768: 1024] += ones_temp[:, 512:768, 768: 1024]
        img_f[:, 512:768, w-256: w] += img15
        count[:, 512:768, w-256: w] += ones_temp[:, 512:768, w-256: w]
        img_f[:, h-256:h, 0: 256] += img16
        count[:, h-256:h, 0: 256] += ones_temp[:, h-256:h, 0: 256]
        img_f[:, h-256:h, 256: 512] += img17
        count[:, h-256:h, 256: 512] += ones_temp[:, h-256:h, 256: 512]
        img_f[:, h-256:h, 512: 768] += img18
        count[:, h-256:h, 512: 768] += ones_temp[:, h-256:h, 512: 768]
        img_f[:, h-256:h, 768: 1024] += img19
        count[:, h-256:h, 768: 1024] += ones_temp[:, h-256:h, 768: 1024]
        img_f[:, h-256:h, w-256: w] += img20
        count[:, h-256:h, w-256: w] += ones_temp[:, h-256:h, w-256: w]

        img_f = img_f / count
        # img_g = img_f
        # for i in range(6):
        #     img_f[:, 256 + i, 0:w] = img_g[:, 256 - i, 0:w] = (img_g[:, 256 + i, 0:w] + img_g[:, 256 - i, 0:w]) / 2
        #     img_f[:, 512 + i, 0:w] = img_g[:, 512 - i, 0:w] = (img_g[:, 512 + i, 0:w] + img_g[:, 512 - i, 0:w]) / 2
        #     img_f[:, 768 + i, 0:w] = img_g[:, 768 - i, 0:w] = (img_g[:, 768 + i, 0:w] + img_g[:, 768 - i, 0:w]) / 2
        #
        #     img_f[:, 0:h, 256 + i] = img_f[:, 0:h, 256 - i] = (img_g[:, 0:h, 256 + i] + img_g[:, 0:h, 256 - i]) / 2
        #     img_f[:, 0:h, 512 + i] = img_f[:, 0:h, 512 - i] = (img_g[:, 0:h, 512 + i] + img_g[:, 0:h, 512 - i]) / 2
        #     img_f[:, 0:h, 768 + i] = img_f[:, 0:h, 768 - i] = (img_g[:, 0:h, 768 + i] + img_g[:, 0:h, 768 - i]) / 2
        #     img_f[:, 0:h, 1024 + i] = img_f[:, 0:h, 1024 - i] = (img_g[:, 0:h, 1024 + i] + img_g[:, 0:h, 1024 - i]) / 2

        img_f_list.append(img_f)
    return img_f_list


def recons_fusion_images6(img_lists, h, w):
    img_f_list = []
    ones_temp = torch.ones(1, h, w).to(args.device)
    for i in range(len(img_lists[0])):

        img1 = img_lists[0][i]
        img2 = img_lists[1][i]

        img_f = torch.zeros(1, h, w).to(args.device)
        count = torch.zeros(1, h, w).to(args.device)
        print(img_f.size())

        img_f[:, h-256:h, 0: 256] += img1
        count[:, h-256:h, 0: 256] += ones_temp[:, h-256:h, 0: 256]
        img_f[:, h-256:h, w-256: w] += img2
        count[:, h-256:h, w-256: w] += ones_temp[:, h-256:h, w-256: w]

        img_f = img_f / count

        # img_g = img_f
        # for i in range(6):
        #     img_f[:, 0:h, 256 + i] = img_f[:, 0:h, 256 - i] = (img_g[:, 0:h, 256 + i] + img_g[:, 0:h, 256 - i]) / 2

        img_f_list.append(img_f)
    return img_f_list


def recons_fusion_images7(img_lists, h, w):
    img_f_list = []
    ones_temp = torch.ones(1, h, w).to(args.device)
    for i in range(len(img_lists[0])):

        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]

        img_f = torch.zeros(1, h, w).to(args.device)
        count = torch.zeros(1, h, w).to(args.device)
        print(img_f.size())

        img_f[:, h-256:h, 0: 256] += img1
        count[:, h - 256:h, 0: 256] += ones_temp[:, h - 256:h, 0: 256]
        img_f[:, h-256:h, 256: 512] += img2
        count[:, h - 256:h, 256: 512] += ones_temp[:, h - 256:h, 256: 512]
        img_f[:, h-256:h, w-256: w] += img3
        count[:, h - 256:h, w-256: w] += ones_temp[:, h - 256:h, w-256: w]

        # img_g = img_f
        # for i in range(6):
        #     img_f[:, 0:h, 256 + i] = img_f[:, 0:h, 256 - i] = (img_g[:, 0:h, 256 + i] + img_g[:, 0:h, 256 - i]) / 2
        #     img_f[:, 0:h, 512 + i] = img_f[:, 0:h, 512 - i] = (img_g[:, 0:h, 512 + i] + img_g[:, 0:h, 512 - i]) / 2
        img_f = img_f / count
        img_f_list.append(img_f)
    return img_f_list


def recons_fusion_images8(img_lists, h, w):
    img_f_list = []
    ones_temp = torch.ones(1, h, w).to(args.device)
    for i in range(len(img_lists[0])):

        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]

        img_f = torch.zeros(1, h, w).to(args.device)
        count = torch.zeros(1, h, w).to(args.device)
        print(img_f.size())

        img_f[:, h - 256:h, 0: 256] += img1
        count[:, h - 256:h, 0: 256] += ones_temp[:, h - 256:h, 0: 256]
        img_f[:, h - 256:h, 256: 512] += img2
        count[:, h - 256:h, 256: 512] += ones_temp[:, h - 256:h, 256: 512]
        img_f[:, h - 256:h, w - 256: w] += img3
        count[:, h - 256:h, w - 256: w] += ones_temp[:, h - 256:h, w - 256: w]

        # img_g = img_f
        # for i in range(6):
        #     img_f[:, 0:h, 256 + i] = img_f[:, 0:h, 256 - i] = (img_g[:, 0:h, 256 + i] + img_g[:, 0:h, 256 - i]) / 2
        #     img_f[:, 0:h, 512 + i] = img_f[:, 0:h, 512 - i] = (img_g[:, 0:h, 512 + i] + img_g[:, 0:h, 512 - i]) / 2
        img_f = img_f / count
        img_f_list.append(img_f)
    return img_f_list



def recons_fusion_images(img_lists, h, w):
    img_f_list = []
    h_cen = int(np.floor(h / 2))
    w_cen = int(np.floor(w / 2))
    ones_temp = torch.ones(1, 1, h, w).to(args.device)
    for i in range(len(img_lists[0])):
        # img1, img2, img3, img4
        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]

        # save_image_test(img1, './outputs/test/block1.png')
        # save_image_test(img2, './outputs/test/block2.png')
        # save_image_test(img3, './outputs/test/block3.png')
        # save_image_test(img4, './outputs/test/block4.png')

        img_f = torch.zeros(1, 1, h, w).to(args.device)
        count = torch.zeros(1, 1, h, w).to(args.device)

        img_f[:, :, 0:h_cen + 3, 0: w_cen + 3] += img1
        count[:, :, 0:h_cen + 3, 0: w_cen + 3] += ones_temp[:, :, 0:h_cen + 3, 0: w_cen + 3]
        img_f[:, :, 0:h_cen + 3, w_cen - 2: w] += img2
        count[:, :, 0:h_cen + 3, w_cen - 2: w] += ones_temp[:, :, 0:h_cen + 3, w_cen - 2: w]
        img_f[:, :, h_cen - 2:h, 0: w_cen + 3] += img3
        count[:, :, h_cen - 2:h, 0: w_cen + 3] += ones_temp[:, :, h_cen - 2:h, 0: w_cen + 3]
        img_f[:, :, h_cen - 2:h, w_cen - 2: w] += img4
        count[:, :, h_cen - 2:h, w_cen - 2: w] += ones_temp[:, :, h_cen - 2:h, w_cen - 2: w]
        img_f = img_f / count
        img_f_list.append(img_f)
    return img_f_list

def save_image_test(img_fusion, output_path):
    img_fusion = img_fusion.float()
    if args.cuda:
        img_fusion = img_fusion.cpu().data[0].numpy()
    else:
        img_fusion = img_fusion.clamp(0, 255).data[0].numpy()

    img_fusion = (img_fusion - np.min(img_fusion)) / (np.max(img_fusion) - np.min(img_fusion))
    img_fusion = img_fusion * 255
    # img_fusion = 255 - img_fusion
    print(img_fusion.shape)
    img_fusion = img_fusion.reshape([1,img_fusion.shape[0], img_fusion.shape[1]])#有些出来的是二维，需要变成3维
    img_fusion = img_fusion.transpose(1, 2, 0).astype('uint8')
    if img_fusion.shape[2] == 1:
        img_fusion = img_fusion.reshape([img_fusion.shape[0], img_fusion.shape[1]])
    cv2.imwrite(output_path, img_fusion)



def get_train_images(paths, height=256, width=256, flag=False):
    if isinstance(paths, str):
        paths = [paths]
    images_ir = []
    images_vi = []
    for path in paths:
        image = get_image(path, height, width, flag)
        image = np.reshape(image, [1, height, width])
        # imsave('./outputs/ir_gray.jpg', image)
        # image = image.transpose(2, 0, 1)
        images_ir.append(image)

        path_vi = path.replace('lwir', 'visible')
        image = get_image(path_vi, height, width, flag)
        image = np.reshape(image, [1, height, width])
        # imsave('./outputs/vi_gray.jpg', image)
        # image = image.transpose(2, 0, 1)
        images_vi.append(image)

    images_ir = np.stack(images_ir, axis=0)
    images_ir = torch.from_numpy(images_ir).float()

    images_vi = np.stack(images_vi, axis=0)
    images_vi = torch.from_numpy(images_vi).float()
    return images_ir, images_vi


# def get_train_images_auto(paths, height=256, width=256, flag=False):
#     if isinstance(paths, str):
#         paths = [paths]
#     images = []
#     for path in paths:
#         image = get_image(path, height, width, flag)
#         if flag is True:
#             image = np.transpose(image, (2, 0, 1))
#         else:
#             image = np.reshape(image, [1, height, width])
#         images.append(image)
#
#     images = np.stack(images, axis=0)
#     images = torch.from_numpy(images).float()
#     return images

def get_train_images_auto(paths, height=256, width=256, flag=False):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, flag)
        # image = image.sum(axis=2)
        if flag is True:
            image = np.transpose(image, (2, 0, 1))
        else:

            # image = np.sum(image, axis=2)

            if image.ndim==3:
                image = np.sum(image, axis=2)

            image = np.reshape(image, [1, height, width])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images.astype(np.float32))
    return images


# 自定义colormap
def colormap():
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#FFFFFF', '#98F5FF', '#00FF00', '#FFFF00','#FF0000', '#8B0000'], 256)

def spd_loss(matrixA, matrixB):
    # matrixA = matrixA.cpu().detach().numpy()
    # matrixB = matrixB.cpu().detach().numpy()

    A = np.array(matrixA)
    U, s, VT = la.svd(A)

    Sigma = np.zeros(np.shape(A))
    Sigma[:len(s), :len(s)] = np.diag(s)

    # 求矩阵的负二分之一次方
    eigenvalues, eigenvectors = np.linalg.eig(Sigma)
    root_eigenvalues = np.sqrt(1 / eigenvalues)
    matrix_pow = eigenvectors @ np.diag(root_eigenvalues) @ np.linalg.inv(eigenvectors)

    B = np.array(matrixB)
    C = U @ matrix_pow @ VT @ B @ U @ matrix_pow @ VT

    U, s, VT = la.svd(C)
    s = np.log(s)
    Sigma = np.zeros(np.shape(A))
    Sigma[:len(s), :len(s)] = np.diag(s)

    C = U @ Sigma @ VT

    f_norm = 0.5 * np.linalg.norm(C, 'fro')

    return f_norm

def init_vgg16(vgg, model_dir):
	vgg_load = torch.load(model_dir)
	count = 0
	for name, param in vgg_load.items():
		if count >= 20:
			break
		if count == 0:
			vgg.conv1_1.weight.data = param
		if count == 1:
			vgg.conv1_1.bias.data = param
		if count == 2:
			vgg.conv1_2.weight.data = param
		if count == 3:
			vgg.conv1_2.bias.data = param

		if count == 4:
			vgg.conv2_1.weight.data = param
		if count == 5:
			vgg.conv2_1.bias.data = param
		if count == 6:
			vgg.conv2_2.weight.data = param
		if count == 7:
			vgg.conv2_2.bias.data = param

		if count == 8:
			vgg.conv3_1.weight.data = param
		if count == 9:
			vgg.conv3_1.bias.data = param
		if count == 10:
			vgg.conv3_2.weight.data = param
		if count == 11:
			vgg.conv3_2.bias.data = param
		if count == 12:
			vgg.conv3_3.weight.data = param
		if count == 13:
			vgg.conv3_3.bias.data = param

		if count == 14:
			vgg.conv4_1.weight.data = param
		if count == 15:
			vgg.conv4_1.bias.data = param
		if count == 16:
			vgg.conv4_2.weight.data = param
		if count == 17:
			vgg.conv4_2.bias.data = param
		if count == 18:
			vgg.conv4_3.weight.data = param
		if count == 19:
			vgg.conv4_3.bias.data = param
		count = count + 1

# def spd_loss(P1,P2):
#     P1 = P1.cpu().detach().numpy()
#     P2 = P2.cpu().detach().numpy()
#     P1_sqrt = np.linalg.matrix_power(P1, -1 / 2)
#
#     # 计算仿射不变距离δR(P1; P2)
#     middle_matrix = P1_sqrt @ P2 @ P1_sqrt
#     # 取对数前确认矩阵是对称正定的，确保可以应用矩阵对数
#     middle_matrix_log = np.linalg.logm(middle_matrix)
#     affine_invariant_distance = 1 / 2 * np.linalg.norm(middle_matrix_log, 'fro')
#     return  affine_invariant_distance

def matrix_logarithm(matrix):
    # 计算矩阵的对数，输入假设为正定矩阵
    # U, D, U_inv = torch.svd(matrix)  # 计算奇异值分解

    U,D,U_inv = np.linalg.svd(matrix.cpu().detach().numpy(), full_matrices=True)

    U = torch.from_numpy(U).to(args.device)
    D = torch.from_numpy(D).to(args.device)
    U_inv = torch.from_numpy(U_inv)

    D_log = torch.diag_embed(torch.log(D)**2)  # 对角线元素取对数

    # for i in range(256):
    #     print(D_log[i,i])

    return U @ D_log @ U.transpose(0,1) # 计算U * log(D) * U_inv



def correlation_loss(Z, Z_bar):

    Z = Z.squeeze(0)
    Z_bar = Z_bar.squeeze(0)

    log_Z = matrix_logarithm(Z)
    log_Z_bar = matrix_logarithm(Z_bar)


    log_Z = torch.sum(log_Z, dim=0, keepdim=True)
    log_Z_bar = torch.sum(log_Z_bar, dim=0, keepdim=True)
    log = torch.sum((log_Z * log_Z_bar), dim=0, keepdim=True)
    if log.ndim == 3:
        log = log.squeeze(0)
    if log_Z.ndim == 3:
        log_Z = log_Z.squeeze(0)
    if log_Z_bar.ndim == 3:
        log_Z_bar = log_Z_bar.squeeze(0)

    Tr_all = torch.trace(log)
    Tr_Z = torch.trace(log_Z)
    Tr_Z_bar = torch.trace(log_Z_bar)

    # print(Tr_Z_bar)

    loss = - Tr_all / ((Tr_Z ** 0.5) * (Tr_Z_bar ** 0.5))


    # loss = torch.trace(Z)+torch.trace(Z_bar)
    # loss = loss.mean()
    # print(loss)
    return loss
