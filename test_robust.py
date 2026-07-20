

import os
import torch
from torch.autograd import Variable
from net import GrFormer
import utils_robust as utils
from args_fusion import args
import numpy as np
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
from thop import profile, clever_format
from PIL import Image

def tensor_to_pil(tensor, mode='RGB'):
    """
    将torch张量转换为PIL图像
    Args:
        tensor: torch.Tensor, 形状应为 [C, H, W]
        mode: PIL图像模式，如 'RGB', 'L' 等
    Returns:
        PIL.Image对象
    """
    # 确保张量在CPU上且是float类型
    tensor = tensor.cpu().detach()

    # 检查张量形状
    if tensor.dim() != 3:
        raise ValueError(f"输入张量应为3维 [C, H, W]，但得到的是 {tensor.shape}")

    # 如果是Byte类型（0-255），转换为float（0-1）
    if tensor.dtype == torch.uint8:
        tensor = tensor.float() / 255.0

    # 如果张量值范围不是0-1，进行归一化或clamp
    if tensor.min() < 0 or tensor.max() > 1:
        # 方法1：clamp到0-1范围
        tensor = tensor.clamp(0, 1)
        # 或者方法2：归一化到0-1范围
        # tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

    # 转换为numpy数组，并调整维度为 [H, W, C]
    # 注意：PIL需要 [H, W, C] 格式
    if tensor.shape[0] == 3:  # RGB图像
        # 从 [C, H, W] 转换为 [H, W, C]
        numpy_array = tensor.permute(1, 2, 0).numpy()
    else:
        # 对于单通道图像
        numpy_array = tensor.squeeze(0).numpy()

    # 转换为0-255的uint8
    numpy_array = (numpy_array * 255).astype(np.uint8)

    # 创建PIL图像
    pil_image = Image.fromarray(numpy_array, mode=mode)

    return pil_image

def clamp(value, min=0., max=1.0):
    """
    将像素值强制约束在[0,1], 以免出现异常斑点
    :param value:
    :param min:
    :param max:
    :return:
    """
    return torch.clamp(value, min=min, max=max)

def RGB2YCrCb(rgb_image):
    """
    将RGB格式转换为YCrCb格式

    :param rgb_image: RGB格式的图像数据
    :return: Y, Cr, Cb
    """
    R = rgb_image[0:1]
    G = rgb_image[1:2]
    B = rgb_image[2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    # Y = clamp(Y)
    # Cr = clamp(Cr)
    # Cb = clamp(Cb)



    # image = torch.sum(Cb, dim=0, keepdim=True)
    #
    # image = image.squeeze().detach().cpu().numpy()
    #
    # image = (image - image.min()) / (image.max() - image.min())
    # plt.imshow(image, cmap='gray')
    # plt.axis('off')
    # plt.show()

    return Y, Cb, Cr


def YCrCb2RGB(Y, Cb, Cr):
    """
    将YcrCb格式转换为RGB格式

    :param Y:
    :param Cb:
    :param Cr:
    :return:
    """
    ycrcb = torch.cat([Y, Cr, Cb], dim=0)
    C, W, H = ycrcb.shape
    im_flat = ycrcb.reshape(3, -1).transpose(0, 1)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.transpose(0, 1).reshape(C, W, H)
    out = (out - out.min()) / (out.max() - out.min())
    out = clamp(out)
    return out

def set_seed_thread(seed):
    seed = seed
    random.seed(seed)
    # th.cuda.set_device(args.gpu)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def load_model1(path, deepsupervision):
	input_nc = 1
	output_nc = 1
	nb_filter=[16, 64, 32, 16]

	nest_model = GrFormer(nb_filter, input_nc, output_nc, deepsupervision)
	nest_model.load_state_dict(torch.load(path, weights_only=False))

	para = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(nest_model._get_name(), para * type_size / 1000 / 1000))





	nest_model.eval()
	nest_model.to(args.device)

	return nest_model



def run_demo(nest_model, infrared_path, visible_path, output_path_root, index, f_type, mode):
	img_ir, h, w, c, viimage, meta_ir = utils.get_test_image1(infrared_path)
	img_vi, h, w, c, viimage, meta_vi = utils.get_test_image1(visible_path)
	meta = meta_ir if meta_ir is not None else meta_vi

	if viimage is not None and len(viimage.shape) == 3:
		# 转换为torch张量并调整维度顺序

		# image = torch.sum(viimage, dim=0, keepdim=True)
		# image = viimage.squeeze().detach().cpu().numpy()
		#
		# image = (image - image.min()) / (image.max() - image.min())
		# plt.imshow(image, cmap='gray')
		# plt.axis('off')
		# plt.show()

		viimage_tensor = torch.from_numpy(viimage).permute(2, 0, 1)

		if args.cuda:
			viimage_tensor = viimage_tensor.to(args.device)

		# 提取可见光图像的Cb和Cr通道
		Y_vi, Cb_vi, Cr_vi = RGB2YCrCb(viimage_tensor)

	# image = torch.sum(Y_vi, dim=0, keepdim=True)
	# image = image.squeeze().detach().cpu().numpy()
	#
	# image = (image - image.min()) / (image.max() - image.min())
	# plt.imshow(image)
	# plt.axis('off')
	# plt.show()

	# c = 1

	if c == 0:
		if args.cuda:
			img_ir = img_ir.to(args.device)
			img_vi = img_vi.to(args.device)
		img_ir = Variable(img_ir, requires_grad=False)
		img_vi = Variable(img_vi, requires_grad=False)

		bbb = nest_model.encoderspdtest(img_vi, img_ir)[0]

		img_fusion = bbb


	else:
		img_fusion_blocks = []
		img_fusion_list = []
		for i in range(c):
			img_vi_temp = img_vi[i]
			img_ir_temp = img_ir[i]
			if args.cuda:
				img_vi_temp = img_vi_temp.to(args.device)
				img_ir_temp = img_ir_temp.to(args.device)
			img_vi_temp = Variable(img_vi_temp, requires_grad=False)
			img_ir_temp = Variable(img_ir_temp, requires_grad=False)


			bbb = nest_model.forward(img_ir_temp, img_vi_temp)[0]


			img_fusion_blocks.append(bbb)
		print(h, "h")
		print(w, "w")
		if meta is not None and meta.get("use_generic"):
			img_fusion_list = utils.recons_fusion_images_any(img_fusion_blocks, h, w, meta)
		else:
			if h == 256 and w == 256:
				img_fusion_list = utils.recons_fusion_images11(img_fusion_blocks, h, w)
			if 256 < h <= 512 and 256 < w <= 512:
				img_fusion_list = utils.recons_fusion_images1(img_fusion_blocks, h, w)

			if h == 576 and w == 768:
				img_fusion_list = utils.recons_fusion_images576768(img_fusion_blocks, h, w)

			if h == 450 and w == 620:
				img_fusion_list = utils.recons_fusion_images450620(img_fusion_blocks, h, w)

			if 512 < h <= 768 and 512 < w <= 768:
				img_fusion_list = utils.recons_fusion_images2(img_fusion_blocks, h, w)
			if 512 < h < 768 and 768 < w <= 1024:
				img_fusion_list = utils.recons_fusion_images3(img_fusion_blocks, h, w)

			if 256 < h <= 512 and 512 < w < 768 and h != 450 and w != 620:
				img_fusion_list = utils.recons_fusion_images4(img_fusion_blocks, h, w)
			if 768 <= h <= 1024 and 1024 <= w <= 1280:
				img_fusion_list = utils.recons_fusion_images5(img_fusion_blocks, h, w)
			if 0 < h < 256 and 256 < w < 512:
				img_fusion_list = utils.recons_fusion_images6(img_fusion_blocks, h, w)
			if 0 < h < 256 and 512 < w < 768:
				img_fusion_list = utils.recons_fusion_images7(img_fusion_blocks, h, w)
			if h == 256 and 512 < w < 768:
				img_fusion_list = utils.recons_fusion_images8(img_fusion_blocks, h, w)

		# if index < 10:
		# 	file_name = '0' + str(index) + '.png'
		# else:
		# 	file_name = str(index) + '.png'
		# output_path = output_path_root + file_name

		# # save images
		# utils.save_image_test(img_fusion, output_path)
		# utils.tensor_save_rgbimage(img_fusion, output_path)
		# if args.cuda:
		# 	img = bbb.cpu().clamp(0, 255).data[0].numpy()
		# else:
		# 	img = bbb.clamp(0, 255).data[0].numpy()
		# img = img.transpose(1, 2, 0).astype('uint8')
		# utils.save_images(output_path, img)
		#
		# print(output_path)
	output_count = 0
	for img_fusion in img_fusion_list:

		# img_fusion是融合后的Y通道
		Y_fused = img_fusion

		# 调整尺寸匹配
		if Y_fused.shape[-2:] != Cb_vi.shape[-2:]:
			# 如果需要调整尺寸
			Y_fused_resized = torch.nn.functional.interpolate(
				Y_fused.unsqueeze(0),
				size=(h, w),
				mode='bilinear',
				align_corners=False
			).squeeze(0)
		else:
			Y_fused_resized = Y_fused

		# image = torch.sum(Y_fused_resized, dim=0, keepdim=True)

		# 将Y通道与Cb、Cr通道结合，转换回RGB
		rgb_fused_image = YCrCb2RGB(Y_fused_resized, Cb_vi, Cr_vi)
		if index < 10:
			file_name = '0' + str(index) + '.png'
		else:
			file_name = str(index) + '.png'
		output_path = output_path_root + file_name
		output_count += 1
		# save images
		pil_image = tensor_to_pil(rgb_fused_image)
		pil_image.save(output_path)
		print(f"Saved: {output_path}")


def main():
	# set_seed_thread(3407)
	# run demo
	test_path = "images/TNO40/"
	network_type = 'SwinFuse'
	fusion_type = ['l1_mean']

	output_path = 'outputsTNO40/'

	# in_c = 3 for RGB imgs; in_c = 1 for gray imgs
	in_chans = 1

	num_classes = in_chans
	mode = 'L'
	model_path = args.model_default

	with torch.no_grad():
		print('SSIM weight ----- ' + args.ssim_path[1])
		ssim_weight_str = args.ssim_path[1]
		f_type = fusion_type[0]

		model1 = load_model1(model_path, num_classes)




		# begin = time.time()
		# for a in range(10):
		for i in range(362):
			# for i in range(1000, 1221):
			# for i in range(1000, 1040):
			index = i + 1
			infrared_path = os.path.join(test_path, 'IR', f"{index:02d}.png")
    		visible_path = os.path.join(test_path, 'VIS', f"{index:02d}.png")
			# infrared_path = test_ir_path + 'roadscene' + '_' + str(index) + '.png'
			# visible_path = test_vis_path + 'roadscene' + '_' + str(index) + '.png'
			# infrared_path = test_ir_path + 'video' + '_' + str(index) + '.png'
			# visible_path = test_vis_path + 'video' + '_' +str(index) + '.png'
			run_demo(model1, infrared_path, visible_path, output_path, index, f_type,mode)
	# end = time.time()
	# print("consumption time of generating:%s " % (end - begin))
	print('Done......')



if __name__ == '__main__':
	main()
