

import os
import torch
from torch.autograd import Variable
from net import GrFormer
import utils
from args_fusion import args
import numpy as np
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
from thop import profile, clever_format


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
	nest_model.load_state_dict(torch.load(path))

	para = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(nest_model._get_name(), para * type_size / 1000 / 1000))





	nest_model.eval()
	nest_model.to(args.device)

	return nest_model



def run_demo(nest_model, infrared_path, visible_path, output_path_root, index, f_type,mode):
	img_ir,h,w,c = utils.get_test_image1(infrared_path)
	img_vi,h,w,c = utils.get_test_image1(visible_path)

	# c = 1

	if c is 0:
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
		if h == 256 and w == 256:
			img_fusion_list = utils.recons_fusion_images11(img_fusion_blocks, h, w)
		if 256 < h <= 512 and 256 < w <= 512:
			img_fusion_list = utils.recons_fusion_images1(img_fusion_blocks, h, w)

		if h== 576 and w == 768:
			img_fusion_list = utils.recons_fusion_images576768(img_fusion_blocks, h, w)

		if h == 450  and w == 620:
			img_fusion_list = utils.recons_fusion_images450620(img_fusion_blocks, h, w)

		if 512 < h <= 768 and 512 < w <= 768:
			img_fusion_list = utils.recons_fusion_images2(img_fusion_blocks, h, w)
		if 512 < h < 768 and 768 < w <= 1024:
			img_fusion_list = utils.recons_fusion_images3(img_fusion_blocks, h, w)

		if 256 < h < 512 and 512 < w < 768 and h!= 450 and w != 620:
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

		# image = img_fusion.squeeze().detach().cpu().numpy()
		#
		# image = (image - image.min()) / (image.max() - image.min())
		# plt.imshow(image, cmap='jet')
		# # plt.imshow(image, cmap='gray')
		# plt.axis('off')
		# plt.show()

		# print(img_fusion)
		# x_min2 = torch.min(img_fusion)
		# x_max2 = torch.max(img_fusion)
		# img_fusion = (img_fusion- x_min2) / (x_max2 - x_min2) * 255
		# img_fusion = -img_fusion
		if index < 10:
			file_name = '0' + str(index) + '.png'
		else:
			file_name = str(index) + '.png'
		output_path = output_path_root + file_name
		output_count += 1
		# save images
		utils.save_image_test(img_fusion, output_path)
		print(output_path)


def main():
	# set_seed_thread(3407)
	# run demo
	test_path = "images/test-TNO/"
	network_type = 'SwinFuse'
	fusion_type = ['l1_mean']

	output_path = 'outputsTNO/attention_avg/'

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
			infrared_path = test_path + 'IR' + str(index) + '.png'
			visible_path = test_path + 'VIS' + str(index) + '.png'
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
