
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import cv2
import sys
import time
import numpy as np
from tqdm import tqdm, trange
import scipy.io as scio
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import utils
from net import GrFormer, Vgg16
from args_fusion import args
import pytorch_msssim
import matplotlib.pyplot as plt
import torchvision.utils
import seaborn as sns
import pandas as pd
import torchvision.transforms as transforms
# from scipy.misc import imread, imsave, imresize
import torch.nn.functional as F



def set_seed_thread(seed):
    seed = seed
    random.seed(seed)
    # th.cuda.set_device(args.gpu)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def sigmoid1(x):
    return 1 / (1 + torch.exp(-x))


def PatchFlattener1(x,patch_size):
    batch_size, channels, height, width = x.size()
    assert height % patch_size == 0 and width % patch_size == 0, "Invalid patch size"
    num_patches = (height // patch_size) * (width // patch_size)
    patch_height = height // patch_size
    patch_width = width // patch_size
    # Rearrange patches into columns
    x = x.view(batch_size, channels, patch_height, patch_size, patch_width, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    x = x.view(batch_size, num_patches, -1)
    return x

def covariance(x):  # input[1,16,224,224]

    s = torch.squeeze(x, dim=0)
    s1 = s
    s = s.cpu().detach().numpy()

    cov = np.cov(s, rowvar=True)
    # cov = np.cov(sir, svi, rowvar=True)[:256, 256:]

    cov = torch.from_numpy(cov)

    x = torch.matmul(cov.to(args.device),s1.to(args.device))
    return [x]

class Sobelxy(nn.Module):
    def __init__(self, device):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        # 这里不行就采用expend_dims
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).to(device=device)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).to(device=device)
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx), torch.abs(sobely)

# class Sobelxy(nn.Module):
#     def __init__(self):
#         super(Sobelxy, self).__init__()
#         kernelx = [[-1, 0, 1],
#                   [-2,0 , 2],
#                   [-1, 0, 1]]
#         kernely = [[1, 2, 1],
#                   [0,0 , 0],
#                   [-1, -2, -1]]
#         kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
#         kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
#         self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
#         self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
#     def forward(self,x):
#         sobelx=F.conv2d(x, self.weightx, padding=1)
#         sobely=F.conv2d(x, self.weighty, padding=1)
#         return torch.abs(sobelx)+torch.abs(sobely)

def main():

	original_imgs_path = utils.list_images(args.dataset)
	train_num = 80000
	original_imgs_path = original_imgs_path[:train_num]
	random.shuffle(original_imgs_path)
	for i in range(2,3):
		# i = 3
		train(i, original_imgs_path)

	# train(i, original_imgs_path)

def train(i, original_imgs_path):

	batch_size = args.batch_size

	# load network model
	# nest_model = FusionNet_gra()
	input_nc = 1
	output_nc = 1
	# true for deeply supervision
	# In our paper, deeply supervision strategy was not used.
	deepsupervision = False
	nb_filter=[16, 64, 32, 16]

	nest_model = GrFormer(nb_filter, input_nc, output_nc, deepsupervision)


	if args.resume is not None:
		print('Resuming, initializing using weight from {}.'.format(args.resume))
		nest_model.load_state_dict(torch.load(args.resume))
	print(nest_model)
	optimizer = Adam(nest_model.parameters(), args.lr)

	mse_loss = torch.nn.MSELoss()
	ssim_loss = pytorch_msssim.msssim

	vgg = Vgg16()
	utils.init_vgg16(vgg, os.path.join(args.vgg_model_dir, "vgg16.pth"))

	if args.cuda:
		nest_model.to(args.device)
		vgg.to(args.device)

	tbar = trange(args.epochs)
	print('Start training.....')

	Loss_pixel = []
	Loss_grad = []
	Loss_ssim1 = []
	Loss_ssim2 = []
	Loss_spd = []
	Loss_corr=[]
	Loss_all = []
	count_loss = 0
	all_ssim_loss = 0.
	all_pixel_loss = 0.
	all_grad_loss = 0.
	all_spd_loss = 0.
	all_corr_loss = 0.
	for e in tbar:
		print('Epoch %d.....' % e)
		# load training database
		image_set_ir, batches = utils.load_dataset(original_imgs_path, batch_size)

		nest_model.train()
		count = 0
		for batch in range(batches):
			image_paths = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]


			img = utils.get_train_images_auto(image_paths, height=args.HEIGHT, width=args.WIDTH, flag=False)

			image_paths_ir = [x.replace('VIS', 'IR') for x in image_paths]
			img_ir = utils.get_train_images_auto(image_paths_ir, height=args.HEIGHT, width=args.WIDTH, flag=False)

			image_paths_vi = [x.replace('IR', 'VIS') for x in image_paths]
			img_vi = utils.get_train_images_auto(image_paths_vi, height=args.HEIGHT, width=args.WIDTH, flag=False)




			count += 1
			optimizer.zero_grad()


			batch_ir = Variable(img_ir, requires_grad=False)		#原来是False
			batch_vi = Variable(img_vi, requires_grad=False)		#原来是False
			img = Variable(img, requires_grad=False)				#原来是False



			if args.cuda:
				img = img.to(args.device)
				batch_ir = batch_ir.to(args.device)
				batch_vi = batch_vi.to(args.device)


			outputs = nest_model.forward(batch_ir,batch_vi)


			x1 = Variable(batch_ir.data.clone(), requires_grad=True)		#原来是False
			x = Variable(img.data.clone(), requires_grad=False)  # 原来是False
			x2 = Variable(batch_vi.data.clone(), requires_grad=True)

			# xspd1 = Variable(spdd11.data.clone(), requires_grad=True)
			# xspd2 = Variable(spdd12.data.clone(), requires_grad=True)
			ssim_loss_value1 = 0.
			ssim_loss_value2 = 0.
			pixel_loss_value = 0.
			grad_loss_value = 0.
			spd_loss_value = 0.
			corr_loss_value= 0.

			sobelconv = Sobelxy(args.device)

			for output in outputs:


				outputsave = output/255.

				vi_grad_x, vi_grad_y = sobelconv(x1)
				ir_grad_x, ir_grad_y = sobelconv(x2)
				fu_grad_x, fu_grad_y = sobelconv(output)
				grad_joint_x = torch.max(vi_grad_x, ir_grad_x)
				grad_joint_y = torch.max(vi_grad_y, ir_grad_y)


				vgg_outs = vgg(output)
				vgg_irs = vgg(x1)
				vgg_vis = vgg(x2)


				# pixel_loss_temp = mse_loss(output, x1 + x2)
				pixel_loss_temp = F.l1_loss(output, torch.max(x1,x2))
				ssim_loss_temp1 = ssim_loss(output, x1.float(), normalize=True)
				ssim_loss_temp2 = ssim_loss(output, x2.float(), normalize=True)
				grad_loss_temp = F.l1_loss(grad_joint_x, fu_grad_x) + F.l1_loss(grad_joint_y, fu_grad_y)
				# grad_loss_temp = F.l1_loss(x_grad_joint, generate_img_grad)
				# spd_loss_temp = utils.spd_loss(covir, covout) + utils.spd_loss(covvi, covout)
				# spd_loss_temp = F.l1_loss(covir,covout)

				t_idx = 0
				for fea_out, fea_ir, fea_vi in zip(vgg_outs, vgg_irs, vgg_vis):

					if t_idx == 2:
						gram_out = utils.gram_matrix(fea_out)
						gram_ir = utils.gram_matrix(fea_ir)
						gram_vi = utils.gram_matrix(fea_vi)

						spd_loss_temp1 = F.l1_loss(gram_vi, gram_out)

						corr_loss_temp1 = utils.correlation_loss(gram_vi,gram_out).requires_grad_(True)



					if t_idx == 3:
						gram_out = utils.gram_matrix(fea_out)
						gram_ir = utils.gram_matrix(fea_ir)
						gram_vi = utils.gram_matrix(fea_vi)

						# covir = covariance(fea_ir)[0]
						# covvi = SPD_model.spdconvx(fea_vi)[0]
						# covout = SPD_model.spdconvx(fea_out)[0]
						#
						# covir = covir.requires_grad_(True)
						# covvi = covvi.requires_grad_(True)
						# covout = covout.requires_grad_(True)

						spd_loss_temp2 = F.l1_loss(gram_vi, gram_out)
						corr_loss_temp2 = utils.correlation_loss(gram_vi,gram_out).requires_grad_(True)


					t_idx += 1



				ssim_loss_value1 += (1 - ssim_loss_temp1)
				ssim_loss_value2 += (1 - ssim_loss_temp2)
				pixel_loss_value += pixel_loss_temp
				grad_loss_value += grad_loss_temp
				spd_loss_value += spd_loss_temp1
				spd_loss_value += spd_loss_temp2
				corr_loss_value += corr_loss_temp1
				corr_loss_value += corr_loss_temp2


				# pixel_loss_temp = mse_loss(output,x)
				# ssim_loss_temp = ssim_loss(output,x, normalize=True)
				# pixel_loss_value += pixel_loss_temp
				# ssim_loss_value += (1 - ssim_loss_temp)

			# total loss
			total_loss = 20 * pixel_loss_value + 20 * grad_loss_value + 30 * spd_loss_value + 2000 * ssim_loss_value1

			total_loss.backward()
			optimizer.step()


			all_ssim_loss += ssim_loss_value1.item()
			# all_ssim_loss += ssim_loss_value2.item()
			all_pixel_loss += pixel_loss_value.item()
			all_grad_loss += grad_loss_value.item()
			all_spd_loss += spd_loss_value.item()
			all_corr_loss += corr_loss_value.item()
			if (batch + 1) % args.log_interval == 0:
				mesg = "{}\t SSIM weight {}\tEpoch {}:\t[{}/{}]\t pixel loss: {:.6f}\t  grad loss: {:.6f}\t spd loss: {:.6f}\t total: {:.6f}".format(
					time.ctime(), i, e + 1, count, batches,
								  # all_pixel_loss / args.log_interval,
								  # (args.ssim_weight[i] * all_ssim_loss) / args.log_interval,
								  # (args.ssim_weight[i] * all_ssim_loss + all_pixel_loss) / args.log_interval
								100 * pixel_loss_value / args.log_interval,
								100 * grad_loss_value / args.log_interval,
								50 * spd_loss_value / args.log_interval,
								(100 * pixel_loss_value + 100 * grad_loss_value+ 50 * spd_loss_value) / args.log_interval
				)
				tbar.set_description(mesg)
				Loss_pixel.append(100 * all_pixel_loss / args.log_interval)
				# Loss_ssim1.append(100 * all_ssim_loss / args.log_interval)
				Loss_grad.append(100 * all_grad_loss / args.log_interval)
				Loss_spd.append(10 * all_spd_loss / args.log_interval)
				# Loss_corr.append(50 * all_corr_loss / args.log_interval)
				Loss_all.append((100 * all_pixel_loss + 100* all_grad_loss  + 50 * all_spd_loss) / args.log_interval)
				count_loss = count_loss + 1
				all_ssim_loss = 0.
				all_pixel_loss = 0.
				all_grad_loss = 0.
				all_spd_loss = 0.
				all_corr_loss = 0.

			if (batch + 1) % (200 * args.log_interval) == 0:
				# save model
				nest_model.eval()
				nest_model.cpu()

				save_model_filename = args.ssim_path[i] + '/' + "Epoch_" + str(e) + "_iters_" + str(count) + "_" + \
									  str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[
										  i] + ".model"
				save_model_path = os.path.join(args.save_model_dir_autoencoder, save_model_filename)
				torch.save(nest_model.state_dict(), save_model_path)
				# save loss data
				# pixel loss
				loss_data_pixel = Loss_pixel
				loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "loss_pixel_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path[i] + ".mat"
				scio.savemat(loss_filename_path, {'loss_pixel': loss_data_pixel})
				# SSIM loss
				loss_data_ssim = Loss_ssim1
				loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "loss_ssim_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path[i] + ".mat"
				scio.savemat(loss_filename_path, {'loss_ssim': loss_data_ssim})
				# grad loss
				loss_data_grad = Loss_grad
				loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "loss_grad_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':',
																											  '_') + "_" + \
									 args.ssim_path[i] + ".mat"
				scio.savemat(loss_filename_path, {'loss_grad': loss_data_grad})
				# # spd loss
				# loss_data_spd = Loss_spd
				# loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "loss_spd_epoch_" + str(
				# 	args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':',
				# 																							  '_') + "_" + \
				# 					 args.ssim_path[i] + ".mat"
				# scio.savemat(loss_filename_path, {'loss_spd': loss_data_spd})
				# all loss
				loss_data = Loss_all

				loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "loss_all_epoch_" + str(e) + "_iters_" + \
									 str(count) + "-" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path[i] + ".mat"
				scio.savemat(loss_filename_path, {'loss_all': loss_data})



				nest_model.train()
				nest_model.to(args.device)

				tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)
		torchvision.utils.save_image(outputsave, './saveimage/{}.png'.format(e))
		# torchvision.utils.save_image(outputsave2, './saveimage2/{}.png'.format(e))
	# pixel loss
	loss_data_pixel = Loss_pixel
	loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "Final_loss_pixel_epoch_" + str(
		args.epochs) + "_" + str(
		time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".mat"
	scio.savemat(loss_filename_path, {'final_loss_pixel': loss_data_pixel})

	loss_data_ssim = Loss_ssim1
	loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "Final_loss_ssim_epoch_" + str(
		args.epochs) + "_" + str(
		time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".mat"
	scio.savemat(loss_filename_path, {'final_loss_ssim': loss_data_ssim})

	# grad loss
	loss_data_grad = Loss_grad
	loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "Final_loss_grad_epoch_" + str(
		args.epochs) + "_" + str(
		time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".mat"
	scio.savemat(loss_filename_path, {'final_loss_grad': loss_data_grad})

	# # spd loss
	# loss_data_spd = Loss_spd
	# loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "Final_loss_spd_epoch_" + str(
	# 	args.epochs) + "_" + str(
	# 	time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".mat"
	# scio.savemat(loss_filename_path, {'final_loss_spd': loss_data_spd})

	loss_data = Loss_all

	loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "Final_loss_all_epoch_" + str(
		args.epochs) + "_" + str(
		time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".mat"
	scio.savemat(loss_filename_path, {'final_loss_all': loss_data})
	# save model
	nest_model.eval()
	nest_model.cpu()

	save_model_filename = args.ssim_path[i] + '/' "Final_epoch_" + str(args.epochs) + "_" + \
						  str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".model"
	save_model_path = os.path.join(args.save_model_dir_autoencoder, save_model_filename)
	torch.save(nest_model.state_dict(), save_model_path)

	print("\nDone, trained model saved at", save_model_path)


def check_paths(args):
	try:
		if not os.path.exists(args.vgg_model_dir):
			os.makedirs(args.vgg_model_dir)
		if not os.path.exists(args.save_model_dir):
			os.makedirs(args.save_model_dir)
	except OSError as e:
		print(e)
		sys.exit(1)


if __name__ == "__main__":
	main()
