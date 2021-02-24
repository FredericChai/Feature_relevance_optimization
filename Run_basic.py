import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

os.system('python basic_train.py --arch resnet101_pretrained --epoch 150 --gpu_id 0 -b 32  ')
# os.system('python basic_train.py --arch densenet121_pretrained --epoch 160 --gpu_id 0 -b 32  ')
# os.system('python basic_train.py --arch densenet121_pretrained --epoch 150 --gpu_id 0 -b 32  ')
# os.system('python basic_train.py --arch vgg16_pretrained --epoch 150 --gpu_id 0 -b 32  ')

# root_path = "./"
# gpu_id = 1
# epoch = 200
# arch = 'resnet34'
# ratio = 0.1 #prune 30% of the filters
# Py = 'basic_train.py'
# Spar_ep = 2
# Source = 'checkpoint/resnet34_pretrained-checkpoint-epoch250'+"/model_best.pth.tar"
# def get_cmd(root_path,gpu_id,
# 			epoch,arch,ratio,Py,
# 			Source,Spar_ep):
# 	cmd_header = "python {root}{py} ".format(root =root_path,
# 												py = Py)
# 	cmd_param_1 = "--epoch {ep} --dataset {dt} --gpu_id {gpu} --sparse_epoch {sp_ep} ".format(ep =epoch,
# 														gpu =gpu_id,
# 														sp_ep = Spar_ep,
# 														dt = root_path)
# 	cmd_param_2 = "--arch {arch} --sparse_ratio {ratio} ".format(arch =arch,
# 																ratio =ratio)
# 	cmd_param_3 = "--source_checkpoint {root}{source}".format(root =root_path,
# 																	source =Source)
# 	return cmd_header+cmd_param_1+cmd_param_2+cmd_param_3

# cmd = get_cmd(root_path,gpu_id,
# 			epoch,arch,ratio,Py,
# 			Source,Spar_ep)
# os.system(cmd)
