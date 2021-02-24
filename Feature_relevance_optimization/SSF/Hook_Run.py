import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
root_path = "/mnt/HDD1/Frederic/ite_sparse_classification/CriteriaCompare/"
gpu_id = 1
epoch = 1
arch = 'resnet34'
ratio = 0.2
Py = 'ReleMain.py'
batch = 8
Source = "resnet34_pretrained-checkpoint-epoch200"+"/model_best.pth.tar"
def get_cmd(root_path,gpu_id,
			epoch,arch,ratio,Py,
			Source,batch):
	cmd_header = "python {root}{py} ".format(root =root_path,
												py = Py)
	cmd_param_1 = "--epoch {ep} --gpu_id {gpu} ".format(ep =epoch,
														gpu =gpu_id)
	cmd_param_2 = "--arch {arch} --sparse_ratio {ratio} -b {batch} ".format(arch =arch,
																ratio =ratio,
																batch = batch)
	cmd_param_3 = "--source_checkpoint {root}checkpoint/{source}".format(root =root_path,
																	source =Source)
	return cmd_header+cmd_param_1+cmd_param_2+cmd_param_3

cmd = get_cmd(root_path,gpu_id,
			epoch,arch,ratio,Py,
			Source,batch)
os.system(cmd)
