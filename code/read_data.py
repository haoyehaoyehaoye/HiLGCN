import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
import scipy.sparse as sp
import world
import torch as t
import torch.utils.data as data
import torch.utils.data as dataloader
import matplotlib.pyplot as plt
def calculateGraph(data):
	path_our='/home/user1809/Zhouxiang/TQL/LightGCN-PyTorch-master/data/'+data+'/train.txt'
	path_auto='/home/user1809/Zhouxiang/TQL/LightGCN-PyTorch-master/data/'+data+'/train_1.txt'
	list_d_auto=[0 for i in range(10000)]
	list_d_our=[0 for i in range(10000)]
	max_auto=0
	max_our=0
	with open(path_auto) as f:
		for l in f.readlines():
			if len(l) > 0:
				#l = l.strip('\n').split(' ')
				l = l.strip('\n').rstrip('\t').split('\t')
				items = [int(i) for i in l[1:]]
				#print(items)
				#print(len(items))
				max_auto=max(max_auto,len(items))
				uid = int(l[0])
				list_d_auto[len(items)]+=1
	with open(path_our) as f:
		for l in f.readlines():
			if len(l) > 0:
				l = l.strip('\n').split(' ')
				#l = l.strip('\n').rstrip('\t').split('\t')
				items = [int(i) for i in l[1:]]
				#print(items)
				#print(len(items))
				max_our=max(max_our,len(items))
				uid = int(l[0])
				list_d_our[len(items)]+=1
	#print(list_d_auto)	
	number_auto_u=0
	for item in list_d_auto:
		if(item !=0):
			number_auto_u+=item
	print(f"auto用户数{number_auto_u}")
	number_our_u=0
	for item in list_d_our:
		if(item !=0):
			number_our_u+=item
	print(f"our用户数{number_our_u}")
	#print(list_d_our)
	list_d_our=list_d_our[:min(max_auto,max_our,100)]
	list_d_auto=list_d_auto[:min(max_auto,max_our,100)]
	# 生成x值列表，从0开始，长度与data相同
	x = list(range(len(list_d_our)))
	# 绘制曲线图
	plt.plot(x, list_d_our, label='User-Item Interaction of dataset-yelp-1')
	# 填充曲线下方的区域
	plt.fill_between(x, list_d_our, color='skyblue', alpha=0.4)
	plt.plot(x, list_d_auto, label='User-Item Interaction of dataset-yelp-2')
	# 填充曲线下方的区域
	plt.fill_between(x, list_d_auto, color='red', alpha=0.4)
	# 添加标题和标签
	plt.title('Interaction Plot of Dataset:Yelp')
	plt.xlabel('Interaction Number')
	plt.ylabel('User Number')
	# 显示图例
	plt.legend()
	# 展示图表
	plt.savefig('my_plot.png', dpi=1000)  # 指定保存路径和文件名，dpi参数可设置图像质量
	# 清理当前图形，防止后续代码中影响新图形
	plt.clf()
def buildData(data):
	if data == 'yelp2018':
		predir = '/home/user1809/Zhouxiang/TQL/LightGCN-PyTorch-master/code/AutoCF_ori/Datasets/sparse_yelp/'
	elif data == 'gowalla':
		predir = '/home/user1809/Zhouxiang/TQL/LightGCN-PyTorch-master/code/AutoCF_ori/Datasets/sparse_gowalla/'
	elif data == 'amazon':
		predir = '/home/user1809/Zhouxiang/TQL/LightGCN-PyTorch-master/code/AutoCF_ori/Datasets/sparse_amazon/'
	trnfile = predir + 'trnMat.pkl'
	tstfile = predir + 'tstMat.pkl'
	with open(trnfile, 'rb') as f:
		recommendations_trn = pickle.load(f)
	with open(tstfile, 'rb') as f:
		recommendations_tst = pickle.load(f)
	path_train='/home/user1809/Zhouxiang/TQL/LightGCN-PyTorch-master/data/'+data+'/train_1.txt'
	path_test='/home/user1809/Zhouxiang/TQL/LightGCN-PyTorch-master/data/'+data+'/test_1.txt'
	users, items = recommendations_trn.shape
	# 将数据转换为txt格式
	with open(path_train, 'w') as f:
		for user in range(users):
			# 获取用户交互的物品
			user_items = recommendations_trn.getrow(user).nonzero()[1]
			if len(user_items) == 0:
				continue  #跳过空行
			f.write(str(user) + '\t')

			# 写入用户交互的物品
			for item in user_items:
				f.write(str(item) + '\t')
			f.write('\n')

	users, items = recommendations_tst.shape
	# 将数据转换为txt格式
	with open(path_test, 'w') as f:
		for user in range(users):
			# 获取用户交互的物品
			user_items = recommendations_tst.getrow(user).nonzero()[1]
			if len(user_items) == 0:
				continue  #跳过空行
			# 写入用户ID
			f.write(str(user) + '\t')
			# 写入用户交互的物品
			for item in user_items:
				f.write(str(item) + '\t')
			f.write('\n')
def dict_to_csr(data_dict):
		rows = []
		cols = []
		data = []
		users = list(data_dict.keys())
		max_col = 0
		for i, user in enumerate(users):
			items = data_dict[user]
			for item in items:
				rows.append(i)
				cols.append(int(item))
				data.append(1)  # Assuming interaction, you can modify as per your data
				max_col = max(max_col, int(item))
		return csr_matrix((data, (rows, cols)), shape=(len(users), max_col+1))

def buildPKL(data):
	if data == 'yelp2018':
		predir='/home/user1809/Zhouxiang/TQL/LightGCN-PyTorch-master/data/yelp2018/'
		#predir = '/home/user1809/Zhouxiang/TQL/LightGCN-PyTorch-master/code/AutoCF_ori/Datasets/sparse_yelp/'
	elif data == 'gowalla':
		predir='/home/user1809/Zhouxiang/TQL/LightGCN-PyTorch-master/data/gowalla/'
		#predir = '/home/user1809/Zhouxiang/TQL/LightGCN-PyTorch-master/code/AutoCF_ori/Datasets/sparse_gowalla/'
	elif data == 'amazon':
		predir='/home/user1809/Zhouxiang/TQL/LightGCN-PyTorch-master/data/amazon-book/'
		#predir = '/home/user1809/Zhouxiang/TQL/LightGCN-PyTorch-master/code/AutoCF_ori/Datasets/sparse_amazon/'
	trnfile = predir + 'train.txt'
	tstfile = predir + 'test.txt'
	trn_data_dict = {}
	with open(trnfile, 'r') as f:
		for line in f:
			line = line.strip().split()
			user = line[0]
			items = line[1:]
			trn_data_dict[user] = items
	tst_data_dict = {}
	with open(tstfile, 'r') as f:
		for line in f:
			line = line.strip().split()
			user = line[0]
			items = line[1:]
			tst_data_dict[user] = items	
	trn_csr_data = dict_to_csr(trn_data_dict)
	tst_csr_data = dict_to_csr(tst_data_dict)
	with open(predir+'trn_o.pkl', 'wb') as f:
		pickle.dump(trn_csr_data, f)
	with open(predir+'tst_o.pkl', 'wb') as f:
		pickle.dump(tst_csr_data, f)

def getMoreInfo(data):
	#用处为得到数据集中用户数，物品数，交互总数，平均交互数
	print(f"data")
	if data == 'yelp2018':
		predir='/home/user1809/Zhouxiang/TQL/LightGCN-PyTorch-master/data/yelp2018/'
	elif data == 'gowalla':
		predir='/home/user1809/Zhouxiang/TQL/LightGCN-PyTorch-master/data/gowalla/'
	dataset_1=predir+'train_1.txt'   #acf
	dataset_2=predir+'train.txt'     #hilgcn
	item_number_1=0
	item_number_2=0
	user_number_1=0
	user_number_2=0
	iteraction_number_1=0
	max_iteraction_number_1=0
	iteraction_number_2=0
	max_iteraction_number_2=0
	with open(dataset_1) as f:
		for l in f.readlines():
			if len(l) > 0:
				#l = l.strip('\n').split(' ')
				l = l.strip('\n').rstrip('\t').split('\t')
				items = [int(i) for i in l[1:]]
				iteraction_number_1+=len(items)
				max_iteraction_number_1=max(max_iteraction_number_1,len(items))
				for item in items:
					item_number_1=max(item_number_1,item)
				user_number_1+=1
	print(f"子数据集-1：用户{user_number_1},物品{item_number_1},交互数{iteraction_number_1},最大交互数{max_iteraction_number_1},平均交互{iteraction_number_1/user_number_1}")			
	with open(dataset_2) as f:
		for l in f.readlines():
			if len(l) > 0:
				l = l.strip('\n').split(' ')
				#l = l.strip('\n').rstrip('\t').split('\t')
				items = [int(i) for i in l[1:]]
				iteraction_number_2+=len(items)
				max_iteraction_number_2=max(max_iteraction_number_2,len(items))
				for item in items:
					item_number_2=max(item_number_2,item)
				user_number_2+=1
	print(f"子数据集-2：用户{user_number_2},物品{item_number_2},交互数{iteraction_number_2},最大交互数{max_iteraction_number_2},平均交互{iteraction_number_2/user_number_2}")

def drawParametersPlot_layer():   #这个函数用来画超参数_层数的折线图
	predir='/home/user1809/Zhouxiang/TQL/LightGCN-PyTorch-master/'
	x=[1,2,4,6,8,10]
	yelp_o_recall=[0.07174,0.07280,0.07288,0.07259,0.07259,0.07261]
	yelp_o_ndcg=[0.03210,0.03258,0.03261,0.03247,0.03244,0.03245]
	yelp_a_recall=[0.07357,0.07460,0.07825,0.07897,0.07939,0.07923]
	yelp_a_ndcg=[0.00768,0.00776,0.00810,0.00812,0.00817,0.00816]
	gowalla_a_recall=[0.27160,0.27372,0.27328,0.27357,0.27343,0.27346]
	gowalla_a_ndcg=[0.05026,0.05059,0.05081,0.05040,0.05039,0.05040]
	gowalla_o_recall=[0.19469,0.19473,0.19350,0.19323,0.19324,0.19327]
	gowalla_o_ndcg=[0.06000,0.06003,0.05972,0.05955,0.05956,0.05956]

	# 绘制Yelp Recall ,两个图
	plt.plot(x, yelp_o_recall, label='Dataset_1 Recall',color=(161/255,143/255,86/255), marker='d',  linestyle='-',linewidth=4,markersize=6)
	plt.title('Yelp Recall of Dataset-1')
	plt.legend()
	plt.grid(True)
	plt.savefig(predir+'Yelp_Recall_1.png')
	plt.close()

	plt.plot(x, yelp_a_recall, label='Dataset_2 Recall' ,color=(96/255,131/255,134/255), marker='s', linestyle='-',linewidth=4,markersize=6)
	plt.title('Yelp Recall of Dataset-2')
	plt.legend()
	plt.grid(True)
	plt.savefig(predir+'Yelp_Recall_2.png')
	plt.close()

	# 绘制Yelp ndcg ,两个图
	plt.plot(x, yelp_o_ndcg, label='Dataset_1 NDCG',color=(102/255,95/255,138/255), marker='d',  linestyle='-',linewidth=4,markersize=6)
	plt.title('Yelp NDCG of Dataset-1')
	plt.legend()
	plt.grid(True)
	plt.savefig(predir+'Yelp_NDCG_1.png')
	plt.close()


	plt.plot(x, yelp_a_ndcg, label='Dataset_2 NDCG' ,color=(176/255,116/255,114/255), marker='s', linestyle='-',linewidth=4,markersize=6)
	plt.title('Yelp NDCG of Dataset-2')
	plt.legend()
	plt.grid(True)
	plt.savefig(predir+'Yelp_NDCG_2.png')
	plt.close()

	# 绘制Gowalla Recall ,两个图
	plt.plot(x, gowalla_o_recall, label='Dataset_1 Recall',color=(161/255,143/255,86/255), marker='d',  linestyle='-',linewidth=4,markersize=6)
	plt.title('Gowalla Recall of Dataset-1')
	plt.legend()
	plt.grid(True)
	plt.savefig(predir+'Gowalla_Recall_1.png')
	plt.close()

	plt.plot(x, gowalla_a_recall, label='Dataset_2 Recall' ,color=(96/255,131/255,134/255), marker='s', linestyle='-',linewidth=4,markersize=6)
	plt.title('Gowalla Recall of Dataset-2')
	plt.legend()
	plt.grid(True)
	plt.savefig(predir+'Gowalla_Recall_2.png')
	plt.close()

	# 绘制Gowalla ndcg ,两个图
	plt.plot(x, gowalla_o_ndcg, label='Dataset_1 NDCG',color=(102/255,95/255,138/255), marker='d',  linestyle='-',linewidth=4,markersize=6)
	plt.title('Gowalla NDCG of Dataset-1')
	plt.legend()
	plt.grid(True)
	plt.savefig(predir+'Gowalla_NDCG_1.png')
	plt.close()


	plt.plot(x, gowalla_a_ndcg, label='Dataset_2 NDCG' ,color=(176/255,116/255,114/255), marker='s', linestyle='-',linewidth=4,markersize=6)
	plt.title('Gowalla NDCG of Dataset-2')
	plt.legend()
	plt.grid(True)
	plt.savefig(predir+'Gowalla_NDCG_2.png')
	plt.close()

if __name__ == '__main__':
	datalist = ['gowalla', 'yelp2018']
	#datalist = ['yelp2018']
	#for data in datalist:
		#buildData(data)
		#buildPKL(data)
		#calculateGraph(data)
		#print(f"finish {data}")
	#	getMoreInfo(data)
	#buildData(datalist[1])

	drawParametersPlot_layer()   #画层数的超参数实验
	