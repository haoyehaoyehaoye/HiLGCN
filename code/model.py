"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world
import torch
from dataloader_mine import BasicDataset
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils2 import cust_mul


class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)

class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict,
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader_mine.Loader = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']    #False
        self.groups = self.config['groups'] ### 子图加加加
        self.cl_rate = self.config['lambda']  # 对应字母λ
        self.eps = self.config['eps']  # 对应字母ε
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        ### 子图加加加
        self.fc = torch.nn.Linear(self.latent_dim, self.latent_dim)
        self.leaky = torch.nn.LeakyReLU()
        self.fc_g = torch.nn.Linear(self.latent_dim, self.groups)
        self.f = nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=0.4)
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = world.device
        self.single = False


        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self,perturbed=False):
        """
        propagate methods for lightGCN
        """       
        # 获取用户和物品的初始嵌入向量
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        # 将用户和物品的嵌入向量拼接成一个大的嵌入矩阵
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        # embs = [all_emb] ### 改改改

        # 根据是否是训练阶段和是否使用dropout来决定是否对图进行dropout操作
        if self.config['dropout']:
            if self.training:
                #print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            #print(f'实际上没有drop')
            g_droped = self.Graph

        #### 改改改改
        # Compute ego + side embeddings
        # 计算自我嵌入（ego embedding）和旁观嵌入（side embedding）
        ego_embed = all_emb
        #print(f'self.Graph维度为{self.Graph.shape}')
        #print(f'drop后g_droped维度{g_droped.shape}')
        #print(f'drop后all_emb维度{all_emb.shape}')
        side_embed = torch.sparse.mm(g_droped, all_emb)
        # 将自我嵌入和旁观嵌入相加后通过一个全连接层，然后应用LeakyReLU激活函数和dropout
        temp = self.dropout(self.leaky(self.fc(ego_embed + side_embed)))
        # 通过另一个全连接层来计算分组得分
        group_scores = self.dropout(self.fc_g(temp))
        # group_scores = self.fc_g(temp)
        # 根据分组得分计算每个节点属于哪个分组的one-hot编码
        a_top, a_top_idx = torch.topk(group_scores, k=1, sorted=False)
        one_hot_emb = torch.eq(group_scores, a_top).float()

        # 将one-hot编码分为用户和物品两部分
        u_one_hot, i_one_hot = torch.split(one_hot_emb, [self.num_users, self.num_items])
        # 将物品部分的one-hot编码设置为全1（因为物品不参与分组）
        i_one_hot = torch.ones(i_one_hot.shape).to(self.device)
        # 重新拼接用户和物品的one-hot编码
        one_hot_emb = torch.cat([u_one_hot, i_one_hot]).t()

        # Create Subgraphs
        subgraph_list = []
        # 根据one-hot编码创建子图
        for g in range(self.groups):
            temp = cust_mul(g_droped, one_hot_emb[g], 1)
            temp = cust_mul(temp, one_hot_emb[g], 0)
            subgraph_list.append(temp)
        # 初始化保存所有层所有分组的嵌入向量的列表
        all_emb_list = [[None for _ in range(self.groups)] for _ in range(self.n_layers)]
        for g in range(0, self.groups):
            # all_emb_list[0][g] = ego_embed ### 原来
            ###改改改
            if perturbed:
                random_noise = torch.rand_like(side_embed).to(self.device)
                side_embed += torch.sign(side_embed) * F.normalize(random_noise, dim=-1) * self.eps
            all_emb_list[0][g] = side_embed
        # 对每个分组在每一层进行消息传递
        for k in range(1, self.n_layers):
            for g in range(self.groups):
                if perturbed:
                    random_noise = torch.rand_like(all_emb_list[k - 1][g]).to(self.device)
                    # all_emb_list[k - 1][g] += torch.sign(all_emb_list[k - 1][g]) * F.normalize(random_noise, dim=-1) * self.eps
                    all_emb_list[k][g] = torch.sparse.mm(subgraph_list[g], all_emb_list[k - 1][g]+torch.sign(all_emb_list[k - 1][g]) * F.normalize(random_noise, dim=-1) * self.eps)
                else:
                    all_emb_list[k][g] = torch.sparse.mm(subgraph_list[g], all_emb_list[k - 1][g])
        # 合并不同层的结果
        all_emb_list = [torch.sum(torch.stack(x), 0) for x in all_emb_list]
        # 根据是否是单层模式来获取最终的嵌入向量
        if self.single:
            all_emb = all_emb_list[-1]
        else:
            weights = [0.2, 0.2, 0.2, 0.2, 0.2]
            all_emb_list = [x * w for x, w in zip(all_emb_list, weights)]
            all_emb = torch.sum(torch.stack(all_emb_list), 0)
            # all_emb = torch.mean(torch.stack(all_emb_list),0)
            # all_emb = all_emb_list[-1]
        # 将最终的嵌入向量分为用户和物品两部分
        users, items = torch.split(all_emb, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma