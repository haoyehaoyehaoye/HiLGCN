import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
import re
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

#weight_file = utils.getFileName()
weight_file = '/home/user1809/Zhouxiang/TQL/LightGCN-PyTorch-master/model/contrast_ex_layer=10_group=1_epoch=20_yelp_a.pth'
print(f"load and save to {weight_file}")

load_flag=False
if load_flag:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cuda')))
        world.cprint(f"loaded model weights from {weight_file}")
        print('load success')
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment))
else:
    w = None
    world.cprint("not enable tensorflowboard")
def trainModel():
    flag_pre_loss=100
    flag_break=0
    try:
        #for epoch in range(world.TRAIN_epochs):
        value_recall=0
        value_precision=0
        value_ndcg=0
        for epoch in range(1000):
            start = time.time()
            #if epoch %20 == 0:
            #    cprint(f"[TEST],epoch{epoch}")
            #    Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            #    torch.save(Recmodel.state_dict(), weight_file)
            if epoch >= 1:
                #cprint(f"[TEST],epoch{epoch}")
                result = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
                value_precision=max(value_precision,result['precision'][0])
                value_recall=max(value_recall,result['recall'][0])
                value_ndcg=max(value_ndcg,result['ndcg'][0])
                torch.save(Recmodel.state_dict(), weight_file)
            if(epoch%5==0):
                cprint(f"[TEST],epoch{epoch}")
                print(f"precision:{value_precision},recall:{value_recall},ndcg:{value_ndcg}")
                value_recall=0
                value_precision=0
                value_ndcg=0
            output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
            output_information = re.search(r'\d+\.\d+', output_information).group()
            loss=float(output_information)
            if(flag_pre_loss-loss<0.0001):
                flag_break+=1
            #if(flag_break>=3):
            #    print('finished')
            #    break
            flag_pre_loss=loss
            #print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
            torch.save(Recmodel.state_dict(), weight_file)
    finally:
        if world.tensorboard:
            w.close()
def testModel():
    Procedure.Test(dataset, Recmodel, 1, w, world.config['multicore'])            
if __name__ == "__main__":
    #testModel()
    trainModel()