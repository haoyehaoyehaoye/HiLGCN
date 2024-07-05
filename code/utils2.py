import torch
import torch.nn.functional as F
import torch.nn as nn

def cust_mul(s, d, dim):
    i = s._indices()
    v = s._values()
    dv = d[i[dim,:]]
    return torch.sparse.FloatTensor(i, v * dv, s.size())

def InfoNCE(view1, view2, temperature, b_cos = True):
    if b_cos:
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    cl_loss = -torch.log(pos_score / ttl_score+10e-6)
    return torch.mean(cl_loss)