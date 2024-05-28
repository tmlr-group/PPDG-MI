import torch
import torch.nn.functional as F


def pairwise_cosine_dist(x, y):
    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)
    return 1 - torch.matmul(x, y.T)


def ct(source, target):
    sim_mat = torch.matmul(target, source.T)
    s_dist = F.softmax(sim_mat, dim=0)
    t_dist = F.softmax(sim_mat, dim=1)

    s_par = 0.5
    cost_mat = pairwise_cosine_dist(target, source)
    source_loss = (s_par * cost_mat * s_dist).sum(0).mean()
    target_loss = ((1 - s_par) * cost_mat * t_dist).sum(1).mean()
    loss = source_loss + target_loss

    return loss
