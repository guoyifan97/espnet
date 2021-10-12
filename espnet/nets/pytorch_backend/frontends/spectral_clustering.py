import torch
from torch.autograd import Variable


import numpy as np


def distance_matrix(mat):
    d= ((mat.unsqueeze (0)-mat.unsqueeze (1))**2).sum (2)**0.5
    return d
 

# Generate Clusters
mat = torch.cat([torch.randn(500,2)+torch.Tensor([-2,-3]),   torch.randn(500,2)+torch.Tensor([2,1])]) 

##-------------------------------------------
#         Compute distance matrix and then the Laplacian
##-------------------------------------------
d= distance_matrix(mat);
da=d<2;

D= ((da.float()).sum(1)).diag()
L = D -da.float()


Lsym=torch.mm(torch.mm(torch.diag(torch.pow(torch.diag(D),-0.5)),L),torch.diag(torch.pow(torch.diag(D),-0.5)));

[u,s,v]=torch.svd(Lsym)

