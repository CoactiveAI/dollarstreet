import torch
import torch.backends.cudnn as cudnn

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    cudnn.benchmark = True
else:
    device = torch.device('cpu')
