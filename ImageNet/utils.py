import torch
import numpy as np
import torch.nn.functional as F
import torch.autograd as autograd
from args import args as parser_args


__all__ = ['sign', 'half', 'halfmask']  # all optional custom layers


# sign binarization (-1, +1)
class sign(autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        out = scores.sign()
        return out

    @staticmethod
    def backward(ctx, g):
        # straight-through estimation
        return g


# half binarization, filter-wise (-1, +1)
class half(autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        out = scores.clone()
        a = scores[0].nelement()
        j = int(0.5 * scores[0].nelement())
        # half and half
        flat_out = out.view(out.size(0), -1)  # convert to matrix with c_out * (c_in *k*k)
        sort_fea, index_fea = scores.view(scores.size(0), -1).sort()  # sort over each channel/filter
        B_creat = np.concatenate((-torch.ones([scores.size(0), j]),
                                  torch.ones([scores.size(0), a - j])), 1)
        B_creat = torch.FloatTensor(B_creat)
        allones = torch.zeros(flat_out.size())
        if parser_args.cuda:
            B_creat = B_creat.cuda()
            allones = allones.cuda()
        out = allones.scatter_(1, index_fea, B_creat)
        out = out.view(scores.size())

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g


# half + mask/pruning: (-1, 0, +1)
class halfmask(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        out = scores.clone()
        _, idx = scores.view(scores.size(0), -1).sort()
        a = scores[0].nelement()

        j = int((1 + k)/2. * a)
        i = int((1 - k)/2. * a)

        flat_out = out.view(out.size(0), -1)
        sort_fea, index_fea = scores.view(scores.size(0), -1).sort()
        B_creat = np.concatenate((-torch.ones([scores.size(0), i]),
                                  torch.zeros([scores.size(0), j - i]),
                                  torch.ones([scores.size(0), a-j])), 1)
        B_creat = torch.FloatTensor(B_creat)
        allones = torch.zeros(flat_out.size())
        if parser_args.cuda:
            B_creat = B_creat.cuda()
            allones = allones.cuda()
        out = allones.scatter_(1, index_fea, B_creat)
        out = out.view(scores.size())
        return out

    @staticmethod
    def backward(ctx, g):
        return g, None


# only mask (0, 1)
class mask(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        out = scores.abs().clone()
        _, idx = out.flatten().sort()
        j = int(k * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1
        out = out * scores.sign()

        return out

    @staticmethod
    def backward(ctx, g):
        return g, None