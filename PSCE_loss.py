import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import MSELoss, SmoothL1Loss, L1Loss
import numpy as np

import os
import pytorch_lightning as pl


class SyncFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]
        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)
        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)
        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]

class PixelwiseContrastiveLoss(torch.nn.Module):
    def __init__(self,
                 neg_multiplier,n_max_pos=128,boundary_aware=False,
                 boundary_loc='both',sampling_type='full',temperature=0.1):
        super(PixelwiseContrastiveLoss, self).__init__()
        self.cosine = nn.CosineSimilarity(dim=-1, eps=1e-8)
        self.n_max_pos = n_max_pos
        self.n_max_neg = n_max_pos * neg_multiplier
        self.boundary_aware = boundary_aware
        self.boundary_loc = boundary_loc
        self.sampling_type = sampling_type # 'full', 'random', 'linear'
        self.temperature = temperature

    def extract_boundary(self, real_label, is_pos=True):
        if not is_pos:
            real_label = 1 - real_label
        gt_b = F.max_pool2d(1 - real_label, kernel_size=5, stride=1, padding=2)
        gt_b_in = 1 - gt_b
        gt_b -= 1 - real_label
        return gt_b, gt_b_in

    def sample_pixels(self, label, n):
        cand_pixels = torch.nonzero(label)
        sample_idx = torch.randperm(cand_pixels.shape[0])[:n]
        sample_pixels = cand_pixels[sample_idx]
        return sample_pixels

    def _sample_balance(self, cand_pixels, n):
        batch_idx = cand_pixels[:,0]
        bs = batch_idx.max() + 1
        n_per_sample = n // bs
        sample_idx = []
        accum = 0
        for b in range(bs):
            n_features = int((batch_idx == b).sum().cpu())
            temp_idx = np.random.permutation(n_features)[:n_per_sample] + accum
            sample_idx += temp_idx.tolist()
            accum += n_features
        return sample_idx

    def split_n(self, n, boundary_type, limit, split_param=None):
        if n < limit:
            valid_n = n
        else:
            valid_n = limit
        if boundary_type == 'full':
            return valid_n, n-valid_n
        elif boundary_type == 'exclude':
            return 0, valid_n
        elif boundary_type == 'random':
            n_bd = int(torch.rand(1) * valid_n)
            n_not_bd = valid_n - n_bd
            return n_bd, n_not_bd
        elif boundary_type == 'linear':
            current_epoch, max_epoch = split_param
            n_bd = int(current_epoch/max_epoch * valid_n)
            n_not_bd = valid_n - n_bd
            return n_bd, n_not_bd
        elif boundary_type == 'fixed':
            n_bd = int(0.2 * valid_n)
            n_not_bd = valid_n - n_bd
            return n_bd, n_not_bd

    def forward(self, predict_seg_map, real_label, split_param=None, vector="embedding"):
        # print('predict_seg_map', predict_seg_map.shape)                          #(5,128,128,128)
        if self.boundary_aware:
            if self.boundary_loc == 'pos':
                pos_b, pos_b_in = self.extract_boundary(real_label)
                n_pos_bd, n_pos_not_bd = self.split_n(self.n_max_pos,
                                                      self.sampling_type,
                                                      limit=pos_b.sum(),
                                                      split_param=split_param)
                neg_b, neg_b_in = 1-real_label, 1-real_label
                n_neg_bd, n_neg_not_bd = 0, self.n_max_neg
            elif self.boundary_loc == 'neg':
                neg_b, neg_b_in = self.extract_boundary(real_label, is_pos=False)
                n_neg_bd, n_neg_not_bd = self.split_n(self.n_max_neg,
                                                      self.sampling_type,
                                                      limit=neg_b.sum(),
                                                      split_param=split_param)
                pos_b, pos_b_in = real_label, real_label
                n_pos_bd, n_pos_not_bd = 0, self.n_max_pos
            elif self.boundary_loc == 'both':
                pos_b, pos_b_in = self.extract_boundary(real_label)
                neg_b, neg_b_in = self.extract_boundary(real_label, is_pos=False)
                n_pos_bd, n_pos_not_bd = self.split_n(self.n_max_pos,
                                                      self.sampling_type,
                                                      limit=pos_b.sum(),
                                                      split_param=split_param)
                n_neg_bd = n_pos_bd
                n_neg_not_bd = self.n_max_neg - n_neg_bd
        else:
            pos_b, pos_b_in = real_label, real_label
            neg_b, neg_b_in = 1-real_label, 1-real_label
            n_pos_bd, n_pos_not_bd = 0, self.n_max_pos
            n_neg_bd, n_neg_not_bd = 0, self.n_max_neg

        pos_b_pixels = self.sample_pixels(pos_b, n_pos_bd)
        pos_b_in_pixels = self.sample_pixels(pos_b_in, n_pos_not_bd)
        pos_pixels = torch.cat((pos_b_pixels, pos_b_in_pixels), dim=0).detach()
        # print('pos_pixels', pos_pixels.shape)              #[64,4]
        pos_pixels1 = pos_pixels.t()
        # print('pos_pixels1', pos_pixels1.shape)              #[4,64]
        pos_pixels = tuple(pos_pixels.t())
        # print('pos_pixels[0]', pos_pixels[0].shape)               #[64]   [batch_1,batch_2, ..., batch_64]
        # print('pos_pixels[1]', pos_pixels[1].shape)               #[64]   [channel_1,channel_2, ..., channel_64]
        # print('pos_pixels[2]', pos_pixels[2].shape)               #[64]   [height_1,height_2, ..., height_64]
        # print('pos_pixels[3]', pos_pixels[3].shape)               #[64]   [width_1,width_2, ..., width_64]

        neg_b_pixels = self.sample_pixels(neg_b, n_neg_bd)
        neg_b_in_pixels = self.sample_pixels(neg_b_in, n_neg_not_bd)
        neg_pixels = torch.cat((neg_b_pixels, neg_b_in_pixels), dim=0).detach()
        neg_pixels = tuple(neg_pixels.t())

        if vector == "embedding" or vector == "first"  :
            positive_logits = predict_seg_map[pos_pixels[0], :, pos_pixels[2], pos_pixels[3]]
            negative_logits = predict_seg_map[neg_pixels[0], :, neg_pixels[2], neg_pixels[3]]
            # print('positive_logits', positive_logits.shape)           #[64, 128]
            # print('negative_logits', negative_logits.shape)           #[128, 128]
        elif vector == "second" :
            positive_logits = predict_seg_map[pos_pixels[0], :, pos_pixels[2], pos_pixels[3]]
            positive_logits = positive_logits[:int(len(positive_logits)/4)]
            negative_logits = predict_seg_map[neg_pixels[0], :, neg_pixels[2], neg_pixels[3]]
            negative_logits = negative_logits[:int(len(negative_logits)/4)]
            # print('positive_logits', positive_logits.shape)           #[16, 128]
            # print('negative_logits', negative_logits.shape)           #[32, 128]
        elif vector == "third" :
            positive_logits = predict_seg_map[pos_pixels[0], :, pos_pixels[2], pos_pixels[3]]
            positive_logits = positive_logits[:int(len(positive_logits)/16)]
            negative_logits = predict_seg_map[neg_pixels[0], :, neg_pixels[2], neg_pixels[3]]
            negative_logits = negative_logits[:int(len(negative_logits)/16)]
        elif vector == "four" :
            positive_logits = predict_seg_map[pos_pixels[0], :, pos_pixels[2], pos_pixels[3]]
            positive_logits = positive_logits[:int(len(positive_logits)/64)]
            negative_logits = predict_seg_map[neg_pixels[0], :, neg_pixels[2], neg_pixels[3]]
            negative_logits = negative_logits[:int(len(negative_logits)/64)]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            all_positive_logits = SyncFunction.apply(positive_logits)
            all_negative_logits = SyncFunction.apply(negative_logits)
        else:
            all_positive_logits = positive_logits
            all_negative_logits = negative_logits
               
        # print('positive_logits', len(positive_logits))
        pos_nll = self._compute_loss(positive_logits,
                                     all_positive_logits,
                                     all_negative_logits)
        return pos_nll

    def _compute_loss(self, pos, all_pos, all_negs):
        # print('pos', len(pos))
        # print('all_pos', len(all_pos))
        # print('all_negs', len(all_negs))
        positive_sim = self.cosine(pos.unsqueeze(1),
                                    all_pos.unsqueeze(0))
        exp_positive_sim = torch.exp(positive_sim/self.temperature)
        off_diagonal = torch.ones(exp_positive_sim.shape).type_as(exp_positive_sim)
        off_diagonal = off_diagonal.fill_diagonal_(0.0)
        exp_positive_sim = exp_positive_sim * off_diagonal
        positive_row_sum = torch.sum(exp_positive_sim, dim=1)
        negative_sim = self.cosine(pos.unsqueeze(1),
                                    all_negs.unsqueeze(0))
        exp_negative_sim = torch.exp(negative_sim/self.temperature)
        negative_row_sum = torch.sum(exp_negative_sim, dim=1)
        likelihood = positive_row_sum / (positive_row_sum + negative_row_sum)
        nll = -torch.log(likelihood).mean()
        return nll


class PixelwiseContrastiveLoss1(nn.Module):
    def __init__(self, n_prototypes =1, embedding=512, prototypes=None,
                 margin=100, miner=True, weights=[1.0, 1.0],
                 ignore_index=-1,squared=True, dist="euclidean", 
                 temperature =8, normalize=False, device="cuda"):
        super(PixelwiseContrastiveLoss1, self).__init__()
        self.n_prototypes = n_prototypes
        self.embedding = embedding
        self.prototypes = (nn.Parameter(torch.rand((n_prototypes, embedding), device=device)).requires_grad_(True)
                           if prototypes is None else nn.Parameter(prototypes).requires_grad_(False))
        self.squared = squared
        self.dist = dist
        self.normalize = normalize
        self.margin = margin
        self.miner = miner 
        self.weights = weights 
        self.ignore_index = ignore_index
                
    def forward(self, embeddings, label):     
        
        if self.normalize:
            embeddings = F.normalize(embeddings, dim=1)
        b, c, h, w = embeddings.shape
        embeddings = embeddings.view(b, c, h * w).transpose(1, 2).contiguous().view(b * h * w, c)   

        if self.dist == "cosine":
            dists = 1 - nn.CosineSimilarity(dim=-1)(embeddings[:, None, :], self.prototypes[None, :, :])
        else:
            dists = torch.norm(embeddings[:, None, :] - self.prototypes[None, :, :], dim=-1)            
        if self.squared:
            self.temperature = 2
            # dists = dists ** 2
            dists = torch.exp(dists*self.temperature)
            
        dists = dists.view(b, h * w, self.n_prototypes).transpose(1, 2).contiguous().view(b, self.n_prototypes, h, w)       
        dist= -dists        
        loss = self.contrastive_loss(-dist, label)
        return loss
    
    def contrastive_loss(self, data, labels):
        if len(data.shape) == 4:
            data = data.flatten()
            data= -data
        if len(labels.shape) == 4:
            labels = labels.flatten()
        coord = torch.where(labels != self.ignore_index)
        labels = labels[coord]
        data = data[coord] 
        # print('miner',self.miner)
        if self.miner:
            data, labels, weights = self.miner1(data, labels)
        loss_contrastive = torch.mean(weights[1] * labels * torch.pow(data, 2) +
                                      weights[0] * (1 - labels) *
                                      torch.pow(torch.clamp(self.margin - data, min=0.0), 2))
        return loss_contrastive

    def miner1(self, data, labels):
        labels = labels.long()  
        if (labels < 0).any():
            raise ValueError("Labels contain negative values, which are not allowed for torch.bincount.")
        all_pos_values = -data[labels.bool()]
        # mean_value = torch.mean(all_pos_values)
        # print('all_pos_values',mean_value)
        all_neg_values = -data[(1 - labels).bool()]
        # mean_value = torch.mean(all_neg_values)
        # print('all_neg_values',mean_value)
        neg_hard = all_neg_values[all_neg_values < self.margin]
        if neg_hard.shape[0] == 0:
            print('neg_hard.shape is zero')
            total = torch.bincount(labels)[0] + torch.bincount(labels)[1]            
            weights = torch.FloatTensor([1.0 + torch.true_divide(torch.bincount(labels)[1], total),
                                         1.0 + torch.true_divide(torch.bincount(labels)[0], total)]).cuda()
            return data, labels, weights
        neg_labels = torch.zeros(neg_hard.shape, device='cuda:0')
        pos_labels = torch.ones(all_pos_values.shape, device='cuda:0')
        total = neg_labels.shape[0] + pos_labels.shape[0]
        weights = torch.FloatTensor([1.0+torch.true_divide(pos_labels.shape[0], total),
                                     1.0+torch.true_divide(neg_labels.shape[0], total)]).cuda()
        return torch.cat([neg_hard, all_pos_values]), torch.cat([neg_labels, pos_labels]), weights


def compute_per_channel_dice(input, target, epsilon=1e-5, ignore_index=None, weight=None):
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"
    if ignore_index is not None:
        mask = target.clone().ne_(ignore_index)
        mask.requires_grad = False
        input = input * mask
        target = target * mask
    input = flatten(input)
    target = flatten(target)
    input = input.float()
    target = target.float()
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect
    denominator = (input + target).sum(-1)
    return 2. * intersect / denominator.clamp(min=epsilon)

class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5, weight=None, ignore_index=None, sigmoid_normalization=True,
                 skip_last_target=False):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index     
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)
        self.skip_last_target = skip_last_target

    def forward(self, input, target):
        input = self.normalization(input)
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
        else:
            weight = None
        if self.skip_last_target:
            target = target[:, :-1, ...]
        per_channel_dice = compute_per_channel_dice(input, target, epsilon=self.epsilon, ignore_index=self.ignore_index,
                                                    weight=weight)
        return torch.mean(1. - per_channel_dice)

def flatten(tensor):
    C = tensor.size(1)
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    transposed = tensor.permute(axis_order)
    return transposed.view(C, -1)


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.filters import frangi
import cv2

class ContourWeightedLoss(nn.Module):
    def __init__(self, epsilon=1e-5, edge_weight=1.5, 
                 frangi_sigmas=np.linspace(1,4,5),
                 frangi_threshold=0.01,
                 frangi_alpha=0.5, 
                 frangi_beta=0.5, 
                 frangi_gamma=15):
        super().__init__()
        self.epsilon = epsilon
        self.edge_weight = edge_weight
        self.frangi_params = {
            'sigmas': frangi_sigmas,
            'black_ridges': False,
            'alpha': frangi_alpha,
            'beta': frangi_beta,
            'gamma': frangi_gamma
        }
        self.frangi_threshold = frangi_threshold
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        
        # Focal Loss parameters (fixed values optimized for DSA)
        self.focal_alpha = 0.8 # Class balancing
        self.focal_gamma = 1.0  # Focusing parameter

    def get_vessel_contours(self, target):
        device = target.device
        target_np = target.squeeze(1).cpu().numpy()
        contours = np.zeros_like(target_np)
        
        for b in range(target.size(0)):
            vessel_mask = (target_np[b] > 0.5).astype(np.float32)
            
            # Enhanced Frangi filtering
            frangi_response = frangi(
                vessel_mask,
                **self.frangi_params
            )
            
            # Normalize and enhance response
            frangi_response = cv2.normalize(
                frangi_response, None, 0, 1.0, cv2.NORM_MINMAX
            )
            
            # Soft thresholding
            contours[b] = np.clip(
                (frangi_response - self.frangi_threshold) / (1 - self.frangi_threshold), 
                0, 1
            )
            
        return torch.from_numpy(contours).float().to(device)

    def focal_loss(self, input, target):
        """Calculate Focal Loss while preserving gradients"""
        bce = self.bce_loss(input, target)
        prob = torch.sigmoid(input)
        p_t = prob * target + (1 - prob) * (1 - target)
        alpha_t = self.focal_alpha * target + (1 - self.focal_alpha) * (1 - target)
        return alpha_t * (1 - p_t) ** self.focal_gamma * bce

    def forward(self, input, target):
        # Get enhanced contours
        with torch.no_grad():
            contours = self.get_vessel_contours(target)  # [B, H, W]
            prob = torch.sigmoid(input.detach())
            
        # Dynamic edge weighting
        edge_map = contours.unsqueeze(1) * (prob > 0.5).float()
        weights = 1.0 + (self.edge_weight - 1.0) * edge_map

        # Calculate both BCE and Focal losses
        bce = self.bce_loss(input, target)
        focal = self.focal_loss(input, target)
        
        # Apply contour-aware weighting
        weighted_bce = (weights * bce).mean()
        weighted_focal = (weights * focal).mean()

        # Enhanced Dice calculation
        prob_flat = prob.view(-1)
        target_flat = target.view(-1)
        contour_weights = contours.view(-1)
        
        intersection = prob_flat * target_flat
        union = prob_flat + target_flat
        
        # Contour-weighted Dice
        dice_loss = 1.0 - (2. * (intersection * contour_weights).sum() + self.epsilon) / \
                   ((union * contour_weights).sum() + self.epsilon)

        # Combined loss (BCE + Focal + Dice)
        return 0.2 * weighted_bce + 0.1 * weighted_focal + 0.7 * dice_loss
