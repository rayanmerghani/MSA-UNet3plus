import os
import torch
import numpy as np
import glob
np.random.seed(123)

from PSCE_main import SegModel
from util import compute_performance

import torch.nn.functional as F
from skimage.segmentation import mark_boundaries
from skimage import io
from skimage.util import img_as_ubyte

torch.use_deterministic_algorithms(False)

trained_root = 'D:/segmentation_practical/Coronary_DSA/SCE/supervised-contrastive-embedding-main1/result/5AttU_Net/origin/version_4/checkpoints'
ckpt_file = glob.glob(os.path.join(trained_root, '*.ckpt'))
model_path  = ckpt_file[0]
# trained_root = 'D:/segmentation_practical/Coronary_DSA/SCE/supervised-contrastive-embedding-main/result/lightning_logs/version_0/checkpoints'
# model_path = os.path.join(trained_root, 'last.ckpt')
data_path ='D:/segmentation_practical/Coronary_DSA/SCE/supervised-contrastive-embedding-main/DSA/right/fold_5'
batch_size = 32
fig_root = 'D:/segmentation_practical/Coronary_DSA/SCE/supervised-contrastive-embedding-main/predict1'

def extract_boundary(label, is_pos=True):
    if not is_pos:
        label = 1 - label

    gt_b = F.max_pool2d(1-label, kernel_size=3, stride=1, padding=1)
    gt_b_in = 1 - gt_b
    gt_b -= 1 - label
    return gt_b, gt_b_in

def sample_balance(label, n):
    cand_pixels = torch.nonzero(label)
    batch_idx = cand_pixels[:,0]
    bs = batch_idx.max()
    sample_idx = []
    accum = 0
    for b in range(bs+1):
        
        n_features = int((batch_idx == b).sum().cpu())
        temp_idx = np.random.permutation(n_features)[:n] + accum
        sample_idx += temp_idx.tolist()
        accum += n_features

    sample_pixels = tuple(cand_pixels[sample_idx].t())

    return sample_pixels

def split_features(embedding, label, n=1):
    pos_b, pos_b_in = extract_boundary(label)
    neg_b, neg_b_in = extract_boundary(label, is_pos=False)
    try:
        pos_b_pixels = sample_balance(pos_b, n)
        pos_b_in_pixels = sample_balance(pos_b_in, n)
        neg_b_pixels = sample_balance(neg_b, n)
        neg_b_in_pixels = sample_balance(neg_b_in, n)
        pos_b_features = embedding[pos_b_pixels[0],:,pos_b_pixels[2],pos_b_pixels[3]]
        pos_b_in_features = embedding[pos_b_in_pixels[0],:,pos_b_in_pixels[2],pos_b_in_pixels[3]]
        neg_b_features = embedding[neg_b_pixels[0],:,neg_b_pixels[2],neg_b_pixels[3]]
        neg_b_in_features = embedding[neg_b_in_pixels[0],:,neg_b_in_pixels[2],neg_b_in_pixels[3]]
    except:
        return None, None, None, None
    return pos_b_features, pos_b_in_features, neg_b_features, neg_b_in_features

# def save_fig(inputs, labels, outputs, fnames, dice_values, asd_values, save_org=False):
#     inputs = (inputs.cpu().numpy()*0.5 + 0.5) * 255
#     inputs = inputs.astype(np.uint8)
#     labels = labels.cpu().numpy()*255
#     labels = labels.astype(np.uint8)
#     outputs = (outputs.cpu().detach().numpy() > 0.0) * 255
#     outputs = outputs.astype(np.uint8)
    
#     for idx, i in enumerate(outputs):
#         fname = fnames[idx]+'.png'
#         io.imsave(os.path.join(fig_root,fname), img_as_ubyte(i[0]))

def save_fig(inputs, labels, outputs, fnames, dice_values, asd_values, save_org=False):
    outputs = (outputs.cpu().detach().numpy() > 0.0) * 255
    outputs = outputs.astype(np.uint8)
    if save_org:
        for idx, i in enumerate(outputs):
            io.imsave(os.path.join(fig_root, fnames[idx]+'_seg.png'), i[0])
    else:
        for idx, i in enumerate(outputs):
            dice_str = str(int(dice_values[idx] * 1000))
            if np.isnan(asd_values[idx]):
                asd_str = 'nan'
            else:
                asd_str = str(int(asd_values[idx] * 100))
            fname = fnames[idx]+'_'+dice_str+'_'+asd_str+'.png'
            io.imsave(os.path.join(fig_root, fname), i[0])

def main():
    # model = SegModel.load_from_checkpoint(model_path,
    #                                   data_path=data_path,
    #                                   batch_size=batch_size,
    #                                   num_layers=3,
    #                                   arch = 'isunet',
    #                                   encoder = 'resnet34',
    #                                   source_data_path2 = None,
    #                                   features_start=64)
    
    model = SegModel.load_from_checkpoint(model_path,
                                      data_path=data_path,
                                      batch_size=batch_size,                                 
                                      arch = 'AttU_Net',
                                      encoder = 'resnet34',
                                      source_data_path2 = None,
                                      features_start=32,
                                      loss_weight = 1.0,
                                      loss_weight1 = 1.0,
                                      n_prototypes = 2,
                                      embedding = 512,
                                      margin = 5.5,
                                      dist = 'cosine',
                                      miner = 'True',
                                      )

    

                         
    model.cuda(device=0)
    model.eval()
    loader = model.test_dataloader()
    fnames = []
    recall_value = []
    precision_value = []
    f1_value = []
    dice_value = []
    asd_value = []
    acd_value = []
    pos_b_features = []
    pos_b_in_features = []
    neg_b_features = []
    neg_b_in_features = []
    count = 1
    for (inputs, labels, resized_labels, fname) in loader:
        bs = inputs.shape[0]
        inputs = inputs.cuda()
        labels = labels.cuda()
        resized_labels = resized_labels.cuda()
        outputs, embeddings = model.forward(inputs)
        metrics = compute_performance(outputs, labels,
                                      metric=['confusion','dice','asd'],
                                      prefix='test', reduction='none')
        print('metrics',metrics)
        recall = np.array(metrics['test_recall'].cpu())[:,0]
        precision = np.array(metrics['test_precision'].cpu())[:,0]
        f1 = np.array(metrics['test_f1'].cpu())[:,0]
        dice = np.array(metrics['test_dice'].cpu())[:,0]
        asd = np.array(metrics['test_asd'].cpu())[:,0]
        acd = np.array(metrics['test_acd'].cpu())[:,0]
        recall_value.extend(recall)
        precision_value.extend(precision)
        f1_value.extend(f1)
        dice_value.extend(dice)
        asd_value.extend(asd)
        acd_value.extend(acd)
        fnames.extend(fname)
        save_fig(inputs, labels, outputs, fname, dice, asd, save_org=False)
        if count % 10 == 0:
            print(f'[{count}/{len(loader)}] done')
        count += 1
    recall = np.mean(recall_value)
    precision = np.mean(precision_value)
    f1 = np.mean(f1_value)
    dice = np.mean(dice_value)
    asd = np.mean([i for i in asd_value if ~np.isnan(i)])
    acd = np.mean([i for i in acd_value if ~np.isnan(i)])
    print(f'Recall: {recall}, Precision: {precision}, F1: {f1}')
    print(f'Dice: {dice}, ASD: {asd}, ACD: {acd}')
    print(recall,precision,f1,dice,asd,acd)

if __name__ == '__main__':
    main()
