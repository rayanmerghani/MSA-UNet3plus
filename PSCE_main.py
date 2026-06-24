import os
import torch
import numpy as np
import random

# Set all random seeds and enforce determinism at the very beginning
os.environ["PL_FAULT_TOLERANT_TRAINING"] = "0"  # Disable PL 2.0+ features
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # For synchronous CUDA ops
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Set random seeds
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Configure PyTorch for deterministic operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True, warn_only=True)

import comet_ml
from argparse import ArgumentParser

import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data import distributed
import time
#from torchsummary import summary

import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import OrganData
from PSCE_loss import DiceLoss, PixelwiseContrastiveLoss,PixelwiseContrastiveLoss1
from util import compute_performance



from U_Net import U_Net
from m_myunet3plus35 import m_myunet3plus35


class SegModel(pl.LightningModule):

    def __init__(self,
                 data_path: str,
                 arch: str,
                 encoder: str,
                 batch_size: int,
                 lr: float = 0.01,
                 optim: str = 'sgd',
                 loss_weight: float = 0.1,
                 loss_weight1: float = 0.1,
                 n_max_pos: int = 128,
                 boundary_aware: bool = False,
                 boundary_loc: str = 'both',
                 sampling_type: str = 'full',
                 neg_multiplier: int = 1,
                 n_prototypes: int = 1,
                 miner: bool = True,
                 margin: int = 500,
                 embedding: int = 512,
                 dist: str = 'euclidean',
                 num_layers: int = 4,
                 features_start: int = 32,
                 use_ddp: bool = False,
                 **kwargs):
        super().__init__()
        self.data_path = data_path
        self.arch = arch
        self.encoder = encoder
        self.batch_size = batch_size
        self.lr = lr
        self.optim = optim
        self.loss_weight = loss_weight
        self.loss_weight1 = loss_weight1
        self.n_max_pos = n_max_pos
        self.boundary_aware = boundary_aware
        self.boundary_loc = boundary_loc
        self.sampling_type = sampling_type
        self.neg_multiplier = neg_multiplier
        self.n_prototypes = n_prototypes
        self.miner = miner
        self.margin = margin
        self.embedding = embedding
        self.dist = dist
        self.num_layers = num_layers
        self.features_start = features_start
        self.use_ddp = use_ddp
        if 'max_epochs' in kwargs.keys():
            self.max_epochs = kwargs['max_epochs']

        if 'target_data_path' in kwargs.keys():
            self.target_data_path = kwargs["target_data_path"]


        
        if self.arch == 'U_Net':
            self.net = U_Net()
    
        elif self.arch == 'm_myunet3plus35':
            self.net = m_myunet3plus35()
        
        

        self.train_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.4, contrast=0.4),
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5])])

        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5])])

        self.trainset = OrganData(data_path=os.path.join(self.data_path,'train'),
                                    transform=self.train_transform)
        self.testset = OrganData(data_path=os.path.join(self.data_path,'test'),
                                    transform=self.val_transform,mode='test')
        print(os.path.join(self.data_path,'train'))

        if kwargs['source_data_path2'] is not None:
            train_set_list = [self.trainset]
            test_set_list = [self.testset]
            
            self.data_path2 = kwargs['source_data_path2']
            train_set_list.append(OrganData(data_path=os.path.join(self.data_path2, 'train'),
                                            transform=self.train_transform))
            test_set_list.append(OrganData(data_path=os.path.join(self.data_path2, 'test'),
                                            transform=self.val_transform,mode='test'))
            
            if kwargs['source_data_path3'] is not None:
                self.data_path3 = kwargs['source_data_path3']
                train_set_list.append(OrganData(data_path=os.path.join(self.data_path3, 'train'),
                                            transform=self.train_transform))
                test_set_list.append(OrganData(data_path=os.path.join(self.data_path3, 'test'),
                                            transform=self.val_transform,mode='test'))
            
            self.trainset = ConcatDataset(train_set_list)
            self.testset = ConcatDataset(test_set_list)

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.cont_loss1 = PixelwiseContrastiveLoss1(n_prototypes=self.n_prototypes, 
                                                  embedding = self.embedding,
                                                  margin = self.margin,
                                                  miner = self.miner,
                                                  dist = self.dist)
        
        self.cont_loss = PixelwiseContrastiveLoss(neg_multiplier=self.neg_multiplier,
                                                  n_max_pos=self.n_max_pos,
                                                  boundary_aware=self.boundary_aware,
                                                  boundary_loc=self.boundary_loc,
                                                  sampling_type=self.sampling_type)

    def forward(self, x):
        # print('*******************************************')
        # print('image bEFORE model',x.shape)
        return self.net(x)

    def training_step(self, batch, batch_idx):
        img, label, resized_label = batch
        # print('*******************************************')
        # print('image before model',img.shape)
        # print('label before model',label.shape)
        # print('resized_label before model',resized_label.shape)
        
        img = img.float()

        output, embeddings = self(img)
        # print('*******************************************')
        # print('predicted output from model',output.shape)
        # print('embedding from model',embedding.shape)

        cont_loss = self.cont_loss(embeddings, resized_label,
                                   split_param=(self.current_epoch, self.max_epochs))

        cont_loss1 = self.cont_loss1(embeddings, resized_label)
        
        if self.loss_weight == 0.0 and self.loss_weight1 == 0.0:
            loss = 1*self.bce_loss(output, label) + \
                1*self.dice_loss(output, label) 
               
        else:
            loss = 1*self.bce_loss(output, label) + \
                1*self.dice_loss(output, label) + \
                self.loss_weight * cont_loss +\
                self.loss_weight1 * cont_loss1
                
                                  
        self.log('train_cont_loss', cont_loss,
                 prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_loss', loss,
                 prog_bar=True, on_step=False, on_epoch=True)
        # metrics
        metric_log = compute_performance(output, label,
                                         metric=['dice'],
                                         prefix='train')
        # self.log_dict(metric_log,
        #               on_step=True,
        #               on_epoch=True,
        #               prog_bar=False,
        #               sync_dist=False)
        
        scalar_metrics = {}
        for key, value in metric_log.items():
            if isinstance(value, torch.Tensor):
                scalar_metrics[key] = value.mean().item()  # or value.sum().item() if you want to log the sum
            else:
                scalar_metrics[key] = value
                
        self.log_dict(scalar_metrics,
                      on_step=False,
                      on_epoch=True,
                      prog_bar=True,
                      sync_dist=False)
        
        # import pandas as pd
        # import os
        # filename = "train_metrics_log.csv"
        # df = pd.DataFrame([scalar_metrics])
        # if os.path.exists(filename):
        #     df.to_csv(filename, mode='a', header=False, index=False)
        # else:
        #     df.to_csv(filename, mode='w', header=True, index=False)
        
        
        return loss

    def validation_step(self, batch, batch_idx):
        img, label, _, _ = batch
        img = img.float()

        output, _ = self(img)
        # output, embedding = self(img)  
        loss = self.bce_loss(output, label)

        self.log('val_loss', loss,
                 prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        # metrics
        metric_log = compute_performance(output, label,
                                         metric=['confusion','dice','asd','acd'],
                                         prefix='val')
        # self.log_dict(metric_log,
        #               on_step=False,
        #               on_epoch=True,
        #               prog_bar=False,
        #               sync_dist=True)
        
        scalar_metrics = {}
        for key, value in metric_log.items():
            if isinstance(value, torch.Tensor):
                scalar_metrics[key] = value.mean().item()  # or value.sum().item() if you want to log the sum
            else:
                scalar_metrics[key] = value
                
        self.log_dict(scalar_metrics,
                      on_step=False,
                      on_epoch=True,
                      prog_bar=True,
                      sync_dist=False)
        
        # import pandas as pd
        # import os
        # filename = "val_metrics_log.csv"
        # df = pd.DataFrame([scalar_metrics])
        # if os.path.exists(filename):
        #     df.to_csv(filename, mode='a', header=False, index=False)
        # else:
        #     df.to_csv(filename, mode='w', header=True, index=False)
        
        # # if batch_idx == 0:  # Only calculate t-SNE for the first batch of each validation
        # #     self.extract_and_visualize_embeddings(embedding, label)
            
           
        return loss

    # def extract_and_visualize_embeddings(self, embeddings, labels):
    #     # Flatten embeddings to 2D for t-SNE
    #     embeddings = embeddings.view(embeddings.size(0), -1)
    #     print('embeddings',embeddings.shape)

    #     # Run t-SNE
    #     tsne = TSNE(n_components=2,  random_state=0)
    #     transformed_embeddings = tsne.fit_transform(embeddings.detach().cpu().numpy())

    #     # Plot the t-SNE result
    #     plt.figure(figsize=(10, 10))
    #     scatter = plt.scatter(transformed_embeddings[:, 0], transformed_embeddings[:, 1], c=labels.cpu().numpy(), cmap='viridis')
    #     plt.title('t-SNE visualization of embeddings')
    #     plt.xlabel('t-SNE feature 1')
    #     plt.ylabel('t-SNE feature 2')
    #     plt.colorbar(scatter)
    #     plt.savefig('t_sne_visualization.png')  # Save the plot to a file
    #     plt.close()
        
        

    def configure_optimizers(self):
        if self.optim == 'sgd':
            opt = torch.optim.SGD(self.net.parameters(),
                                lr=self.lr,
                                momentum=0.9,
                                weight_decay=0.0001)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(opt,
                            milestones=[int(6/10*self.max_epochs),
                                        int(8/10*self.max_epochs)],
                            gamma=0.1)
            return [opt], [scheduler]
        if self.optim == 'adam':
            opt = torch.optim.Adam(self.net.parameters(),
                                    lr=self.lr,
                                    eps=1e-4,
                                    weight_decay=0.0005)
            #opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(opt,
                            milestones=[int(5/8*self.max_epochs),
                                        int(7/8*self.max_epochs)],
                            gamma=0.2)
            return [opt], [scheduler]
    
    
    # def configure_optimizers(self):
    #     if self.optim == 'sgd':
    #         opt = torch.optim.SGD(
    #             self.net.parameters(),
    #             lr=self.lr,
    #             momentum=0.9,
    #             weight_decay=0.0001
    #         )
    #         scheduler = {
    #             'scheduler': torch.optim.lr_scheduler.MultiStepLR(
    #                 opt,
    #                 milestones=[int(6/10 * self.max_epochs), int(8/10 * self.max_epochs)],
    #                 gamma=0.1
    #             ),
    #             'interval': 'epoch',  # Update at the end of each epoch
    #             'name': 'lr_scheduler'  # Optional but useful for logging
    #         }
    #         return [opt], [scheduler]  # Return optimizers and schedulers as lists
    
    #     elif self.optim == 'adam':
    #         opt = torch.optim.Adam(
    #             self.net.parameters(),
    #             lr=self.lr,
    #             eps=1e-4,
    #             weight_decay=0.0005
    #         )
    #         scheduler = {
    #             'scheduler': torch.optim.lr_scheduler.MultiStepLR(
    #                 opt,
    #                 milestones=[int(5/8 * self.max_epochs), int(7/8 * self.max_epochs)],
    #                 gamma=0.2
    #             ),
    #             'interval': 'epoch',
    #             'name': 'lr_scheduler'
    #         }
    #         return [opt], [scheduler]
    



    # def train_dataloader(self):
    #     if self.use_ddp:
    #         train_sampler = distributed.DistributedSampler(self.trainset)
    #     else:
    #         train_sampler = None
    #     return DataLoader(self.trainset, batch_size=self.batch_size,
    #                       shuffle=(train_sampler is None),
    #                       num_workers=0,
    #                       sampler=train_sampler,
    #                       pin_memory=True)
    
    def train_dataloader(self):
        if self.use_ddp:
            train_sampler = distributed.DistributedSampler(
                self.trainset, 
                shuffle=False,  # Disable shuffling for reproducibility
                seed=seed       # Fixed seed for sampler
            )
        else:
            train_sampler = None
            
        return DataLoader(
            self.trainset, 
            batch_size=self.batch_size,
            shuffle=(train_sampler is None),  # Only shuffle if no sampler
            num_workers=0,  # Disable multi-processing for reproducibility
            sampler=train_sampler,
            pin_memory=True
        )

    # def val_dataloader(self):
    #     if self.use_ddp:
    #         val_sampler = distributed.DistributedSampler(self.testset)
    #     else:
    #         val_sampler = None
    #     return DataLoader(self.testset, batch_size=self.batch_size,
    #                       shuffle=False,
    #                       num_workers=0,
    #                       sampler=val_sampler,
    #                       pin_memory=True)
    
    def val_dataloader(self):
        if self.use_ddp:
            val_sampler = distributed.DistributedSampler(
                self.testset,
                shuffle=False,
                seed=seed
            )
        else:
            val_sampler = None
            
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # Disable multi-processing
            sampler=val_sampler,
            pin_memory=True
        )

    def test_dataloader(self):
        if self.use_ddp:
            test_sampler = distributed.DistributedSampler(self.testset)
        else:
            test_sampler = None
        return DataLoader(self.testset, batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=0,
                          sampler=test_sampler,
                          pin_memory=True)


    def target_test_dataloader(self):
        self.target_testset = OrganData(data_path=self.target_data_path,
                                        transform=self.val_transform, mode='test')

        if self.use_ddp:
            test_sampler = distributed.DistributedSampler(self.target_testset)
        else:
            test_sampler = None
        return DataLoader(self.target_testset, batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=0,
                          sampler=test_sampler,
                          pin_memory=True)


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument("--data-path", type=str, default='/daintlab/data/lung_segmentation_dataset/JSRT_dataset')
        parser.add_argument("--source-data-path2", type=str, default=None)
        parser.add_argument("--source-data-path3", type=str, default=None)
        parser.add_argument("--target-data-path", type=str, default=None)
        parser.add_argument("--arch", type=str, choices=['unet','unetpp','dlabv3','dlabv3p','manet','pspnet','fpn', 'linknet', 'pan', 'vnet','isunetv1','swinunet'])
        parser.add_argument("--encoder", type=str, choices=['resnet34','resnet50'])
        parser.add_argument("--batch-size", type=int, default=32)
        parser.add_argument("--lr", type=float, default=0.01)
        parser.add_argument("--optim", type=str, default='sgd')
        parser.add_argument("--loss-weight", type=float, default=1.0)
        parser.add_argument("--loss-weight1", type=float, default=1.0)
        parser.add_argument("--boundary-aware", action='store_true',default='False')
        parser.add_argument("--boundary-loc", type=str, default='both')
        parser.add_argument("--sampling-type", type=str, default='full')
        parser.add_argument("--n-max-pos", type=int, default=64)
        parser.add_argument("--neg-multiplier", type=int, default=2)
        parser.add_argument("--miner", action='store_true',default='True')
        parser.add_argument("--n-prototypes", type=int, default=1)
        parser.add_argument("--embedding", type=int, default=512)
        parser.add_argument("--margin", type=int, default=300)
        parser.add_argument("--dist", type=str, default='euclidean')
        parser.add_argument("--num-layers", type=int, default=5)
        parser.add_argument("--features-start", type=int, default=64)
        parser.add_argument("--max-epochs", type=int, default=120)
        parser.add_argument("--deterministic", type=bool, default=True)

        return parser


def main(hparams):
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
   
    project_name = os.environ.get("COMET_PROJECT_NAME")

    if hparams.logging:
        logger = CometLogger(api_key=os.environ.get('COMET_API_KEY'),
                            workspace='swlee',
                            project_name=project_name,
                            experiment_name=hparams.exp, online =False)
        hparams.logger = logger
        hparams.logger.log_hyperparams(hparams)

        

    # ckpt_callback = ModelCheckpoint(save_last=True)
    ckpt_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1,
                                    mode='min', save_last=None)
    hparams.callbacks = [ckpt_callback]

    hparams.sync_batchnorm = True
    hparams.precision = 16

    model = SegModel(**vars(hparams))
    
    total_params = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total_params/1e6))

    csv_logger = CSVLogger(save_dir=hparams.default_root_dir, name="")
    hparams.loggers = [csv_logger]  # Add to existing loggers if needed
   
    # trainer = pl.Trainer(
    # max_epochs=hparams.max_epochs,
    # accelerator="gpu",
    # logger=hparams.loggers,
    # callbacks=[ckpt_callback],  # <-- Pass callbacks here
    # enable_progress_bar=True,
    # enable_model_summary=True,
    # log_every_n_steps=21,
    # benchmark=False)
    
    trainer = pl.Trainer(
       max_epochs=hparams.max_epochs,
       accelerator="gpu",
       logger=hparams.loggers,
       callbacks=[ckpt_callback],
       log_every_n_steps=21,
       # deterministic=True,  # Critical for reproducibility
       benchmark=False,    # Disable cuDNN benchmarking
       enable_progress_bar=True,
       enable_model_summary=True,
   )
    
    # trainer = pl.Trainer(max_epochs=hparams.max_epochs,devices=hparams.gpus,
    #     accelerator="gpu",log_every_n_steps=21,default_root_dir=hparams.default_root_dir,
    #     logger=hparams.loggers)
    start_time = time.time()
    
    trainer.fit(model)
    end_time = time.time()
    total_training_time = end_time - start_time
    print("Total training time: {:.2f} seconds".format(total_training_time))
    # return dice

if __name__ == '__main__':

    #seed = int(os.environ.get("PL_GLOBAL_SEED"))
    #pl.utilities.seed.seed_everything(seed=seed)
    
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    parser = ArgumentParser(add_help=False)
    parser.add_argument("--default-root-dir", type=str, default='./logs')
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--replace-sampler-ddp", type=bool, default=False)
    parser.add_argument("--use-ddp", type=bool, default=True)
    parser.add_argument("--accelerator", type=str, default='ddp')
    parser.add_argument("--exp", type=str, default='test')
    parser.add_argument("--logging", action='store_true')
    parser = SegModel.add_model_specific_args(parser)
    hparams = parser.parse_args()
    
    hparams.deterministic = True
    hparams.benchmark = False
    
    hparams.default_root_dir = "D:/segmentation_practical/Coronary_DSA/SCE/supervised-contrastive-embedding-main/result2"
    #hparams.exp = "manet-cont-only-np256-nm2"  
    hparams.exp = "unet-bd-random-np256-nm2"  
    # hparams.data_path = "D:/segmentation_practical/Coronary_DSA/SCE/supervised-contrastive-embedding-main/DSA/right/fold_1"
    # hparams.data_path = "D:/segmentation_practical/Coronary_DSA/SCE/supervised-contrastive-embedding-main/DSA/right/fold_1"
    hparams.arch = 'm_myunet3plus35'
    hparams.encoder = 'resnet34'
    hparams.gpus = 1
    hparams.use_ddp = False
    hparams.max_epochs = 100
    hparams.batch_size = 5
    hparams.optim = 'adam' 
    hparams.lr = 0.001
    hparams.loss_weight = 1.0
    hparams.loss_weight1 = 1.0
    hparams.accelerator = 'cuda'
    hparams.neg_multiplier = 2
    hparams.n_max_pos = 64
    # hparams.boundary_loc = 'both'
    hparams.boundary_aware = 'False'
    # hparams.sampling_type = 'fixed'
    hparams.n_prototypes = 2
    hparams.embedding = 128 
    hparams.margin = 6.3
    hparams.dist = 'cosine'
    hparams.miner = 'True'
    
    for i in range(1,6):
        
        # Update data path for each fold
        hparams.data_path = f"D:/segmentation_practical/Coronary_DSA/SCE/supervised-contrastive-embedding-main/DSA/right/fold_{i}"
        print(f"Running for fold_{i} with data path: {hparams.data_path}")
        main(hparams)
    # main(hparams)

# import itertools

# if __name__ == '__main__':
#     seed = 42  # You can choose any number
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
    
#     # Define the hyperparameter ranges
#     # margin_values = [2.44, 2.45, 2.46, 2.47, 2.48, 2.49, 2.51, 2.52, 2.53, 2.54, 2.55, 2.56, 2.57] 
#     # margin_values = [5, 5.05, 5.1, 5.15, 5.2, 5.25, 5.3, 5.35, 5.4, 5.45, 5.5, 5.55, 5.6] 
#     # margin_values = [2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.5, 2.55, 2.6, 2.65, 2.7, 2.75, 2.8, 2.85, 2.9, 2.95, 3] 
#     # margin_values = [2.5, 2.55, 2.6, 2.65, 2.7, 2.75, 2.8, 2.85, 2.9, 2.95, 3] 
#     # margin_values = [8.0, 8.1, 8.15, 8.2, 8.25, 8.3, 8.35, 8.4, 8.45, 8.5, 8.55, 9.0] 
#     # margin_values = [6, 6.1, 6.15, 6.2, 6.25, 6.3, 6.35, 6.4, 6.45, 6.5, 6.55]
#     # margin_values = [3.2, 3.25, 3.3, 3.35, 3.4, 3.45, 3.5, 3.55, 3.6, 3.65, 3.7, 3.75, 3.8, 3.85, 3.9]
#     # margin_values = [3.46, 3.47, 3.48, 3.49, 3.51, 3.52, 3.53, 3.54, 3.56, 3.57, 3.58, 3.59]
#     # margin_values = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1]
#     # margin_values = [3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5]
#     # margin_values = [3.97, 3.98, 3.99, 4.41, 4.42, 4.43, 4.44, 4.45]
#     # margin_values = [7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0]
#     # margin_values = [5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5]
#     # margin_values = [1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4]
#     # margin_values = [3.3, 3.35, 3.45, 3.5, 3.55, 4]
#     # margin_values = [1.15, 1.16, 1.17, 1.18, 1.19, 1.21, 1.22, 1.23, 1.24, 1.25]
#     # margin_values = [1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19]
#     # margin_values = [8.0, 8.2, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9]
#     # margin_values = [2.5, 2.51, 2.52, 2.53, 2.54, 2.55, 2.56, 2.57, 2.58, 2.59, 2.6, 2.61, 2.62, 2.63, 2.64, 2.65, 2.66, 2.67, 2.68, 2.69, 2.7]
#     # margin_values = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5]
#     # margin_values = [9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0]
#     # margin_values = [8, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0] 
#     # margin_values = [6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0] 
#     # margin_values = [2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8] 
#     # margin_values = [2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1] 
#     # margin_values = [6.3, 6.6, 6.7, 6.8, 6.9, 7.0] 
#     margin_values = [8.2, 8.3, 8.4, 8.5, 8.6, 8.7] 
#     n_prototypes_values = [1, 2, 3]  
#     # n_prototypes_values = [3]  
#     # n_prototypes_values = [5]

#     # Create a grid of hyperparameter combinations
#     hyperparameter_combinations = itertools.product(margin_values, n_prototypes_values)

#     for margin, n_prototypes in hyperparameter_combinations:
#         parser = ArgumentParser(add_help=False)
#         parser.add_argument("--default-root-dir", type=str, default='./logs')
#         parser.add_argument("--gpus", type=int, default=8)
#         parser.add_argument("--replace-sampler-ddp", type=bool, default=False)
#         parser.add_argument("--use-ddp", type=bool, default=True)
#         parser.add_argument("--accelerator", type=str, default='ddp')
#         parser.add_argument("--exp", type=str, default='test')
#         parser.add_argument("--logging", action='store_true')
#         parser = SegModel.add_model_specific_args(parser)
#         hparams = parser.parse_args()
        
#         hparams.default_root_dir = "D:/segmentation_practical/Coronary_DSA/SCE/supervised-contrastive-embedding-main/result"
#         hparams.exp = "unet-bd-random-np256-nm2"  
#         hparams.data_path = "D:/segmentation_practical/Coronary_DSA/SCE/supervised-contrastive-embedding-main/DSA/right/fold_1"
#         hparams.arch = 'DSASeg10'
#         hparams.encoder = 'resnet34'
#         hparams.gpus = 1
#         hparams.use_ddp = False
#         hparams.max_epochs = 100
#         hparams.batch_size = 5
#         hparams.optim = 'adam' 
#         hparams.lr = 0.001
#         hparams.loss_weight = 1
#         hparams.loss_weight1 = 1
#         hparams.accelerator = 'cuda'
#         hparams.neg_multiplier = 2
#         hparams.n_max_pos = 64
#         hparams.n_prototypes = n_prototypes  
#         hparams.embedding = 48
#         hparams.margin = margin  
#         hparams.dist = 'cosine'
#         hparams.miner = 'True'
        
#         main(hparams)













