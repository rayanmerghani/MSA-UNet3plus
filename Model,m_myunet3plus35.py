import torch
import torch.nn as nn
import torch.nn.functional as F
from MCDAU_blocks3 import *

class conv_block(nn.Module):
    def __init__(self, in_c, out_c, act=True, dropout_prob=0.0):
        super().__init__()

        layers = [nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)]

        if act == True:
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            
        # layers.append(nn.Dropout2d(dropout_prob))
        
        if dropout_prob > 0:
            layers.append(nn.Dropout(dropout_prob))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)
    
class m_myunet3plus35(nn.Module):
    def __init__(self, n_classes=1, bn=True, BatchNorm=False):
        super().__init__()

        """ Encoder """
        self.e1 = M_Encoder(1, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm, res= True)  # 512
        self.e2 = M_Encoder(32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm, res = True)  # 256
        self.e3 = M_Encoder(64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm, res= True)  # 128
        # self.e4 = nn.Conv2d(128, 512, kernel_size=1, padding=0, stride=1, bias=True)
        
        self.bn = bn
        self.BatchNorm = BatchNorm
        
        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=4, padding=0)
        self.deconv2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0)
        self.deconv3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0)
        
        self.s1 = nn.Conv2d(224, 32, kernel_size=1, stride=1, padding=0)
        self.s2 = nn.Conv2d(192, 64, kernel_size=1, stride=1, padding=0)
        
        """ Bottleneck """
        self.center1 = Bottleneck(128, 128, [1, 2, 1], kernel_size=3, bn=self.bn, BatchNorm=self.BatchNorm)
        self.center2 = Bottleneck(128, 128, [2, 4, 2], kernel_size=3, bn=self.bn, BatchNorm=self.BatchNorm)
        self.center3 = Bottleneck(128, 128, [4, 8, 4], kernel_size=3, bn=self.bn, BatchNorm=self.BatchNorm)

        self.aspp = ASPP(128, [4, 8, 4])
        
        # self.ccaf1 = CCAF1(32,32)
        self.ccaf2 = CCAF2(64,64)
        self.ccaf3 = CCAF3(128,128)
        self.ccaf4 = CCAF4(128,128)
        
        """ Decoder 4 """
        # self.e1_d4 = M_Decoder(64, 64)
        # self.e2_d4 = M_Decoder(128, 64)
        # self.e3_d4 = M_Decoder(256, 64)
        # self.e4_d4 = M_Decoder(512, 64)
        # self.e5_d4 = M_Decoder(1024, 64)

        # self.d4 = M_Decoder(64*4, 64)

        """ Decoder 3 """
        # self.e1_d3 = conv_block(32, 32)
        # self.e2_d3 = conv_block(64, 32)
        self.e3_d3 = conv_block(128, 32)
        self.e4_d3 = conv_block(128, 32)
        # self.e5_d3 = M_Decoder(1024, 64)

        self.d3 = conv_block(32*2, 32)

        """ Decoder 2 """
        # self.e1_d2 = conv_block(32, 32)
        self.e2_d2 = conv_block(64, 32)
        self.e3_d2 = conv_block(32, 32)
        self.e4_d2 = conv_block(128, 32)
        # self.e5_d2 = M_Decoder(1024, 64)

        self.d2 = conv_block(32*3, 32)

        """ Decoder 1 """
        self.e1_d1 = conv_block(32, 32)
        self.e2_d1 = conv_block(32, 32)
        self.e3_d1 = conv_block(32, 32)
        self.e4_d1 = conv_block(128, 32)
        # self.e5_d1 = M_Decoder(1024, 64)

        self.d1 = conv_block(32*4, 32)

        """ Output """
        # self.e5 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        # self.e6 = FPI()
        # self.e7 = nn.ConvTranspose2d(96, 96,kernel_size=4, stride=2,padding=1)  
        # self.e8 = DMFF()
        
        self.y1 = nn.Conv2d(32, n_classes, kernel_size=3, padding=1)
        
    def forward(self, x):
        """ Encoder """
        # print('x', x.shape)
        _, _, img_shape, _ = x.size()
        # print('img_shape', img_shape)

        conv1, out = self.e1(x) 
        # print('conv1',conv1.shape)
        # print('out', out.shape)

        conv2, out = self.e2(out)
        # print('conv2',conv2.shape)
        # print('out', out.shape)

        conv3, out = self.e3(out)
        out3 = conv3
        # out_emb = self.e4(conv3)
        # print('conv3',conv3.shape)
        # print('out',out.shape)
        
        # conv_transpose = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        # conv12 = conv_transpose(conv2)
        # # print('conv12', conv12.shape)
        # # print('conv1', conv1.shape)
        # conv12 = torch.cat((conv12, conv1), dim=1)
        # conv_layer = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        # conv12 = conv_layer(conv12)
        # # print('conv12', conv12.shape)
        # conv_transpose1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        # conv23 = conv_transpose1(conv3)
        # # print('conv23', conv23.shape)
        # # print('conv2', conv2.shape)
        # conv23 = torch.cat((conv23, conv2), dim=1)
        # conv_layer = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        # conv23 = conv_layer(conv23)
        # # print('conv23', conv23.shape)
        
        
        """ Bottleneck """
        out = self.center1(out)
        # # print('out',out.shape)
        out = self.center2(out)
        # # print('out',out.shape)
        out = self.center3(out)
        # # print('out',out.shape)
        out = self.aspp(out)
        # out1= out
        # # print('out',out.shape)
        
        # print('conv1',conv1.shape)
        # print('conv2',conv2.shape)
        # print('conv3',conv3.shape)
        # print('out1',out1.shape)
        
        s31 = self.deconv1(conv3)
        s31 = torch.cat([s31, conv1], dim=1)
        s21 = self.deconv2(conv2)
        tmp11 = torch.cat([s21, s31], dim=1)
        tmp11 = self.s1(tmp11)
        s32 = self.deconv3(conv3)
        tmp22 = torch.cat([s32, conv2], dim=1)
        tmp22 = self.s2(tmp22)
        # print('tmp11',tmp11.shape)
        # print('tmp22',tmp22.shape)
        
        
        # tmp1 = self.ccaf1(conv1)
        tmp1 = tmp11
        tmp2 = self.ccaf2(tmp22)
        tmp3 = self.ccaf3(conv3)

        out = self.ccaf4(out)
        # print('tmp1',tmp1.shape)
        # print('tmp2',tmp2.shape)
        # print('tmp3',tmp3.shape)
        # print('out',out.shape)
        
        # tmp1 = tmp11
        # tmp2 = tmp22
        # tmp3 = conv3
        # out = out

        # """ Decoder 4 """
        # e1_d4 = F.max_pool2d(tmp1, kernel_size=8, stride=8)
        # e1_d4 = self.e1_d4(e1_d4)

        # e2_d4 = F.max_pool2d(tmp2, kernel_size=4, stride=4)
        # e2_d4 = self.e2_d4(e2_d4)

        # e3_d4 = F.max_pool2d(tmp3, kernel_size=2, stride=2)
        # e3_d4 = self.e3_d4(e3_d4)

        # e4_d4 = self.e4_d4(out)

        # # e5_d4 = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=True)
        # # e5_d4 = self.e5_d4(e5_d4)

        # d4 = torch.cat([e1_d4, e2_d4, e3_d4, e4_d4], dim=1)
        # d4 = self.d4(d4)

        """ Decoder 3 """
        # e1_d3 = F.max_pool2d(tmp1, kernel_size=4, stride=4)
        # e1_d3 = self.e1_d3(e1_d3)

        # e2_d3 = F.max_pool2d(tmp2, kernel_size=2, stride=2)
        # e2_d3 = self.e2_d3(e2_d3)

        e3_d3 = self.e3_d3(tmp3 )

        e4_d3 = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=True)
        e4_d3 = self.e4_d3(e4_d3)

        # e5_d3 = F.interpolate(e5, scale_factor=4, mode="bilinear", align_corners=True)
        # e5_d3 = self.e5_d3(e5_d3)
        # print('e1_d3', e1_d3.shape)
        # print('e2_d3', e1_d3.shape)
        # print('e3_d3', e1_d3.shape)
        # print('e4_d3', e1_d3.shape)

        d3 = torch.cat([e3_d3, e4_d3], dim=1)
        d3 = self.d3(d3)

        """ Decoder 2 """
        # e1_d2 = F.max_pool2d(tmp1, kernel_size=2, stride=2)
        # e1_d2 = self.e1_d2(e1_d2)

        e2_d2 = self.e2_d2(tmp2)

        e3_d2 = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=True)
        e3_d2 = self.e3_d2(e3_d2)

        e4_d2 = F.interpolate(out, scale_factor=4, mode="bilinear", align_corners=True)
        e4_d2 = self.e4_d2(e4_d2)

        # e5_d2 = F.interpolate(e5, scale_factor=8, mode="bilinear", align_corners=True)
        # e5_d2 = self.e5_d2(e5_d2)
        # print('e4_d2', e1_d2.shape)
        # print('e4_d2', e1_d2.shape)
        # print('e4_d2', e1_d2.shape)
        # print('e4_d2', e1_d2.shape)

        d2 = torch.cat([e2_d2, e3_d2, e4_d2], dim=1)
        d2 = self.d2(d2)

        """ Decoder 1 """
        e1_d1 = self.e1_d1(tmp1)

        e2_d1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=True)
        e2_d1 = self.e2_d1(e2_d1)

        e3_d1 = F.interpolate(d3, scale_factor=4, mode="bilinear", align_corners=True)
        e3_d1 = self.e3_d1(e3_d1)

        e4_d1 = F.interpolate(out, scale_factor=8, mode="bilinear", align_corners=True)
        e4_d1 = self.e4_d1(e4_d1)

        # e5_d1 = F.interpolate(e5, scale_factor=16, mode="bilinear", align_corners=True)
        # e5_d1 = self.e5_d1(e5_d1)
        # print('e4_d1', e1_d1.shape)
        # print('e4_d1', e1_d1.shape)
        # print('e4_d1', e1_d1.shape)
        # print('e4_d1', e1_d1.shape)

        d1 = torch.cat([e1_d1, e2_d1, e3_d1, e4_d1], dim=1)
        d1 = self.d1(d1)
        
        # print('d1',d1.shape)
        # print('d2',d2.shape)
        # print('d3',d3.shape)

        """ Output """
        # d33 = self.e5(d3)
        # #print('d33',d33.shape)
        # d23 = torch.cat([d2, d33], dim=1)
        # #print('d23',d23.shape)
        # d23f = self.e6(d23)
        # d23f = self.e7(d23f)
        # d23f = torch.cat([d1, d23f], dim=1)
        # #print('d23f',d23f.shape)
        # d123 = self.e8 (d23f)
        # #print('d123',d123.shape)
        # d123f = torch.cat([d123, d23f], dim=1)
        # #print('d123f',d123f.shape)
        
        y1 = self.y1(d1)
        # out_emb = self.e4(conv3)
        # out_emb = self.e4(out1)
        # out_emb = self.e4(out3)
        # embedding = F.interpolate(out_emb, size=(128, 128), mode='bilinear', align_corners=False)
        embedding = F.interpolate(out3, size=(128, 128), mode='bilinear', align_corners=False)

        return y1, embedding

 
if __name__ == '__main__':
    model = m_myunet3plus35()
    # print(model)
    total_params = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total_params/1e6))
    res, res1 = model(torch.randn(5, 1, 256, 256))
    print(res.shape)
    print(res1.shape)