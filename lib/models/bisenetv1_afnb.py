#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from lib.models.resnet import Resnet18

from torch.nn import BatchNorm2d


class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class UpSample(nn.Module):

    def __init__(self, n_chan, factor=2):
        super(UpSample, self).__init__()
        out_chan = n_chan * factor * factor
        self.proj = nn.Conv2d(n_chan, out_chan, 1, 1, 0)
        self.up = nn.PixelShuffle(factor)
        self.init_weight()

    def forward(self, x):
        feat = self.proj(x)
        feat = self.up(feat)
        return feat

    def init_weight(self):
        nn.init.xavier_normal_(self.proj.weight, gain=1.)


class BiSeNetOutput(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=32, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.up_factor = up_factor
        out_chan = n_classes
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, out_chan, kernel_size=1, bias=True)
        self.up = nn.Upsample(scale_factor=up_factor,
                              mode='bilinear', align_corners=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        x = self.up(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = BatchNorm2d(out_chan)
        #  self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        #  atten = self.sigmoid_atten(atten)
        atten = atten.sigmoid()
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class ContextPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ContextPath, self).__init__()
        self.resnet = Resnet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)
        self.up32 = nn.Upsample(scale_factor=2.)
        self.up16 = nn.Upsample(scale_factor=2.)

        self.init_weight()

    def forward(self, x):
        feat8, feat16, feat32 = self.resnet(x)

        avg = torch.mean(feat32, dim=(2, 3), keepdim=True)
        avg = self.conv_avg(avg)

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg
        feat32_up = self.up32(feat32_sum)
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = self.up16(feat16_sum)
        feat16_up = self.conv_head16(feat16_up)

        return feat16_up, feat32_up  # x8, x16

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class SpatialPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)
        self.init_weight()

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv_out(feat)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    def __init__(self, sizes=(1, 3, 6, 8), dimension=2):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center


class _SelfAttentionBlock(nn.Module):
    def __init__(self, x_channels, skip_channels, key_channels, value_channels, out_channels=None, scale=1,
                 psp_size=(1, 3, 6, 8)):
        super(_SelfAttentionBlock, self).__init__()
        self.scale = scale
        self.x_channels = x_channels
        self.skip_channels = skip_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.out_channels = out_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.x_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(key_channels),
            nn.ReLU()
        )
        self.f_query = nn.Sequential(
            nn.Conv2d(in_channels=self.skip_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(key_channels),
            nn.ReLU()
        )
        self.f_value = nn.Conv2d(in_channels=self.x_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
                           kernel_size=1, stride=1, padding=0)
        self.psp = PSPModule(psp_size)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x, skip):
        b, h, w = skip.size(0), skip.size(2), skip.size(3)
        # if self.scale > 1:
        # skip = self.pool(skip)
        f_value = self.f_value(x)
        # print(f_value.shape)
        value = self.psp(f_value)
        # print("psp")
        query = self.f_query(skip).view(b, self.key_channels, -1)
        query = query.permute(0, 2, 1)

        key = self.f_key(x)
        # value=self.psp(value)#.view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        key = self.psp(key)  # .view(batch_size, self.key_channels, -1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(b, self.value_channels, *skip.size()[2:])
        context = self.W(context)
        return context


class SelfAttentionBlock2D(_SelfAttentionBlock):
    def __init__(self, x_channels, skip_channels, key_channels, value_channels, out_channels, scale=1,
                 psp_size=(1, 3, 6, 8)):
        super(SelfAttentionBlock2D, self).__init__(x_channels=x_channels,
                                                   skip_channels=skip_channels,
                                                   key_channels=key_channels,
                                                   value_channels=value_channels,
                                                   out_channels=out_channels,
                                                   scale=scale,
                                                   psp_size=psp_size)


class AFNB(nn.Module):
    def __init__(self, x_channels, skip_channels, out_channels, key_channels, value_channels, scale):
        super(AFNB, self).__init__()
        #print("\tx_channels = {}".format(x_channels))
        #print("\tskip_channels = {}".format(skip_channels))
        #print("\tkey_channels = {}".format(key_channels))
        #print("\tvalue_channels = {}".format(value_channels))
        #print("\tout_channels = {}\n".format(out_channels))
        # print("\tscale = {}\n".format(scale))

        self.stage = SelfAttentionBlock2D(x_channels=x_channels,
                                          skip_channels=skip_channels,
                                          key_channels=key_channels,
                                          value_channels=value_channels,
                                          out_channels=out_channels,
                                          scale=scale)
        self.bn = nn.BatchNorm2d(out_channels)
        # self.conv_bn = nn.Sequential(
        #    nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, padding=0),
        #    nn.BatchNorm2d(out_channels),
        # nn.Dropout2d()
        # )

    def forward(self, x, skip):
        # print("afnb")
        priors_skip = self.stage(x, skip)
        # output = self.bn(priors_skip + skip)
        # output = self.conv_bn(torch.cat([priors_skip, skip], dim=1))
        return priors_skip


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        ## use conv-bn instead of 2 layer mlp, so that tensorrt 7.2.3.4 can work for fp16
        self.conv = nn.Conv2d(out_chan,
                              out_chan,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.afnb = AFNB(in_chan, out_chan, out_chan, out_chan, out_chan, 1)
        #  self.conv1 = nn.Conv2d(out_chan,
        #          out_chan//4,
        #          kernel_size = 1,
        #          stride = 1,
        #          padding = 0,
        #          bias = False)
        #  self.conv2 = nn.Conv2d(out_chan//4,
        #          out_chan,
        #          kernel_size = 1,
        #          stride = 1,
        #          padding = 0,
        #          bias = False)
        #  self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        print(fcat.shape)
        feat = self.convblk(fcat)
        afnb_feat = self.afnb(fcat,feat)

        # atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        # atten = self.conv(atten)
        # atten = self.bn(atten)
        #  atten = self.conv1(atten)
        #  atten = self.relu(atten)
        #  atten = self.conv2(atten)
        # atten = atten.sigmoid()
        # feat_atten = torch.mul(feat, atten)
        # feat_out = feat_atten + feat
        # return feat_out
        return afnb_feat+feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class BiSeNetV1_afnb(nn.Module):

    def __init__(self, n_classes, aux_mode='train', *args, **kwargs):
        super(BiSeNetV1_afnb, self).__init__()
        self.cp = ContextPath()
        self.sp = SpatialPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes, up_factor=8)
        self.aux_mode = aux_mode
        if self.aux_mode == 'train':
            self.conv_out16 = BiSeNetOutput(128, 64, n_classes, up_factor=8)
            self.conv_out32 = BiSeNetOutput(128, 64, n_classes, up_factor=16)
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat_cp8, feat_cp16 = self.cp(x)
        feat_sp = self.sp(x)
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        if self.aux_mode == 'train':
            feat_out16 = self.conv_out16(feat_cp8)
            feat_out32 = self.conv_out32(feat_cp16)
            return feat_out, feat_out16, feat_out32
        elif self.aux_mode == 'eval':
            return feat_out,
        elif self.aux_mode == 'pred':
            feat_out = feat_out.argmax(dim=1)
            return feat_out
        else:
            raise NotImplementedError

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (FeatureFusionModule, BiSeNetOutput)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


if __name__ == "__main__":
    net = BiSeNetV1_afnb(2)
    net.cuda()
    net.eval()
    in_ten = torch.randn(16, 3, 512, 512).cuda()
    out, out16, out32 = net(in_ten)
    print(out.shape)
    print(out16.shape)
    print(out32.shape)

    net.get_params()
