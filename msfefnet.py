from ..block.conv import conv3x3
from .base import BaseNet
from ..block.SEAttention import SEAttention
import torch
from torch import nn
import torch.nn.functional as F


class MSFEFNet(BaseNet):
    def __init__(self, backbone, pretrained, nclass, lightweight):
        super(MSFEFNet, self).__init__(backbone, pretrained)

        channel_list = self.backbone.channels

        self.head_bin = MSFEFHead(channel_list, 1, lightweight)

    def base_forward(self, x1, x2, muti_loss=False):
        b, c, h, w = x1.shape

        feature_list1 = self.backbone.base_forward(x1)
        feature_list2 = self.backbone.base_forward(x2)

        feature_list = []
        feature_list.append(torch.abs(feature_list1[-4] - feature_list2[-4]))
        feature_list.append(torch.abs(feature_list1[-3] - feature_list2[-3]))
        feature_list.append(torch.abs(feature_list1[-2] - feature_list2[-2]))
        feature_list.append(torch.abs(feature_list1[-1] - feature_list2[-1]))

        aux_out, out_bin = self.head_bin(feature_list)
        out_bin = F.interpolate(out_bin, size=(h, w), mode='bilinear', align_corners=False)
        out_bin = torch.sigmoid(out_bin)
        out_bin_dsn0 = torch.sigmoid(F.interpolate(aux_out[0], size=(h, w), mode='bilinear', align_corners=False))
        out_bin_dsn1 = torch.sigmoid(F.interpolate(aux_out[1], size=(h, w), mode='bilinear', align_corners=False))
        out_bin_dsn2 = torch.sigmoid(F.interpolate(aux_out[2], size=(h, w), mode='bilinear', align_corners=False))
        out_bin_dsn3 = torch.sigmoid(F.interpolate(aux_out[3], size=(h, w), mode='bilinear', align_corners=False))
        if muti_loss:
            return [out_bin_dsn0.squeeze(1), out_bin_dsn1.squeeze(1), out_bin_dsn2.squeeze(1),
                    out_bin_dsn3.squeeze(1)], out_bin.squeeze(1)
        else:
            return out_bin.squeeze(1)


class MSFEFHead(nn.Module):
    def __init__(self, in_channels_list, out_channels, lightweight):
        super(MSFEFHead, self).__init__()
        print(in_channels_list)
        inter_channel_list = [sum(in_channels_list[1:]) // 16] * 4
        self.dsn0 = self.dsn(inter_channel_list[0])
        self.dsn1 = self.dsn(inter_channel_list[1])
        self.dsn2 = self.dsn(inter_channel_list[2])
        self.dsn3 = self.dsn(inter_channel_list[3])

        in_channel = sum(in_channels_list[1:]) // 4
        inter_channels = in_channel // 4
        self.MSFEF_Pyramid = MSFEF_Pyramid(in_channels_list)

        self.conv5 = nn.Sequential(
            conv3x3(inter_channels, inter_channels, lightweight),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(inter_channels, out_channels, 1))

    def dsn(self, inter_channel):
        return nn.Sequential(
            nn.Conv2d(inter_channel, inter_channel // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inter_channel // 4),
            nn.Dropout2d(0.1),
            nn.Conv2d(inter_channel // 4, 1, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        aux, out = self.MSFEF_Pyramid(x)
        out = self.conv5(out)
        aux0 = self.dsn0(aux[0])
        aux1 = self.dsn1(aux[1])
        aux2 = self.dsn2(aux[2])
        aux3 = self.dsn3(aux[3])
        return [aux0, aux1, aux2, aux3], out


class MSFEF_Pyramid(nn.Module):
    def __init__(self, in_channels_list, conv_kernels=[3, 5, 7, 9], conv_groups=[2, 4, 8, 16]):
        super(MSFEF_Pyramid, self).__init__()
        self.rank_strategy = [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]]
        in_channel = (in_channels_list[-1] + in_channels_list[-2] + in_channels_list[-3] + in_channels_list[-4]) // 4
        out_channel = in_channel // 4
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, conv_kernels[0], padding=conv_kernels[0] // 2,
                      groups=conv_groups[0], bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            SEAttention(channel=out_channel, reduction=8))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, conv_kernels[1], padding=conv_kernels[1] // 2,
                      groups=conv_groups[1], bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            SEAttention(channel=out_channel, reduction=8))
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, conv_kernels[2], padding=conv_kernels[2] // 2,
                      groups=conv_groups[2], bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            SEAttention(channel=out_channel, reduction=8))
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, conv_kernels[3], padding=conv_kernels[3] // 2,
                      groups=conv_groups[3], bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            SEAttention(channel=out_channel, reduction=8))
        self.se_attention = SEAttention(channel=out_channel)

    def downsample(self, in_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, out_features):
        c1, c2, c3, c4 = out_features[-4], out_features[-3], out_features[-2], out_features[-1]

        h = c1.shape[-2]
        w = c1.shape[-1]

        c1 = F.interpolate(c1, (h, w), mode="bilinear", align_corners=False)
        c2 = F.interpolate(c2, (h, w), mode="bilinear", align_corners=False)
        c3 = F.interpolate(c3, (h, w), mode="bilinear", align_corners=False)
        c4 = F.interpolate(c4, (h, w), mode="bilinear", align_corners=False)

        c1 = torch.split(c1, c1.shape[1] // 4, dim=1)
        c2 = torch.split(c2, c2.shape[1] // 4, dim=1)
        c3 = torch.split(c3, c3.shape[1] // 4, dim=1)
        c4 = torch.split(c4, c4.shape[1] // 4, dim=1)

        out_list = []
        for rank in self.rank_strategy:
            out = {}
            for idx, i in enumerate(rank):
                out[[c1, c2, c3, c4][idx][i]] = i
            out = sorted(out.items(), key=lambda kv: (kv[1], kv[0]))
            out = [per[0] for per in out]
            out_list.append(torch.cat(out, dim=1))
        out1, out2, out3, out4 = out_list

        out1 = self.layer1(out1)
        out2 = self.layer2(out2)
        out3 = self.layer3(out3)
        out4 = self.layer4(out4)

        out = out1 + out2 + out3 + out4
        out = self.se_attention(out)
        return [out1, out2, out3, out4], out


if __name__ == '__main__':
    x1 = torch.rand([1, 3, 256, 256])
    x2 = torch.rand([1, 3, 256, 256])
    model = MSFEFNet('resnet50', False, 2, True)
    # print(model.named_children())
    # for out in model(x1, x2)[0]:
    #     print(out.shape)
    # for name, module in model.backbone.named_children():
    #     print(name)
    # for index, layer in enumerate(model.backbone):
    #     print(index,layer)

    from thop import profile

    # model = resnet50()
    # input = torch.randn(1, 3, 224, 224)
    macs, params = profile(model, inputs=(x1, x2))
    print(
        "%s | %.2f | %.2f" % ('ourmodel', params / (1000 ** 2), macs / (1000 ** 3))
    )
