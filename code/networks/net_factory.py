from networks.unet import UNet
from networks.VNet import VNet, CAML3d_v1


def net_factory(net_type="unet", in_chns=1, class_num=4, mode = "train", **kwargs):
    if net_type == "vnet" and mode == "train":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "caml3d_v1" and mode == "train":
        net = CAML3d_v1(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True, **kwargs).cuda()
    elif net_type == "caml3d_v1" and mode == "test":
        net = CAML3d_v1(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False, **kwargs).cuda()
    return net
