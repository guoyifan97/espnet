# Implementation of the TCN proposed in
# Luo. et al.  "Conv-tasnet: Surpassing ideal time–frequency
# magnitude masking for speech separation."
#
# The code is based on:
# https://github.com/kaituoxu/Conv-TasNet/blob/master/src/conv_tasnet.py
#


import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = torch.finfo(torch.get_default_dtype()).eps

class TAC(nn.Module):
    def __init__(
        self,
        feature_dim,
        nonlinear="relu",
    ):
        super().__init__()
        # B*C N T -> B*C N T
        self.linear1 = nn.Conv1d(feature_dim, feature_dim, 1)

        assert nonlinear in ["relu", "prelu"]
        self.nonlinear1 = {
            "relu": nn.ReLU,
            "prelu": nn.PReLU,
        }[nonlinear]()

        # B N T -> B N T
        self.linear2 = nn.Conv1d(feature_dim, feature_dim, 1)
        self.nonlinear2 = {
            "relu": nn.ReLU,
            "prelu": nn.PReLU,
        }[nonlinear]()

        # B 2*N T -> B N T
        self.linear3 = nn.Conv1d(2*feature_dim, feature_dim, 1)
        self.nonlinear3 = {
            "relu": nn.ReLU,
            "prelu": nn.PReLU,
        }[nonlinear]()


    def forward(self, x):
        """
            Args:
                x: B C N T
            
            Returns:
                x: B C N T

        """
        B, C, N, T = x.shape

        residual = x

        # B C N T -> B*C N T
        x = x.contiguous().view(-1, N, T)

        # B*C N T -> B*C N T
        x = self.nonlinear1(self.linear1(x))

        # B*C N T -> B C N T
        x = x.contiguous().view(B, C, N, T)

        # B C N T -> B N T
        x_avg = self.nonlinear2(self.linear2(torch.mean(x, dim=1, keepdim=False)))
        # B N T -> B 1 N T -> B C N T
        x_avg = x_avg.unsqueeze(1).expand(B, C, N, T)

        # B C N T -> B C 2*N T
        x = torch.cat([x, x_avg], dim=2)

        # B C 2*N T -> B*C 2*N T
        x = x.contiguous().view(-1, 2*N, T)
        # B*C 2*N T -> B*C N T -> B C N T
        x = self.nonlinear3(self.linear3(x)).contiguous().view(B, C, N, T)
        
        # B C N T
        return x + residual




class TemporalConvNetTAC(nn.Module):
    def __init__(
        self, N, B, H, P, X, R, C, norm_type="gLN", causal=False, mask_nonlinear="relu", tac_nonlinear="prelu",
    ):
        """Basic Module of tasnet.

        Args:
            N: Number of filters in autoencoder
            B: Number of channels in bottleneck 1 * 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            norm_type: BN, gLN, cLN
            causal: causal or non-causal
            mask_nonlinear: use which non-linear function to generate mask
        """
        super().__init__()
        # Hyper-parameter
        self.num_spk = C
        self.mask_nonlinear = mask_nonlinear
        # Components
        # [M, N, K] -> [M, N, K]
        self.layer_norm = ChannelwiseLayerNorm(N)
        # [M, N, K] -> [M, B, K]
        self.bottleneck_conv1x1 = nn.Conv1d(N, B, 1, bias=False)
        # [M, B, K] -> [M, B, K]
        repeats = []
        for r in range(R):
            blocks = []
            for x in range(X):
                dilation = 2 ** x
                padding = (P - 1) * dilation if causal else (P - 1) * dilation // 2
                blocks += [
                    TemporalBlock(
                        B,
                        H,
                        P,
                        stride=1,
                        padding=padding,
                        dilation=dilation,
                        norm_type=norm_type,
                        causal=causal,
                    ),
                ]
            repeats += [nn.Sequential(*blocks), TAC(B, tac_nonlinear)]
        self.temporal_conv_net_tac = nn.Sequential(*repeats)
        # self.network = nn.Sequential(
        #     layer_norm, bottleneck_conv1x1, temporal_conv_net_tac,
        # )
        # [M, B, K] -> [M, C*N, K]
        self.mask_conv1x1 = nn.Conv1d(B, C * N, 1, bias=False)


    def forward(self, mixture_w):
        """Keep this API same with TasNet.

        Args:
            mixture_w: [B, C, N, T], M is batch size, Channel is num_channels, N is the feature_dim

        Returns:
            est_mask: [B, num_spk, C, N, T]
        """
        B, C, N, T = mixture_w.size()

        output = self.layer_norm(mixture_w.contiguous().view(-1, N, T)) # B*C N T

        output = self.bottleneck_conv1x1(output) # B*C N T

        output = self.temporal_conv_net_tac(output.contiguous().view(B, C, -1, T))  # B C N T -> B C N T

        score = self.mask_conv1x1(output.contiguous().view(B*C, -1, T)) # B C N T -> B*C N T -> B*C num_spk*N T
        score = score.view(B*C, self.num_spk, N, T)  # B*C num_spk*N T -> B*C num_spk N T
        if self.mask_nonlinear == "softmax":
            est_mask = F.softmax(score, dim=1)
        elif self.mask_nonlinear == "relu":
            est_mask = F.relu(score)
        elif self.mask_nonlinear == "sigmoid":
            est_mask = torch.sigmoid(score)
        elif self.mask_nonlinear == "tanh":
            est_mask = torch.tanh(score)
        else:
            raise ValueError("Unsupported mask non-linear function")
        return est_mask


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        norm_type="gLN",
        causal=False,
    ):
        super().__init__()
        # [M, B, K] -> [M, H, K]
        conv1x1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        prelu = nn.PReLU()
        norm = chose_norm(norm_type, out_channels)
        # [M, H, K] -> [M, B, K]
        dsconv = DepthwiseSeparableConv(
            out_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            norm_type,
            causal,
        )
        # Put together
        self.net = nn.Sequential(conv1x1, prelu, norm, dsconv)

    def forward(self, x):
        """Forward.

        Args:
            x: [B, C, N, T]

        Returns:
            [B, C, N, T]
        """
        # B C N T
        B, C, N, T = x.shape
        residual = x
        # B C N T -> B*C N T -> B C N T
        out = self.net(x.contiguous().view(-1, N, T)).contiguous().view(B, C, N, T)

        # TODO(Jing): when P = 3 here works fine, but when P = 2 maybe need to pad?
        return out + residual  # look like w/o F.relu is better than w/ F.relu
        # return F.relu(out + residual)


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        norm_type="gLN",
        causal=False,
    ):
        super().__init__()
        # Use `groups` option to implement depthwise convolution
        # [M, H, K] -> [M, H, K]
        depthwise_conv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        if causal:
            chomp = Chomp1d(padding)
        prelu = nn.PReLU()
        norm = chose_norm(norm_type, in_channels)
        # [M, H, K] -> [M, B, K]
        pointwise_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        # Put together
        if causal:
            self.net = nn.Sequential(depthwise_conv, chomp, prelu, norm, pointwise_conv)
        else:
            self.net = nn.Sequential(depthwise_conv, prelu, norm, pointwise_conv)

    def forward(self, x):
        """Forward.

        Args:
            x: [M, H, K]

        Returns:
            result: [M, B, K]
        """
        return self.net(x)


class Chomp1d(nn.Module):
    """To ensure the output length is the same as the input."""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """Forward.

        Args:
            x: [M, H, Kpad]

        Returns:
            [M, H, K]
        """
        return x[:, :, : -self.chomp_size].contiguous()


def check_nonlinear(nolinear_type):
    if nolinear_type not in ["softmax", "relu"]:
        raise ValueError("Unsupported nonlinear type")


def chose_norm(norm_type, channel_size):
    """The input of normalization will be (M, C, K), where M is batch size.

    C is channel size and K is sequence length.
    """
    if norm_type == "gLN":
        return GlobalLayerNorm(channel_size)
    elif norm_type == "cLN":
        return ChannelwiseLayerNorm(channel_size)
    elif norm_type == "BN":
        # Given input (M, C, K), nn.BatchNorm1d(C) will accumulate statics
        # along M and K, so this BN usage is right.
        return nn.BatchNorm1d(channel_size)
    else:
        raise ValueError("Unsupported normalization type")


class ChannelwiseLayerNorm(nn.Module):
    """Channel-wise Layer Normalization (cLN)."""

    def __init__(self, channel_size):
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """Forward.

        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length

        Returns:
            cLN_y: [M, N, K]
        """
        mean = torch.mean(y, dim=1, keepdim=True)  # [M, 1, K]
        var = torch.var(y, dim=1, keepdim=True, unbiased=False)  # [M, 1, K]
        cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return cLN_y


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)."""

    def __init__(self, channel_size):
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """Forward.

        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length

        Returns:
            gLN_y: [M, N, K]
        """
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)  # [M, 1, 1]
        var = (
            (torch.pow(y - mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        )
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y
