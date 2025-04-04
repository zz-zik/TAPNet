# -*- coding: utf-8 -*-
"""
@Project : TFPNet4-DAF
@FileName: adaptive_decomposition.py
@Time    : 2024/10/19 上午11:39
@Author  : ZhouFei
@Email   : zhoufei.net@outlook.com
@Desc    : consultation https://github.com/xujian000/DAF-Net/blob/main/model/net.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
import numbers


class AttentionBase(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
    ):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )
        out = self.proj(out)
        return out


class Mlp(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, ffn_expansion_factor=2, bias=False
    ):
        super().__init__()
        hidden_features = int(in_features * ffn_expansion_factor)
        self.project_in = nn.Conv2d(
            in_features, hidden_features * 2, kernel_size=1, bias=bias
        )
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features,
            bias=bias,
        )
        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias
        )

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class BaseFeatureExtractor(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        ffn_expansion_factor=1.0,
        qkv_bias=False,
    ):
        super(BaseFeatureExtractor, self).__init__()
        self.norm1 = LayerNorm(dim, "BiasFree")
        self.attn = AttentionBase(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        self.norm2 = LayerNorm(dim, "BiasFree")
        self.mlp = Mlp(
            in_features=dim,
            ffn_expansion_factor=ffn_expansion_factor,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        return self.bottleneckBlock(x)


class DetailNode(nn.Module):
    def __init__(self, dim=64):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=dim//2, oup=dim//2, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=dim//2, oup=dim//2, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=dim//2, oup=dim//2, expand_ratio=2)
        self.shffleconv = nn.Conv2d(
            dim, dim, kernel_size=1, stride=1, padding=0, bias=True
        )

    def separateFeature(self, x):
        z1, z2 = x[:, : x.shape[1] // 2], x[:, x.shape[1] // 2 : x.shape[1]]
        return z1, z2

    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2


class DetailFeatureExtractor(nn.Module):
    def __init__(self, num_layers=3, dim=64):
        super(DetailFeatureExtractor, self).__init__()
        INNmodules = [DetailNode(dim=dim) for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)

    def forward(self, x):
        z1, z2 = x[:, : x.shape[1] // 2], x[:, x.shape[1] // 2 : x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class Restormer_Encoder(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=64,
        num_blocks=[4, 4],
        heads=[8, 8, 8],
        ffn_expansion_factor=2,
        bias=False,
        LayerNorm_type="BiasFree",  # WithBias
    ):

        super(Restormer_Encoder, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[0])
            ]
        )
        self.baseFeature = BaseFeatureExtractor(dim=dim, num_heads=heads[2])
        self.detailFeature = DetailFeatureExtractor(num_layers=1, dim=dim)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        base_feature = self.baseFeature(out_enc_level1)
        detail_feature = self.detailFeature(out_enc_level1)
        return base_feature, detail_feature, out_enc_level1


class Restormer_Decoder(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=64,
        num_blocks=[4, 4],
        heads=[8, 8, 8],
        ffn_expansion_factor=2,
        bias=False,
        LayerNorm_type="BiasFree",  # WithBias
    ):

        super(Restormer_Decoder, self).__init__()
        self.reduce_channel = nn.Conv2d(
            int(dim * 2), int(dim), kernel_size=1, bias=bias
        )
        self.encoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )
        self.output = nn.Sequential(
            nn.Conv2d(
                int(dim), int(dim) // 2, kernel_size=3, stride=1, padding=1, bias=bias
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                int(dim) // 2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias,
            ),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp_img, base_feature, detail_feature):
        out_enc_level0 = torch.cat((base_feature, detail_feature), dim=1)
        out_enc_level0 = self.reduce_channel(out_enc_level0)
        out_enc_level1 = self.encoder_level2(out_enc_level0)
        if inp_img is not None:
            out_enc_level1 = self.output(out_enc_level1) + inp_img
        else:
            out_enc_level1 = self.output(out_enc_level1)
        return self.sigmoid(out_enc_level1), out_enc_level0


if __name__ == "__main__":
    import time
    from thop import profile
    height = 512
    width = 640
    window_size = 8
    modelE = Restormer_Encoder(inp_channels=1, dim=64, num_blocks=[1, 1]).cuda()
    modelD = Restormer_Decoder(out_channels=1, dim=64, num_blocks=[1, 1]).cuda()
    modelB = BaseFeatureExtractor(dim=256, num_heads=8)
    modelB = modelB.cuda()
    modelsD = DetailFeatureExtractor(num_layers=3, dim=256)
    modelsD = modelsD.cuda()

    input_vi = torch.randn(2, 1, height, width).cuda()
    input_ir = torch.randn(2, 1, height, width).cuda()
    start_time = time.time()
    base_feature, detail_feature, out_enc_level1 = modelE(input_ir)
    start_time2 = time.time()
    print(base_feature.shape, detail_feature.shape, out_enc_level1.shape, start_time2-start_time)

    flops, params = profile(modelE, inputs=(input_vi,))
    print(f"FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")

    # 测试Decoder
    output, out_enc_level0 = modelD(input_vi, base_feature, detail_feature)
    start_time3 = time.time()
    print(output.shape, out_enc_level0.shape, start_time3-start_time2)

    # 测试BaseFeatureExtractor
    out_enc_level1 = torch.randn(2, 256, 32, 40).cuda()
    output_B = modelB(out_enc_level1)
    start_time4 = time.time()
    print(output_B.shape, start_time4-start_time3)

    # 测试DetailFeatureExtractor
    output_D = modelsD(out_enc_level1)
    start_time5 = time.time()
    print(output_D.shape, start_time5-start_time4)




