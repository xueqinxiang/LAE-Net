import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from .CCN import cConv2d

#======================================================20240604
from torch.nn.utils.spectral_norm import SpectralNorm
from torchvision.transforms import RandomCrop, Normalize
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from networks.shared import ResidualBlock, FullyConnectedLayer
from networks.vit_utils import make_vit_backbone, forward_vit
from training.diffaug import DiffAugment

class DCBlock(nn.Module):
    def __init__(self, in_ch, cond_dim=512):
        super(DCBlock, self).__init__()
        self.conv = cConv2d(in_ch, 8, 3, 1, 1, cond_dim=cond_dim)
        self.conv_weight = nn.Conv2d(8, 1, 3, 1, 1, bias=False)
        self.conv_bias = nn.Conv2d(8, 1, 3, 1, 1, bias=False)

    def forward(self, img, mask, c):
        h = self.conv(mask, c)
        weight = self.conv_weight(h)
        bias = self.conv_bias(h)
        return weight * img + bias


class CAFF(nn.Module):
    def __init__(self, num_features, cond_dim=512):
        super(CAFF, self).__init__()

        self.num_features = num_features
        
        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(cond_dim, cond_dim)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(cond_dim, num_features)),
            ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(cond_dim, cond_dim)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(cond_dim, num_features)),
            ]))

        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, c):
        gamma = self.fc_gamma(c).view(-1, self.num_features, 1, 1)
        beta = self.fc_beta(c).view(-1, self.num_features, 1, 1)
        return gamma * x + beta


class UNetDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNetDown, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias = False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, h):
        m = h
        h = self.conv(h)
        h = self.bn(h)
        h = nn.LeakyReLU(0.2, inplace=True)(h)
        return h, m


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 4, 1, 0)

    def forward(self, h):
        h = self.conv2(self.act(self.conv1(h)))
        return h


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.main = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(in_dim, in_dim)),
            ('relu1',nn.LeakyReLU(0.2, inplace=True)),
            ('linear2',nn.Linear(in_dim, out_dim)),
            ]))

    def forward(self, h):
        h = self.main(h)
        return h


class UNetYp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNetYp, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1)

    def forward(self, h, skip_input):
        h = F.interpolate(h, scale_factor=2)
        h = self.conv(h)
        h = nn.LeakyReLU(0.2, inplace=True)(h)
        h = h + skip_input
        return h


class DEBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DEBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.CAFF = CAFF(out_ch)
        self.DCBlk = DCBlock(out_ch)
        self.fc = nn.Sequential(OrderedDict([
            ('linear2',nn.Linear(512*4, out_ch)),
            ('sigmoid',nn.Sigmoid()),
            ]))

        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc.linear2.weight.data)
        nn.init.zeros_(self.fc.linear2.bias.data)

    def forward(self, h, skip, mask, y, ti):
        h = F.interpolate(h, scale_factor=2)
        h = self.conv(h)
        h_m = nn.LeakyReLU(0.2, inplace=True)(self.CAFF(h, y))
        h_s = nn.LeakyReLU(0.2, inplace=True)(self.DCBlk(h, mask, y))
        weights = self.fc(ti).unsqueeze(-1).unsqueeze(-1)
        h = weights*h_m + (1-weights)*h_s
        h = h + skip
        return h


class CWP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CWP, self).__init__()
        self.img_enc = ConvBlock(in_ch, out_ch)
        self.txt_enc = MLP(out_ch, out_ch)
        self.fuse = MLP(512*2, 512*2)
        self.fc = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(512*4, 512*4)),
            ('relu1',nn.LeakyReLU(0.2,inplace=True)),
        ]))

    def forward(self, v, t):
        img_emb = self.img_enc(v).view(v.size(0),-1)
        txt_emb = self.txt_enc(t)
        ti = self.fuse(torch.cat((img_emb, txt_emb), dim=1))
        tid = ti[:,:512] - ti[:,512:]
        tim = ti[:,:512] * ti[:,512:]
        ti = self.fc(torch.cat((tid, tim, img_emb, txt_emb), dim=1))
        return ti


class NetG(nn.Module):
    def __init__(self, ngf=64, nz=100, lstm = None):
        super(NetG, self).__init__()
        self.ngf = ngf
        self.lstm = lstm
        self.ext = nn.Conv2d(3, ngf * 1, 3, 1, 1)#256

        self.down1 = UNetDown(ngf * 1,  ngf * 2)#128
        self.down2 = UNetDown(ngf * 2,  ngf * 4)#64
        self.down3 = UNetDown(ngf * 4,  ngf * 8)#32
        self.down4 = UNetDown(ngf * 8,  ngf * 8)#16
        self.down5 = UNetDown(ngf * 8,  ngf * 8)#8
        self.down6 = UNetDown(ngf * 8,  ngf * 8)#4

        self.yp1 = UNetYp(ngf * 8, ngf * 8)#8
        self.yp2 = UNetYp(ngf * 8, ngf * 8)#16
        self.yp3 = UNetYp(ngf * 8, ngf * 8)#32
        self.yp4 = UNetYp(ngf * 8, ngf * 4)#64
        self.yp5 = UNetYp(ngf * 4, ngf * 2)#128
        self.yp6 = UNetYp(ngf * 2, ngf * 1)#256

        self.up1 = DEBlock(ngf * 8, ngf * 8)#8
        self.up2 = DEBlock(ngf * 8, ngf * 8)#16
        self.up3 = DEBlock(ngf * 8, ngf * 8)#32
        self.up4 = DEBlock(ngf * 8, ngf * 4)#64
        self.up5 = DEBlock(ngf * 4, ngf * 2)#128
        self.up6 = DEBlock(ngf * 2, ngf * 1)#256

        self.cwp = CWP(ngf * 8, 512)

        self.conv_img = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ngf * 1, 3, 3, 1, 1),
            nn.Tanh(),
        )
        
    def forward(self, img, c):
        c = c.float()  # float16 -> float32
        h = self.ext(img)

        d1, s1 = self.down1(h)
        d2, s2 = self.down2(d1)
        d3, s3 = self.down3(d2)
        d4, s4 = self.down4(d3)
        d5, s5 = self.down5(d4)
        d6, s6 = self.down6(d5)


        m1 = self.yp1(d6, s6)#8
        m2 = self.yp2(m1, s5)#16
        m3 = self.yp3(m2, s4)#32
        m4 = self.yp4(m3, s3)#64
        m5 = self.yp5(m4, s2)#128
        m6 = self.yp6(m5, s1)#256

        ti = self.cwp(d6, c)

        lstm_input = c
        c,_  =  self.lstm(lstm_input)
        u1 = self.up1(d6, s6, m1, c, ti)#8

        lstm_input = c
        c,_  =  self.lstm(lstm_input)
        u2 = self.up2(u1, s5, m2, c, ti)#16

        lstm_input = c
        c,_  =  self.lstm(lstm_input)
        u3 = self.up3(u2, s4, m3, c, ti)#32

        lstm_input = c
        c,_  =  self.lstm(lstm_input)
        u4 = self.up4(u3, s3, m4, c, ti)#64

        lstm_input = c
        c,_  =  self.lstm(lstm_input)
        u5 = self.up5(u4, s2, m5, c, ti)#128

        lstm_input = c
        c,_  =  self.lstm(lstm_input)
        u6 = self.up6(u5, s1, m6, c, ti)#256

        fake = self.conv_img(u6)
        return fake


#NetD Version1
# 定义鉴别器网络D
# class NetD(nn.Module):
#     def __init__(self, ndf):
#         super(NetD, self).__init__()
#
#         self.block0 = D_Block(3, ndf, 4, 2, 1, bn=False)#128
#         self.block1 = D_Block(ndf * 1, ndf * 2, 4, 2, 1)#64
#         self.block2 = D_Block(ndf * 2, ndf * 4, 4, 2, 1)#32
#         self.block3 = D_Block(ndf * 4, ndf * 8, 4, 2, 1)#16
#         self.block4 = D_Block(ndf * 8, ndf * 8, 4, 2, 1)#8
#         self.block5 = D_Block(ndf * 8, ndf * 8, 4, 2, 1)#4
#
#     def forward(self,out):
#
#         h = self.block0(out)
#         h = self.block1(h)
#         h = self.block2(h)
#         h = self.block3(h)
#         h = self.block4(h)
#         h = self.block5(h)
#         return h


class NetC(nn.Module):
    def __init__(self, ndf):
        super(NetC, self).__init__()
        self.joint_conv = nn.Sequential(
            #nn.Conv2d(ndf * 8+256, ndf * 2, 3, 1, 1, bias=False),
            nn.Conv2d(ndf * 8 + 512, ndf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf * 2, 1, 4, 1, 0, bias=False),
        )

    def forward(self, out, y):
        #y = y.view(-1, 256, 1, 1)
        y = y.view(-1, 512, 1, 1)
        y = y.repeat(1, 1, 4, 4)
        h_c_code = torch.cat((out, y), 1)
        out = self.joint_conv(h_c_code)
        return out 


#NetD Version1
# class D_Block(nn.Module):
#     def __init__(self, in_ch, out_ch, kernel_size, stride, padding, bn=True):
#         super(D_Block, self).__init__()
#         self.bn = bn
#         self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
#         if bn==True:
#             self.batchnorm = nn.BatchNorm2d(out_ch)
#         else:
#             self.batchnorm = None
#
#     def forward(self, h, y=None):
#         h = self.conv(h)
#         if self.bn==True:
#             h = self.batchnorm(h)
#         h = nn.LeakyReLU(0.2, inplace=True)(h)
#         return h

def get_D_in_out_chs(nf, imsize):
    layer_num = int(np.log2(imsize))-1
    channel_nums = [nf*min(2**idx, 8) for idx in range(layer_num)]
    in_out_pairs = zip(channel_nums[:-1], channel_nums[1:])
    return in_out_pairs

class D_Block(nn.Module):
    def __init__(self, fin, fout, downsample=True):
        super(D_Block, self).__init__()
        self.downsample = downsample
        self.learned_shortcut = (fin != fout)
        self.conv_r = nn.Sequential(
            nn.Conv2d(fin, fout, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fout, fout, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_s = nn.Conv2d(fin,fout, 1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        res = self.conv_r(x)
        if self.learned_shortcut:
            x = self.conv_s(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        #return x + res
        return x + self.gamma*res

class NetD(nn.Module):
    def __init__(self, ndf, imsize=256, ch_size=3):
        super(NetD, self).__init__()
        self.conv_img = nn.Conv2d(ch_size, ndf, 3, 1, 1)
        # build DBlocks
        self.DBlocks = nn.ModuleList([])
        in_out_pairs = get_D_in_out_chs(ndf, imsize)
        for idx, (in_ch, out_ch) in enumerate(in_out_pairs):
            self.DBlocks.append(D_Block(in_ch, out_ch))

    def forward(self,x):
        out = self.conv_img(x)
        for DBlock in self.DBlocks:
            out = DBlock(out)
        return out



class CLIP_IMG_ENCODER(nn.Module):
    def __init__(self, CLIP):
        super(CLIP_IMG_ENCODER, self).__init__()
        model = CLIP.visual
        # print(model)
        self.define_module(model)
        for param in self.parameters():
            param.requires_grad = False

    def define_module(self, model):
        self.conv1 = model.conv1
        self.class_embedding = model.class_embedding
        self.positional_embedding = model.positional_embedding
        self.ln_pre = model.ln_pre
        self.transformer = model.transformer
        self.ln_post = model.ln_post
        self.proj = model.proj

    @property
    def dtype(self):
        return self.conv1.weight.dtype

    def transf_to_CLIP_input(self,inputs):
        device = inputs.device
        if len(inputs.size()) != 4:
            raise ValueError('Expect the (B, C, X, Y) tensor.')
        else:
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])\
                .unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)
            var = torch.tensor([0.26862954, 0.26130258, 0.27577711])\
                .unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)
            inputs = F.interpolate(inputs*0.5+0.5, size=(224, 224))
            inputs = ((inputs+1)*0.5-mean)/var
            return inputs

    def forward(self, img: torch.Tensor):
        x = self.transf_to_CLIP_input(img)
        x = x.type(self.dtype)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        grid =  x.size(-1)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        # NLD -> LND
        x = x.permute(1, 0, 2)
        # Local features
        #selected = [1,4,7,12]
        selected = [1,4,8]
        local_features = []
        for i in range(12):
            x = self.transformer.resblocks[i](x)
            if i in selected:
                local_features.append(x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(-1, 768, grid, grid).contiguous().type(img.dtype))
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return torch.stack(local_features, dim=1), x.type(img.dtype)


class CLIP_TXT_ENCODER(nn.Module):
    def __init__(self, CLIP):
        super(CLIP_TXT_ENCODER, self).__init__()
        self.define_module(CLIP)
        # print(model)
        for param in self.parameters():
            param.requires_grad = False

    def define_module(self, CLIP):
        self.transformer = CLIP.transformer
        self.vocab_size = CLIP.vocab_size
        self.token_embedding = CLIP.token_embedding
        self.positional_embedding = CLIP.positional_embedding
        self.ln_final = CLIP.ln_final
        self.text_projection = CLIP.text_projection

    @property
    def dtype(self):
        return self.transformer.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        sent_emb = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return sent_emb, x


    #======================================================20240604
class SpectralConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SpectralNorm.apply(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12)


class BatchNormLocal(nn.Module):
    def __init__(self, num_features: int, affine: bool = True, virtual_bs: int = 8, eps: float = 1e-5):
        super().__init__()
        self.virtual_bs = virtual_bs
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()

        # Reshape batch into groups.
        G = np.ceil(x.size(0)/self.virtual_bs).astype(int)
        x = x.view(G, -1, x.size(-2), x.size(-1))

        # Calculate stats.
        mean = x.mean([1, 3], keepdim=True)
        var = x.var([1, 3], keepdim=True, unbiased=False)
        x = (x - mean) / (torch.sqrt(var + self.eps))

        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]

        return x.view(shape)

def make_block(channels: int, kernel_size: int) -> nn.Module:
    return nn.Sequential(
        SpectralConv1d(
            channels,
            channels,
            kernel_size = kernel_size,
            padding = kernel_size//2,
            padding_mode = 'circular',
        ),
        BatchNormLocal(channels),
        nn.LeakyReLU(0.2, True),
    )
class DiscHead(nn.Module):
    def __init__(self, channels: int, c_dim: int, cmap_dim: int = 64):
        super().__init__()
        self.channels = channels
        self.c_dim = c_dim
        self.cmap_dim = cmap_dim

        self.main = nn.Sequential(
            make_block(channels, kernel_size=1),
            ResidualBlock(make_block(channels, kernel_size=9))
        )

        if self.c_dim > 0:
            self.cmapper = FullyConnectedLayer(self.c_dim, cmap_dim)
            self.cls = SpectralConv1d(channels, cmap_dim, kernel_size=1, padding=0)
        else:
            self.cls = SpectralConv1d(channels, 1, kernel_size=1, padding=0)

        self.joint_conv = nn.Sequential(
            nn.Conv2d(2 * 64, 2 * 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2 * 64, 64, 3, 1, 1, bias=False),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.main(x)
        out = self.cls(h)
        channel = out.shape[1]
        batch = out.shape[0]

        if self.c_dim > 0:
            cmap = self.cmapper(c).unsqueeze(-1)
            cmap = cmap.unsqueeze(-1)
            cmap = cmap.repeat(1, 1, 14, 14)
            out = out.reshape(batch, channel, 14, 14)
            h_c_code = torch.cat((out, cmap), 1)
            out = self.joint_conv(h_c_code)
            out = out.sum(1, keepdim=True)
            out = out.reshape(batch, 1, -1)
            # out = (out * cmap).sum(1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        return out

class DINO(torch.nn.Module):
    def __init__(self, hooks: list[int] = [2, 5, 8, 11], hook_patch: bool = True):
        super().__init__()
        self.n_hooks = len(hooks) + int(hook_patch)

        self.model = make_vit_backbone(
            timm.create_model('vit_small_patch16_224_dino', pretrained=True),
            patch_size=[16, 16], hooks=hooks, hook_patch=hook_patch,
        )
        self.model = self.model.eval().requires_grad_(False)

        self.img_resolution = self.model.model.patch_embed.img_size[0]
        self.embed_dim = self.model.model.embed_dim
        self.norm = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' input: x in [0, 1]; output: dict of activations '''
        x = F.interpolate(x, self.img_resolution, mode='area')
        x = self.norm(x)
        features = forward_vit(self.model, x)
        return features

class NetD_DINO(nn.Module):
    def __init__(self, c_dim: int, diffaug: bool = True, p_crop: float = 0.5):
        super().__init__()
        self.c_dim = c_dim
        self.diffaug = diffaug
        self.p_crop = p_crop

        self.dino = DINO()

        heads = []
        for i in range(self.dino.n_hooks):
            heads += [str(i), DiscHead(self.dino.embed_dim, c_dim)],
        self.heads = nn.ModuleDict(heads)

    def train(self, mode: bool = True):
        self.dino = self.dino.train(False)
        self.heads = self.heads.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Apply augmentation (x in [-1, 1]).
        if self.diffaug:
            x = DiffAugment(x, policy='color,translation,cutout')

        # Transform to [0, 1].
        x = x.add(1).div(2)

        # Take crops with probablity p_crop if the image is larger.
        if x.size(-1) > self.dino.img_resolution and np.random.random() < self.p_crop:
            x = RandomCrop(self.dino.img_resolution)(x)

        # Forward pass through DINO ViT.
        features = self.dino(x)

        # Apply discriminator heads.
        logits = []
        for k, head in self.heads.items():
            logits.append(head(features[k], c).view(x.size(0), -1))
        logits = torch.cat(logits, dim=1)

        return logits
