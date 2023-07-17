import torch.nn as nn
import torch, math
from model.net_blocks import *
from model.swin_blocks import BasicLayer, BasicLayer_up
from typing import Any, Optional, Tuple, Type
import numpy as np

class Attn(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    
class SwinAttn(nn.Module):
    def __init__(self, args, num_cls) -> None:
        super().__init__()
        '''
        split image into 16*16/8*8/4*4 tokens, and compute attention and downsample, to generate 8*8 image embeddings (128*128 cropped image, original size is 1024*1024)
        '''
        self.num_token = 8 # 16*16, 8*8, 4*4 token embeddings 

        self.patch_embed = PatchEmbed(img_size=args.imgsz, patch_size=args.tokensz, in_chans=3)

        self.num_layers = int(math.log2((args.imgsz//args.tokensz)//8))
        # encoder
        patches_resolution = self.patch_embed.patches_resolution
        depths = [2, 2, 2, 2]
        num_heads = [3, 6, 12, 24]
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(args.dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=2,
                               num_heads=3,
                               window_size=7,
                               mlp_ratio=4.,
                               qkv_bias=True, qk_scale=None,
                               drop=0., attn_drop=0.,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=nn.LayerNorm,
                               downsample=PatchMerging if (i_layer < self.num_layers) else None,
                               use_checkpoint=False)
            self.layers.append(layer)
        
        self.norm = nn.LayerNorm(args.dim*2**(self.num_layers))
            
        # decoder
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()

        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2*int(args.dim*2**(2-1-i_layer)),
            int(args.dim*2**(self.num_layers-1-i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer ==0 :
                layer_up = PatchExpand(input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))), dim=int(args.dim * 2 ** (self.num_layers-1-i_layer)), dim_scale=2, norm_layer=nn.LayerNorm)
            else:
                layer_up = BasicLayer_up(dim=int(args.dim * 2 ** (self.num_layers-1-i_layer)),
                                input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                                                    patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))),
                                depth=depths[(self.num_layers-1-i_layer)],
                                num_heads=num_heads[(self.num_layers-1-i_layer)],
                                window_size=7,
                                mlp_ratio=4.,
                                qkv_bias=True, qk_scale=None,
                                drop=0., attn_drop=0.,
                                drop_path=dpr[sum(depths[:(self.num_layers-1-i_layer)]):sum(depths[:(self.num_layers-1-i_layer) + 1])],
                                norm_layer=nn.LayerNorm,
                                upsample=PatchExpand if (i_layer < self.num_layers-1) else None,
                                use_checkpoint=False)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm_up = nn.LayerNorm(args.dim)
        self.conv = nn.Conv2d(in_channels=args.dim*2**(self.num_layers),out_channels=args.dim,kernel_size=3,bias=False, padding=1)
        self.output = nn.Conv2d(in_channels=args.dim,out_channels=num_cls,kernel_size=1,bias=False)
        # self.mlp = Mlp(args.dim*(2**3), out_features=1)

    def forward(self, x):
        B, C, H, W = x.shape
        # print(x.shape)
        x = self.patch_embed(x)
        # print(x.shape)

        # encoder
        x_downsample = []
        
        for layer in self.layers:
            x_downsample.append(x) # x_ds[0]:(B, 28*28, 96), x_ds[1]:(B, 14*14, 96*2), 
            x = layer(x)
        x = self.norm(x) # x:(B, 14*14, 96*2)
        # print(x.shape)
        
        # decoder
        # for inx, layer_up in enumerate(self.layers_up):
        #     if inx == 0:
        #         x = layer_up(x)
        #     else:
        #         # x = self.connects[inx - 1](x, x_downsample[3-inx])
        #         x = torch.cat([x,x_downsample[1-inx]],-1)
        #         x = self.concat_back_dim[inx](x)
        #         x = layer_up(x)
        # x = self.norm_up(x)
        # print(torch.unique(x))
        x = self.conv(x.view(B, 96*2**(self.num_layers), self.num_token, self.num_token))
        x = self.output(x.view(B, 96, self.num_token, self.num_token))
        # print(torch.unique(x))
        # x = nn.Sigmoid()(x)

        # x = x.view(B, 1, self.num_token, self.num_token)
        return x

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C
    
class SwinAttn_pyramid(nn.Module):
    def __init__(self, args, num_cls) -> None:
        super().__init__()
        '''
        split image into 16*16/8*8/4*4 tokens, and compute attention and downsample, to generate 8*8 image embeddings (128*128 cropped image, original size is 1024*1024)
        '''
        self.num_token = 8 # 16*16, 8*8, 4*4 token embeddings 

        self.patch_embed = PatchEmbed(img_size=args.imgsz, patch_size=args.tokensz, in_chans=3)

        self.num_layers = int(math.log2((args.imgsz//args.tokensz)//8))
        # encoder
        patches_resolution = self.patch_embed.patches_resolution
        depths = [2, 2, 2, 2, 2, 2]
        num_heads = [3, 6, 12, 24]
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(args.dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=2,
                               num_heads=3,
                               window_size=7,
                               mlp_ratio=4.,
                               qkv_bias=True, qk_scale=None,
                               drop=0., attn_drop=0.,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=nn.LayerNorm,
                               downsample=PatchMerging if (i_layer < self.num_layers) else None,
                               use_checkpoint=False)
            self.layers.append(layer)
        
        self.norm = nn.LayerNorm(args.dim*2**(self.num_layers))
        self.norm64 = nn.LayerNorm(args.dim*2**(self.num_layers - 1))
        self.norm32 = nn.LayerNorm(args.dim*2**(self.num_layers - 2))
            
        # decoder
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()

        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2*int(args.dim*2**(2-1-i_layer)),
            int(args.dim*2**(self.num_layers-1-i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer ==0 :
                layer_up = PatchExpand(input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))), dim=int(args.dim * 2 ** (self.num_layers-1-i_layer)), dim_scale=2, norm_layer=nn.LayerNorm)
            else:
                layer_up = BasicLayer_up(dim=int(args.dim * 2 ** (self.num_layers-1-i_layer)),
                                input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                                                    patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))),
                                depth=depths[(self.num_layers-1-i_layer)],
                                num_heads=num_heads[(self.num_layers-1-i_layer)],
                                window_size=7,
                                mlp_ratio=4.,
                                qkv_bias=True, qk_scale=None,
                                drop=0., attn_drop=0.,
                                drop_path=dpr[sum(depths[:(self.num_layers-1-i_layer)]):sum(depths[:(self.num_layers-1-i_layer) + 1])],
                                norm_layer=nn.LayerNorm,
                                upsample=PatchExpand if (i_layer < self.num_layers-1) else None,
                                use_checkpoint=False)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm_up = nn.LayerNorm(args.dim)
        self.conv = nn.Conv2d(in_channels=args.dim*2**(self.num_layers),out_channels=args.dim,kernel_size=3,bias=False, padding=1)
        self.conv64 = nn.Conv2d(in_channels=args.dim*2**(self.num_layers-1),out_channels=args.dim,kernel_size=3,bias=False, padding=1)
        self.conv32 = nn.Conv2d(in_channels=args.dim*2**(self.num_layers-2),out_channels=args.dim,kernel_size=3,bias=False, padding=1)
        self.output = nn.Conv2d(in_channels=args.dim,out_channels=num_cls,kernel_size=1,bias=False)
        self.output64 = nn.Conv2d(in_channels=args.dim,out_channels=num_cls,kernel_size=1,bias=False)
        self.output32 = nn.Conv2d(in_channels=args.dim,out_channels=num_cls,kernel_size=1,bias=False)
        # self.mlp = Mlp(args.dim*(2**3), out_features=1)

    def forward(self, x):
        B, C, H, W = x.shape
        # print(x.shape)
        x = self.patch_embed(x)
        # print(x.shape)

        # encoder
        x_downsample = []
        
        for layer in self.layers:
            x_downsample.append(x) # x_ds[0]:(B, 28*28, 96), x_ds[1]:(B, 14*14, 96*2), 
            x = layer(x)
        x = self.norm(x) # x:(B, 14*14, 96*2)
        # print(x.shape)
        x64 = self.norm64(x_downsample[-1])
        x32 = self.norm32(x_downsample[-2])
        

        # decoder
        # for inx, layer_up in enumerate(self.layers_up):
        #     if inx == 0:
        #         x = layer_up(x)
        #     else:
        #         # x = self.connects[inx - 1](x, x_downsample[3-inx])
        #         x = torch.cat([x,x_downsample[1-inx]],-1)
        #         x = self.concat_back_dim[inx](x)
        #         x = layer_up(x)
        # x = self.norm_up(x)
        # print(torch.unique(x))
        x = self.conv(x.view(B, 96*2**(self.num_layers), self.num_token, self.num_token))
        x = self.output(x.view(B, 96, self.num_token, self.num_token))
        # print(torch.unique(x))
        # x = nn.Sigmoid()(x)
        x64 = self.conv64(x64.view(B, 96*2**(self.num_layers-1), self.num_token*2, self.num_token*2))
        x64 = self.output64(x64.view(B, 96, self.num_token*2, self.num_token*2))
        x32 = self.conv32(x32.view(B, 96*2**(self.num_layers-2), self.num_token*4, self.num_token*4))
        x32 = self.output32(x32.view(B, 96, self.num_token*4, self.num_token*4))
        # x = x.view(B, 1, self.num_token, self.num_token)
        return x, x64, x32
    
class SwinAttn_pyramid_sc(nn.Module):
    def __init__(self, args, num_cls) -> None:
        super().__init__()
        '''
        split image into 16*16/8*8/4*4 tokens, and compute attention and downsample, to generate 8*8 image embeddings (128*128 cropped image, original size is 1024*1024)
        '''
        self.num_token = 8 # 16*16, 8*8, 4*4 token embeddings 

        self.patch_embed = PatchEmbed(img_size=args.imgsz, patch_size=args.tokensz, in_chans=3)

        self.num_layers = int(math.log2((args.imgsz//args.tokensz)//8))
        # encoder
        patches_resolution = self.patch_embed.patches_resolution
        depths = [2, 2, 2, 2, 2, 2]
        num_heads = [3, 6, 12, 24]
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(args.dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=2,
                               num_heads=3,
                               window_size=7,
                               mlp_ratio=4.,
                               qkv_bias=True, qk_scale=None,
                               drop=0., attn_drop=0.,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=nn.LayerNorm,
                               downsample=PatchMerging if (i_layer < self.num_layers) else None,
                               use_checkpoint=False)
            self.layers.append(layer)
        
        self.norm = nn.LayerNorm(args.dim*2**(self.num_layers))
        self.norm64 = nn.LayerNorm(args.dim*2**(self.num_layers - 1))
        self.norm32 = nn.LayerNorm(args.dim*2**(self.num_layers - 2))
            
        # decoder
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()

        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2*int(args.dim*2**(2-1-i_layer)),
            int(args.dim*2**(self.num_layers-1-i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer ==0 :
                layer_up = PatchExpand(input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-i_layer)),
                patches_resolution[1] // (2 ** (self.num_layers-i_layer))), dim=int(args.dim * 2 ** (self.num_layers-i_layer)), dim_scale=2, norm_layer=nn.LayerNorm)
            else:
                layer_up = BasicLayer_up(dim=int(args.dim * 2 ** (self.num_layers-i_layer)),
                                input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-i_layer)),
                                                    patches_resolution[1] // (2 ** (self.num_layers-i_layer))),
                                depth=depths[(self.num_layers-1-i_layer)],
                                num_heads=num_heads[(self.num_layers-1-i_layer)],
                                window_size=7,
                                mlp_ratio=4.,
                                qkv_bias=True, qk_scale=None,
                                drop=0., attn_drop=0.,
                                drop_path=dpr[sum(depths[:(self.num_layers-i_layer)]):sum(depths[:(self.num_layers-i_layer) + 1])],
                                norm_layer=nn.LayerNorm,
                                upsample=PatchExpand if (i_layer < self.num_layers) else None,
                                use_checkpoint=False)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm_up = nn.LayerNorm(args.dim)
        self.conv = nn.Conv2d(in_channels=args.dim*2**(self.num_layers),out_channels=args.dim,kernel_size=3,bias=False, padding=1)
        self.conv64 = nn.Conv2d(in_channels=args.dim*2**(self.num_layers-1),out_channels=args.dim,kernel_size=3,bias=False, padding=1)
        self.conv32 = nn.Conv2d(in_channels=args.dim*2**(self.num_layers-2),out_channels=args.dim,kernel_size=3,bias=False, padding=1)
        self.output = nn.Conv2d(in_channels=args.dim,out_channels=num_cls,kernel_size=1,bias=False)
        self.output64 = nn.Conv2d(in_channels=args.dim,out_channels=num_cls,kernel_size=1,bias=False)
        self.output32 = nn.Conv2d(in_channels=args.dim,out_channels=num_cls,kernel_size=1,bias=False)
        # self.mlp = Mlp(args.dim*(2**3), out_features=1)

        self.concat_64_128 = nn.Linear(args.dim*2**(self.num_layers),args.dim*2**(self.num_layers - 1))
        self.concat_32_64 = nn.Linear(args.dim*2**(self.num_layers - 1),args.dim*2**(self.num_layers - 2))

    def forward(self, x):
        B, C, H, W = x.shape
        # print(x.shape)
        x = self.patch_embed(x)
        # print(x.shape)

        # encoder
        x_downsample = []
        
        for layer in self.layers:
            x_downsample.append(x) # x_ds[0]:(B, 28*28, 96), x_ds[1]:(B, 14*14, 96*2), 
            x = layer(x)
        x = self.norm(x) # x:(B, 14*14, 96*2)
        # print(x.shape)
        # x64 = self.norm64(x_downsample[-1])
        # x32 = self.norm32(x_downsample[-2])
        # print(x.shape)
        x64up = self.layers_up[0](x)
        x64up = torch.cat([x64up, x_downsample[-1]], -1)
        x64up = self.concat_64_128(x64up)
        x64 = self.norm64(x64up)

        x32up = self.layers_up[1](x64up)
        x32up = torch.cat([x32up, x_downsample[-2]], -1)
        x32up = self.concat_32_64(x32up)
        x32 = self.norm32(x32up)
        # decoder
        # for inx, layer_up in enumerate(self.layers_up):
        #     if inx == 0:
        #         x = layer_up(x)
        #     else:
        #         # x = self.connects[inx - 1](x, x_downsample[3-inx])
        #         x = torch.cat([x,x_downsample[1-inx]],-1)
        #         x = self.concat_back_dim[inx](x)
        #         x = layer_up(x)
        # x = self.norm_up(x)
        # print(torch.unique(x))
        x = self.conv(x.view(B, 96*2**(self.num_layers), self.num_token, self.num_token))
        x = self.output(x.view(B, 96, self.num_token, self.num_token))
        # print(torch.unique(x))
        # x = nn.Sigmoid()(x)
        x64 = self.conv64(x64.view(B, 96*2**(self.num_layers-1), self.num_token*2, self.num_token*2))
        x64 = self.output64(x64.view(B, 96, self.num_token*2, self.num_token*2))
        x32 = self.conv32(x32.view(B, 96*2**(self.num_layers-2), self.num_token*4, self.num_token*4))
        x32 = self.output32(x32.view(B, 96, self.num_token*4, self.num_token*4))
        # x = x.view(B, 1, self.num_token, self.num_token)
        return x, x64, x32

class mask_token_inference(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sigmoid = nn.Sigmoid()

    def forward(self, fea):
        B, N, C = fea.shape
        x = self.norm(fea)
        T_s, F_s = x[:, 0, :].unsqueeze(1), x[:, 1:, :]
        # T_s [B, 1, 384]  F_s [B, 14*14, 384]

        q = self.q(F_s).reshape(B, N-1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(T_s).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(T_s).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = self.sigmoid(attn)
        attn = self.attn_drop(attn)

        infer_fea = (attn @ v).transpose(1, 2).reshape(B, N-1, C)
        infer_fea = self.proj(infer_fea)
        infer_fea = self.proj_drop(infer_fea)

        infer_fea = infer_fea + fea[:, 1:, :]
        return infer_fea

class SwinAttn_pyramid_token(nn.Module):
    def __init__(self, args, num_cls) -> None:
        super().__init__()
        '''
        split image into 16*16/8*8/4*4 tokens, and compute attention and downsample, to generate 8*8 image embeddings (128*128 cropped image, original size is 1024*1024)
        '''
        self.num_token = 8 # 16*16, 8*8, 4*4 token embeddings 

        self.patch_embed = PatchEmbed(img_size=args.imgsz, patch_size=args.tokensz, in_chans=3)

        self.num_layers = int(math.log2((args.imgsz//args.tokensz)//8))
        # encoder
        patches_resolution = self.patch_embed.patches_resolution
        depths = [2, 2, 2, 2, 2, 2]
        num_heads = [3, 6, 12, 24]
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(args.dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=2,
                               num_heads=3,
                               window_size=7,
                               mlp_ratio=4.,
                               qkv_bias=True, qk_scale=None,
                               drop=0., attn_drop=0.,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=nn.LayerNorm,
                               downsample=PatchMerging if (i_layer < self.num_layers) else None,
                               use_checkpoint=False)
            self.layers.append(layer)
        
        self.norm = nn.LayerNorm(args.dim*2**(self.num_layers))
        self.norm64 = nn.LayerNorm(args.dim*2**(self.num_layers - 1))
        self.norm32 = nn.LayerNorm(args.dim*2**(self.num_layers - 2))
            
        # decoder
        self.mask_token = nn.Embedding(1, 96*(2**self.num_layers))
        self.fea_mlp128 = Mlp(in_features=96*(2**self.num_layers), hidden_features=96*(2**self.num_layers), out_features=96*(2**self.num_layers))
        self.encoder128 = BasicLayer(dim=int(args.dim * 2 ** 5),
                               input_resolution=(patches_resolution[0] // (2 ** 5),
                                                 patches_resolution[1] // (2 ** 5)),
                               depth=2,
                               num_heads=6,
                               window_size=7,
                               mlp_ratio=4.,
                               qkv_bias=True, qk_scale=None,
                               drop=0., attn_drop=0.,
                               drop_path=dpr[sum(depths[:5]):sum(depths[:5 + 1])],
                               norm_layer=nn.LayerNorm,
                               downsample=None,
                               use_checkpoint=False)
        self.mask_pre128 = mask_token_inference(dim=96*(2**self.num_layers), num_heads=1)
        self.mlp_norm128 = nn.LayerNorm(96*(2**self.num_layers))
        self.mlp128 = Mlp(in_features=96*(2**self.num_layers), hidden_features=96, out_features=96)
        self.linear128 = nn.Linear(96, num_cls)

        # self.layer_up64 = 
        self.fea_mlp64 = Mlp(in_features=96*(2**(self.num_layers-1)), hidden_features=96*(2**self.num_layers), out_features=96*(2**self.num_layers))
        self.encoder64 = BasicLayer(dim=int(args.dim * 2 ** 5),
                               input_resolution=(patches_resolution[0] // (2 ** 4),
                                                 patches_resolution[1] // (2 ** 4)),
                               depth=2,
                               num_heads=6,
                               window_size=7,
                               mlp_ratio=4.,
                               qkv_bias=True, qk_scale=None,
                               drop=0., attn_drop=0.,
                               drop_path=dpr[sum(depths[:4]):sum(depths[:4 + 1])],
                               norm_layer=nn.LayerNorm,
                               downsample=None,
                               use_checkpoint=False)
        self.mask_pre64 = mask_token_inference(dim=96*(2**self.num_layers), num_heads=1)
        self.mlp_norm64 = nn.LayerNorm(96*(2**self.num_layers))
        self.mlp64 = Mlp(in_features=96*(2**self.num_layers), hidden_features=96, out_features=96)
        self.linear64 = nn.Linear(96, num_cls)

        self.fea_mlp32 = Mlp(in_features=96*(2**(self.num_layers-2)), hidden_features=96*(2**self.num_layers), out_features=96*(2**self.num_layers))
        self.encoder32 = BasicLayer(dim=int(args.dim * 2 ** 5),
                               input_resolution=(patches_resolution[0] // (2 ** 3),
                                                 patches_resolution[1] // (2 ** 3)),
                               depth=2,
                               num_heads=6,
                               window_size=7,
                               mlp_ratio=4.,
                               qkv_bias=True, qk_scale=None,
                               drop=0., attn_drop=0.,
                               drop_path=dpr[sum(depths[:3]):sum(depths[:3 + 1])],
                               norm_layer=nn.LayerNorm,
                               downsample=None,
                               use_checkpoint=False)
        self.mask_pre32 = mask_token_inference(dim=96*(2**self.num_layers), num_heads=1)
        self.mlp_norm32 = nn.LayerNorm(96*(2**self.num_layers))
        self.mlp32 = Mlp(in_features=96*(2**self.num_layers), hidden_features=96, out_features=96)
        self.linear32 = nn.Linear(96, num_cls)
        

    def forward(self, x):
        B, C, H, W = x.shape
        # print(x.shape)
        x = self.patch_embed(x)
        # print(x.shape)

        # encoder
        x_downsample = []
        
        for layer in self.layers:
            x_downsample.append(x) # x_ds[0]:(B, 28*28, 96), x_ds[1]:(B, 14*14, 96*2), 
            x = layer(x)
        x = self.norm(x) # x:(B, 14*14, 96*2)
        # print(x.shape)
        x64 = self.norm64(x_downsample[-1])
        x32 = self.norm32(x_downsample[-2])
        
        # decoder
        mask_tokens = self.mask_token.weight
        mask_tokens = mask_tokens.unsqueeze(0).expand(B, -1, -1)

        # predict 8*8 mask
        fea_128 = torch.cat((mask_tokens, self.fea_mlp128(x)), dim=1)
        # fea_128 = self.encoder128(fea_128) # [B, 64+1, 96*32]

        mask_tokens = fea_128[:, 0, :].unsqueeze(1)
        mask_128 = self.mask_pre128(fea_128) # [B, 64, 96*32]
        mask_128 = self.mlp128(self.mlp_norm128(mask_128))
        mask_128 = self.linear128(mask_128)
        B, N, C = mask_128.shape
        mask_128 = mask_128.transpose(1, 2).reshape(B, C, 8, 8)
        
        # predict 16*16 mask
        # fea_128_up = fea_128[:, 1:, :]
        # fea_128_up = 
        fea_64 = torch.cat((mask_tokens, self.fea_mlp64(x64)), dim=1)
        # fea_64 = self.encoder64(fea_64) 

        mask_tokens = fea_64[:, 0, :].unsqueeze(1)
        mask_64 = self.mask_pre64(fea_64) 
        mask_64 = self.mlp64(self.mlp_norm64(mask_64))
        mask_64 = self.linear64(mask_64)
        mask_64 = mask_64.transpose(1, 2).reshape(B, C, 16, 16)

        # predict 32*32 mask
        fea_32 = torch.cat((mask_tokens, self.fea_mlp32(x32)), dim=1)
        # fea_32 = self.encoder32(fea_32) 

        mask_tokens = fea_32[:, 0, :].unsqueeze(1)
        mask_32 = self.mask_pre32(fea_32) 
        mask_32 = self.mlp32(self.mlp_norm32(mask_32))
        mask_32 = self.linear32(mask_32)
        mask_32 = mask_32.transpose(1, 2).reshape(B, C, 32, 32)
        
        return mask_128, mask_64, mask_32