from model.net_blocks import BasicLayer, PatchEmbed, Mlp, PatchMerging

import torch, math
import torch.nn as nn

class mask_token_inference(nn.Module):
    r""" cross-attention between classfification token and image representation
    """
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
        # T_s [B, 1, c]  F_s [B, h*w, c]

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
    
class PatchSelector(nn.Module):
    r""" Tile Selection Module
    Split image into non-overlapping tiles, 4*4/8*4..., classify each tile

    Args:
        args.
        num_cls: number of output classes (BCELoss-->1)
    """
    def __init__(self, args, num_cls) -> None:
        super().__init__()
        self.num_patch = args.num_patch # the size of the final encoded representations, i.e. number of tiles (4*4)
        self.patch_embed = PatchEmbed(img_size=args.imgsz, patch_size=args.tokensz, in_chans=3, embed_dim=args.dim)
        self.num_layers = int(math.log2((args.imgsz//args.tokensz)//args.num_patch))

        # encoder
        patches_resolution = self.patch_embed.patches_resolution
        depths = [2]*self.num_layers
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
        
        self.norm3 = nn.LayerNorm(args.dim*2**(self.num_layers))
        self.norm2 = nn.LayerNorm(args.dim*2**(self.num_layers - 1))
        self.norm1 = nn.LayerNorm(args.dim*2**(self.num_layers - 2))
            
        # classifier
        self.mask_token = nn.Embedding(1, args.dim*(2**self.num_layers))
        self.fea_mlp3 = Mlp(in_features=args.dim*(2**self.num_layers), hidden_features=args.dim*(2**self.num_layers), out_features=args.dim*(2**self.num_layers))
        
        self.mask_pre3 = mask_token_inference(dim=args.dim*(2**self.num_layers), num_heads=1)
        self.mlp_norm3 = nn.LayerNorm(args.dim*(2**self.num_layers))
        self.mlp3 = Mlp(in_features=args.dim*(2**self.num_layers), hidden_features=args.dim, out_features=args.dim)
        self.linear3 = nn.Linear(args.dim, num_cls)

        self.fea_mlp2 = Mlp(in_features=args.dim*(2**(self.num_layers-1)), hidden_features=args.dim*(2**self.num_layers), out_features=args.dim*(2**self.num_layers))
        self.mask_pre2 = mask_token_inference(dim=args.dim*(2**self.num_layers), num_heads=1)
        self.mlp_norm2 = nn.LayerNorm(args.dim*(2**self.num_layers))
        self.mlp2 = Mlp(in_features=args.dim*(2**self.num_layers), hidden_features=args.dim, out_features=args.dim)
        self.linear2 = nn.Linear(args.dim, num_cls)

        self.fea_mlp1 = Mlp(in_features=args.dim*(2**(self.num_layers-2)), hidden_features=args.dim*(2**self.num_layers), out_features=args.dim*(2**self.num_layers))
        self.mask_pre1 = mask_token_inference(dim=args.dim*(2**self.num_layers), num_heads=1)
        self.mlp_norm1 = nn.LayerNorm(args.dim*(2**self.num_layers))
        self.mlp1 = Mlp(in_features=args.dim*(2**self.num_layers), hidden_features=args.dim, out_features=args.dim)
        self.linear1 = nn.Linear(args.dim, num_cls)
        

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)

        # encoder
        x_downsample = []
        
        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)
        x3 = self.norm3(x)
        x2 = self.norm2(x_downsample[-1])
        x1 = self.norm1(x_downsample[-2])
        
        # decoder
        mask_tokens = self.mask_token.weight
        mask_tokens = mask_tokens.unsqueeze(0).expand(B, -1, -1)

        # predict 8*8 representations
        fea_3 = torch.cat((mask_tokens, self.fea_mlp3(x3)), dim=1)

        mask_tokens = fea_3[:, 0, :].unsqueeze(1)
        mask_3 = self.mask_pre3(fea_3)
        mask_3 = self.mlp3(self.mlp_norm3(mask_3))
        mask_3 = self.linear3(mask_3)
        B, N, C = mask_3.shape
        mask_3 = mask_3.transpose(1, 2).reshape(B, C, self.num_patch, self.num_patch)
        
        # predict 16*16 representations
        fea_2 = torch.cat((mask_tokens, self.fea_mlp2(x2)), dim=1)

        mask_tokens = fea_2[:, 0, :].unsqueeze(1)
        mask_2 = self.mask_pre2(fea_2) 
        mask_2 = self.mlp2(self.mlp_norm2(mask_2))
        mask_2 = self.linear2(mask_2)
        mask_2 = mask_2.transpose(1, 2).reshape(B, C, self.num_patch*2, self.num_patch*2)

        # predict 32*32 representations
        fea_1 = torch.cat((mask_tokens, self.fea_mlp1(x1)), dim=1)

        mask_tokens = fea_1[:, 0, :].unsqueeze(1)
        mask_1 = self.mask_pre1(fea_1) 
        mask_1 = self.mlp1(self.mlp_norm1(mask_1))
        mask_1 = self.linear1(mask_1)
        mask_1 = mask_1.transpose(1, 2).reshape(B, C, self.num_patch*4, self.num_patch*4)
        
        return mask_3, mask_2, mask_1