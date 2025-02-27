import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath, trunc_normal_
from mamba_ssm import Mamba  # Ensure mamba_ssm is installed and accessible

# Depth-Wise Convolution Module
class DepthwiseConvolution(nn.Module):
    def __init__(self, dim=None):
        super(DepthwiseConvolution, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x, H, W):
        B, N, C = x.shape
        if N != H * W:
            raise ValueError(f"DepthwiseConvolution expects N=H*W, but got N={N}, HxW={H}x{W}")
        x = x.transpose(1, 2).view(B, C, H, W)  #(B, C, H, W)
        x = F.layer_norm(x, [H, W])            # LN over spatial dims
        x = self.dwconv(x)                     # Depthwise conv
        x = x.flatten(2).transpose(1, 2)       # (B, N, C)
        return x

# Mamba SSM-based Layer Wrapper
class MambaSSMLayer(nn.Module):
    """
    A thin wrapper around the Mamba SSM-based layer.
    Expects input as either (B, C, H, W) or (B, N, C).
    Internally reshapes (B, C, H, W) -> (B, N, C) -> Mamba -> (B, N, C).
    Output shape matches the input dimension form:
       - If input is 4D, returns (B, N, C).
       - If input is 3D, returns (B, N, C).
    """
    def __init__(self, d_model, d_state, d_conv, expand):
        super(MambaSSMLayer, self).__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        reshaped_4d = (x.dim() == 4)
        if reshaped_4d:
            B, C, H, W = x.shape
            x = x.view(B, C, H * W).transpose(1, 2)  # (B, N, C)
        elif x.dim() != 3:
            raise ValueError(
                f"MambaSSMLayer expects input with 3 or 4 dimensions, got {x.dim()}"
            )
        # (B, N, C)
        x = self.mamba(x)
        return x

# Local-Global Feature Mixing Module
class LocalGlobalFeatureMixing(nn.Module):
    def __init__(self, input_dim, num_heads=8):
        super(LocalGlobalFeatureMixing, self).__init__()
        self.local_conv = nn.Conv1d(
            input_dim, input_dim, kernel_size=3, padding=1, groups=input_dim
        )
        self.global_attn = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(input_dim)
        self.fc = nn.Linear(input_dim * 2, input_dim)
        self.act = nn.GELU()
        self.dwconv = DepthwiseConvolution(input_dim)

    def forward(self, x):
        # x: (B, N, C)
        # Local branch
        x_local = x.transpose(1, 2)  # (B, C, N)
        x_local = self.local_conv(x_local)
        x_local = x_local.transpose(1, 2)  # (B, N, C)

        # Global branch (Self-attention)
        x_global, _ = self.global_attn(x, x, x)

        # Combine local and global
        x_cat = torch.cat([x_local, x_global], dim=-1)  # (B, N, 2C)
        x_out = self.fc(x_cat)                          #(B, N, C)
        x_out = self.act(self.norm(x_out))

        # Apply DepthwiseConvolution for enhanced spatial feature extraction
        B, N, C = x_out.shape
        H = W = int(math.sqrt(N))
        if H * W != N:
            raise ValueError(
                f"Feature map is not square. Expecting N=H*W. Got N={N}, H*W={H*W}."
            )
        x_out = self.dwconv(x_out, H, W)  # (B, N, C)

        return x_out

# Adaptive Multi-Branch Mamba Feature Fusion
class AdaptiveMultiBranchMambaFeatureFusion(nn.Module):
    """
    Takes (B, C, H, W) as input and internally reshapes to (B, H*W, C),
    splits channels across multiple branches, passes them through MambaSSMLayer,
    merges again, and outputs (B, C_out, H, W).
    """
    def __init__(self, input_dim, output_dim, num_branches=4, d_state=16, d_conv=4, expand=2):
        super(AdaptiveMultiBranchMambaFeatureFusion, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_branches = num_branches
        self.norm = nn.LayerNorm(input_dim)

        assert input_dim % num_branches == 0, (
            f"input_dim({input_dim}) must be divisible by num_branches({num_branches})"
        )
        self.mamba_layers = nn.ModuleList([ 
            MambaSSMLayer(
                d_model=input_dim // num_branches,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            ) for _ in range(num_branches)
        ])

        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

        # Adaptive Gating Mechanisms
        # Q-branch gating
        self.conv_q_dw = nn.Conv1d(
            in_channels=input_dim, 
            out_channels=num_branches, 
            kernel_size=3, 
            padding=1, 
            groups=num_branches
        )
        self.conv_q_pw = nn.Conv1d(
            in_channels=num_branches, 
            out_channels=num_branches, 
            kernel_size=1
        )
        # R-branch gating
        self.conv_r_dw = nn.Conv1d(
            in_channels=input_dim, 
            out_channels=input_dim, 
            kernel_size=3, 
            padding=1, 
            groups=input_dim
        )
        self.conv_r_pw = nn.Conv1d(
            in_channels=input_dim, 
            out_channels=input_dim, 
            kernel_size=1
        )
        # Final depthwise conv
        self.dw_conv_final_dw = nn.Conv1d(
            in_channels=2 * input_dim, 
            out_channels=2 * input_dim, 
            kernel_size=3, 
            padding=1, 
            groups=2 * input_dim
        )
        self.dw_conv_final_pw = nn.Conv1d(
            in_channels=2 * input_dim, 
            out_channels=input_dim, 
            kernel_size=1
        )
        self.sigmoid = nn.Sigmoid()

        # Local-Global Feature Mixing
        self.local_global_mixing = LocalGlobalFeatureMixing(input_dim)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        N = H * W

        # Flatten => (B, N, C)
        x = x.view(B, C, N).transpose(1, 2)
        x_norm = self.norm(x)

        # Split into multiple branches along channel dimension
        x_chunks = torch.chunk(x_norm, self.num_branches, dim=2)

        # Pass each chunk through MambaSSMLayer + skip
        x_mamba_chunks = [
            self.mamba_layers[i](x_chunk) + self.skip_scale * x_chunk
            for i, x_chunk in enumerate(x_chunks)
        ]

        # Concatenate => (B, N, C)
        x_concat = torch.cat(x_mamba_chunks, dim=2)

        # Adaptive Gating: Q
        x_concat_t = x_concat.transpose(1, 2)  # (B, C, N)
        Q_dw = self.conv_q_dw(x_concat_t)      # (B, num_branches, N)
        Q = self.conv_q_pw(Q_dw)               # (B, num_branches, N)
        g = self.sigmoid(Q)                    # (B, num_branches, N)

        # Gate each chunk
        g_chunks = torch.chunk(g, self.num_branches, dim=1)
        x_mamba_chunks_gated = [
            g_chunk.permute(0, 2, 1) * x_mamba_chunk
            for g_chunk, x_mamba_chunk in zip(g_chunks, x_mamba_chunks)
        ]
        x_gated = torch.cat(x_mamba_chunks_gated, dim=2)  # (B, N, C)

        # Adaptive Gating: R
        x_gated_t = x_gated.transpose(1, 2)    # (B, C, N)
        x_gated_t_residual = x_gated_t
        R_dw = self.conv_r_dw(x_gated_t)       # (B, C, N)
        R = self.conv_r_pw(R_dw)               # (B, C, N)
        R = R + x_gated_t_residual             # (B, C, N)
        R = R.transpose(1, 2)                  # (B, N, C)

        # Final depthwise conv
        concat_NR = torch.cat([x, R], dim=2)         # (B, N, 2C)
        concat_NR_t = concat_NR.transpose(1, 2)      # (B, 2C, N)
        x_mamba_t_dw = self.dw_conv_final_dw(concat_NR_t)   # (B, 2C, N)
        x_mamba_t = self.dw_conv_final_pw(x_mamba_t_dw)     # (B, C, N)
        # Residual from only the original x portion (first C channels)
        x_mamba_t = x_mamba_t + concat_NR_t[:, :C, :]
        x_mamba = x_mamba_t.transpose(1, 2)          # (B, N, C)

        # Local-Global Feature Mixing
        x_mixed = self.local_global_mixing(x_mamba)

        # Final LN + projection
        x_mixed = self.norm(x_mixed)
        x_mixed = self.proj(x_mixed)

        # Reshape back to (B, output_dim, H, W)
        out = x_mixed.transpose(1, 2).view(B, self.output_dim, H, W)
        return out

# Cross-Gated Attention Gate using Depthwise Convolutions
class CrossGatedAttentionGate(nn.Module):
    """
    A refined attention gate that applies cross-gating between g and x
    on a per-branch basis, then merges the results to produce a final
    single-channel attention mask. That mask is applied to x (the skip connection).
    """
    def __init__(self, F_g, F_l, F_int, num_branches=4, d_state=16, d_conv=4, expand=2):
        super(CrossGatedAttentionGate, self).__init__()
        self.F_g = F_g
        self.F_l = F_l
        self.F_int = F_int
        self.num_branches = num_branches

        # MambaSSMLayer for gating signal g
        assert F_g % num_branches == 0, f"F_g={F_g} must be divisible by num_branches={num_branches}"
        self.mamba_g = nn.ModuleList([
            MambaSSMLayer(
                d_model=F_g // num_branches,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            ) for _ in range(num_branches)
        ])

        # MambaSSMLayer for skip connection x
        assert F_l % num_branches == 0, f"F_l={F_l} must be divisible by num_branches={num_branches}"
        self.mamba_x = nn.ModuleList([
            MambaSSMLayer(
                d_model=F_l // num_branches,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            ) for _ in range(num_branches)
        ])

        # Replace lightweight 1x1 conv with depthwise conv for each branch
        self.conv_g_dw = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(F_g // num_branches, F_g // num_branches, kernel_size=3, stride=1, padding=1, groups=F_g // num_branches),
                nn.BatchNorm2d(F_g // num_branches),
                nn.ReLU(inplace=True)
            ) for _ in range(num_branches)
        ])
        self.conv_x_dw = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(F_l // num_branches, F_l // num_branches, kernel_size=3, stride=1, padding=1, groups=F_l // num_branches),
                nn.BatchNorm2d(F_l // num_branches),
                nn.ReLU(inplace=True)
            ) for _ in range(num_branches)
        ])

        # Project cross_i => F_int using depthwise then pointwise conv => BN => ReLU
        self.branch_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(F_g // num_branches, F_g // num_branches, kernel_size=3, stride=1, padding=1, groups=F_g // num_branches),
                nn.BatchNorm2d(F_g // num_branches),
                nn.ReLU(inplace=True),
                nn.Conv2d(F_g // num_branches, F_int, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(F_int),
                nn.ReLU(inplace=True),
            ) for _ in range(num_branches)
        ])

        # Final combination: we get (B, num_branches*F_int, H, W), reduce to (B, F_int, H, W)
        # then produce a single-channel attention mask
        self.combine = nn.Sequential(
            nn.Conv2d(F_int * num_branches, F_int, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(F_int),
            nn.ReLU(inplace=True),
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        """
        g: (B, F_g,H, W)
        x: (B, F_l, H, W)
        returns: x * attention_mask
        """
        B, _, H, W = g.shape

        # Split gating signal and skip connection features into branches
        g_chunks = torch.chunk(g, self.num_branches, dim=1)
        x_chunks = torch.chunk(x, self.num_branches, dim=1)

        # For merging cross-gated branches.
        cross_branch_list = []

        for i in range(self.num_branches):
            # -- Gating branch --
            g_i = g_chunks[i]  # (B, F_g/num_branches, H, W)
            g_i_mamba = self.mamba_g[i](g_i)
            _, n_g, c_g = g_i_mamba.shape
            if n_g != H * W:
                raise ValueError(f"G branch mismatch: N={n_g}, but H*W={H}*{W}={H*W}")
            g_i_mamba = g_i_mamba.transpose(1, 2).view(B, c_g, H, W)

            # -- Skip branch --
            x_i = x_chunks[i]  # (B, F_l/num_branches, H, W)
            x_i_mamba = self.mamba_x[i](x_i)
            _, n_x, c_x = x_i_mamba.shape
            if n_x != H * W:
                raise ValueError(f"X branch mismatch: N={n_x}, but H*W={H}*{W}={H*W}")
            x_i_mamba = x_i_mamba.transpose(1, 2).view(B, c_x, H, W)

            # Cross gating using depthwise convs:
            g_i_conv = self.conv_g_dw[i](g_i_mamba)  # (B, c_g, H, W)
            x_i_conv = self.conv_x_dw[i](x_i_mamba)  # (B, c_x, H, W)
            cross_i = x_i_mamba * torch.sigmoid(g_i_conv) + g_i_mamba * torch.sigmoid(x_i_conv)

            # Project cross_i => F_int with branch_proj[i]
            cross_i_proj = self.branch_proj[i](cross_i)  # (B, F_int, H, W)
            cross_branch_list.append(cross_i_proj)

        # Combine across branches => (B, F_int * num_branches, H, W)
        cross_cat = torch.cat(cross_branch_list, dim=1)

        # Final attention mask
        psi = self.combine(cross_cat)  # (B, 1, H, W)

        # Apply attention mask to skip connection x
        return x * psi


# MambaLiteUNet Model 
class MambaLiteUNet(nn.Module): 
    def __init__(
        self,
        num_classes=1,
        input_channels=3,
        c_list=[16, 32, 48, 64, 96, 128]
    ):
        super(MambaLiteUNet, self).__init__()

        #print(c_list)

        # Encoder layers
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], kernel_size=3, stride=1, padding=1),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], kernel_size=3, stride=1, padding=1),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], kernel_size=3, stride=1, padding=1),
        )
        self.encoder4 = AdaptiveMultiBranchMambaFeatureFusion(
            input_dim=c_list[2],
            output_dim=c_list[3],
            num_branches=4,
            d_state=16,
            d_conv=4,
            expand=2
        )
        self.encoder5 = AdaptiveMultiBranchMambaFeatureFusion(
            input_dim=c_list[3],
            output_dim=c_list[4],
            num_branches=4,
            d_state=16,
            d_conv=4,
            expand=2
        )
        self.encoder6 = AdaptiveMultiBranchMambaFeatureFusion(
            input_dim=c_list[4],
            output_dim=c_list[5],
            num_branches=4,
            d_state=16,
            d_conv=4,
            expand=2
        )

        # Decoder layers
        self.decoder1 = AdaptiveMultiBranchMambaFeatureFusion(
            input_dim=c_list[5],
            output_dim=c_list[4],
            num_branches=4,
            d_state=16,
            d_conv=4,
            expand=2
        )
        self.decoder2 = AdaptiveMultiBranchMambaFeatureFusion(
            input_dim=c_list[4],
            output_dim=c_list[3],
            num_branches=4,
            d_state=16,
            d_conv=4,
            expand=2
        )
        self.decoder3 = AdaptiveMultiBranchMambaFeatureFusion(
            input_dim=c_list[3],
            output_dim=c_list[2],
            num_branches=4,
            d_state=16,
            d_conv=4,
            expand=2
        )
        self.decoder4 = nn.Conv2d(c_list[2], c_list[1], kernel_size=3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(c_list[1], c_list[0], kernel_size=3, stride=1, padding=1)

        # Cross-Gated Attention Gates (with Cross-Gating)
        self.attention5 = CrossGatedAttentionGate(
            F_g=c_list[4],
            F_l=c_list[4],
            F_int=c_list[4] // 2,
            num_branches=4,
            d_state=16,
            d_conv=4,
            expand=2
        )
        self.attention4 = CrossGatedAttentionGate(
            F_g=c_list[3],
            F_l=c_list[3],
            F_int=c_list[3] // 2,
            num_branches=4,
            d_state=16,
            d_conv=4,
            expand=2
        )
        self.attention3 = CrossGatedAttentionGate(
            F_g=c_list[2],
            F_l=c_list[2],
            F_int=c_list[2] // 2,
            num_branches=4,
            d_state=16,
            d_conv=4,
            expand=2
        )
        self.attention2 = CrossGatedAttentionGate(
            F_g=c_list[1],
            F_l=c_list[1],
            F_int=c_list[1] // 2,
            num_branches=4,
            d_state=16,
            d_conv=4,
            expand=2
        )

        # Normalizations
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])

        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        # Final convolution
        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # Encoder
        out = self.ebn1(self.encoder1(x))
        out = F.gelu(F.max_pool2d(out, 2, 2))
        t1 = out

        out = self.ebn2(self.encoder2(out))
        out = F.gelu(F.max_pool2d(out, 2, 2))
        t2 = out

        out = self.ebn3(self.encoder3(out))
        out = F.gelu(F.max_pool2d(out, 2, 2))
        t3 = out

        out = self.ebn4(self.encoder4(out))
        out = F.gelu(F.max_pool2d(out, 2, 2))
        t4 = out

        out = self.ebn5(self.encoder5(out))
        out = F.gelu(F.max_pool2d(out, 2, 2))
        t5 = out
        # Consider as Bottleneck part
        out = self.encoder6(out)
        out = F.gelu(out)

        # Decoder with Cross-Gated Attention
        out5 = self.decoder1(out)
        out5 = F.gelu(self.dbn1(out5))
        att_t5 = self.attention5(g=out5, x=t5)
        out5 = out5 + att_t5

        out4 = self.decoder2(out5)
        out4 = F.gelu(self.dbn2(out4))
        out4 = F.interpolate(out4, scale_factor=2, mode='bilinear', align_corners=True)
        att_t4 = self.attention4(g=out4, x=t4)
        out4 = out4 + att_t4

        out3 = self.decoder3(out4)
        out3 = F.gelu(self.dbn3(out3))
        out3 = F.interpolate(out3, scale_factor=2, mode='bilinear', align_corners=True)
        att_t3 = self.attention3(g=out3, x=t3)
        out3 = out3 + att_t3

        out2 = self.dbn4(self.decoder4(out3))
        out2 = F.gelu(out2)
        out2 = F.interpolate(out2, scale_factor=2, mode='bilinear', align_corners=True)
        att_t2 = self.attention2(g=out2, x=t2)
        out2 = out2 + att_t2

        out1 = self.dbn5(self.decoder5(out2))
        out1 = F.gelu(out1)
        out1 = F.interpolate(out1, scale_factor=2, mode='bilinear', align_corners=True)
        out1 = out1 + t1  

        # Final prediction
        out0 = F.interpolate(self.final(out1), scale_factor=2, mode='bilinear', align_corners=True)
        return torch.sigmoid(out0)
