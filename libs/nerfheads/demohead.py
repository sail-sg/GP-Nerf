# Copyright 2022 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import spconv
import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.nerfheads.networks import MultiHeadAttention, SparseConvNet

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        "include_input": True,
        "input_dims": input_dims,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


# default tensorflow initialization of linear layers
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


# @torch.jit.script
def fused_mean_variance(x):
    mean = x.mean(-2).unsqueeze(-2)
    var = torch.mean((x - mean) ** 2, dim=2, keepdim=True)
    return mean, var


class NeRFSigmaHead(nn.Module):
    def __init__(
        self,
        in_feat_ch=32,
        n_smpl=6890,
        code_dim=16,
        attn_n_heads=4,
        spconv_n_layers=4,
        spconv_out_dim=[32, 32, 32, 32],
    ):
        super(NeRFSigmaHead, self).__init__()
        self.c = nn.Embedding(n_smpl, code_dim)
        self.xyzc_attn = MultiHeadAttention(
            attn_n_heads,
            code_dim,
            code_dim // attn_n_heads,
            code_dim // attn_n_heads,
            kv_dim=in_feat_ch,
            sum=False,
        )
        self.xyzc_net = SparseConvNet(
            n_layers=spconv_n_layers, in_dim=code_dim, out_dim=spconv_out_dim
        )
        self.out_geometry_fc = nn.Sequential(
            nn.Linear(sum(spconv_out_dim), 64),
            nn.ELU(inplace=True),
        )
        self.out_geometry_fc_out = nn.Sequential(
            nn.Linear(64 + (in_feat_ch + 3) * 2, 64),
            nn.ELU(inplace=True),
            nn.Linear(64, 32),
            nn.ELU(inplace=True),
            nn.Linear(32, 16),
            nn.ELU(inplace=True),
            nn.Linear(16, 1),
            nn.ReLU(),
        )

        self.out_geometry_fc.apply(weights_init)
        self.out_geometry_fc_out.apply(weights_init)

    def forward(self, sp_input, grid_coords, rgb_feat, mask):
        # import pdb; pdb.set_trace()
        grid_coords = grid_coords[:, None, None].float()
        xyzc_features = self.xyzc_net(
            sp_input["xyzc"], grid_coords
        )  # [batchsize, channels, point_nums]
        xyzc_features = xyzc_features.permute(
            0, 2, 1
        ).contiguous()  # [batchsize, point_nums, channels]
        num_views = rgb_feat.shape[2]
        mean, var = fused_mean_variance(rgb_feat)
        globalfeat = torch.cat([mean, var], dim=-1)
        n_rays, n_samples = rgb_feat.shape[:2]
        sigma_feat = self.out_geometry_fc(xyzc_features)
        sigma_feat = sigma_feat.view(n_rays, n_samples, -1)
        globalfeat = torch.cat([sigma_feat.unsqueeze(-2), globalfeat], dim=-1)
        sigma_x = globalfeat.squeeze(2)
        sigma = self.out_geometry_fc_out(sigma_x)
        num_valid_obs = torch.sum(mask, dim=2)
        sigma_out = sigma.masked_fill(
            num_valid_obs < 1, 0.0
        )  # set the sigma of invalid point to zero
        return sigma_feat, sigma_out

    def forward_bk(self, sp_input, grid_coords, smpl_feat_sampled, mask):
        coord = sp_input["coord"]
        out_sh = sp_input["out_sh"]
        batch_size = sp_input["batch_size"]

        code = self.c(torch.arange(0, 6890).to(grid_coords.device))
        code_query = code.unsqueeze(1)
        smpl_feat_sampled = smpl_feat_sampled.flatten(0, 1)
        xyzc_fuse = self.xyzc_attn(code_query, smpl_feat_sampled, smpl_feat_sampled)[
            0
        ].squeeze(1)
        code = xyzc_fuse
        # get latent volume features
        xyzc = spconv.SparseConvTensor(code, coord, out_sh, batch_size)
        grid_coords = grid_coords[:, None, None].float()
        xyzc_features = self.xyzc_net(
            xyzc, grid_coords
        )  # [batchsize, channels, point_nums]
        xyzc_features = xyzc_features.permute(
            0, 2, 1
        ).contiguous()  # [batchsize, point_nums, channels]
        sigma_feat = self.out_geometry_fc(xyzc_features).view(
            -1, mask.shape[1], 1
        )  # [n_rays, n_samples, 1]

        return sigma_feat


class NeRFRGBHead(nn.Module):
    def __init__(self, in_feat_ch=32):
        super(NeRFRGBHead, self).__init__()
        self.view_embedder, self.view_dim = get_embedder(8)
        self.base_fc = nn.Sequential(
            nn.Linear((in_feat_ch + 3) * 3, 64),
            nn.ELU(inplace=True),
            nn.Linear(64, 32),
            nn.ELU(inplace=True),
        )
        self.vis_fc = nn.Sequential(
            nn.Linear(32, 32),
            nn.ELU(inplace=True),
            nn.Linear(32, 32),
            nn.ELU(inplace=True),
        )
        self.rgb_fc = nn.Sequential(
            nn.Linear(96, 32),
            nn.ELU(inplace=True),
            nn.Linear(32, 16),
            nn.ELU(inplace=True),
            nn.Linear(16, 3),
        )

        self.base_fc.apply(weights_init)
        self.vis_fc.apply(weights_init)
        self.rgb_fc.apply(weights_init)

    def forward(self, rgb_feat):
        """
        :param rgb_feat: rgbs and image features [n_rays, n_samples, n_views, n_feat]
        :param mask: mask for whether each projection is valid or not. [n_rays, n_samples, n_views, 1]
        :return: rgb and density output, [n_rays, n_samples, 4]
        """
        num_views = rgb_feat.shape[2]
        rgb_in = rgb_feat[..., :3]
        # compute mean and variance across different views for each point
        mean, var = fused_mean_variance(rgb_feat)  # [n_rays, n_samples, 1, n_feat]
        globalfeat = torch.cat([mean, var], dim=-1)  # [n_rays, n_samples, 1, 2*n_feat]
        x = torch.cat(
            [globalfeat.expand(-1, -1, num_views, -1), rgb_feat], dim=-1
        )  # [n_rays, n_samples, n_views, 3*n_feat]

        x = self.base_fc(x)
        x_vis = self.vis_fc(x * 1.0 / num_views)
        x = x + x_vis

        rgb_out = self.rgb_fc(x.flatten(2, 3)).sigmoid()

        return rgb_in, rgb_out  # , sigma_out


class NeRFHead(nn.Module):
    def __init__(
        self,
        in_feat_ch=32,
        n_smpl=6890,
        code_dim=16,
        attn_n_heads=4,
        spconv_n_layers=4,
        spconv_out_dim=[32, 32, 32, 32],
        use_rgbhead=True,
    ):
        super(NeRFHead, self).__init__()
        self.sigmahead = NeRFSigmaHead(
            in_feat_ch=in_feat_ch,
            n_smpl=n_smpl,
            code_dim=code_dim,
            attn_n_heads=attn_n_heads,
            spconv_n_layers=spconv_n_layers,
            spconv_out_dim=spconv_out_dim,
        )
        self.use_rgbhead = use_rgbhead
        self.rgbhead = NeRFRGBHead(in_feat_ch=in_feat_ch)

    def forward(self, rgb_feat):
        rgb_in, rgb_out = self.rgbhead(rgb_feat)

        return rgb_out, rgb_in

    def forward_bk(
        self, sp_input, grid_coords, smpl_feat_sampled, rgb_feat, viewdir, mask
    ):
        sigma_feat = self.sigmahead(sp_input, grid_coords, smpl_feat_sampled, mask)
        rgb_in, rgb_out, sigma_out = self.rgbhead(rgb_feat, sigma_feat, viewdir, mask)
        out = torch.cat([rgb_out, sigma_out], dim=-1)
        return out, rgb_in


def build_head(cfg):
    head_config = {
        "in_feat_ch": cfg.encoder.out_ch,
        "use_rgbhead": cfg.head.rgb.use_rgbhead,
        "n_smpl": cfg.head.sigma.n_smpl,
        "code_dim": 32,
        "attn_n_heads": cfg.head.sigma.n_heads,
        "spconv_n_layers": cfg.head.sigma.n_layers,
        "spconv_out_dim": cfg.head.sigma.outdims,
    }
    nerfhead = NeRFHead(**head_config)
    return nerfhead
