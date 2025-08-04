import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import save_image
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from .diffusion.respace import space_timesteps
from .diffusion import create_diffusion

device = torch.device("cuda:0")
diffusion = create_diffusion(timestep_respacing="")

def adjust_brightness_self(adjust_brightness_param, img):
    #print("++++++++++++++++++++++++++++++++++++")
    adjust_brightness_param = adjust_brightness_param.to(img.device).to(img.dtype)
    for _ in img.shape[1:]:
        adjust_brightness_param = torch.unsqueeze(adjust_brightness_param, dim=-1)
    #print("adjust_brightness_param shape:",adjust_brightness_param.shape)
    img=adjust_brightness_param+img
    #img=sudo_sigmoid(img)
    #img=(torch.sigmoid(img) - 0.2689) / 0.6118
    #print("img shape:",img.shape)
    return img

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, batch_size, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Linear(batch_size, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        #print("label shape:",labels.shape)
        use_dropout = self.dropout_prob > 0
        #if (train and use_dropout) or (force_drop_ids is not None):
        #    labels = self.token_drop(labels, force_drop_ids)
        #print("label shape:",labels.shape)
        embeddings = self.embedding_table(labels)
        #print("brightness label shape after embeeding",embeddings.shape)

        return embeddings

# Function to calculate brightness of a batch of images
def calculate_brightness(images):
    """
    Calculates the brightness of a batch of images.
    :param images: A batch of images in tensor format.
    :return: A tensor containing the brightness values of each image in the batch.
    """
    # Convert images to grayscale
    gray_images = torch.mean(images, dim=1, keepdim=True)
    # Calculate brightness as the mean pixel value of the grayscale images
    brightness = torch.mean(gray_images, dim=(1, 2, 3))
    return brightness

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU()
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        #print("c shape:",c.shape)
        #stop
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels,num_classes):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear_final = nn.Linear(64, num_classes, bias=True)
        self.linear = nn.Linear(hidden_size, num_classes, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size,2 * hidden_size , bias=True)
        )
        self.sigmoid = nn.Sequential(
            nn.Sigmoid()
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        x = x.view(x.size(0), -1)
        #print("x shape:",x.shape)
        x=self.linear_final(x)
        x=self.sigmoid(x)
        return x

def Encoder(x):
    noise=torch.rand(x.shape[0])
    #noise=(noise-0.5)*2
    #noise=(noise+1)/4
    noise=noise+0.5
    return noise

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=256,
        patch_size=32,
        in_channels=3,
        hidden_size=256,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1,
        batch_size=8,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, batch_size,hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels,num_classes)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        image=x
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y
        #print("t shape:",t.shape)
        #print("x shape:",x.shape)# (N, D)
        #print("c shape:",c.shape)
        #stop
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = torch.squeeze(self.final_layer(x, c))
        x=(x-0.5)*2

        return x

class DiffusionModel_Bright(nn.Module):
    def __init__(self, num_encoders=3, num_decoders=3,dix=0):
        super().__init__()
        self.num_encoders = num_encoders
        self.num_decoders = num_decoders
        self.decoder1 = DiT()
        self.decoder2 = DiT()
        self.decoder3 = DiT()

    def forward(self, x, idx):
        idx=idx
        image=x
        enocde_param_all=0

        encode_param=Encoder(image)
        image=adjust_brightness_self(encode_param,image)
        enocde_param_all=enocde_param_all+encode_param

        encoded_image=image

        batch_brightness = calculate_brightness(image)
        t = torch.randint(0, diffusion.num_timesteps, (image.shape[0],), device=device)
        decode_param1 = self.decoder1(image, t, batch_brightness)
        image = adjust_brightness_self(decode_param1, image)

        batch_brightness = calculate_brightness(image)
        t = torch.randint(0, diffusion.num_timesteps, (image.shape[0],), device=device)
        decode_param2 = self.decoder2(image, t, batch_brightness)
        image = adjust_brightness_self(decode_param2, image)

        batch_brightness = calculate_brightness(image)
        t = torch.randint(0, diffusion.num_timesteps, (image.shape[0],), device=device)
        decode_param3 = self.decoder3(image, t, batch_brightness)
        image = adjust_brightness_self(decode_param3, image)

        decode_difference=torch.mean(image-encoded_image)
        main_difference=torch.mean(image-x)

        #if idx% 512==0:
            #print("enocde_param_all:",enocde_param_all[3].item())
            #print("decode param:",(decode_param1[3]+decode_param2[3]+decode_param3[3]).item())
            #print("decode_param1:",decode_param1[3].item())
            #print("decode_param2:", decode_param2[3].item())
            #print("decode_param3:", decode_param3[3].item())
            #save_image(x[3], "%s/origin_%d.jpg" % ("brightness",idx),normalize=False)
            #save_image(encoded_image[3], "%s/encoded_%d.jpg" % ("brightness", idx),normalize=False)
            #save_image(image[3], "%s/decoded_%d.jpg" % ("brightness", idx),normalize=False)

        return image










def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def DiT_model(**kwargs):
    return DiT(depth=8, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


