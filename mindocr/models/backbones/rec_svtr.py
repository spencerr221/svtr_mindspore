import mindspore
from mindspore import Parameter,nn, Tensor
from mindspore.common.initializer import initializer, TruncatedNormal,Normal,Constant,HeUniform
import mindspore.common.initializer as init
import mindspore.numpy as np
import mindspore.ops as ops
from ._registry import register_backbone, register_backbone_class

__all__ = ['SVTRNet', 'rec_svtr']

# trunc_normal_ = TruncatedNormal(sigma=0.02)
normal = Normal
# zeros_ = Constant(value=0)
# ones_ = Constant(value=1)
zeros = ops.Zeros()
gelu=nn.GELU(False)

def drop_path(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = Tensor(1 - drop_prob)
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)  #?
    random_tensor = keep_prob + np.rand(shape, dtype=x.dtype)
    random_tensor = ops.floor(random_tensor)  # binarize

    output = x / (keep_prob) * random_tensor
    return output

class ConvBNLayer(nn.Cell):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 0,
                 bias_attr: bool = False,
                 groups: int = 1,
                 act=gelu):
        super(ConvBNLayer,self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            pad_mode="pad",     #the 'pad' must be zero when 'pad_mode' is not 'pad', but got 'pad': 1 and 'pad_mode': same.
            padding=padding,
            group=groups,
            weight_init=HeUniform(), #nonlinearity='relu'
            # weight_init=3,
            has_bias=bias_attr)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = act

    def construct(self, inputs: Tensor) -> Tensor:
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.act(out)
        return out

class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def construct(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)

class Identity(nn.Cell):
    def __init__(self):
        super(Identity, self).__init__()

    def construct(self, input: Tensor) -> Tensor:
        return input

class Mlp(nn.Cell):
    def __init__(self,
                 in_features: int,
                 hidden_features: int = None,
                 out_features: int = None,
                 act_layer=gelu,
                 drop: float=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features, weight_init='zeros')
        self.act = act_layer
        self.fc2 = nn.Dense(hidden_features, out_features, weight_init='zeros')
        self.drop = nn.Dropout(1-drop)

    def construct(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ConvMixer(nn.Cell):
    def __init__(
            self,
            dim,
            num_heads=8,
            HW=[8, 25],
            local_k=[3, 3], ):
        super().__init__()
        self.HW = HW
        self.dim = dim
        self.local_mixer = nn.Conv2d(
            dim,
            dim,
            local_k,
            1, pad_mode='same',padding=[local_k[0] // 2, local_k[0] // 2,local_k[1] // 2,local_k[1] // 2],   #mindspore needs 4 int in tuple    #TODO: pad_mode
            group=num_heads,
            weight_init=HeUniform()
            # weight_init=3
        )

    def construct(self, x: Tensor) -> Tensor:
        h = self.HW[0]
        w = self.HW[1]
        x = x.transpose([0, 2, 1]).reshape([x.shape[0], self.dim, h, w])  #TODO: transpose
        x = self.local_mixer(x)
        Bm, Cm, Hm, Wm = x.shape
        flat_x = ops.reshape(x, (Bm, Cm, Hm * Wm))
        x=flat_x.transpose([0, 2, 1])
        return x

class Attention(nn.Cell):
    def __init__(self,
                 dim,
                 num_heads=8,
                 mixer='Global',
                 HW=None,
                 local_k=[7, 11],
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Dense(dim, dim * 3, weight_init='zeros', bias_init=qkv_bias)   #bias_attr=bias_init?
        self.attn_drop = nn.Dropout(1-attn_drop)
        self.proj = nn.Dense(dim, dim, weight_init='zeros')
        self.proj_drop = nn.Dropout(1-proj_drop)
        self.HW = HW
        if HW is not None:
            H = HW[0]
            W = HW[1]
            self.N = H * W
            self.C = dim
        if mixer == 'Local' and HW is not None:
            hk = local_k[0]
            wk = local_k[1]
            mask = ops.ones((H * W, H + hk - 1, W + wk - 1), mindspore.float32)
            for h in range(0, H):
                for w in range(0, W):
                    mask[h * W + w, h:h + hk, w:w + wk] = 0.
            mask_mid = mask[:, hk // 2:H + hk // 2, wk // 2:W + wk // 2]
            mask_ms=ops.flatten(mask_mid)
            mask_inf = np.full([H * W, H * W], float('-inf'), dtype='float32')   #param
            mask = np.where(mask_ms < 1, mask_ms, mask_inf)
            # self.mask = mask.unsqueeze([0, 1])    # unsqueeze
            mask_ed_mid = ops.ExpandDims()(mask, 0)
            mask_ed_fin = ops.ExpandDims()(mask_ed_mid, 1)
            self.mask = mask_ed_fin
        self.mixer = mixer

    def construct(self, x: Tensor) -> Tensor:
        if self.HW is not None:
            N = self.N
            C = self.C
        else:
            _, N, C = x.shape
        Bs, Cs, Hs = self.qkv(x).shape
        qkv = ops.reshape(self.qkv(x), (Bs, N, 3, self.num_heads, C // self.num_heads))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn=ops.BatchMatMul()(q, (k.transpose((0, 1, 3, 2))))  # matmul
        if self.mixer == 'Local':
            attn += self.mask
        softmax_fn=ops.Softmax(-1)
        attn = softmax_fn(attn)
        attn = self.attn_drop(attn)
        # x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((0, N, C))
        attn_mat=ops.BatchMatMul()(attn, v)
        attn_mat=ops.transpose(attn_mat,(0, 2, 1, 3))
        Ba, Na, Ca, Ha=attn.shape
        x=ops.reshape(attn_mat,(Ba, N, C))
        # x = attn_mat.transpose((0, 2, 1, 3)).reshape((0, N, C))    #matmul   #TODO transpose

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Cell):
    def __init__(self,
                 dim,
                 num_heads,
                 mixer='Global',
                 local_mixer=[7, 11],
                 HW=None,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=gelu,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-6,
                 prenorm=True):
        super().__init__()
        if isinstance(norm_layer, str):
            self.norm1 = eval(norm_layer)([dim], epsilon=epsilon)
        else:
            self.norm1 = norm_layer([dim])
        if mixer == 'Global' or mixer == 'Local':
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                mixer=mixer,
                HW=HW,
                local_k=local_mixer,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop)
        elif mixer == 'Conv':
            self.mixer = ConvMixer(
                dim, num_heads=num_heads, HW=HW, local_k=local_mixer)
        else:
            raise TypeError("The mixer must be one of [Global, Local, Conv]")

        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        if isinstance(norm_layer, str):
            self.norm2 = eval(norm_layer)([dim], epsilon=epsilon)     #eval
        else:
            self.norm2 = norm_layer([dim])
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
        self.prenorm = prenorm

    def construct(self, x: Tensor) -> Tensor:
        if self.prenorm:
            x = self.norm1(x + self.drop_path(self.mixer(x)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Cell):
    """ Image to Patch Embedding
    """

    def __init__(self,
                 img_size=[32, 100],
                 in_channels=3,
                 embed_dim=768,
                 sub_num=2,
                 patch_size=[4, 4],
                 mode='pope'):
        super().__init__()
        num_patches = (img_size[1] // (2 ** sub_num)) * \
                      (img_size[0] // (2 ** sub_num))
        self.img_size = img_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.norm = None
        if mode == 'pope':
            if sub_num == 2:
                self.proj = nn.SequentialCell([
                    ConvBNLayer(
                        in_channels=in_channels,
                        out_channels=embed_dim // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=gelu,
                        bias_attr=True),   # bias_atrr: none in paddle means zeros
                    ConvBNLayer(
                        in_channels=embed_dim // 2,
                        out_channels=embed_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=gelu,
                        bias_attr=True)])
            if sub_num == 3:
                self.proj = nn.SequentialCell([
                    ConvBNLayer(
                        in_channels=in_channels,
                        out_channels=embed_dim // 4,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=gelu,
                        bias_attr=True),
                    ConvBNLayer(
                        in_channels=embed_dim // 4,
                        out_channels=embed_dim // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=gelu,
                        bias_attr=True),
                    ConvBNLayer(
                        in_channels=embed_dim // 2,
                        out_channels=embed_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=gelu,
                        bias_attr=True)])
        elif mode == 'linear':
            self.proj = nn.Conv2d(
                1, embed_dim, kernel_size=patch_size, stride=patch_size,
                weight_init='xavier_uniform', has_bias=True)
            self.num_patches = img_size[0] // patch_size[0] * img_size[
                1] // patch_size[1]

    def construct(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        mid_x=self.proj(x)
        Bm, Cm, Hm, Wm =mid_x.shape
        flat_x=ops.reshape(mid_x, (Bm, Cm, Hm * Wm))
        x = flat_x.transpose((0, 2, 1))
        return x

class SubSample(nn.Cell):
    def __init__(self,
                 in_channels,
                 out_channels,
                 types='Pool',
                 stride=(2,1),
                 sub_norm='nn.LayerNorm',
                 act=None):
        super().__init__()
        self.types = types
        if types == 'Pool':
            print("oooops")
            # self.avgpool = nn.AvgPool2d(
            #     kernel_size=[3, 5], stride=stride, padding=[1, 2])   #not support list
            # self.maxpool = nn.MaxPool2d(
            #     kernel_size=[3, 5], stride=stride, padding=[1, 2])   #not support list
            # self.proj = nn.Dense(in_channels, out_channels)
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                pad_mode="pad",
                weight_init=HeUniform(),
                has_bias=True
                # weight_init=3
            )
        self.norm = eval(sub_norm)([out_channels])
        if act is not None:
            self.act = act()
        else:
            self.act = None


    def construct(self, x: Tensor) -> Tensor:

        if self.types == 'Pool':
            x1 = self.avgpool(x)
            x2 = self.maxpool(x)
            x = (x1 + x2) * 0.5
            out = self.proj(x.flatten(2).transpose((0, 2, 1)))    #TODO flatten transpose
        else:
            x = self.conv(x)
            Bm, Cm, Hm, Wm = x.shape
            flat_x = ops.reshape(x, (Bm, Cm, Hm * Wm))
            out = flat_x.transpose((0, 2, 1))
        out = self.norm(out)
        if self.act is not None:
            out = self.act(out)

        return out

@register_backbone_class
class SVTRNet(nn.Cell):
    def __init__(
            self,
            img_size=[32, 100],
            in_channels=3,
            embed_dim=[64, 128, 256],
            depth=[3, 6, 3],
            num_heads=[2, 4, 8],
            mixer=['Local'] * 6 + ['Global'] *6,  # Local atten, Global atten, Conv
            local_mixer=[[7, 11], [7, 11], [7, 11]],
            patch_merging='Conv',  # Conv, Pool, None
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            last_drop=0.1,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer='nn.LayerNorm',
            sub_norm='nn.LayerNorm',
            epsilon=1e-6,
            out_channels=192,
            out_char_num=25,
            block_unit='Block',
            act='gelu',
            last_stage=True,
            sub_num=2,
            prenorm=True,
            use_lenhead=False,
            **kwargs):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.prenorm = prenorm
        patch_merging = None if patch_merging != 'Conv' and patch_merging != 'Pool' else patch_merging
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            in_channels=in_channels,
            embed_dim=embed_dim[0],
            sub_num=sub_num)
        num_patches = self.patch_embed.num_patches
        self.HW = [img_size[0] // (2**sub_num), img_size[1] // (2**sub_num)]

        self.pos_embed=Parameter(zeros((1,num_patches,embed_dim[0]),mindspore.float32))

        self.pos_drop = nn.Dropout(keep_prob=1-drop_rate)
        Block_unit = eval(block_unit)
        start=Tensor(0, mindspore.float32)
        stop=Tensor(drop_path_rate, mindspore.float32)
        dpr = ops.linspace(start, stop, num=sum(depth))
        self.blocks1 = nn.CellList([
            Block_unit(
                dim=embed_dim[0],
                num_heads=num_heads[0],
                mixer=mixer[0:depth[0]][i],
                HW=self.HW,
                local_mixer=local_mixer[0],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=eval(act),
                attn_drop=attn_drop_rate,
                drop_path=dpr[0:depth[0]][i],
                norm_layer=norm_layer,
                epsilon=epsilon,
                prenorm=prenorm) for i in range(depth[0])
        ])
        if patch_merging is not None:
            self.sub_sample1 = SubSample(
                embed_dim[0],
                embed_dim[1],
                sub_norm=sub_norm,
                stride=(2, 1),
                types=patch_merging)
            HW = [self.HW[0] // 2, self.HW[1]]
        else:
            HW = self.HW
        self.patch_merging = patch_merging
        self.blocks2 = nn.CellList([
            Block_unit(
                dim=embed_dim[1],
                num_heads=num_heads[1],
                mixer=mixer[depth[0]:depth[0] + depth[1]][i],
                HW=HW,
                local_mixer=local_mixer[1],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=eval(act),
                attn_drop=attn_drop_rate,
                drop_path=dpr[depth[0]:depth[0] + depth[1]][i],
                norm_layer=norm_layer,
                epsilon=epsilon,
                prenorm=prenorm) for i in range(depth[1])
        ])
        if patch_merging is not None:
            self.sub_sample2 = SubSample(
                embed_dim[1],
                embed_dim[2],
                sub_norm=sub_norm,
                stride=(2, 1),
                types=patch_merging)
            HW = [self.HW[0] // 4, self.HW[1]]
        else:
            HW = self.HW
        self.blocks3 = nn.CellList([
            Block_unit(
                dim=embed_dim[2],
                num_heads=num_heads[2],
                mixer=mixer[depth[0] + depth[1]:][i],
                HW=HW,
                local_mixer=local_mixer[2],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=eval(act),
                attn_drop=attn_drop_rate,
                drop_path=dpr[depth[0] + depth[1]:][i],
                norm_layer=norm_layer,
                epsilon=epsilon,
                prenorm=prenorm) for i in range(depth[2])
        ])
        self.last_stage = last_stage
        if last_stage:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, out_char_num))
            self.last_conv = nn.Conv2d(
                in_channels=embed_dim[2],
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                weight_init='xavier_uniform',
                has_bias=False)
            self.hardswish = nn.HSwish()
            self.dropout = nn.Dropout(keep_prob=1-last_drop)   #TODO: no mode
        if not prenorm:
            self.norm = eval(norm_layer)([embed_dim[-1]], epsilon=epsilon)
        self.use_lenhead = use_lenhead
        if use_lenhead:
            self.len_conv = nn.Dense(embed_dim[2], self.out_channels)
            self.hardswish_len = nn.HSwish()
            self.dropout_len = nn.Dropout(
                keep_prob=1-last_drop) #TODO: no mode
        self.pos_embed = init.initializer(TruncatedNormal(sigma=0.02), self.pos_embed.shape, mindspore.float32)
        # self.pos_embed = init.initializer(init.Constant(0), self.pos_embed.shape, mindspore.float32)  TODO fortest
        # trunc_normal_(Tensor(self.pos_embed))
        self._init_weights()

    def _init_weights(self) -> None:
        for name, m in self.cells_and_names():
            if isinstance(m, nn.Dense):
                m.weight.set_data(init.initializer(TruncatedNormal(sigma=0.02), m.weight.shape))
                # m.weight.set_data(init.initializer(init.Constant(1), m.weight.shape)) TODO fortest
                # trunc_normal_(m.weight)
                if m.bias is not None:
                    m.bias.set_data(init.initializer(init.Constant(0), m.bias.shape))
            elif isinstance(m, nn.LayerNorm):
                m.beta.set_data(init.initializer(init.Constant(0), m.beta.shape))
                m.gamma.set_data(init.initializer(init.Constant(1), m.gamma.shape))

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        if self.patch_merging is not None:

            x = self.sub_sample1(
                x.transpose([0, 2, 1]).reshape(
                    [x.shape[0], self.embed_dim[0], self.HW[0], self.HW[1]]))
        for blk in self.blocks2:
            x = blk(x)
        if self.patch_merging is not None:
            x = self.sub_sample2(
                x.transpose([0, 2, 1]).reshape(
                    [x.shape[0], self.embed_dim[1], self.HW[0] // 2, self.HW[1]]))
        for blk in self.blocks3:
            x = blk(x)
        if not self.prenorm:
            x = self.norm(x)
        return x

    def construct(self, x):
        x = self.forward_features(x)
        if self.use_lenhead:
            len_x = self.len_conv(x.mean(1))
            len_x = self.dropout_len(self.hardswish_len(len_x))
        if self.last_stage:
            if self.patch_merging is not None:
                h = self.HW[0] // 4
            else:
                h = self.HW[0]
            x = self.avg_pool(
                x.transpose([0, 2, 1]).reshape([x.shape[0], self.embed_dim[2], h, self.HW[1]]))
            x = self.last_conv(x)
            x = self.hardswish(x)
            x = self.dropout(x)
        # if self.use_lenhead:     #TODO: for graph mode
        #     return x,len_x
        return [x]
    
@register_backbone
def rec_svtr(pretrained: bool = True, **kwargs):
    model = SVTRNet(**kwargs)

    # load pretrained weights
    if pretrained:
        raise NotImplementedError

    return model