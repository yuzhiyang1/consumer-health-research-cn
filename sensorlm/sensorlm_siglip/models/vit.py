# Copyright 2024 Big Vision Authors.
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

"""A refactored and simplified ViT.

However, the names of modules are made to match the old ones for easy loading.
"""

from typing import Optional, Sequence, Union

from absl import logging
from big_vision import utils
from big_vision.models import common
import flax
import flax.linen as nn
import flax.training.checkpoints
import jax
import jax.numpy as jnp
import numpy as np
import scipy.ndimage


def posemb_sincos_2d(h, w, width, temperature=10_000., dtype=jnp.float32):
  """Follows the MoCo v3 logic."""
  y, x = jnp.mgrid[:h, :w]

  assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
  omega = jnp.arange(width // 4) / (width // 4 - 1)
  omega = 1. / (temperature**omega)
  y = jnp.einsum("m,d->md", y.flatten(), omega)
  x = jnp.einsum("m,d->md", x.flatten(), omega)
  pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
  return jnp.asarray(pe, dtype)[None, :, :]


def get_posemb(self, typ, seqshape, width, name, dtype=jnp.float32):
  if typ == "learn":
    return self.param(name, nn.initializers.normal(stddev=1/np.sqrt(width)),
                      (1, np.prod(seqshape), width), dtype)
  elif typ == "sincos2d":
    return posemb_sincos_2d(*seqshape, width, dtype=dtype)
  else:
    raise ValueError(f"Unknown posemb type: {typ}")


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""
  mlp_dim: Optional[int] = None  # 默认为 4x 输入阴暗
  dropout: float = 0.0
  dtype_mm: str = "float32"

  @nn.compact
  def __call__(self, x, deterministic=True):
    """Applies Transformer MlpBlock module."""
    inits = dict(
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
    )

    d = x.shape[-1]
    x = nn.Dense(self.mlp_dim or 4 * d, dtype=self.dtype_mm, **inits)(x)
    # 在某些极大规模的情况下，需要从9月起2024:
    x = nn.with_logical_constraint(x, ("act_batch", "act_len", "act_emb"))
    x = nn.gelu(x)
    x = nn.Dropout(rate=self.dropout)(x, deterministic)
    x = nn.Dense(d, dtype=self.dtype_mm, **inits)(x)
    return x


class Encoder1DBlock(nn.Module):
  """Single transformer encoder block (MHSA + MLP)."""
  mlp_dim: Optional[int] = None  # 默认为 4x 输入阴暗
  num_heads: int = 12
  dropout: float = 0.0
  dtype_mm: str = "float32"

  @nn.compact
  def __call__(self, x, deterministic=True):
    out = {}
    x = nn.with_logical_constraint(x, ("act_batch", "act_len", "act_emb"))
    y = nn.LayerNorm()(x)
    y = out["sa"] = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        kernel_init=nn.initializers.xavier_uniform(),
        deterministic=deterministic,
        dtype=self.dtype_mm,
    )(y, y)
    y = nn.with_logical_constraint(y, ("act_batch", "act_len", "act_emb"))
    y = nn.Dropout(rate=self.dropout)(y, deterministic)
    x = out["+sa"] = x + y

    y = nn.LayerNorm()(x)
    y = out["mlp"] = MlpBlock(
        mlp_dim=self.mlp_dim, dropout=self.dropout,
        dtype_mm=self.dtype_mm,
    )(y, deterministic)
    y = nn.with_logical_constraint(y, ("act_batch", "act_len", "act_emb"))
    y = nn.Dropout(rate=self.dropout)(y, deterministic)
    x = out["+mlp"] = x + y
    x = nn.with_logical_constraint(x, ("act_batch", "act_len", "act_emb"))
    return x, out


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation."""
  depth: int
  mlp_dim: Optional[int] = None  # 默认为 4x 输入阴暗
  num_heads: int = 12
  dropout: float = 0.0
  scan: bool = False
  remat_policy: str = "nothing_saveable"
  dtype_mm: str = "float32"

  @nn.compact
  def __call__(self, x, deterministic=True):
    out = {}

    if self.scan:
      block = nn.remat(
          Encoder1DBlock,
          prevent_cse=False,
          static_argnums=(2,),  # 0=self, 2=deterministic
          policy=getattr(jax.checkpoint_policies, self.remat_policy, None),
          )
      x, scan_out = nn.scan(
          block,
          variable_axes={"params": 0},
          split_rngs={"params": True, "dropout": True},
          in_axes=nn.broadcast,
          length=self.depth)(
              name="encoderblock",
              dtype_mm=self.dtype_mm,
              mlp_dim=self.mlp_dim,
              num_heads=self.num_heads,
              dropout=self.dropout)(x, deterministic)
      for lyr in range(self.depth):
        out[f"block{lyr:02d}"] = jax.tree.map(lambda o, l=lyr: o[l], scan_out)
    else:
      # 输入编码器
      for lyr in range(self.depth):
        block_cur = Encoder1DBlock(
            name=f"encoderblock_{lyr}",
            dtype_mm=self.dtype_mm,
            mlp_dim=self.mlp_dim, num_heads=self.num_heads,
            dropout=self.dropout)
        x, out[f"block{lyr:02d}"] = block_cur(x, deterministic)
      out["pre_ln"] = x  # 最后一个街区是别的地方，但里面没有号码

    return nn.LayerNorm(name="encoder_norm")(x), out


class MAPHead(nn.Module):
  """Multihead Attention Pooling."""
  mlp_dim: Optional[int] = None  # 默认为 4x 输入阴暗
  num_heads: int = 12

  @nn.compact
  def __call__(self, x):
    # TODO
    n, l, d = x.shape  # pylint: disable=unused-variable
    probe = self.param("probe", nn.initializers.xavier_uniform(),
                       (1, 1, d), x.dtype)
    probe = jnp.tile(probe, [n, 1, 1])

    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        kernel_init=nn.initializers.xavier_uniform())(probe, x)

    # TODO: 辍学的头？
    y = nn.LayerNorm()(x)
    x = x + MlpBlock(mlp_dim=self.mlp_dim)(y)
    return x[:, 0]


class _Model(nn.Module):
  """ViT model."""

  num_classes: Optional[int] = None
  patch_size: Sequence[int] = (16, 16)
  width: int = 768
  depth: int = 12
  mlp_dim: Optional[int] = None  # 默认为 4x 输入阴暗
  num_heads: int = 12
  posemb: str = "learn"  # 也可以是"银河2d"
  rep_size: Union[int, bool] = False
  dropout: float = 0.0
  pool_type: str = "gap"  # 也可以是"地图"或"tok"
  head_zeroinit: bool = True
  scan: bool = False
  # 或 " dots_with_no_batch_dims_saveable] " ,以达到更高的速度(费用高昂)
  remat_policy: str = "nothing_saveable"
  dtype_mm: str = "float32"

  @nn.compact
  def __call__(self, image, *, train=False):
    out = {}

    image = jnp.asarray(image, self.dtype_mm)

    # 补丁提取
    x = out["stem"] = nn.Conv(
        self.width, self.patch_size, strides=self.patch_size,
        padding="VALID", name="embedding", dtype=self.dtype_mm)(image)

    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    # 在添加额外符号前添加假名( P)。
    x = out["with_posemb"] = x + get_posemb(
        self, self.posemb, (h, w), c, "pos_embedding", x.dtype)

    if self.pool_type == "tok":
      cls = self.param("cls", nn.initializers.zeros, (1, 1, c), x.dtype)
      x = jnp.concatenate([jnp.tile(cls, [n, 1, 1]), x], axis=1)

    n, l, c = x.shape  # pylint: disable=unused-variable
    x = nn.Dropout(rate=self.dropout)(x, not train)

    x, out["encoder"] = Encoder(
        depth=self.depth,
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        dropout=self.dropout,
        scan=self.scan,
        remat_policy=self.remat_policy,
        dtype_mm=self.dtype_mm,
        name="Transformer")(
            x, deterministic=not train)
    encoded = out["encoded"] = x

    if self.pool_type == "map":
      x = out["head_input"] = MAPHead(
          num_heads=self.num_heads, mlp_dim=self.mlp_dim)(x)
    elif self.pool_type == "gap":
      x = out["head_input"] = jnp.mean(x, axis=1)
    elif self.pool_type == "0":
      x = out["head_input"] = x[:, 0]
    elif self.pool_type == "tok":
      x = out["head_input"] = x[:, 0]
      encoded = encoded[:, 1:]
    elif self.pool_type == "none":
      pass
    else:
      raise ValueError(f"Unknown pool type: '{self.pool_type}'")

    x_2d = jnp.reshape(encoded, [n, h, w, -1])

    if self.rep_size:
      rep_size = self.width if self.rep_size is True else self.rep_size
      hid = nn.Dense(rep_size, name="pre_logits")
      # NOTE:过去我们没有将tanh列入pre_logits.
      # 对于少样本,它不应该有什么关系，因为它反正白化了。
      x_2d = nn.tanh(hid(x_2d))
      x = nn.tanh(hid(x))

    out["pre_logits_2d"] = x_2d
    out["pre_logits"] = x

    if self.num_classes:
      kw = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}
      head = nn.Dense(self.num_classes, name="head", **kw)
      x_2d = out["logits_2d"] = head(x_2d)
      x = out["logits"] = head(x)

    return x, out


def Model(num_classes=None, *, variant=None, **kw):  # pylint: disable=invalid-name
  """Factory function, because linen really don't like what I'm doing!"""
  return _Model(num_classes, **{**decode_variant(variant), **kw})


def decode_variant(variant):
  """Converts a string like "B", "B/32", or "B/10/2" into a params dict."""
  if variant is None:
    return {}

  # SensorLM: 添加矩形补丁大小支持
  parts = variant.split("/")
  v = parts[0]
  patch = {}

  if len(parts) == 2:
    patch_size = int(parts[1])
    patch = {"patch_size": (patch_size, patch_size)}
  elif len(parts) == 3:
    patch_h = int(parts[1])
    patch_w = int(parts[2])
    patch = {"patch_size": (patch_h, patch_w)}

  return {
      # pylint:disable=line-too-long
      # 参考:表 2 页:1 https://arxiv.org/abs/2106.04560.
      "width": {"mu": 32, "Ti": 192, "S": 384, "M": 512, "B": 768, "L": 1024, "So400m": 1152, "H": 1280, "g": 1408, "g-opt": 1536, "G": 1664, "G-opt": 1536, "e": 1792}[v],
      "depth": {"mu": 1, "Ti": 12, "S": 12, "M": 12, "B": 12, "L": 24, "So400m": 27, "H": 32, "g": 40, "g-opt": 40, "G": 48, "G-opt": 48, "e": 56}[v],
      "mlp_dim": {"mu": 128, "Ti": 768, "S": 1536, "M": 2048, "B": 3072, "L": 4096, "So400m": 4304, "H": 5120, "g": 6144, "g-opt": 6144, "G": 8192, "G-opt": 8192, "e": 15360}[v],
      "num_heads": {"mu": 2, "Ti": 3, "S": 6, "M": 8, "B": 12, "L": 16, "So400m": 16, "H": 16, "g": 16, "g-opt": 16, "G": 16, "G-opt": 16, "e": 16}[v],
      # pylint:enable=line-too-long
      **patch
  }


def resample_posemb(old, new):
  """This function implements "high-res finetuning" for transformer models."""
  # 调整嵌入位置的网格。Param 形状是( ) 1 ,N, 1024 )
  if old.shape == new.shape:
    return old

  logging.info("ViT: resize %s to %s", old.shape, new.shape)
  gs_old = int(np.sqrt(old.shape[1]))
  gs_new = int(np.sqrt(new.shape[1]))
  logging.info("ViT: grid-size from %s to %s", gs_old, gs_new)
  grid = old.reshape(gs_old, gs_old, -1)

  zoom = (gs_new/gs_old, gs_new/gs_old, 1)
  grid = scipy.ndimage.zoom(grid, zoom, order=1)
  grid = grid.reshape(1, gs_new*gs_new, -1)
  return grid


def fix_old_checkpoints(params):
  """Fix small bwd incompat that can't be resolved with names in model def."""

  params = flax.core.unfreeze(
      flax.training.checkpoints.convert_pre_linen(params))

  # 原ViT纸质变体在模块中摆出：
  if "posembed_input" in params["Transformer"]:
    logging.info("ViT: Loading and fixing VERY old posemb")
    posemb = params["Transformer"].pop("posembed_input")
    params["pos_embedding"] = posemb["pos_embedding"]

  # 在 2022在 Encoder 中摆出假名之前被广泛使用的版本：
  if "pos_embedding" in params["Transformer"]:
    logging.info("ViT: Loading and fixing old posemb")
    params["pos_embedding"] = params["Transformer"].pop("pos_embedding")

  # 旧的vit.py,用来先装配[cls]符号，然后添加假名。
  # 这意味着一个 B/ 32@ 224px 将拥有 7x7+ 1 假名。这是无用的和笨拙的。
  # 所以我们换了个姿势，然后把那些旧的都找回来。
  # 检查台通过人工合成[cls]符号及其假名进入。
  if "pos_embedding" in params:
    pe = params["pos_embedding"]
    if int(np.sqrt(pe.shape[1])) ** 2 + 1 == int(pe.shape[1]):
      logging.info("ViT: Loading and fixing combined cls+posemb")
      pe_cls, params["pos_embedding"] = pe[:, :1], pe[:, 1:]
      if "cls" in params:
        params["cls"] += pe_cls

  # MAP - ViT-G开发期间的头部变体有内含：
  if "probe" in params:
    params["MAPHead_0"] = {
        k: params.pop(k) for k in
        ["probe", "MlpBlock_0", "MultiHeadDotProductAttention_0", "LayerNorm_0"]
    }

  return params


def pyloop_to_scan(params_pyloop):
  """Converts a python for-loop ViT checkpoint to a lax.scan based one."""
  # 在高层次上，它们是一样的，除了循环是分开的
  # 每个编码器块的阵列 pytrees,而扫描器只有一个
  # 编码器封存的pytree,所有块的参数调和。

  params_scan = jax.tree.map(lambda x: x, params_pyloop)  # 结构复制
  t = params_scan["Transformer"]

  # 在检查站中查找编码器块的最高索引(它们始于0):
  encoderblocks = {k for k in t if k.startswith("encoderblock_")}
  depth = 1 + max({int(k.split("_")[-1]) for k in encoderblocks})

  def stack(*values):
    return np.stack(values)

  # 将所有编码器块叠成一个：
  t["encoderblock"] = jax.tree.map(
      stack, *[t[f"encoderblock_{lyr}"] for lyr in range(depth)])

  for lyr in range(depth):
    del t[f"encoderblock_{lyr}"]

  return params_scan


def scan_to_pyloop(params_scan):
  """Converts a lax.scan ViT checkpoint to a python for-loop based one."""
  # 见pyloop_to_scan中的评论。

  params_scan = jax.tree.map(lambda x: x, params_scan)  # 结构复制
  t = params_scan["Transformer"]

  # 找出有多少编码器块
  depth = len(t["encoderblock"]["LayerNorm_0"]["bias"])

  # 创建许多编码器，每一个都用他们的小树皮片。
  for lyr in range(depth):
    block = jax.tree.map(lambda x, lyr=lyr: x[lyr], t["encoderblock"])
    t[f"encoderblock_{lyr}"] = block

  del t["encoderblock"]
  return params_scan


def load(init_params, init_file, model_cfg, dont_load=()):  # pylint: disable=invalid-name because we had to CamelCase above.
  """Load init from checkpoint, both old model and this one. +Hi-res posemb."""
  init_file = VANITY_NAMES.get(init_file, init_file)
  restored_params = utils.load_params(init_file)

  restored_params = fix_old_checkpoints(restored_params)

  # 检测是否试图将非扫描检查点装入扫描模型。
  if (model_cfg.get("scan") and
      "encoderblock" not in restored_params["Transformer"]):
    restored_params = pyloop_to_scan(restored_params)
  if (not model_cfg.get("scan")
      and "encoderblock" in restored_params["Transformer"]):
    restored_params = scan_to_pyloop(restored_params)

  # 可能对某些参数(如头)使用随机输入。
  restored_params = common.merge_params(restored_params, init_params, dont_load)

  # 必要时再做个假肢
  # TODO:取自model_cfg,以避免需要init_params.
  if init_params and "pos_embedding" in init_params:
    restored_params["pos_embedding"] = resample_posemb(
        old=restored_params["pos_embedding"],
        new=init_params["pos_embedding"])

  return restored_params


# 一些犬类纸质检查站的快捷键名称：
VANITY_NAMES = {
    # pylint: disable=line-too-long
    # https://arxiv.org/abs/2106.10270推荐的模型
    # 更多模型在 https://github.com/google-research/vision_transformer
    "howto-i21k-Ti/16": "gs://vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz",
    "howto-i21k-S/32": "gs://vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_none-wd_0.1-do_0.0-sd_0.0.npz",
    "howto-i21k-S/16": "gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz",
    "howto-i21k-B/32": "gs://vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz",
    "howto-i21k-B/16": "gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz",
    "howto-i21k-B/8": "gs://vit_models/augreg/B_8-i21k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.0-sd_0.0.npz",
    "howto-i21k-L/16": "gs://vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_strong1-wd_0.1-do_0.0-sd_0.0.npz",

    # (原始内容存档于2018-09-26). Better plain vit-s16 基线 from https://arxiv.org/abs/2205.01580.
    "i1k-s16-90ep": "gs://big_vision/vit_s16_i1k_90ep.npz",
    "i1k-s16-150ep": "gs://big_vision/vit_s16_i1k_150ep.npz",
    "i1k-s16-300ep": "gs://big_vision/vit_s16_i1k_300ep.npz",

    # DeiT - 3 检查站 https://github.com/facebookresearch/deit/blob/main/README_revenge.md
    # 第一层转换为在 -1,1 中接受输入
    "deit3_S_224_1k": "gs://big_vision/zoo/deit3/bv_deit_3_small_224_1k.npz",
    "deit3_S_224_21k": "gs://big_vision/zoo/deit3/bv_deit_3_small_224_21k.npz",
    "deit3_S_384_1k": "gs://big_vision/zoo/deit3/bv_deit_3_small_384_1k.npz",
    "deit3_S_384_21k": "gs://big_vision/zoo/deit3/bv_deit_3_small_384_21k.npz",
    "deit3_B_224_1k": "gs://big_vision/zoo/deit3/bv_deit_3_base_224_1k.npz",
    "deit3_B_224_21k": "gs://big_vision/zoo/deit3/bv_deit_3_base_224_21k.npz",
    "deit3_B_384_1k": "gs://big_vision/zoo/deit3/bv_deit_3_base_384_1k.npz",
    "deit3_B_384_21k": "gs://big_vision/zoo/deit3/bv_deit_3_base_384_21k.npz",
    "deit3_L_224_1k": "gs://big_vision/zoo/deit3/bv_deit_3_large_224_1k.npz",
    "deit3_L_224_21k": "gs://big_vision/zoo/deit3/bv_deit_3_large_224_21k.npz",
    "deit3_L_384_1k": "gs://big_vision/zoo/deit3/bv_deit_3_large_384_1k.npz",
    "deit3_L_384_21k": "gs://big_vision/zoo/deit3/bv_deit_3_large_384_21k.npz",

    # SigLIP来自https://arxiv.org/abs/2303.15343的图像编码检查站
    "SigLIP B/16 224": "gs://big_vision/siglip/webli_en_b16_224_63724782.npz:img",
    "SigLIP B/16 256": "gs://big_vision/siglip/webli_en_b16_256_60500360.npz:img",
    "SigLIP B/16 384": "gs://big_vision/siglip/webli_en_b16_384_68578854.npz:img",
    "SigLIP B/16 512": "gs://big_vision/siglip/webli_en_b16_512_68580893.npz:img",
    "SigLIP L/16 256": "gs://big_vision/siglip/webli_en_l16_256_60552751.npz:img",
    "SigLIP L/16 384": "gs://big_vision/siglip/webli_en_l16_384_63634585.npz:img",
    "SigLIP So400m/14 224": "gs://big_vision/siglip/webli_en_so400m_224_57633886.npz:img",
    "SigLIP So400m/14 384": "gs://big_vision/siglip/webli_en_so400m_384_58765454.npz:img",
    "SigLIP B/16-i18n 256": "gs://big_vision/siglip/webli_i18n_b16_256_66117334.npz:img",

    # SigLIP 2来自https://arxiv.org/abs/2502.14786的图像编码检查站。
    "SigLIP2 B/16 224": "gs://big_vision/siglip2/siglip2_b16_224.npz:img",
    "SigLIP2 B/16 256": "gs://big_vision/siglip2/siglip2_b16_256.npz:img",
    "SigLIP2 B/16 384": "gs://big_vision/siglip2/siglip2_b16_384.npz:img",
    "SigLIP2 B/16 512": "gs://big_vision/siglip2/siglip2_b16_512.npz:img",
    "SigLIP2 B/32 256": "gs://big_vision/siglip2/siglip2_b32_256.npz:img",
    "SigLIP2 L/16 256": "gs://big_vision/siglip2/siglip2_l16_256.npz:img",
    "SigLIP2 L/16 384": "gs://big_vision/siglip2/siglip2_l16_384.npz:img",
    "SigLIP2 L/16 512": "gs://big_vision/siglip2/siglip2_l16_512.npz:img",
    "SigLIP2 So400m/14 224": "gs://big_vision/siglip2/siglip2_so400m14_224.npz:img",
    "SigLIP2 So400m/14 384": "gs://big_vision/siglip2/siglip2_so400m14_384.npz:img",
    "SigLIP2 So400m/16 256": "gs://big_vision/siglip2/siglip2_so400m16_256.npz:img",
    "SigLIP2 So400m/16 384": "gs://big_vision/siglip2/siglip2_so400m16_384.npz:img",
    "SigLIP2 So400m/16 512": "gs://big_vision/siglip2/siglip2_so400m16_512.npz:img",
    "SigLIP2 g-opt/16 256": "gs://big_vision/siglip2/siglip2_g-opt16_256.npz:img",
    "SigLIP2 g-opt/16 384": "gs://big_vision/siglip2/siglip2_g-opt16_384.npz:img",
    # SigLIP 2 NaFlex图像编码检查站。
    # 这些需要`proj.image_text.naflex_vit.py`作为图像编码器模型
    # 以及非标准预处理，见configs/proj/image_text/README_siglip2.md.
    "SigLIP2 B/16 NaFlex": "gs://big_vision/siglip2/siglip2_b16_naflex.npz:img",
    "SigLIP2 So400m/16 NaFlex": "gs://big_vision/siglip2/siglip2_so400m16_naflex.npz:img",
    # pylint: enable=line-too-long
}
