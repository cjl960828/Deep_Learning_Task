### EfficientNet V2

- 网络亮点

  - 提出新的渐进学习方法，可以根据输入分辨率而动态调节正则方法
    - Dropout
    - Data Argumentation
    - Mixup

- EficientNet V1 网络中存在的问题

  1. 当输入图像分辨率太大时，训练速度会变慢 [size 表示输入分辨率，OOM 表示 out of memory]

     ![分辨率对训练速度的影响](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/EfficientNet_v2_OOM.png)

  2. 网络浅层采用 DW 卷积会降低计算速度：将 MBConv 结构替换为 Fused-MBConv

     - 图解

       - $1\times 1$ 卷积和 DW 卷积 $\rightarrow$ $3 \times 3$ 卷积

     - 图示

       ![Fused-MBConv](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/EfficientNet_v2_FMBConv.png)

  3. 同等增加网络的宽度和深度的策略不合理：采用非均匀的缩放策略对深度和宽度进行放缩 [原来同时扩大一倍，现在非均匀放缩]

- NAS 结构搜索： trainning-aware NAS framework 

  - 考虑的三个维度

    1. 准确率 $\rightarrow$ 模型性能最佳评价指标
    2. 参数有效性 $\rightarrow$ 防止增加大量参数去小幅提升性能
    3. 训练有效性 $\rightarrow$ $1\times 1$ + DW 卷积 $\rightarrow$ $3 \times 3$

  - 模型设计空间

    1. 卷积操作类型：MBConv，Fused-MBConv
    2. 网络层数
    3. 卷积核大小：$3 \times 3$, $5 \times 5$
    4. 扩展比：MBConv 中的第一个 $1 \times 1$ 扩展卷积 + Fused-MBConv 中的第一个 $3 \times 3$ 扩展卷积 {1, 4, 6}

  - 减少搜素空间

    1. 移除不需要的搜索选项，例如 pooling skip 操作
    2. 重用 EfficientNet 中搜索的 channel sizes

  - 结构搜索过程

    1. 在搜索空间中随机采样 1000 个模型

    2. 使用小分辨率的图像在每个模型上训练 1000 个 epochs

       1. 奖励函数：$A$ 表示模型准确率，$S$ 表示标准训练一个 Step 所需的时间，$P$ 表示模型参数大小
          $$
          A \cdot S^w \cdot P^v
          $$

       2. $w = -0.07$，$v = -0.05$

- EfficientNet V2-S 网络结构

  - 图解

    - EfficientNet V2 中同时采用 MBConv 与 Fused-MBConv 结构（网络浅层使用 Fused-MBConv）
    - EfficientNet V2 中使用更小的扩展率 (EfficientNet V2 中采用的是 4，EfficientNet V1 中采用的是 6)
    - EfficientNet V2 中采用更多的 $3 \times 3$ 来增加感受野
    - 移除 EfficientNet V1 中的最后一个 MBConv 结构对应的 stage，即最后一个步距为 1 的 stage

  - 图示

    ![EfficientNet V2-S 网络结构](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/EfficientNet_v2_s.png)

  - 结构说明
    - Conv3x3 ：普通的 $3 \times 3$ 卷积 + 激活函数（SiLU）+ BN
    - Fused-MBConv* 以及 MBConv* ：* 号表示扩展率
    - $k3 \times 3$ ：表示采用 $3 \times 3$ 的卷积
    - expansion = 1 时，没有扩展卷积，当 stride = 1 且输入维度等于输出维度时，才有 shortcut 连接
    - SE0.25: 0.25 表示第一个 FC 层的输出维度为输入维度的 $\frac 14$
    - Channel 表示该 stage 的输出维度
    - Layer 表示重复 n 次 Operator

- Pytorch 版本的 EfficientNet V2 结构搭建

  ```python
  from collections import OrderedDict
  from functools import partial
  from typing import Callable, Optional
  
  import torch.nn as nn
  import torch
  from torch import Tensor
  
  
  # 采用 Stochastic Depth
  def drop_path(x, drop_prob: float = 0., training: bool = False):
      """
      Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
      "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
      This function is taken from the rwightman.
      It can be seen here:
      https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
      """
      if drop_prob == 0. or not training:
          return x
      keep_prob = 1 - drop_prob
      shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
      random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
      random_tensor.floor_()  # binarize
      output = x.div(keep_prob) * random_tensor
      return output
  
  
  # 采用 Stochastic Depth
  class DropPath(nn.Module):
      """
      Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
      "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
      """
      def __init__(self, drop_prob=None):
          super(DropPath, self).__init__()
          self.drop_prob = drop_prob
  
      def forward(self, x):
          return drop_path(x, self.drop_prob, self.training)
  
      
  # Conv + BN + SiLU
  class ConvBNAct(nn.Module):
      def __init__(self,
                   in_planes: int,
                   out_planes: int,
                   kernel_size: int = 3,
                   stride: int = 1,
                   groups: int = 1,
                   norm_layer: Optional[Callable[..., nn.Module]] = None,
                   activation_layer: Optional[Callable[..., nn.Module]] = None):
          super(ConvBNAct, self).__init__()
  
          padding = (kernel_size - 1) // 2
          if norm_layer is None:
              norm_layer = nn.BatchNorm2d
          if activation_layer is None:
              activation_layer = nn.SiLU  # alias Swish  (torch>=1.7)
  
          self.conv = nn.Conv2d(in_channels=in_planes,
                                out_channels=out_planes,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                groups=groups,
                                bias=False)
  
          self.bn = norm_layer(out_planes)
          self.act = activation_layer()
  
      def forward(self, x):
          result = self.conv(x)
          result = self.bn(result)
          result = self.act(result)
  
          return result
  
  
  # SE Net 0.25
  class SqueezeExcite(nn.Module):
      def __init__(self,
                   input_c: int,   # block input channel
                   expand_c: int,  # block expand channel
                   se_ratio: float = 0.25):
          super(SqueezeExcite, self).__init__()
          squeeze_c = int(input_c * se_ratio)
          self.conv_reduce = nn.Conv2d(expand_c, squeeze_c, 1)
          self.act1 = nn.SiLU()  # alias Swish
          self.conv_expand = nn.Conv2d(squeeze_c, expand_c, 1)
          self.act2 = nn.Sigmoid()
  
      def forward(self, x: Tensor) -> Tensor:
          scale = x.mean((2, 3), keepdim=True)
          scale = self.conv_reduce(scale)
          scale = self.act1(scale)
          scale = self.conv_expand(scale)
          scale = self.act2(scale)
          return scale * x
  
  
  # MBConv 结构
  class MBConv(nn.Module):
      def __init__(self,
                   kernel_size: int,
                   input_c: int,
                   out_c: int,
                   expand_ratio: int,
                   stride: int,
                   se_ratio: float,
                   drop_rate: float,
                   norm_layer: Callable[..., nn.Module]):
          super(MBConv, self).__init__()
  
          if stride not in [1, 2]:
              raise ValueError("illegal stride value.")
  
          # 当 stride = 1 并且输入通道等于输出通道时才有 shortcut
          self.has_shortcut = (stride == 1 and input_c == out_c)
  
          activation_layer = nn.SiLU  # alias Swish
          expanded_c = input_c * expand_ratio
  
          # 在 EfficientNet V2 中，MBConv 中不存在 expansion=1 的情况所以 conv_pw 肯定存在
          assert expand_ratio != 1
          # Point-wise expansion
          # Conv 1x1 + 3x3 DW + SE + Conv 1x1 + Dropout
          self.expand_conv = ConvBNAct(input_c,
                                       expanded_c,
                                       kernel_size=1,
                                       norm_layer=norm_layer,
                                       activation_layer=activation_layer)
  
          # Depth-wise convolution
          self.dwconv = ConvBNAct(expanded_c,
                                  expanded_c,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  groups=expanded_c,
                                  norm_layer=norm_layer,
                                  activation_layer=activation_layer)
  
          self.se = SqueezeExcite(input_c, expanded_c, se_ratio) if se_ratio > 0 else nn.Identity()
  
          # Point-wise linear projection
          self.project_conv = ConvBNAct(expanded_c,
                                        out_planes=out_c,
                                        kernel_size=1,
                                        norm_layer=norm_layer,
                                        activation_layer=nn.Identity)  # 注意这里没有激活函数，所有传入Identity
  
          self.out_channels = out_c
  
          # 只有在使用shortcut连接时才使用dropout层
          self.drop_rate = drop_rate
          if self.has_shortcut and drop_rate > 0:
              self.dropout = DropPath(drop_rate)
  
      def forward(self, x: Tensor) -> Tensor:
          result = self.expand_conv(x)
          result = self.dwconv(result)
          result = self.se(result)
          result = self.project_conv(result)
  
          if self.has_shortcut:
              if self.drop_rate > 0:
                  result = self.dropout(result)
              result += x
  
          return result
  
  
  class FusedMBConv(nn.Module):
      def __init__(self,
                   kernel_size: int,
                   input_c: int,
                   out_c: int,
                   expand_ratio: int,
                   stride: int,
                   se_ratio: float,
                   drop_rate: float,
                   norm_layer: Callable[..., nn.Module]):
          super(FusedMBConv, self).__init__()
  
          assert stride in [1, 2]
          assert se_ratio == 0
  
          self.has_shortcut = stride == 1 and input_c == out_c
          self.drop_rate = drop_rate
  
          self.has_expansion = expand_ratio != 1
  
          activation_layer = nn.SiLU  # alias Swish
          expanded_c = input_c * expand_ratio
  
          # 只有当expand ratio不等于1时才有expand conv
          if self.has_expansion:
              # Expansion convolution
              self.expand_conv = ConvBNAct(input_c,
                                           expanded_c,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           norm_layer=norm_layer,
                                           activation_layer=activation_layer)
  
              self.project_conv = ConvBNAct(expanded_c,
                                            out_c,
                                            kernel_size=1,
                                            norm_layer=norm_layer,
                                            activation_layer=nn.Identity)  # 注意没有激活函数
          else:
              # 当只有 project_conv 时的情况
              self.project_conv = ConvBNAct(input_c,
                                            out_c,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            norm_layer=norm_layer,
                                            activation_layer=activation_layer)  # 注意有激活函数
  
          self.out_channels = out_c
  
          # 只有在使用 shortcut 连接时才使用 dropout 层
          self.drop_rate = drop_rate
          if self.has_shortcut and drop_rate > 0:
              self.dropout = DropPath(drop_rate)
  
      def forward(self, x: Tensor) -> Tensor:
          if self.has_expansion:
              result = self.expand_conv(x)
              result = self.project_conv(result)
          else:
              result = self.project_conv(x)
  
          if self.has_shortcut:
              if self.drop_rate > 0:
                  result = self.dropout(result)
  
              result += x
  
          return result
  
  
  class EfficientNetV2(nn.Module):
      def __init__(self,
                   model_cnf: list,
                   num_classes: int = 1000,
                   num_features: int = 1280,
                   dropout_rate: float = 0.2,
                   drop_connect_rate: float = 0.2):
          super(EfficientNetV2, self).__init__()
  
          for cnf in model_cnf:
              assert len(cnf) == 8
  
          norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)
  
          stem_filter_num = model_cnf[0][4]
  
          self.stem = ConvBNAct(3,
                                stem_filter_num,
                                kernel_size=3,
                                stride=2,
                                norm_layer=norm_layer)  # 激活函数默认是SiLU
  
          total_blocks = sum([i[0] for i in model_cnf])
          block_id = 0
          blocks = []
          for cnf in model_cnf:
              repeats = cnf[0]
              op = FusedMBConv if cnf[-2] == 0 else MBConv
              for i in range(repeats):
                  blocks.append(op(kernel_size=cnf[1],
                                   input_c=cnf[4] if i == 0 else cnf[5],
                                   out_c=cnf[5],
                                   expand_ratio=cnf[3],
                                   stride=cnf[2] if i == 0 else 1,
                                   se_ratio=cnf[-1],
                                   drop_rate=drop_connect_rate * block_id / total_blocks,
                                   norm_layer=norm_layer))
                  block_id += 1
          self.blocks = nn.Sequential(*blocks)
  
          head_input_c = model_cnf[-1][-3]
          head = OrderedDict()
  
          head.update({"project_conv": ConvBNAct(head_input_c,
                                                 num_features,
                                                 kernel_size=1,
                                                 norm_layer=norm_layer)})  # 激活函数默认是SiLU
  
          head.update({"avgpool": nn.AdaptiveAvgPool2d(1)})
          head.update({"flatten": nn.Flatten()})
  
          if dropout_rate > 0:
              head.update({"dropout": nn.Dropout(p=dropout_rate, inplace=True)})
          head.update({"classifier": nn.Linear(num_features, num_classes)})
  
          self.head = nn.Sequential(head)
  
          # initial weights
          for m in self.modules():
              if isinstance(m, nn.Conv2d):
                  nn.init.kaiming_normal_(m.weight, mode="fan_out")
                  if m.bias is not None:
                      nn.init.zeros_(m.bias)
              elif isinstance(m, nn.BatchNorm2d):
                  nn.init.ones_(m.weight)
                  nn.init.zeros_(m.bias)
              elif isinstance(m, nn.Linear):
                  nn.init.normal_(m.weight, 0, 0.01)
                  nn.init.zeros_(m.bias)
  
      def forward(self, x: Tensor) -> Tensor:
          x = self.stem(x)
          x = self.blocks(x)
          x = self.head(x)
  
          return x
  
  
  def efficientnetv2_s(num_classes: int = 1000):
      """
      EfficientNetV2
      https://arxiv.org/abs/2104.00298
      """
      # train_size: 300, eval_size: 384
  
      # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
      model_config = [[2, 3, 1, 1, 24, 24, 0, 0],
                      [4, 3, 2, 4, 24, 48, 0, 0],
                      [4, 3, 2, 4, 48, 64, 0, 0],
                      [6, 3, 2, 4, 64, 128, 1, 0.25],
                      [9, 3, 1, 6, 128, 160, 1, 0.25],
                      [15, 3, 2, 6, 160, 256, 1, 0.25]]
  
      model = EfficientNetV2(model_cnf=model_config,
                             num_classes=num_classes,
                             dropout_rate=0.2)
      return model
  
  
  def efficientnetv2_m(num_classes: int = 1000):
      """
      EfficientNetV2
      https://arxiv.org/abs/2104.00298
      """
      # train_size: 384, eval_size: 480
  
      # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
      model_config = [[3, 3, 1, 1, 24, 24, 0, 0],
                      [5, 3, 2, 4, 24, 48, 0, 0],
                      [5, 3, 2, 4, 48, 80, 0, 0],
                      [7, 3, 2, 4, 80, 160, 1, 0.25],
                      [14, 3, 1, 6, 160, 176, 1, 0.25],
                      [18, 3, 2, 6, 176, 304, 1, 0.25],
                      [5, 3, 1, 6, 304, 512, 1, 0.25]]
  
      model = EfficientNetV2(model_cnf=model_config,
                             num_classes=num_classes,
                             dropout_rate=0.3)
      return model
  
  
  def efficientnetv2_l(num_classes: int = 1000):
      """
      EfficientNetV2
      https://arxiv.org/abs/2104.00298
      """
      # train_size: 384, eval_size: 480
  
      # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
      model_config = [[4, 3, 1, 1, 32, 32, 0, 0],
                      [7, 3, 2, 4, 32, 64, 0, 0],
                      [7, 3, 2, 4, 64, 96, 0, 0],
                      [10, 3, 2, 4, 96, 192, 1, 0.25],
                      [19, 3, 1, 6, 192, 224, 1, 0.25],
                      [25, 3, 2, 6, 224, 384, 1, 0.25],
                      [7, 3, 1, 6, 384, 640, 1, 0.25]]
  
      model = EfficientNetV2(model_cnf=model_config,
                             num_classes=num_classes,
                             dropout_rate=0.4)
      return model
  ```

  

- tensorflow2 版本的 EfficientNet V2 结构搭建

  ```python
  """
  official code:
  https://github.com/google/automl/tree/master/efficientnetv2
  """
  
  import itertools
  
  import tensorflow as tf
  from tensorflow.keras import layers, Model, Input
  
  
  CONV_KERNEL_INITIALIZER = {
      'class_name': 'VarianceScaling',
      'config': {
          'scale': 2.0,
          'mode': 'fan_out',
          'distribution': 'truncated_normal'
      }
  }
  
  DENSE_KERNEL_INITIALIZER = {
      'class_name': 'VarianceScaling',
      'config': {
          'scale': 1. / 3.,
          'mode': 'fan_out',
          'distribution': 'uniform'
      }
  }
  
  
  # SE 网络结构 SE 0.25
  class SE(layers.Layer):
      def __init__(self,
                   se_filters: int,
                   output_filters: int,
                   name: str = None):
          super(SE, self).__init__(name=name)
  
          self.se_reduce = layers.Conv2D(filters=se_filters,
                                         kernel_size=1,
                                         strides=1,
                                         padding="same",
                                         activation="swish",
                                         use_bias=True,
                                         kernel_initializer=CONV_KERNEL_INITIALIZER,
                                         name="conv2d")
  
          self.se_expand = layers.Conv2D(filters=output_filters,
                                         kernel_size=1,
                                         strides=1,
                                         padding="same",
                                         activation="sigmoid",
                                         use_bias=True,
                                         kernel_initializer=CONV_KERNEL_INITIALIZER,
                                         name="conv2d_1")
  
      def call(self, inputs, **kwargs):
          # Tensor: [N, H, W, C] -> [N, 1, 1, C]
          se_tensor = tf.reduce_mean(inputs, [1, 2], keepdims=True)
          se_tensor = self.se_reduce(se_tensor)
          se_tensor = self.se_expand(se_tensor)
          return se_tensor * inputs
  
  
  class MBConv(layers.Layer):
      def __init__(self,
                   kernel_size: int,
                   input_c: int,
                   out_c: int,
                   expand_ratio: int,
                   stride: int,
                   se_ratio: float = 0.25,
                   drop_rate: float = 0.,
                   name: str = None):
          super(MBConv, self).__init__(name=name)
  
          if stride not in [1, 2]:
              raise ValueError("illegal stride value.")
  
          self.has_shortcut = (stride == 1 and input_c == out_c)
          expanded_c = input_c * expand_ratio
  
          bid = itertools.count(0)
          get_norm_name = lambda: 'batch_normalization' + ('' if not next(
              bid) else '_' + str(next(bid) // 2))
          cid = itertools.count(0)
          get_conv_name = lambda: 'conv2d' + ('' if not next(cid) else '_' + str(
              next(cid) // 2))
  
          # 在 EfficientNet V2 中，MBConv 中不存在 expansion=1 的情况所以 conv_pw 肯定存在
          assert expand_ratio != 1
          # Point-wise expansion
          self.expand_conv = layers.Conv2D(
              filters=expanded_c,
              kernel_size=1,
              strides=1,
              padding="same",
              use_bias=False,
              name=get_conv_name())
          self.norm0 = layers.BatchNormalization(
              axis=-1,
              momentum=0.9,
              epsilon=1e-3,
              name=get_norm_name())
          self.act0 = layers.Activation("swish")
  
          # Depth-wise convolution
          self.depthwise_conv = layers.DepthwiseConv2D(
              kernel_size=kernel_size,
              strides=stride,
              depthwise_initializer=CONV_KERNEL_INITIALIZER,
              padding="same",
              use_bias=False,
              name="depthwise_conv2d")
          self.norm1 = layers.BatchNormalization(
              axis=-1,
              momentum=0.9,
              epsilon=1e-3,
              name=get_norm_name())
          self.act1 = layers.Activation("swish")
  
          # SE
          num_reduced_filters = max(1, int(input_c * se_ratio))
          self.se = SE(num_reduced_filters, expanded_c, name="se")
  
          # Point-wise linear projection
          self.project_conv = layers.Conv2D(
              filters=out_c,
              kernel_size=1,
              strides=1,
              kernel_initializer=CONV_KERNEL_INITIALIZER,
              padding="same",
              use_bias=False,
              name=get_conv_name())
          self.norm2 = layers.BatchNormalization(
              axis=-1,
              momentum=0.9,
              epsilon=1e-3,
              name=get_norm_name())
  
          self.drop_rate = drop_rate
          if self.has_shortcut and drop_rate > 0:
              # Stochastic Depth
              self.drop_path = layers.Dropout(rate=drop_rate,
                                              noise_shape=(None, 1, 1, 1),  # binary dropout mask
                                              name="drop_path")
  
      def call(self, inputs, training=None):
          x = inputs
  
          x = self.expand_conv(x)
          x = self.norm0(x, training=training)
          x = self.act0(x)
  
          x = self.depthwise_conv(x)
          x = self.norm1(x, training=training)
          x = self.act1(x)
  
          x = self.se(x)
  
          x = self.project_conv(x)
          x = self.norm2(x, training=training)
  
          if self.has_shortcut:
              if self.drop_rate > 0:
                  x = self.drop_path(x, training=training)
  
              x = tf.add(x, inputs)
  
          return x
  
  
  class FusedMBConv(layers.Layer):
      def __init__(self,
                   kernel_size: int,
                   input_c: int,
                   out_c: int,
                   expand_ratio: int,
                   stride: int,
                   se_ratio: float,
                   drop_rate: float = 0.,
                   name: str = None):
          super(FusedMBConv, self).__init__(name=name)
          if stride not in [1, 2]:
              raise ValueError("illegal stride value.")
  
          assert se_ratio == 0.
  
          self.has_shortcut = (stride == 1 and input_c == out_c)
          self.has_expansion = expand_ratio != 1
          expanded_c = input_c * expand_ratio
  
          bid = itertools.count(0)
          get_norm_name = lambda: 'batch_normalization' + ('' if not next(
              bid) else '_' + str(next(bid) // 2))
          cid = itertools.count(0)
          get_conv_name = lambda: 'conv2d' + ('' if not next(cid) else '_' + str(
              next(cid) // 2))
  
          if expand_ratio != 1:
              self.expand_conv = layers.Conv2D(
                  filters=expanded_c,
                  kernel_size=kernel_size,
                  strides=stride,
                  kernel_initializer=CONV_KERNEL_INITIALIZER,
                  padding="same",
                  use_bias=False,
                  name=get_conv_name())
              self.norm0 = layers.BatchNormalization(
                  axis=-1,
                  momentum=0.9,
                  epsilon=1e-3,
                  name=get_norm_name())
              self.act0 = layers.Activation("swish")
  
          self.project_conv = layers.Conv2D(
              filters=out_c,
              kernel_size=1 if expand_ratio != 1 else kernel_size,
              strides=1 if expand_ratio != 1 else stride,
              kernel_initializer=CONV_KERNEL_INITIALIZER,
              padding="same",
              use_bias=False,
              name=get_conv_name())
          self.norm1 = layers.BatchNormalization(
              axis=-1,
              momentum=0.9,
              epsilon=1e-3,
              name=get_norm_name())
  
          if expand_ratio == 1:
              self.act1 = layers.Activation("swish")
  
          self.drop_rate = drop_rate
          if self.has_shortcut and drop_rate > 0:
              # Stochastic Depth
              self.drop_path = layers.Dropout(rate=drop_rate,
                                              noise_shape=(None, 1, 1, 1),  # binary dropout mask
                                              name="drop_path")
  
      def call(self, inputs, training=None):
          x = inputs
          if self.has_expansion:
              x = self.expand_conv(x)
              x = self.norm0(x, training=training)
              x = self.act0(x)
  
          x = self.project_conv(x)
          x = self.norm1(x, training=training)
          if self.has_expansion is False:
              x = self.act1(x)
  
          if self.has_shortcut:
              if self.drop_rate > 0:
                  x = self.drop_path(x, training=training)
  
              x = tf.add(x, inputs)
  
          return x
  
  
  class Stem(layers.Layer):
      def __init__(self, filters: int, name: str = None):
          super(Stem, self).__init__(name=name)
          self.conv_stem = layers.Conv2D(
              filters=filters,
              kernel_size=3,
              strides=2,
              kernel_initializer=CONV_KERNEL_INITIALIZER,
              padding="same",
              use_bias=False,
              name="conv2d")
          self.norm = layers.BatchNormalization(
              axis=-1,
              momentum=0.9,
              epsilon=1e-3,
              name="batch_normalization")
          self.act = layers.Activation("swish")
  
      def call(self, inputs, training=None):
          x = self.conv_stem(inputs)
          x = self.norm(x, training=training)
          x = self.act(x)
  
          return x
  
  
  class Head(layers.Layer):
      def __init__(self,
                   filters: int = 1280,
                   num_classes: int = 1000,
                   drop_rate: float = 0.,
                   name: str = None):
          super(Head, self).__init__(name=name)
          self.conv_head = layers.Conv2D(
              filters=filters,
              kernel_size=1,
              kernel_initializer=CONV_KERNEL_INITIALIZER,
              padding="same",
              use_bias=False,
              name="conv2d")
          self.norm = layers.BatchNormalization(
              axis=-1,
              momentum=0.9,
              epsilon=1e-3,
              name="batch_normalization")
          self.act = layers.Activation("swish")
  
          self.avg = layers.GlobalAveragePooling2D()
          self.fc = layers.Dense(num_classes,
                                 kernel_initializer=DENSE_KERNEL_INITIALIZER)
  
          if drop_rate > 0:
              self.dropout = layers.Dropout(drop_rate)
  
      def call(self, inputs, training=None):
          x = self.conv_head(inputs)
          x = self.norm(x)
          x = self.act(x)
          x = self.avg(x)
  
          if self.dropout:
              x = self.dropout(x, training=training)
  
          x = self.fc(x)
          return x
  
  
  class EfficientNetV2(Model):
      def __init__(self,
                   model_cnf: list,
                   num_classes: int = 1000,
                   num_features: int = 1280,
                   dropout_rate: float = 0.2,
                   drop_connect_rate: float = 0.2,
                   name: str = None):
          super(EfficientNetV2, self).__init__(name=name)
  
          for cnf in model_cnf:
              assert len(cnf) == 8
  
          stem_filter_num = model_cnf[0][4]
          self.stem = Stem(stem_filter_num)
  
          total_blocks = sum([i[0] for i in model_cnf])
          block_id = 0
          self.blocks = []
          # Builds blocks.
          for cnf in model_cnf:
              repeats = cnf[0]
              op = FusedMBConv if cnf[-2] == 0 else MBConv
              for i in range(repeats):
                  self.blocks.append(op(kernel_size=cnf[1],
                                        input_c=cnf[4] if i == 0 else cnf[5],
                                        out_c=cnf[5],
                                        expand_ratio=cnf[3],
                                        stride=cnf[2] if i == 0 else 1,
                                        se_ratio=cnf[-1],
                                        drop_rate=drop_connect_rate * block_id / total_blocks,
                                        name="blocks_{}".format(block_id)))
                  block_id += 1
  
          self.head = Head(num_features, num_classes, dropout_rate)
  
      # def summary(self, input_shape=(224, 224, 3), **kwargs):
      #     x = Input(shape=input_shape)
      #     model = Model(inputs=[x], outputs=self.call(x, training=True))
      #     return model.summary()
  
      def call(self, inputs, training=None):
          x = self.stem(inputs, training)
  
          # call for blocks.
          for _, block in enumerate(self.blocks):
              x = block(x, training=training)
  
          x = self.head(x, training=training)
  
          return x
  
  
  def efficientnetv2_s(num_classes: int = 1000):
      """
      EfficientNetV2
      https://arxiv.org/abs/2104.00298
      """
      # train_size: 300, eval_size: 384
  
      # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
      model_config = [[2, 3, 1, 1, 24, 24, 0, 0],
                      [4, 3, 2, 4, 24, 48, 0, 0],
                      [4, 3, 2, 4, 48, 64, 0, 0],
                      [6, 3, 2, 4, 64, 128, 1, 0.25],
                      [9, 3, 1, 6, 128, 160, 1, 0.25],
                      [15, 3, 2, 6, 160, 256, 1, 0.25]]
  
      model = EfficientNetV2(model_cnf=model_config,
                             num_classes=num_classes,
                             dropout_rate=0.2,
                             name="efficientnetv2-s")
      return model
  
  
  def efficientnetv2_m(num_classes: int = 1000):
      """
      EfficientNetV2
      https://arxiv.org/abs/2104.00298
      """
      # train_size: 384, eval_size: 480
  
      # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
      model_config = [[3, 3, 1, 1, 24, 24, 0, 0],
                      [5, 3, 2, 4, 24, 48, 0, 0],
                      [5, 3, 2, 4, 48, 80, 0, 0],
                      [7, 3, 2, 4, 80, 160, 1, 0.25],
                      [14, 3, 1, 6, 160, 176, 1, 0.25],
                      [18, 3, 2, 6, 176, 304, 1, 0.25],
                      [5, 3, 1, 6, 304, 512, 1, 0.25]]
  
      model = EfficientNetV2(model_cnf=model_config,
                             num_classes=num_classes,
                             dropout_rate=0.3,
                             name="efficientnetv2-m")
      return model
  
  
  def efficientnetv2_l(num_classes: int = 1000):
      """
      EfficientNetV2
      https://arxiv.org/abs/2104.00298
      """
      # train_size: 384, eval_size: 480
  
      # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
      model_config = [[4, 3, 1, 1, 32, 32, 0, 0],
                      [7, 3, 2, 4, 32, 64, 0, 0],
                      [7, 3, 2, 4, 64, 96, 0, 0],
                      [10, 3, 2, 4, 96, 192, 1, 0.25],
                      [19, 3, 1, 6, 192, 224, 1, 0.25],
                      [25, 3, 2, 6, 224, 384, 1, 0.25],
                      [7, 3, 1, 6, 384, 640, 1, 0.25]]
  
      model = EfficientNetV2(model_cnf=model_config,
                             num_classes=num_classes,
                             dropout_rate=0.4,
                             name="efficientnetv2-l")
      return model
  
  
  # m = efficientnetv2_s()
  # m.summary()
  ```

  