### EfficientNet V1

- 网络亮点

  - 可以同时探索输入分辨率，网络的深度 (堆叠卷积的次数) 和宽度 (channel) 的影响

- 不同增强网络性能的方法

  - 图解

    - 图 (a): 原始网络结构
    - 图 (b): 增加每层卷积的卷积核个数，即增加 channel 数
    - 图 (c): 增加卷积层的叠加层数，即增加网络的 depth
    - 图 (d): 增加模型输入的分辨率
    - 图 (e): 采用 NAS 对输入分辨率，网络深度和宽度同时进行研究

  - 图示

    ![增强网路性能的方法](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/EfficientNet_v1_Introbution.png)

- 以往方法存在的问题

  - 图解

    - 增加网络的 `深度`：叠加更多层的卷积层能够得到更加复杂且丰富的特征，但是过深的模型会出现退化以及训练困难的问题
    - 增加网络的 `宽度`：增加各层卷积核的个数能够获得更细粒度的特征，并且容易训练，但是宽但浅的网络无法学习到深层次的特征
    - 增加输入 `分辨率`：增加输入分辨率能够获得更细粒度的特征，但是往往收益较小，即增加大量计算换来小幅性能提升

  - 图示

    ![增加 depth，width，resolution 的性能提升](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/EfficientNet_v1_DWR.png)

  - 总结：仅仅通过增加网络深度、宽度或者输入分辨率，最终准确率趋于 80% 达到饱和

- 同时修改网络深度以及分辨率对性能的影响

  - 图解：通过同时增加网络深度以及输入分辨率能够促使模型突破准确率为 80% 的瓶颈

  - 图示

    ![同时增加网络深度以及输入分辨率](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/EfficientNet_v1_dAr.png)

- EfficientNet V1 网络结构

  ![EfficientNet V1 网络结构](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/EfficientNet_v1_B0.png)

- MBConv 结构

  - 图解 [stride = 1 且输入与输出维度相同时才有残差连接]

    - 记录初始输入，得到大小为 $[C, H, W]$ 的特征 $o_1$
    - 将大小为 $[C, H, W]$ 的特征输入到卷积核大小为 $1 \times 1$ ，步距为 $1$ 的卷积中，得到大小为 $[C‘, H, W]$ 的特征 $t_1$
    - 将特征 $t_1$ 进行 BN 处理以及 Swish 激活，得到大小为 $[C’, H, W]$ 的特征 $t_2$
    - 将特征 $t_2$ 输入到卷积核为 $k \times k$ ，步距为 1/2 的卷积中 (具体的 k 与 s 大小详见上表)，得到大小为 $[C, H, W]$ 的特征 $t_3$
    - 将特征 $t_3$ 输入到 SE 注意力模块中，得到大小为 $[C, H, W]$ 的特征 $t_4$
    - 将特征 $t_4$ 输入到卷积核大小为 $1 \times 1$ ，步距为 $1$ 的卷积中，得到大小为 $[C, H, W]$ 的特征 $t_5$
    - 将特征 $t_5$ 进行 BN 处理，随后采用 Dropout 层对 BN 后的结果进行处理，得到大小为 $[C, H, W]$ 的特征 $t_6$
    - 将最初大小为 $[C, H, W]$ 的特征 $o_1$ 与特征 $t_6$ 进行相加处理，得到大小为 $[C, H, W]$ 的特征 $o_2$

  - 图示

    ![MBConv](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/EfficientNet_v1_MBConv.png)

  - 注意事项
    - 第一个 $1 \times 1$ 的卷积主要用于升维，一般为输入特征的 $n \in \{1, 6\}$ 倍
    - 当 $n = 1$ 时，不需要第一个 $1 \times 1 $ 卷积
    - 当 stride = 1，并且输入特征尺寸与输出尺寸相同时才有 shortcut 连接
    - SE 模块中，第一个 FC 的维度为输入维度的 $\frac 14$，并且激活函数采用 Swish。第二个 FC 的维度与输入特征维度相同，并且激活函数采用 sigmoid
    - Dropout 层在网络中存在 shortcut 连接时才使用

- 8 种不同 EfficientNet v1 网络结构参数对比

  ![EfficientNet V1 B0-B7](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/EfficientNet_v1_B0_B7.png)

- Pytotch 版本的 EfficientNet V1 网络搭建

  ```python
  import math
  import copy
  from functools import partial
  from collections import OrderedDict
  from typing import Optional, Callable
  
  import torch
  import torch.nn as nn
  from torch import Tensor
  from torch.nn import functional as F
  
  
  """
  该方法就是将传入的 channel 的个数调整到 8 的整数倍，使得对硬件友好
  """
  def _make_divisible(ch, divisor=8, min_ch=None):
      """
      This function is taken from the original tf repo.
      It ensures that all layers have a channel number that is divisible by 8
      It can be seen here:
      https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
      """
      if min_ch is None:
          min_ch = divisor
      new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
      # Make sure that round down does not go down by more than 10%.
      if new_ch < 0.9 * ch:
          new_ch += divisor
      return new_ch
  
  
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
  
  """
  定义 BN 和激活函数
  """
  class ConvBNActivation(nn.Sequential):
      def __init__(self,
                   in_planes: int,
                   out_planes: int,
                   kernel_size: int = 3,
                   stride: int = 1,
                   groups: int = 1,
                   norm_layer: Optional[Callable[..., nn.Module]] = None, #BN结构，默认为None
                   activation_layer: Optional[Callable[..., nn.Module]] = None): #ac结构，默认为None
          padding = (kernel_size - 1) // 2
          if norm_layer is None:
              """
              不指定 norm 的话，默认 BN 结构
              """
              norm_layer = nn.BatchNorm2d
          if activation_layer is None:
              """
              如果不指定激活函数，则默认 SiLU 激活函数
              """
              activation_layer = nn.SiLU  # alias Swish  (torch>=1.7) 激活函数使用的是 SiLU，在版本 1.70 以上才有，建议1.7.1版本
  
          # 定义层结构 [in_channels,out_channels,kernel_size,stride,padding,groups,bias=False],--> BN--> ac
          super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes,
                                                           out_channels=out_planes,
                                                           kernel_size=kernel_size,
                                                           stride=stride,
                                                           padding=padding,
                                                           groups=groups,
                                                           bias=False),
                                                 norm_layer(out_planes),
                                                 activation_layer())
  
  """
  定义 SE 模块
  """
  class SqueezeExcitation(nn.Module):
      def __init__(self,
                   input_c: int,   # block input channel
                   expand_c: int,  # block expand channel
                   squeeze_factor: int = 4):
          super(SqueezeExcitation, self).__init__()
          squeeze_c = input_c // squeeze_factor
          self.fc1 = nn.Conv2d(expand_c, squeeze_c, 1)
          self.ac1 = nn.SiLU()  # alias Swish
          self.fc2 = nn.Conv2d(squeeze_c, expand_c, 1)
          self.ac2 = nn.Sigmoid()
  
      def forward(self, x: Tensor) -> Tensor:
          scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
          scale = self.fc1(scale)
          scale = self.ac1(scale)
          scale = self.fc2(scale)
          scale = self.ac2(scale)
          return scale * x
  
  
  class InvertedResidualConfig:
      # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate
      def __init__(self,
                   kernel: int,          # 3 or 5
                   input_c: int,
                   out_c: int,
                   expanded_ratio: int,  # 1 or 6
                   stride: int,          # 1 or 2
                   use_se: bool,         # True
                   drop_rate: float,
                   index: str,           # 1a, 2a, 2b, ...
                   width_coefficient: float):
          self.input_c = self.adjust_channels(input_c, width_coefficient)
          self.kernel = kernel
          self.expanded_c = self.input_c * expanded_ratio
          self.out_c = self.adjust_channels(out_c, width_coefficient)
          self.use_se = use_se
          self.stride = stride
          self.drop_rate = drop_rate
          self.index = index
  
      @staticmethod
      def adjust_channels(channels: int, width_coefficient: float):
          return _make_divisible(channels * width_coefficient, 8)
  
  
  class InvertedResidual(nn.Module):
      def __init__(self,
                   cnf: InvertedResidualConfig,
                   norm_layer: Callable[..., nn.Module]):
          super(InvertedResidual, self).__init__()
  
          if cnf.stride not in [1, 2]:
              raise ValueError("illegal stride value.")
  
          self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)
  
          layers = OrderedDict()
          activation_layer = nn.SiLU  # alias Swish
  
          # expand
          if cnf.expanded_c != cnf.input_c:
              layers.update({"expand_conv": ConvBNActivation(cnf.input_c,
                                                             cnf.expanded_c,
                                                             kernel_size=1,
                                                             norm_layer=norm_layer,
                                                             activation_layer=activation_layer)})
  
          # depthwise
          layers.update({"dwconv": ConvBNActivation(cnf.expanded_c,
                                                    cnf.expanded_c,
                                                    kernel_size=cnf.kernel,
                                                    stride=cnf.stride,
                                                    groups=cnf.expanded_c,
                                                    norm_layer=norm_layer,
                                                    activation_layer=activation_layer)})
  
          if cnf.use_se:
              layers.update({"se": SqueezeExcitation(cnf.input_c,
                                                     cnf.expanded_c)})
  
          # project
          layers.update({"project_conv": ConvBNActivation(cnf.expanded_c,
                                                          cnf.out_c,
                                                          kernel_size=1,
                                                          norm_layer=norm_layer,
                                                          activation_layer=nn.Identity)})
  
          self.block = nn.Sequential(layers)
          self.out_channels = cnf.out_c
          self.is_strided = cnf.stride > 1
  
          # 只有在使用shortcut连接时才使用dropout层
          if self.use_res_connect and cnf.drop_rate > 0:
              self.dropout = DropPath(cnf.drop_rate)
          else:
              self.dropout = nn.Identity()
  
      def forward(self, x: Tensor) -> Tensor:
          result = self.block(x)
          result = self.dropout(result)
          if self.use_res_connect:
              result += x
  
          return result
  
  
  class EfficientNet(nn.Module):
      def __init__(self,
                   width_coefficient: float,
                   depth_coefficient: float,
                   num_classes: int = 1000,
                   dropout_rate: float = 0.2,
                   drop_connect_rate: float = 0.2,
                   block: Optional[Callable[..., nn.Module]] = None,
                   norm_layer: Optional[Callable[..., nn.Module]] = None
                   ):
          super(EfficientNet, self).__init__()
  
          # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate, repeats
          default_cnf = [[3, 32, 16, 1, 1, True, drop_connect_rate, 1],
                         [3, 16, 24, 6, 2, True, drop_connect_rate, 2],
                         [5, 24, 40, 6, 2, True, drop_connect_rate, 2],
                         [3, 40, 80, 6, 2, True, drop_connect_rate, 3],
                         [5, 80, 112, 6, 1, True, drop_connect_rate, 3],
                         [5, 112, 192, 6, 2, True, drop_connect_rate, 4],
                         [3, 192, 320, 6, 1, True, drop_connect_rate, 1]]
  
          def round_repeats(repeats):
              """Round number of repeats based on depth multiplier."""
              return int(math.ceil(depth_coefficient * repeats))
  
          if block is None:
              block = InvertedResidual
  
          if norm_layer is None:
              norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)
  
          adjust_channels = partial(InvertedResidualConfig.adjust_channels,
                                    width_coefficient=width_coefficient)
  
          # build inverted_residual_setting
          bneck_conf = partial(InvertedResidualConfig,
                               width_coefficient=width_coefficient)
  
          b = 0
          num_blocks = float(sum(round_repeats(i[-1]) for i in default_cnf))
          inverted_residual_setting = []
          for stage, args in enumerate(default_cnf):
              cnf = copy.copy(args)
              for i in range(round_repeats(cnf.pop(-1))):
                  if i > 0:
                      # strides equal 1 except first cnf
                      cnf[-3] = 1  # strides
                      cnf[1] = cnf[2]  # input_channel equal output_channel
  
                  cnf[-1] = args[-2] * b / num_blocks  # update dropout ratio
                  index = str(stage + 1) + chr(i + 97)  # 1a, 2a, 2b, ...
                  inverted_residual_setting.append(bneck_conf(*cnf, index))
                  b += 1
  
          # create layers
          layers = OrderedDict()
  
          # first conv
          layers.update({"stem_conv": ConvBNActivation(in_planes=3,
                                                       out_planes=adjust_channels(32),
                                                       kernel_size=3,
                                                       stride=2,
                                                       norm_layer=norm_layer)})
  
          # building inverted residual blocks
          for cnf in inverted_residual_setting:
              layers.update({cnf.index: block(cnf, norm_layer)})
  
          # build top
          last_conv_input_c = inverted_residual_setting[-1].out_c
          last_conv_output_c = adjust_channels(1280)
          layers.update({"top": ConvBNActivation(in_planes=last_conv_input_c,
                                                 out_planes=last_conv_output_c,
                                                 kernel_size=1,
                                                 norm_layer=norm_layer)})
  
          self.features = nn.Sequential(layers)
          self.avgpool = nn.AdaptiveAvgPool2d(1)
  
          classifier = []
          if dropout_rate > 0:
              classifier.append(nn.Dropout(p=dropout_rate, inplace=True))
          classifier.append(nn.Linear(last_conv_output_c, num_classes))
          self.classifier = nn.Sequential(*classifier)
  
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
  
      def _forward_impl(self, x: Tensor) -> Tensor:
          x = self.features(x)
          x = self.avgpool(x)
          x = torch.flatten(x, 1)
          x = self.classifier(x)
  
          return x
  
      def forward(self, x: Tensor) -> Tensor:
          return self._forward_impl(x)
  
  
  def efficientnet_b0(num_classes=1000):
      # input image size 224x224
      return EfficientNet(width_coefficient=1.0,
                          depth_coefficient=1.0,
                          dropout_rate=0.2,
                          num_classes=num_classes)
  
  
  def efficientnet_b1(num_classes=1000):
      # input image size 240x240
      return EfficientNet(width_coefficient=1.0,
                          depth_coefficient=1.1,
                          dropout_rate=0.2,
                          num_classes=num_classes)
  
  def efficientnet_b2(num_classes=1000):
      # input image size 260x260
      return EfficientNet(width_coefficient=1.1,
                          depth_coefficient=1.2,
                          dropout_rate=0.3,
                          num_classes=num_classes)
  
  
  def efficientnet_b3(num_classes=1000):
      # input image size 300x300
      return EfficientNet(width_coefficient=1.2,
                          depth_coefficient=1.4,
                          dropout_rate=0.3,
                          num_classes=num_classes)
  
  
  def efficientnet_b4(num_classes=1000):
      # input image size 380x380
      return EfficientNet(width_coefficient=1.4,
                          depth_coefficient=1.8,
                          dropout_rate=0.4,
                          num_classes=num_classes)
  
  
  def efficientnet_b5(num_classes=1000):
      # input image size 456x456
      return EfficientNet(width_coefficient=1.6,
                          depth_coefficient=2.2,
                          dropout_rate=0.4,
                          num_classes=num_classes)
  
  
  def efficientnet_b6(num_classes=1000):
      # input image size 528x528
      return EfficientNet(width_coefficient=1.8,
                          depth_coefficient=2.6,
                          dropout_rate=0.5,
                          num_classes=num_classes)
  
  
  def efficientnet_b7(num_classes=1000):
      # input image size 600x600
      return EfficientNet(width_coefficient=2.0,
                          depth_coefficient=3.1,
                          dropout_rate=0.5,
                          num_classes=num_classes)
  ```

  