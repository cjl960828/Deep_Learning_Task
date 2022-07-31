### MobileNet_v3

- 网络亮点

  - 更新了 Block(bottleneck)

    - 加入了 SE 模块 (avgpool $\rightarrow$ FC $\rightarrow$ FC)

    - 更新了激活函数 (FC(ReLU), FC(hard-sigmoid))
      $$
      \sigma(x) = \frac {1}{1+e^{-x}} \\ ReLU6(x) = min(max(x, 0), 6) \\ h-sigmoid = \frac {ReLU6(x+3)}{6} \\ h-swish = x \frac  {ReLU6(x+3)}{6}
      $$

  - 使用 NAS 进行参数搜索 (Neural Architecture Search)

  - 重新设计了耗时层的结构

    - 减少了第一个卷积层的卷积核个数 (32 $\rightarrow$ 16)
    - 精简了 Last Stage：将最后一个 bottleneck 网络中的 $ 3 \times 3 + 1\times1$ 网络架构使用 avg pool 操作替代

- MobileNet V3 中的网络变化

  - 两个版本：Large 和 Small，分别适用于不同的场景;
  - 使用 NetAdapt 算法获得卷积核和通道的最佳数量;
  - 继承 V1 的深度可分离卷积;
  - 继承V2 的倒残差结构;
  - 引入 SE 通道注意力结构;
  - 使用了一种新的激活函数 h-swish(x) 代替 Relu6，h 的意思表示 hard;
  - 使用了 $\frac{Relu6(x + 3)}{6}$来近似 SE 模块中的 sigmoid;
  - 修改了 MobileNetV2 后端输出 head;

- 残差结构 (ResNet)，倒残差结构 (MobileNet V2)，SE 倒残差结构 (MobileNet V3) 结构比较

  - 残差块 VS 倒残差块

    ![残差块 VS 倒残差块](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/MobileNet_v2_Inver_resi_block.png)

  - SE 倒残差块

    ![SE 倒残差块](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/MobileNet_v3_Inver_resi_block.png)

- 激活函数比对，采用类线性拟合非线性函数

  ![激活函数比对](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/ActivationFunction_compare.png)

- 最后的输出 head 结构

  - 图解

    - 上图：由 NetAdapt 算法获取的结构
    - 下图：对搜索出的结构修改后得到的结构

  - 图示

    ![输出 head](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/MobileNet_v3_last_stage.png)

- MobileNet V3 small 结构

  ![MobileNet V3 small 结构](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/MobileNet_v3_small_Structure.png)

- MobileNet V3 Large 结构

  ![MobileNet V3 Large 结构](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/MobileNet_v3_large_Structure.png)

- pytorch 版本的 MobileNet V3 网络搭建

  ```python
  from typing import Callable, List, Optional
  
  import torch
  from torch import nn, Tensor
  from torch.nn import functional as F
  from functools import partial
  
  
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
  
  
  class ConvBNActivation(nn.Sequential):
      def __init__(self,
                   in_planes: int,
                   out_planes: int,
                   kernel_size: int = 3,
                   stride: int = 1,
                   groups: int = 1,
                   norm_layer: Optional[Callable[..., nn.Module]] = None,
                   activation_layer: Optional[Callable[..., nn.Module]] = None):
          padding = (kernel_size - 1) // 2
          if norm_layer is None:
              norm_layer = nn.BatchNorm2d
          if activation_layer is None:
              activation_layer = nn.ReLU6
          super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes,
                                                           out_channels=out_planes,
                                                           kernel_size=kernel_size,
                                                           stride=stride,
                                                           padding=padding,
                                                           groups=groups,
                                                           bias=False),
                                                 norm_layer(out_planes),
                                                 activation_layer(inplace=True))
  
  # SE 网络
  class SqueezeExcitation(nn.Module):
      def __init__(self, input_c: int, squeeze_factor: int = 4):
          super(SqueezeExcitation, self).__init__()
          squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
          # 降维后升维
          self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
          self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)
  
      def forward(self, x: Tensor) -> Tensor:
          scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
          scale = self.fc1(scale)
          scale = F.relu(scale, inplace=True)
          scale = self.fc2(scale)
          scale = F.hardsigmoid(scale, inplace=True)
          return scale * x
  
  
  class InvertedResidualConfig:
      def __init__(self,
                   input_c: int,
                   kernel: int,
                   expanded_c: int,
                   out_c: int,
                   use_se: bool,
                   activation: str,
                   stride: int,
                   width_multi: float):
          self.input_c = self.adjust_channels(input_c, width_multi)
          self.kernel = kernel
          self.expanded_c = self.adjust_channels(expanded_c, width_multi)
          self.out_c = self.adjust_channels(out_c, width_multi)
          self.use_se = use_se
          self.use_hs = activation == "HS"  # whether using h-swish activation
          self.stride = stride
  
      @staticmethod
      def adjust_channels(channels: int, width_multi: float):
          return _make_divisible(channels * width_multi, 8)
  
  
  class InvertedResidual(nn.Module):
      def __init__(self,
                   cnf: InvertedResidualConfig,
                   norm_layer: Callable[..., nn.Module]):
          super(InvertedResidual, self).__init__()
  
          if cnf.stride not in [1, 2]:
              raise ValueError("illegal stride value.")
  
          self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)
  
          layers: List[nn.Module] = []
          activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU
  
          # expand
          if cnf.expanded_c != cnf.input_c:
              layers.append(ConvBNActivation(cnf.input_c,
                                             cnf.expanded_c,
                                             kernel_size=1,
                                             norm_layer=norm_layer,
                                             activation_layer=activation_layer))
  
          # depthwise
          layers.append(ConvBNActivation(cnf.expanded_c,
                                         cnf.expanded_c,
                                         kernel_size=cnf.kernel,
                                         stride=cnf.stride,
                                         groups=cnf.expanded_c,
                                         norm_layer=norm_layer,
                                         activation_layer=activation_layer))
  
          if cnf.use_se:
              layers.append(SqueezeExcitation(cnf.expanded_c))
  
          # project
          layers.append(ConvBNActivation(cnf.expanded_c,
                                         cnf.out_c,
                                         kernel_size=1,
                                         norm_layer=norm_layer,
                                         activation_layer=nn.Identity))
  
          self.block = nn.Sequential(*layers)
          self.out_channels = cnf.out_c
          self.is_strided = cnf.stride > 1
  
      def forward(self, x: Tensor) -> Tensor:
          result = self.block(x)
          if self.use_res_connect:
              result += x
  
          return result
  
  
  class MobileNetV3(nn.Module):
      def __init__(self,
                   inverted_residual_setting: List[InvertedResidualConfig],
                   last_channel: int,
                   num_classes: int = 1000,
                   block: Optional[Callable[..., nn.Module]] = None,
                   norm_layer: Optional[Callable[..., nn.Module]] = None):
          super(MobileNetV3, self).__init__()
  
          if not inverted_residual_setting:
              raise ValueError("The inverted_residual_setting should not be empty.")
          elif not (isinstance(inverted_residual_setting, List) and
                    all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
              raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")
  
          if block is None:
              block = InvertedResidual
  
          if norm_layer is None:
              norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
  
          layers: List[nn.Module] = []
  
          # building first layer
          firstconv_output_c = inverted_residual_setting[0].input_c
          layers.append(ConvBNActivation(3,
                                         firstconv_output_c,
                                         kernel_size=3,
                                         stride=2,
                                         norm_layer=norm_layer,
                                         activation_layer=nn.Hardswish))
          # building inverted residual blocks
          for cnf in inverted_residual_setting:
              layers.append(block(cnf, norm_layer))
  
          # building last several layers
          lastconv_input_c = inverted_residual_setting[-1].out_c
          lastconv_output_c = 6 * lastconv_input_c
          layers.append(ConvBNActivation(lastconv_input_c,
                                         lastconv_output_c,
                                         kernel_size=1,
                                         norm_layer=norm_layer,
                                         activation_layer=nn.Hardswish))
          self.features = nn.Sequential(*layers)
          self.avgpool = nn.AdaptiveAvgPool2d(1)
          self.classifier = nn.Sequential(nn.Linear(lastconv_output_c, last_channel),
                                          nn.Hardswish(inplace=True),
                                          nn.Dropout(p=0.2, inplace=True),
                                          nn.Linear(last_channel, num_classes))
  
          # initial weights
          for m in self.modules():
              if isinstance(m, nn.Conv2d):
                  nn.init.kaiming_normal_(m.weight, mode="fan_out")
                  if m.bias is not None:
                      nn.init.zeros_(m.bias)
              elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
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
  
  
  def mobilenet_v3_large(num_classes: int = 1000,
                         reduced_tail: bool = False) -> MobileNetV3:
      """
      Constructs a large MobileNetV3 architecture from
      "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>.
  
      weights_link:
      https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth
  
      Args:
          num_classes (int): number of classes
          reduced_tail (bool): If True, reduces the channel counts of all feature layers
              between C4 and C5 by 2. It is used to reduce the channel redundancy in the
              backbone for Detection and Segmentation.
      """
      width_multi = 1.0
      bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
      adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)
  
      reduce_divider = 2 if reduced_tail else 1
  
      inverted_residual_setting = [
          # input_c, kernel, expanded_c, out_c, use_se, activation, stride
          bneck_conf(16, 3, 16, 16, False, "RE", 1),
          bneck_conf(16, 3, 64, 24, False, "RE", 2),  # C1
          bneck_conf(24, 3, 72, 24, False, "RE", 1),
          bneck_conf(24, 5, 72, 40, True, "RE", 2),  # C2
          bneck_conf(40, 5, 120, 40, True, "RE", 1),
          bneck_conf(40, 5, 120, 40, True, "RE", 1),
          bneck_conf(40, 3, 240, 80, False, "HS", 2),  # C3
          bneck_conf(80, 3, 200, 80, False, "HS", 1),
          bneck_conf(80, 3, 184, 80, False, "HS", 1),
          bneck_conf(80, 3, 184, 80, False, "HS", 1),
          bneck_conf(80, 3, 480, 112, True, "HS", 1),
          bneck_conf(112, 3, 672, 112, True, "HS", 1),
          bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2),  # C4
          bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
          bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
      ]
      last_channel = adjust_channels(1280 // reduce_divider)  # C5
  
      return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                         last_channel=last_channel,
                         num_classes=num_classes)
  
  
  def mobilenet_v3_small(num_classes: int = 1000,
                         reduced_tail: bool = False) -> MobileNetV3:
      """
      Constructs a large MobileNetV3 architecture from
      "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>.
  
      weights_link:
      https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth
  
      Args:
          num_classes (int): number of classes
          reduced_tail (bool): If True, reduces the channel counts of all feature layers
              between C4 and C5 by 2. It is used to reduce the channel redundancy in the
              backbone for Detection and Segmentation.
      """
      width_multi = 1.0
      bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
      adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)
  
      reduce_divider = 2 if reduced_tail else 1
  
      inverted_residual_setting = [
          # input_c, kernel, expanded_c, out_c, use_se, activation, stride
          bneck_conf(16, 3, 16, 16, True, "RE", 2),  # C1
          bneck_conf(16, 3, 72, 24, False, "RE", 2),  # C2
          bneck_conf(24, 3, 88, 24, False, "RE", 1),
          bneck_conf(24, 5, 96, 40, True, "HS", 2),  # C3
          bneck_conf(40, 5, 240, 40, True, "HS", 1),
          bneck_conf(40, 5, 240, 40, True, "HS", 1),
          bneck_conf(40, 5, 120, 48, True, "HS", 1),
          bneck_conf(48, 5, 144, 48, True, "HS", 1),
          bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2),  # C4
          bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1),
          bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1)
      ]
      last_channel = adjust_channels(1024 // reduce_divider)  # C5
  
      return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                         last_channel=last_channel,
                         num_classes=num_classes)
  
  ```

- tensorflow2 版本的 MobileNet V3 网络搭建

  ```python
  from typing import Union
  from functools import partial
  from tensorflow.keras import layers, Model
  
  
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
  
  
  def correct_pad(input_size: Union[int, tuple], kernel_size: int):
      """Returns a tuple for zero-padding for 2D convolution with downsampling.
  
      Arguments:
        input_size: Input tensor size.
        kernel_size: An integer or tuple/list of 2 integers.
  
      Returns:
        A tuple.
      """
  
      if isinstance(input_size, int):
          input_size = (input_size, input_size)
  
      kernel_size = (kernel_size, kernel_size)
  
      adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
      correct = (kernel_size[0] // 2, kernel_size[1] // 2)
      return ((correct[0] - adjust[0], correct[0]),
              (correct[1] - adjust[1], correct[1]))
  
  
  class HardSigmoid(layers.Layer):
      def __init__(self, **kwargs):
          super(HardSigmoid, self).__init__(**kwargs)
          self.relu6 = layers.ReLU(6.)
  
      def call(self, inputs, **kwargs):
          x = self.relu6(inputs + 3) * (1. / 6)
          return x
  
  
  class HardSwish(layers.Layer):
      def __init__(self, **kwargs):
          super(HardSwish, self).__init__(**kwargs)
          self.hard_sigmoid = HardSigmoid()
  
      def call(self, inputs, **kwargs):
          x = self.hard_sigmoid(inputs) * inputs
          return x
  
  
  def _se_block(inputs, filters, prefix, se_ratio=1 / 4.):
      # [batch, height, width, channel] -> [batch, channel]
      x = layers.GlobalAveragePooling2D(name=prefix + 'squeeze_excite/AvgPool')(inputs)
  
      # Target shape. Tuple of integers, does not include the samples dimension (batch size).
      # [batch, channel] -> [batch, 1, 1, channel]
      x = layers.Reshape((1, 1, filters))(x)
  
      # fc1
      x = layers.Conv2D(filters=_make_divisible(filters * se_ratio),
                        kernel_size=1,
                        padding='same',
                        name=prefix + 'squeeze_excite/Conv')(x)
      x = layers.ReLU(name=prefix + 'squeeze_excite/Relu')(x)
  
      # fc2
      x = layers.Conv2D(filters=filters,
                        kernel_size=1,
                        padding='same',
                        name=prefix + 'squeeze_excite/Conv_1')(x)
      x = HardSigmoid(name=prefix + 'squeeze_excite/HardSigmoid')(x)
  
      x = layers.Multiply(name=prefix + 'squeeze_excite/Mul')([inputs, x])
      return x
  
  
  def _inverted_res_block(x,
                          input_c: int,      # input channel
                          kernel_size: int,  # kennel size
                          exp_c: int,        # expanded channel
                          out_c: int,        # out channel
                          use_se: bool,      # whether using SE
                          activation: str,   # RE or HS
                          stride: int,
                          block_id: int,
                          alpha: float = 1.0):
  
      bn = partial(layers.BatchNormalization, epsilon=0.001, momentum=0.99)
  
      input_c = _make_divisible(input_c * alpha)
      exp_c = _make_divisible(exp_c * alpha)
      out_c = _make_divisible(out_c * alpha)
  
      act = layers.ReLU if activation == "RE" else HardSwish
  
      shortcut = x
      prefix = 'expanded_conv/'
      if block_id:
          # expand channel
          prefix = 'expanded_conv_{}/'.format(block_id)
          x = layers.Conv2D(filters=exp_c,
                            kernel_size=1,
                            padding='same',
                            use_bias=False,
                            name=prefix + 'expand')(x)
          x = bn(name=prefix + 'expand/BatchNorm')(x)
          x = act(name=prefix + 'expand/' + act.__name__)(x)
  
      if stride == 2:
          input_size = (x.shape[1], x.shape[2])  # height, width
          x = layers.ZeroPadding2D(padding=correct_pad(input_size, kernel_size),
                                   name=prefix + 'depthwise/pad')(x)
  
      x = layers.DepthwiseConv2D(kernel_size=kernel_size,
                                 strides=stride,
                                 padding='same' if stride == 1 else 'valid',
                                 use_bias=False,
                                 name=prefix + 'depthwise')(x)
      x = bn(name=prefix + 'depthwise/BatchNorm')(x)
      x = act(name=prefix + 'depthwise/' + act.__name__)(x)
  
      if use_se:
          x = _se_block(x, filters=exp_c, prefix=prefix)
  
      x = layers.Conv2D(filters=out_c,
                        kernel_size=1,
                        padding='same',
                        use_bias=False,
                        name=prefix + 'project')(x)
      x = bn(name=prefix + 'project/BatchNorm')(x)
  
      if stride == 1 and input_c == out_c:
          x = layers.Add(name=prefix + 'Add')([shortcut, x])
  
      return x
  
  
  def mobilenet_v3_large(input_shape=(224, 224, 3),
                         num_classes=1000,
                         alpha=1.0,
                         include_top=True):
      """
      download weights url:
      链接: https://pan.baidu.com/s/13uJznKeqHkjUp72G_gxe8Q  密码: 8quu
      """
      bn = partial(layers.BatchNormalization, epsilon=0.001, momentum=0.99)
      img_input = layers.Input(shape=input_shape)
  
      x = layers.Conv2D(filters=16,
                        kernel_size=3,
                        strides=(2, 2),
                        padding='same',
                        use_bias=False,
                        name="Conv")(img_input)
      x = bn(name="Conv/BatchNorm")(x)
      x = HardSwish(name="Conv/HardSwish")(x)
  
      inverted_cnf = partial(_inverted_res_block, alpha=alpha)
      # input, input_c, k_size, expand_c, use_se, activation, stride, block_id
      x = inverted_cnf(x, 16, 3, 16, 16, False, "RE", 1, 0)
      x = inverted_cnf(x, 16, 3, 64, 24, False, "RE", 2, 1)
      x = inverted_cnf(x, 24, 3, 72, 24, False, "RE", 1, 2)
      x = inverted_cnf(x, 24, 5, 72, 40, True, "RE", 2, 3)
      x = inverted_cnf(x, 40, 5, 120, 40, True, "RE", 1, 4)
      x = inverted_cnf(x, 40, 5, 120, 40, True, "RE", 1, 5)
      x = inverted_cnf(x, 40, 3, 240, 80, False, "HS", 2, 6)
      x = inverted_cnf(x, 80, 3, 200, 80, False, "HS", 1, 7)
      x = inverted_cnf(x, 80, 3, 184, 80, False, "HS", 1, 8)
      x = inverted_cnf(x, 80, 3, 184, 80, False, "HS", 1, 9)
      x = inverted_cnf(x, 80, 3, 480, 112, True, "HS", 1, 10)
      x = inverted_cnf(x, 112, 3, 672, 112, True, "HS", 1, 11)
      x = inverted_cnf(x, 112, 5, 672, 160, True, "HS", 2, 12)
      x = inverted_cnf(x, 160, 5, 960, 160, True, "HS", 1, 13)
      x = inverted_cnf(x, 160, 5, 960, 160, True, "HS", 1, 14)
  
      last_c = _make_divisible(160 * 6 * alpha)
      last_point_c = _make_divisible(1280 * alpha)
  
      x = layers.Conv2D(filters=last_c,
                        kernel_size=1,
                        padding='same',
                        use_bias=False,
                        name="Conv_1")(x)
      x = bn(name="Conv_1/BatchNorm")(x)
      x = HardSwish(name="Conv_1/HardSwish")(x)
  
      if include_top is True:
          x = layers.GlobalAveragePooling2D()(x)
          x = layers.Reshape((1, 1, last_c))(x)
  
          # fc1
          x = layers.Conv2D(filters=last_point_c,
                            kernel_size=1,
                            padding='same',
                            name="Conv_2")(x)
          x = HardSwish(name="Conv_2/HardSwish")(x)
  
          # fc2
          x = layers.Conv2D(filters=num_classes,
                            kernel_size=1,
                            padding='same',
                            name='Logits/Conv2d_1c_1x1')(x)
          x = layers.Flatten()(x)
          x = layers.Softmax(name="Predictions")(x)
  
      model = Model(img_input, x, name="MobilenetV3large")
  
      return model
  
  
  def mobilenet_v3_small(input_shape=(224, 224, 3),
                         num_classes=1000,
                         alpha=1.0,
                         include_top=True):
      """
      download weights url:
      链接: https://pan.baidu.com/s/1vrQ_6HdDTHL1UUAN6nSEcw  密码: rrf0
      """
      bn = partial(layers.BatchNormalization, epsilon=0.001, momentum=0.99)
      img_input = layers.Input(shape=input_shape)
  
      x = layers.Conv2D(filters=16,
                        kernel_size=3,
                        strides=(2, 2),
                        padding='same',
                        use_bias=False,
                        name="Conv")(img_input)
      x = bn(name="Conv/BatchNorm")(x)
      x = HardSwish(name="Conv/HardSwish")(x)
  
      inverted_cnf = partial(_inverted_res_block, alpha=alpha)
      # input, input_c, k_size, expand_c, use_se, activation, stride, block_id
      x = inverted_cnf(x, 16, 3, 16, 16, True, "RE", 2, 0)
      x = inverted_cnf(x, 16, 3, 72, 24, False, "RE", 2, 1)
      x = inverted_cnf(x, 24, 3, 88, 24, False, "RE", 1, 2)
      x = inverted_cnf(x, 24, 5, 96, 40, True, "HS", 2, 3)
      x = inverted_cnf(x, 40, 5, 240, 40, True, "HS", 1, 4)
      x = inverted_cnf(x, 40, 5, 240, 40, True, "HS", 1, 5)
      x = inverted_cnf(x, 40, 5, 120, 48, True, "HS", 1, 6)
      x = inverted_cnf(x, 48, 5, 144, 48, True, "HS", 1, 7)
      x = inverted_cnf(x, 48, 5, 288, 96, True, "HS", 2, 8)
      x = inverted_cnf(x, 96, 5, 576, 96, True, "HS", 1, 9)
      x = inverted_cnf(x, 96, 5, 576, 96, True, "HS", 1, 10)
  
      last_c = _make_divisible(96 * 6 * alpha)
      last_point_c = _make_divisible(1024 * alpha)
  
      x = layers.Conv2D(filters=last_c,
                        kernel_size=1,
                        padding='same',
                        use_bias=False,
                        name="Conv_1")(x)
      x = bn(name="Conv_1/BatchNorm")(x)
      x = HardSwish(name="Conv_1/HardSwish")(x)
  
      if include_top is True:
          x = layers.GlobalAveragePooling2D()(x)
          x = layers.Reshape((1, 1, last_c))(x)
  
          # fc1
          x = layers.Conv2D(filters=last_point_c,
                            kernel_size=1,
                            padding='same',
                            name="Conv_2")(x)
          x = HardSwish(name="Conv_2/HardSwish")(x)
  
          # fc2
          x = layers.Conv2D(filters=num_classes,
                            kernel_size=1,
                            padding='same',
                            name='Logits/Conv2d_1c_1x1')(x)
          x = layers.Flatten()(x)
          x = layers.Softmax(name="Predictions")(x)
  
      model = Model(img_input, x, name="MobilenetV3large")
  
      return model
  ```

  