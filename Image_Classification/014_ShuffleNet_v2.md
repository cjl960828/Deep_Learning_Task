### ShuffleNet V2

- 四条指导准则 【MAC 也会影响 FLOPs】

  1. 当卷积层的输入特征图与输出特征图 channel 相等时 MAC 最小 (memory access cost) [保持 FLOPs 不变的前提]
  2. 当组卷积中的 groups 增大时(保持 FLOPs 不变的前提)，此时的 MAC 也会变大
  3. 网络设计的碎片化程序越高，速度越慢 [并行处理越多，速度越慢，在 GPU 上表现明显]
  4. Element-wise 操作带来的影响是不可忽视的

- ShuffleNet V1 VS ShuffleNet V2

  - 图解

    - 图 (a) 和图 (b) 表示 ShuffleNet V1 在不同步距的两种残差结构，图 (c) 和图 (d) 表示 ShuffleNet V2 在不同步距的两种结构
    - 图 (a) 与图 (c) 进行对比
      - ShuffleNet V2 首先将通道一分为二 (Channel Split)，从而使得每一条分支上的输入通道数与输出通道数相同 $\rightarrow$ 满足 `第一条准则`，保证 MAC 最小
      - ShuffleNet V2 将 ShuffleNet V1 中采用的 $1 \times 1$ 的分组卷积替换为 $1 \times 1$ 的普通卷积 $\rightarrow$ 满足 `第二条准则`，避免多组卷积多次访问内存，从而减少 MAC
      - ShuffleNet V2 由于将组卷积替换为了普通卷积，因此在最后一个 $1 \times 1$ 卷积部分需要采用激活函数
      - ShuffleNet V2 最后采用 concat 操作而不是 add 操作，减少了 Element-wise 操作 $\rightarrow$ 满足 `第四条准则`
    - 图 (b) 与图 (d) 进行对比
      - ShuffleNet V2 与 ShuffleNet V1 一样，直接使用初始的输入特征图，但是依旧保证输入特征通道数与输出特征通道数相同 $\rightarrow$ 满足 `第一条准则`，保证 MAC 最小
      - ShuffleNet V2 将 ShuffleNet V1 中采用的 $1 \times 1$ 的分组卷积替换为 $1 \times 1$ 的普通卷积 $\rightarrow$ 满足 `第二条准则`，避免多组卷积多次访问内存，从而减少 MAC
      - ShuffleNet V2 由于将组卷积替换为了普通卷积，因此在最后一个 $1 \times 1$ 卷积部分需要采用激活函数
      - ShuffleNet V2 最后采用 concat 操作 (通道数翻倍) 而不是 add 操作，减少了 Element-wise 操作 $\rightarrow$ 满足 `第四条准则`

  - 图示

    ![ShuffleNet V2 VS ShuffleNet V1](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/ShuffleNet_v2_vs_ShuffleNet_v1.png)

- ShuffleNet V2 网络结构

  ![ShuffleNet V2 网络结构](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/ShuffleNet_v2_Structure.png)

- Pytorch 版本的 ShuffleNet V2 网络搭建

  ```python
  from typing import List, Callable
  
  import torch
  from torch import Tensor
  import torch.nn as nn
  
  
  # 通道打散操作 reshape -> transpose -> reshape
  def channel_shuffle(x: Tensor, groups: int) -> Tensor:
  
      batch_size, num_channels, height, width = x.size()
      
      # 计算各个组的通道数
      channels_per_group = num_channels // groups
  
      # reshape
      # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
      x = x.view(batch_size, groups, channels_per_group, height, width)
  
      x = torch.transpose(x, 1, 2).contiguous()
  
      # flatten
      x = x.view(batch_size, -1, height, width)
  
      return x
  
  
  # 倒残差结构，中间大，两边小，即先升维后降维
  class InvertedResidual(nn.Module):
      def __init__(self, input_c: int, output_c: int, stride: int):
          super(InvertedResidual, self).__init__()
  
          if stride not in [1, 2]:
              raise ValueError("illegal stride value.")
          self.stride = stride
  
          assert output_c % 2 == 0
          branch_features = output_c // 2
          # 当 stride 为 1 时，input_channel 应该是 branch_features 的两倍
          # python 中 '<<' 是位运算，可理解为计算 ×2 的快速方法
          assert (self.stride != 1) or (input_c == branch_features << 1)
  
          # 图 (d) 中的左分支构建
          if self.stride == 2:
              self.branch1 = nn.Sequential(
                  self.depthwise_conv(input_c, input_c, kernel_s=3, stride=self.stride, padding=1),
                  nn.BatchNorm2d(input_c),
                  nn.Conv2d(input_c, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                  nn.BatchNorm2d(branch_features),
                  nn.ReLU(inplace=True)
              )
          else:
              self.branch1 = nn.Sequential()
  
          # 右分支构建
          self.branch2 = nn.Sequential(
              nn.Conv2d(input_c if self.stride > 1 else branch_features, branch_features, kernel_size=1,
                        stride=1, padding=0, bias=False),
              nn.BatchNorm2d(branch_features),
              nn.ReLU(inplace=True),
              self.depthwise_conv(branch_features, branch_features, kernel_s=3, stride=self.stride, padding=1),
              nn.BatchNorm2d(branch_features),
              nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
              nn.BatchNorm2d(branch_features),
              nn.ReLU(inplace=True)  # 使用普通卷积需要采用激活函数
          )
  
      @staticmethod
      def depthwise_conv(input_c: int,
                         output_c: int,
                         kernel_s: int,
                         stride: int = 1,
                         padding: int = 0,
                         bias: bool = False) -> nn.Conv2d:
          return nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=kernel_s,
                           stride=stride, padding=padding, bias=bias, groups=input_c)
  
      def forward(self, x: Tensor) -> Tensor:
          if self.stride == 1:
              x1, x2 = x.chunk(2, dim=1)
              out = torch.cat((x1, self.branch2(x2)), dim=1)
          else:
              out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
  
          out = channel_shuffle(out, 2)
  
          return out
  
  
  class ShuffleNetV2(nn.Module):
      def __init__(self,
                   stages_repeats: List[int],
                   stages_out_channels: List[int],
                   num_classes: int = 1000,
                   inverted_residual: Callable[..., nn.Module] = InvertedResidual):
          super(ShuffleNetV2, self).__init__()
  
          if len(stages_repeats) != 3:
              raise ValueError("expected stages_repeats as list of 3 positive ints")
          if len(stages_out_channels) != 5:
              raise ValueError("expected stages_out_channels as list of 5 positive ints")
          self._stage_out_channels = stages_out_channels
  
          # input RGB image
          input_channels = 3
          output_channels = self._stage_out_channels[0]
  
          self.conv1 = nn.Sequential(
              nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1, bias=False),
              nn.BatchNorm2d(output_channels),
              nn.ReLU(inplace=True)
          )
          input_channels = output_channels
  
          self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
  
          # Static annotations for mypy
          self.stage2: nn.Sequential
          self.stage3: nn.Sequential
          self.stage4: nn.Sequential
  
          stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
          for name, repeats, output_channels in zip(stage_names, stages_repeats,
                                                    self._stage_out_channels[1:]):
              seq = [inverted_residual(input_channels, output_channels, 2)]
              for i in range(repeats - 1):
                  seq.append(inverted_residual(output_channels, output_channels, 1))
              setattr(self, name, nn.Sequential(*seq))
              input_channels = output_channels
  
          output_channels = self._stage_out_channels[-1]
          self.conv5 = nn.Sequential(
              nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
              nn.BatchNorm2d(output_channels),
              nn.ReLU(inplace=True)
          )
  
          self.fc = nn.Linear(output_channels, num_classes)
  
      def _forward_impl(self, x: Tensor) -> Tensor:
          # See note [TorchScript super()]
          x = self.conv1(x)
          x = self.maxpool(x)
          x = self.stage2(x)
          x = self.stage3(x)
          x = self.stage4(x)
          x = self.conv5(x)
          x = x.mean([2, 3])  # global pool
          x = self.fc(x)
          return x
  
      def forward(self, x: Tensor) -> Tensor:
          return self._forward_impl(x)
  
  
  def shufflenet_v2_x1_0(num_classes=1000):
      """
      Constructs a ShuffleNetV2 with 1.0x output channels, as described in
      `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
      <https://arxiv.org/abs/1807.11164>`.
      weight: https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth
  
      :param num_classes:
      :return:
      """
      model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                           stages_out_channels=[24, 116, 232, 464, 1024],
                           num_classes=num_classes)
  
      return model
  
  
  def shufflenet_v2_x0_5(num_classes=1000):
      """
      Constructs a ShuffleNetV2 with 0.5x output channels, as described in
      `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
      <https://arxiv.org/abs/1807.11164>`.
      weight: https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth
  
      :param num_classes:
      :return:
      """
      model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                           stages_out_channels=[24, 48, 96, 192, 1024],
                           num_classes=num_classes)
  
      return model
  ```

- Tensorflow 2 版本的 ShuffleNet V2 网络搭建

  ```python
  import tensorflow as tf
  from tensorflow.keras import layers, Model
  
  
  class ConvBNReLU(layers.Layer):
      def __init__(self,
                   filters: int = 1,
                   kernel_size: int = 1,
                   strides: int = 1,
                   padding: str = 'same',
                   **kwargs):
          super(ConvBNReLU, self).__init__(**kwargs)
  
          self.conv = layers.Conv2D(filters=filters,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    padding=padding,
                                    use_bias=False,
                                    kernel_regularizer=tf.keras.regularizers.l2(4e-5),
                                    name="conv1")
          self.bn = layers.BatchNormalization(momentum=0.9, name="bn")
          self.relu = layers.ReLU()
  
      def call(self, inputs, training=None, **kwargs):
          x = self.conv(inputs)
          x = self.bn(x, training=training)
          x = self.relu(x)
          return x
  
  
  class DWConvBN(layers.Layer):
      def __init__(self,
                   kernel_size: int = 3,
                   strides: int = 1,
                   padding: str = 'same',
                   **kwargs):
          super(DWConvBN, self).__init__(**kwargs)
          self.dw_conv = layers.DepthwiseConv2D(kernel_size=kernel_size,
                                                strides=strides,
                                                padding=padding,
                                                use_bias=False,
                                                kernel_regularizer=tf.keras.regularizers.l2(4e-5),
                                                name="dw1")
          self.bn = layers.BatchNormalization(momentum=0.9, name="bn")
  
      def call(self, inputs, training=None, **kwargs):
          x = self.dw_conv(inputs)
          x = self.bn(x, training=training)
          return x
  
  
  class ChannelShuffle(layers.Layer):
      def __init__(self, shape, groups: int = 2, **kwargs):
          super(ChannelShuffle, self).__init__(**kwargs)
          batch_size, height, width, num_channels = shape
          assert num_channels % 2 == 0
          channel_per_group = num_channels // groups
  
          # Tuple of integers, does not include the samples dimension (batch size).
          self.reshape1 = layers.Reshape((height, width, groups, channel_per_group))
          self.reshape2 = layers.Reshape((height, width, num_channels))
  
      def call(self, inputs, **kwargs):
          x = self.reshape1(inputs)
          x = tf.transpose(x, perm=[0, 1, 2, 4, 3])
          x = self.reshape2(x)
          return x
  
  
  class ChannelSplit(layers.Layer):
      def __init__(self, num_splits: int = 2, **kwargs):
          super(ChannelSplit, self).__init__(**kwargs)
          self.num_splits = num_splits
  
      def call(self, inputs, **kwargs):
          b1, b2 = tf.split(inputs,
                            num_or_size_splits=self.num_splits,
                            axis=-1)
          return b1, b2
  
  
  def shuffle_block_s1(inputs, output_c: int, stride: int, prefix: str):
      if stride != 1:
          raise ValueError("illegal stride value.")
  
      assert output_c % 2 == 0
      branch_c = output_c // 2
  
      x1, x2 = ChannelSplit(name=prefix + "/split")(inputs)
  
      # main branch
      x2 = ConvBNReLU(filters=branch_c, name=prefix + "/b2_conv1")(x2)
      x2 = DWConvBN(kernel_size=3, strides=stride, name=prefix + "/b2_dw1")(x2)
      x2 = ConvBNReLU(filters=branch_c, name=prefix + "/b2_conv2")(x2)
  
      x = layers.Concatenate(name=prefix + "/concat")([x1, x2])
      x = ChannelShuffle(x.shape, name=prefix + "/channelshuffle")(x)
  
      return x
  
  
  def shuffle_block_s2(inputs, output_c: int, stride: int, prefix: str):
      if stride != 2:
          raise ValueError("illegal stride value.")
  
      assert output_c % 2 == 0
      branch_c = output_c // 2
  
      # shortcut branch
      x1 = DWConvBN(kernel_size=3, strides=stride, name=prefix + "/b1_dw1")(inputs)
      x1 = ConvBNReLU(filters=branch_c, name=prefix + "/b1_conv1")(x1)
  
      # main branch
      x2 = ConvBNReLU(filters=branch_c, name=prefix + "/b2_conv1")(inputs)
      x2 = DWConvBN(kernel_size=3, strides=stride, name=prefix + "/b2_dw1")(x2)
      x2 = ConvBNReLU(filters=branch_c, name=prefix + "/b2_conv2")(x2)
  
      x = layers.Concatenate(name=prefix + "/concat")([x1, x2])
      x = ChannelShuffle(x.shape, name=prefix + "/channelshuffle")(x)
  
      return x
  
  
  def shufflenet_v2(num_classes: int,
                    input_shape: tuple,
                    stages_repeats: list,
                    stages_out_channels: list):
      img_input = layers.Input(shape=input_shape)
      if len(stages_repeats) != 3:
          raise ValueError("expected stages_repeats as list of 3 positive ints")
      if len(stages_out_channels) != 5:
          raise ValueError("expected stages_out_channels as list of 5 positive ints")
  
      x = ConvBNReLU(filters=stages_out_channels[0],
                     kernel_size=3,
                     strides=2,
                     name="conv1")(img_input)
  
      x = layers.MaxPooling2D(pool_size=(3, 3),
                              strides=2,
                              padding='same',
                              name="maxpool")(x)
  
      stage_name = ["stage{}".format(i) for i in [2, 3, 4]]
      for name, repeats, output_channels in zip(stage_name,
                                                stages_repeats,
                                                stages_out_channels[1:]):
          for i in range(repeats):
              if i == 0:
                  x = shuffle_block_s2(x, output_c=output_channels, stride=2, prefix=name + "_{}".format(i))
              else:
                  x = shuffle_block_s1(x, output_c=output_channels, stride=1, prefix=name + "_{}".format(i))
  
      x = ConvBNReLU(filters=stages_out_channels[-1], name="conv5")(x)
  
      x = layers.GlobalAveragePooling2D(name="globalpool")(x)
  
      x = layers.Dense(units=num_classes, name="fc")(x)
      x = layers.Softmax()(x)
  
      model = Model(img_input, x, name="ShuffleNetV2_1.0")
  
      return model
  
  
  def shufflenet_v2_x1_0(num_classes=1000, input_shape=(224, 224, 3)):
      # 权重链接: https://pan.baidu.com/s/1M2mp98Si9eT9qT436DcdOw  密码: mhts
      model = shufflenet_v2(num_classes=num_classes,
                            input_shape=input_shape,
                            stages_repeats=[4, 8, 4],
                            stages_out_channels=[24, 116, 232, 464, 1024])
      return model
  
  
  def shufflenet_v2_x0_5(num_classes=1000, input_shape=(224, 224, 3)):
      model = shufflenet_v2(num_classes=num_classes,
                            input_shape=input_shape,
                            stages_repeats=[4, 8, 4],
                            stages_out_channels=[24, 48, 96, 192, 1024])
      return model
  
  
  def shufflenet_v2_x2_0(num_classes=1000, input_shape=(224, 224, 3)):
      model = shufflenet_v2(num_classes=num_classes,
                            input_shape=input_shape,
                            stages_repeats=[4, 8, 4],
                            stages_out_channels=[24, 244, 488, 976, 2048])
      return model
  ```

  