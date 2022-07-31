### MobileNet_v2

- 网络亮点：

  - 提出了倒残差结构
  - 使用了线性瓶颈 Linear Bottlenecks：激活函数 ReLU6=$min(max(x, 0), 6)$

- 残差结构与倒残差结构对比

  - 结构说明

    - Residual Block
      - $1 \times 1$ 卷积进行降维
      - $3 \times 3$ 卷积
      - $1 \times 1$ 卷积进行升维
    - Inverted Residual Block
      - $1 \times 1$ 卷积进行升维
      - $3 \times 3$ DW 卷积
      - $1 \times 1$ 卷积进行降维

  - 图示

    ![ResNet VS Inverted Residual Block](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/MobileNet_v2_Inver_resi_block.png)

- MobileNet V2 倒残差注意点：当 `stride=1` 并且 `输入特征与输出特征维度相同` 时才有 shortcut 连接

  - stride = 1
    1. 将输入进行 1x1 卷积计算，然后使用 ReLU6 激活函数
    2. 将得到的结果输入到 3x3 DW 卷积计算，然后使用 ReLU6 激活函数
    3. 将得到的结果进行 1x1 卷积计算，然后输入到一个 Linear 层中
    4. 将得到的结果与最初的输入进行相加操作
  - stride = 2
    1. 将输入进行 1x1 卷积计算，然后使用 ReLU6 激活函数
    2. 将得到的结果输入到 3x3，stride 为 2 的 DW 卷积，然后使用 ReLU6 激活函数
    3. 将得到的结果进行 1x1 卷积计算，然后输入到一个 Linear 层中

- 不同卷积网络结构对比

  ![不同网络结构](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/MobileNet_v2_diff_conv_structure.png)

- MobileNet V2 网络结构

  - 参数说明

    - t 表示扩展因子 [倒残差升维的倍数]
    - c 表示输出特征矩阵深度 channel
    - n 表示 bottleneck 重复的次数
    - s 表示步距【只对第一个 bottleneck 有效，后面默认为 1】

  - 结构表示

    ![MobileNet V2](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/MobileNet_v2_Structure.png)

- Pytorch 版本的 MobileNet V2 网络搭建

  ```python
  from torch import nn
  import torch
  
  
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
  
  
  class ConvBNReLU(nn.Sequential):
      def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
          padding = (kernel_size - 1) // 2
          super(ConvBNReLU, self).__init__(
              nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
              nn.BatchNorm2d(out_channel),
              nn.ReLU6(inplace=True)
          )
  
  
  class InvertedResidual(nn.Module):
      def __init__(self, in_channel, out_channel, stride, expand_ratio):
          super(InvertedResidual, self).__init__()
          hidden_channel = in_channel * expand_ratio
          self.use_shortcut = stride == 1 and in_channel == out_channel
  
          layers = []
          if expand_ratio != 1:
              # 1x1 pointwise conv
              layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
          layers.extend([
              # 3x3 depthwise conv
              ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
              # 1x1 pointwise conv(linear activation) 
              nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
              nn.BatchNorm2d(out_channel),
          ])
  
          self.conv = nn.Sequential(*layers)
  
      def forward(self, x):
          if self.use_shortcut:
              return x + self.conv(x)
          else:
              return self.conv(x)
  
  
  class MobileNetV2(nn.Module):
      def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
          super(MobileNetV2, self).__init__()
          block = InvertedResidual
          input_channel = _make_divisible(32 * alpha, round_nearest)
          last_channel = _make_divisible(1280 * alpha, round_nearest)
  
          inverted_residual_setting = [
              # t, c, n, s
              [1, 16, 1, 1],
              [6, 24, 2, 2],
              [6, 32, 3, 2],
              [6, 64, 4, 2],
              [6, 96, 3, 1],
              [6, 160, 3, 2],
              [6, 320, 1, 1],
          ]
  
          features = []
          # conv1 layer
          features.append(ConvBNReLU(3, input_channel, stride=2))
          # building inverted residual residual blockes
          for t, c, n, s in inverted_residual_setting:
              output_channel = _make_divisible(c * alpha, round_nearest)
              for i in range(n):
                  stride = s if i == 0 else 1
                  features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                  input_channel = output_channel
          # building last several layers
          features.append(ConvBNReLU(input_channel, last_channel, 1))
          # combine feature layers
          self.features = nn.Sequential(*features)
  
          # building classifier
          self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
          self.classifier = nn.Sequential(
              nn.Dropout(0.2),
              nn.Linear(last_channel, num_classes)
          )
  
          # weight initialization
          for m in self.modules():
              if isinstance(m, nn.Conv2d):
                  nn.init.kaiming_normal_(m.weight, mode='fan_out')
                  if m.bias is not None:
                      nn.init.zeros_(m.bias)
              elif isinstance(m, nn.BatchNorm2d):
                  nn.init.ones_(m.weight)
                  nn.init.zeros_(m.bias)
              elif isinstance(m, nn.Linear):
                  nn.init.normal_(m.weight, 0, 0.01)
                  nn.init.zeros_(m.bias)
  
      def forward(self, x):
          x = self.features(x)
          x = self.avgpool(x)
          x = torch.flatten(x, 1)
          x = self.classifier(x)
          return x
  ```

  

- tensorflow2 版本的 MobileNet V2 网络构建

  ```python
  from tensorflow.keras import layers, Model, Sequential
  
  
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
  
  
  class ConvBNReLU(layers.Layer):
      def __init__(self, out_channel, kernel_size=3, stride=1, **kwargs):
          super(ConvBNReLU, self).__init__(**kwargs)
          self.conv = layers.Conv2D(filters=out_channel, kernel_size=kernel_size,
                                    strides=stride, padding='SAME', use_bias=False, name='Conv2d')
          self.bn = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='BatchNorm')
          self.activation = layers.ReLU(max_value=6.0)
  
      def call(self, inputs, training=False):
          x = self.conv(inputs)
          x = self.bn(x, training=training)
          x = self.activation(x)
          return x
  
  
  class InvertedResidual(layers.Layer):
      def __init__(self, in_channel, out_channel, stride, expand_ratio, **kwargs):
          super(InvertedResidual, self).__init__(**kwargs)
          self.hidden_channel = in_channel * expand_ratio
          self.use_shortcut = stride == 1 and in_channel == out_channel
  
          layer_list = []
          if expand_ratio != 1:
              # 1x1 pointwise conv
              layer_list.append(ConvBNReLU(out_channel=self.hidden_channel, kernel_size=1, name='expand'))
  
          layer_list.extend([
              # 3x3 depthwise conv
              layers.DepthwiseConv2D(kernel_size=3, padding='SAME', strides=stride,
                                     use_bias=False, name='depthwise'),
              layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='depthwise/BatchNorm'),
              layers.ReLU(max_value=6.0),
              # 1x1 pointwise conv(linear)
              layers.Conv2D(filters=out_channel, kernel_size=1, strides=1,
                            padding='SAME', use_bias=False, name='project'),
              layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='project/BatchNorm')
          ])
          self.main_branch = Sequential(layer_list, name='expanded_conv')
  
      def call(self, inputs, training=False, **kwargs):
          if self.use_shortcut:
              return inputs + self.main_branch(inputs, training=training)
          else:
              return self.main_branch(inputs, training=training)
  
  
  def MobileNetV2(im_height=224,
                  im_width=224,
                  num_classes=1000,
                  alpha=1.0,
                  round_nearest=8,
                  include_top=True):
      block = InvertedResidual
      input_channel = _make_divisible(32 * alpha, round_nearest)
      last_channel = _make_divisible(1280 * alpha, round_nearest)
      inverted_residual_setting = [
          # t, c, n, s
          [1, 16, 1, 1],
          [6, 24, 2, 2],
          [6, 32, 3, 2],
          [6, 64, 4, 2],
          [6, 96, 3, 1],
          [6, 160, 3, 2],
          [6, 320, 1, 1],
      ]
  
      input_image = layers.Input(shape=(im_height, im_width, 3), dtype='float32')
      # conv1
      x = ConvBNReLU(input_channel, stride=2, name='Conv')(input_image)
      # building inverted residual residual blockes
      for idx, (t, c, n, s) in enumerate(inverted_residual_setting):
          output_channel = _make_divisible(c * alpha, round_nearest)
          for i in range(n):
              stride = s if i == 0 else 1
              x = block(x.shape[-1],
                        output_channel,
                        stride,
                        expand_ratio=t)(x)
      # building last several layers
      x = ConvBNReLU(last_channel, kernel_size=1, name='Conv_1')(x)
  
      if include_top is True:
          # building classifier
          x = layers.GlobalAveragePooling2D()(x)  # pool + flatten
          x = layers.Dropout(0.2)(x)
          output = layers.Dense(num_classes, name='Logits')(x)
      else:
          output = x
  
      model = Model(inputs=input_image, outputs=output)
      return model
  ```

  