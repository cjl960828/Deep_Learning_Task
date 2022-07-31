### ResNet

- 网络亮点：

  1. 超深的网络结构

  2. 提出了 residual 模块

  3. 使用 Batch Normalization 加速训练，丢弃了 Dropout

     - 当 batch = 1 的时候，BN 没有意义，代码可能会报错

     - BN 对输入的 batch 进行归一化操作，使得输出结果满足均值为 0 方差为 1 的分布规律

- 无限堆叠卷积 + 池化层会出现的问题

  1. 梯度消失或者梯度爆炸 【可以采用 BN 进行解决】

     - 梯度消失：当梯度为 0.9 时，堆叠 $n$ 层后，$0.9^n \approx 0$
     - 梯度爆炸：当梯度为 1.1 时，堆叠 n 层后，$1.1^n \approx \infin$

  2. 退化问题【模型越深，性能越差】

     ![退化问题](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/ResNet4degradation .png)

- ResNet 中提出的解决退化问题的方法 【残差结构】

  ![采用残差结构后堆叠网络层的实验结果](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/ResNet4SolveDegradation.png)

- 残差结构

  - 图解

    - 左图：用于浅层的残差结构 【18 或者 34 层】
    - 右图：用于深层的残差结构 【50层、101 层或者 152 层】

  - 残差块设置作用

    - 左图：用于浅层的残差网络，可以保证模型能够拟合数据
    - 右图：用于深层的残差网络，可以有效减少模型参数和降低运算量

  - 参数量比较 【假设残差块的输入和输出维度均为 256】

    - 左图：$3 \times 3 \times 256 \times 256 \times 2 = 1179648$
    - 右图：$1 \times 1 \times 256 \times 64 + 3 \times 3 \times 64 \times 64 + 1 \times 1 \times 64 \times 256 = 69632$

  - 图示

    ![两种残差结构](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/ResNet4ResidualBlock.png)

- ResNet34 网络结构

  - 结构说明

    - Conv2_x 为第一个残差块组合，其残差连接均为实线连接，上图左图方式
    - Conv3_x - Conv5_x 为后三个残差块组合，其中第一个残差过程是虚线，其余残差过程为实线。原因是虚线部分残差部分的输入维度和输出维度不同，因此需要在残差连接部分连接一个 $1 \times 1$ 的卷积用于维度控制。比如在 Conv3_x 中，输入维度为 64，输出维度为 128，因此需要采用 $1 \times 1$ 的结构进行升维处理

  - 图示

    ![ResNet 34](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/ResNet4Layer34.png)

- 不同层数的 ResNet 网络结构组成

  ![ResNet](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/ResNet.png)

- Pytorch 版本的 ResNet 网络搭建

  ```python
  class BasicBlock(nn.Module):
      # 浅层的 ResNet 采用的 expansion = 1，expansion 表示通道维度变化率
      expansion = 1
  
      def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
          # 由两个卷积块组成的残差结构
          super(BasicBlock, self).__init__()
          self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                 kernel_size=3, stride=stride, padding=1, bias=False)
          self.bn1 = nn.BatchNorm2d(out_channel)
          self.relu = nn.ReLU()
          self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                                 kernel_size=3, stride=1, padding=1, bias=False)
          self.bn2 = nn.BatchNorm2d(out_channel)
          self.downsample = downsample
  
      def forward(self, x):
          identity = x
          if self.downsample is not None:
              identity = self.downsample(x)
  
          out = self.conv1(x)
          out = self.bn1(out)
          out = self.relu(out)
  
          out = self.conv2(out)
          out = self.bn2(out)
  
          out += identity
          out = self.relu(out)
  
          return out
  
  
  class Bottleneck(nn.Module):
      """
      注意：原论文中，在虚线残差结构的主分支上，第一个 1x1 卷积层的步距是 2，第二个 3x3 卷积层步距是 1。
      但在 pytorch 官方实现过程中是第一个 1x1 卷积层的步距是 1，第二个 3x3 卷积层步距是 2，
      这么做的好处是能够在 top1 上提升大概 0.5% 的准确率。
      可参考 Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
      """
      expansion = 4
  
      def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                   groups=1, width_per_group=64):
          super(Bottleneck, self).__init__()
  
          # 用于分组卷积，ResNet 中没有变化
          width = int(out_channel * (width_per_group / 64.)) * groups
  
          self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                                 kernel_size=1, stride=1, bias=False)  # squeeze channels
          self.bn1 = nn.BatchNorm2d(width)
          # -----------------------------------------
          self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                                 kernel_size=3, stride=stride, bias=False, padding=1)
          self.bn2 = nn.BatchNorm2d(width)
          # -----------------------------------------
          self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                                 kernel_size=1, stride=1, bias=False)  # unsqueeze channels
          self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
          self.relu = nn.ReLU(inplace=True)
          self.downsample = downsample
  
      def forward(self, x):
          identity = x
          if self.downsample is not None:
              identity = self.downsample(x)
  
          out = self.conv1(x)
          out = self.bn1(out)
          out = self.relu(out)
  
          out = self.conv2(out)
          out = self.bn2(out)
          out = self.relu(out)
  
          out = self.conv3(out)
          out = self.bn3(out)
  
          out += identity
          out = self.relu(out)
  
          return out
  
  
  class ResNet(nn.Module):
  
      def __init__(self,
                   block,
                   blocks_num,
                   num_classes=1000,
                   include_top=True,
                   groups=1,
                   width_per_group=64):
          super(ResNet, self).__init__()
          self.include_top = include_top
          self.in_channel = 64
  
          self.groups = groups
          self.width_per_group = width_per_group
  
          self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                                 padding=3, bias=False)
          self.bn1 = nn.BatchNorm2d(self.in_channel)
          self.relu = nn.ReLU(inplace=True)
          self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
          self.layer1 = self._make_layer(block, 64, blocks_num[0])
          self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
          self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
          self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
          if self.include_top:
              self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
              self.fc = nn.Linear(512 * block.expansion, num_classes)
  
          for m in self.modules():
              if isinstance(m, nn.Conv2d):
                  nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
  
      def _make_layer(self, block, channel, block_num, stride=1):
          downsample = None
          if stride != 1 or self.in_channel != channel * block.expansion:
              downsample = nn.Sequential(
                  nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                  nn.BatchNorm2d(channel * block.expansion))
  
          layers = []
          layers.append(block(self.in_channel,
                              channel,
                              downsample=downsample,
                              stride=stride,
                              groups=self.groups,
                              width_per_group=self.width_per_group))
          self.in_channel = channel * block.expansion
  
          for _ in range(1, block_num):
              layers.append(block(self.in_channel,
                                  channel,
                                  groups=self.groups,
                                  width_per_group=self.width_per_group))
  
          return nn.Sequential(*layers)
  
      def forward(self, x):
          x = self.conv1(x)
          x = self.bn1(x)
          x = self.relu(x)
          x = self.maxpool(x)
  
          x = self.layer1(x)
          x = self.layer2(x)
          x = self.layer3(x)
          x = self.layer4(x)
  
          if self.include_top:
              x = self.avgpool(x)
              x = torch.flatten(x, 1)
              x = self.fc(x)
  
          return x
  
  
  def resnet34(num_classes=1000, include_top=True):
      # https://download.pytorch.org/models/resnet34-333f7ec4.pth
      return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
  
  
  def resnet50(num_classes=1000, include_top=True):
      # https://download.pytorch.org/models/resnet50-19c8e357.pth
      return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
  
  
  def resnet101(num_classes=1000, include_top=True):
      # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
      return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)
  
  
  def resnext50_32x4d(num_classes=1000, include_top=True):
      # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
      groups = 32
      width_per_group = 4
      return ResNet(Bottleneck, [3, 4, 6, 3],
                    num_classes=num_classes,
                    include_top=include_top,
                    groups=groups,
                    width_per_group=width_per_group)
  
  
  def resnext101_32x8d(num_classes=1000, include_top=True):
      # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
      groups = 32
      width_per_group = 8
      return ResNet(Bottleneck, [3, 4, 23, 3],
                    num_classes=num_classes,
                    include_top=include_top,
                    groups=groups,
                    width_per_group=width_per_group)
  ```

  

- tensorflow2 的 ResNet 网络搭建

  ```python
  from tensorflow.keras import layers, Model, Sequential
  
  
  class BasicBlock(layers.Layer):
      expansion = 1
  
      def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
          super(BasicBlock, self).__init__(**kwargs)
          self.conv1 = layers.Conv2D(out_channel, kernel_size=3, strides=strides,
                                     padding="SAME", use_bias=False)
          self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
          # -----------------------------------------
          self.conv2 = layers.Conv2D(out_channel, kernel_size=3, strides=1,
                                     padding="SAME", use_bias=False)
          self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
          # -----------------------------------------
          self.downsample = downsample
          self.relu = layers.ReLU()
          self.add = layers.Add()
  
      def call(self, inputs, training=False):
          identity = inputs
          if self.downsample is not None:
              identity = self.downsample(inputs)
  
          x = self.conv1(inputs)
          x = self.bn1(x, training=training)
          x = self.relu(x)
  
          x = self.conv2(x)
          x = self.bn2(x, training=training)
  
          x = self.add([identity, x])
          x = self.relu(x)
  
          return x
  
  
  class Bottleneck(layers.Layer):
      """
      注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
      但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
      这么做的好处是能够在top1上提升大概0.5%的准确率。
      可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
      """
      expansion = 4
  
      def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
          super(Bottleneck, self).__init__(**kwargs)
          self.conv1 = layers.Conv2D(out_channel, kernel_size=1, use_bias=False, name="conv1")
          self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1/BatchNorm")
          # -----------------------------------------
          self.conv2 = layers.Conv2D(out_channel, kernel_size=3, use_bias=False,
                                     strides=strides, padding="SAME", name="conv2")
          self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv2/BatchNorm")
          # -----------------------------------------
          self.conv3 = layers.Conv2D(out_channel * self.expansion, kernel_size=1, use_bias=False, name="conv3")
          self.bn3 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv3/BatchNorm")
          # -----------------------------------------
          self.relu = layers.ReLU()
          self.downsample = downsample
          self.add = layers.Add()
  
      def call(self, inputs, training=False):
          identity = inputs
          if self.downsample is not None:
              identity = self.downsample(inputs)
  
          x = self.conv1(inputs)
          x = self.bn1(x, training=training)
          x = self.relu(x)
  
          x = self.conv2(x)
          x = self.bn2(x, training=training)
          x = self.relu(x)
  
          x = self.conv3(x)
          x = self.bn3(x, training=training)
  
          x = self.add([x, identity])
          x = self.relu(x)
  
          return x
  
  
  def _make_layer(block, in_channel, channel, block_num, name, strides=1):
      downsample = None
      if strides != 1 or in_channel != channel * block.expansion:
          downsample = Sequential([
              layers.Conv2D(channel * block.expansion, kernel_size=1, strides=strides,
                            use_bias=False, name="conv1"),
              layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5, name="BatchNorm")
          ], name="shortcut")
  
      layers_list = []
      layers_list.append(block(channel, downsample=downsample, strides=strides, name="unit_1"))
  
      for index in range(1, block_num):
          layers_list.append(block(channel, name="unit_" + str(index + 1)))
  
      return Sequential(layers_list, name=name)
  
  
  def _resnet(block, blocks_num, im_width=224, im_height=224, num_classes=1000, include_top=True):
      # tensorflow中的tensor通道排序是NHWC
      # (None, 224, 224, 3)
      input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")
      x = layers.Conv2D(filters=64, kernel_size=7, strides=2,
                        padding="SAME", use_bias=False, name="conv1")(input_image)
      x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1/BatchNorm")(x)
      x = layers.ReLU()(x)
      x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME")(x)
  
      x = _make_layer(block, x.shape[-1], 64, blocks_num[0], name="block1")(x)
      x = _make_layer(block, x.shape[-1], 128, blocks_num[1], strides=2, name="block2")(x)
      x = _make_layer(block, x.shape[-1], 256, blocks_num[2], strides=2, name="block3")(x)
      x = _make_layer(block, x.shape[-1], 512, blocks_num[3], strides=2, name="block4")(x)
  
      if include_top:
          x = layers.GlobalAvgPool2D()(x)  # pool + flatten
          x = layers.Dense(num_classes, name="logits")(x)
          predict = layers.Softmax()(x)
      else:
          predict = x
  
      model = Model(inputs=input_image, outputs=predict)
  
      return model
  
  
  def resnet34(im_width=224, im_height=224, num_classes=1000, include_top=True):
      return _resnet(BasicBlock, [3, 4, 6, 3], im_width, im_height, num_classes, include_top)
  
  
  def resnet50(im_width=224, im_height=224, num_classes=1000, include_top=True):
      return _resnet(Bottleneck, [3, 4, 6, 3], im_width, im_height, num_classes, include_top)
  
  
  def resnet101(im_width=224, im_height=224, num_classes=1000, include_top=True):
      return _resnet(Bottleneck, [3, 4, 23, 3], im_width, im_height, num_classes, include_top)
  
  ```

  