### GoogLeNet

- 网络亮点：

  - 引入了 Inception 结构，融合了不同尺度的特征信息 [总有一个尺度的特征能够捕获到对象]
  - 使用 1x1 的卷积核进行降维以及映射处理
  - 添加两个辅助分类器帮助训练（防止剃度消失，引入正则化，让底层学习到可分类特征，加快收敛）[预想效果]
  - 丢弃全连接层，使用平均池化层（大大减少了模型的参数）

- Inception v1 网络结构

  - 图解

    - 左图：最初的 Inception 网络结构 【四个分支的输出尺寸 [$H, W$] 一致】
      - 卷积核大小为 1，步距为 1
      - 卷积核大小为 3，步距为 1，填充为 1
      - 卷积核大小为 5，步距为 1，填充为 2
      - 池化核大小为 3，步距为 1，填充为 1
    - 右图：联合降维功能 [$1 \times 1$] 的 Inception 网络结构
      - 卷积核大小为 1，步距为 1
      - 卷积核大小为 1，步距为 1 + 卷积核大小为 3，步距为 1，填充为 1 
      - 卷积核大小为 1，步距为 1 + 卷积核大小为 5，步距为 1，填充为 2
      - 池化核大小为 3，步距为 1，填充为 1 + 卷积核大小为 1，步距为 1

  - 图示

    ![Inception V1](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/GoogLeNet4Inception.png)

  - $1 \times 1 $ 卷积的作用

    1. 降维或者升维：通过修改 $1 \times 1$ 的卷积核个数来控制维度
    2. 跨通道信息交融：$1 \times 1 $ 卷积的卷积核个数与输入特征的通道数相同，因此可以进行通道间的信息交互
    3. 减少参数量：假设将 512 维度的特征图卷积 [$5 \times 5$] 为 64 维度的特征图
       - 原始参数量：$5 \times 5 \times 512 \times 64 = 819200$
       - 降维后参数量：$24 \times 512 + 5 \times 5 \times 24 \times 64 = 50688$
    4. 增加模型深度并提升模型非线性能力：通过修改 $1 \times 1 $ 的卷积核个数来控制模型深度

- 模型结构 【Inception 网络结构】

  - \# $ 1\times 1$：Inception 结构中 $1 \times 1$ 卷积的卷积核数
  - \# $3 \times 3 \ reduce$ ：Inception 结构中 $3 \times 3$ 卷积部分用于降维的 $1 \times 1$ 卷积核数
  - \# $3 \times 3$：Inception 结构中 $3 \times 3$ 的卷积核数
  - \# $5 \times 5 \ reduce$ ：Inception 结构中 $5 \times 5$ 卷积部分用于降维的 $1 \times 1$ 卷积核数
  - \# $5 \times 5$ ：Inception 结构中  $5 \times 5$ 的卷积核数
  - $pool \ proj$：Inception 结构中 $3 \times 3$ 池化层后用于降维的 $1 \times 1$ 卷积核数
  
  ![GoogLeNet 模型组成](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/GoogLeNet.png)

- 辅助分类器

  - 结构组成

    - 卷积核大小为 5，步距为 3 的平均池化层，用于处理输出尺寸
  - 卷积核大小为 1，步距为 1 的卷积层，用于修改模型通道数
    - 全连接层的神经元数为 1024
    - 全连接层的神经元数为 num_calsses，用于做分类结果
  
  - 图示
  
    ![辅助分类器](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/GoogLeNet4AuxiClassifier.png)
  
- pytorch 版本的 GoogLeNet 网络搭建

  ```python
  import torch.nn as nn
  import torch
  import torch.nn.functional as F
  
  
  class GoogLeNet(nn.Module):
      def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
          super(GoogLeNet, self).__init__()
          # 用于表示是否使用 辅助分类器
          self.aux_logits = aux_logits
  
          # 7x7 / 2，用于将图像尺寸缩小为原始尺寸的 1/4
          self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
          self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
  		
          # 首先使用 1x1 卷积进行降维处理，后续将维度进行提升
          self.conv2 = BasicConv2d(64, 64, kernel_size=1)
          self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
          self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
  
          # Inception V1 部分的构建
          self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
          self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
          self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
  
          self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
          self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
          self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
          self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
          self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
          self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
  
          self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
          self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
  
          # 辅助分类器部分
          if self.aux_logits:
              self.aux1 = InceptionAux(512, num_classes)
              self.aux2 = InceptionAux(528, num_classes)
  
          # 全局平均池化 + FC 用于最终的分类
          self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
          self.dropout = nn.Dropout(0.4)
          self.fc = nn.Linear(1024, num_classes)
          if init_weights:
              self._initialize_weights()
  
      def forward(self, x):
          # N x 3 x 224 x 224
          x = self.conv1(x)
          # N x 64 x 112 x 112
          x = self.maxpool1(x)
          # N x 64 x 56 x 56
          x = self.conv2(x)
          # N x 64 x 56 x 56
          x = self.conv3(x)
          # N x 192 x 56 x 56
          x = self.maxpool2(x)
  
          # N x 192 x 28 x 28
          x = self.inception3a(x)
          # N x 256 x 28 x 28
          x = self.inception3b(x)
          # N x 480 x 28 x 28
          x = self.maxpool3(x)
          # N x 480 x 14 x 14
          x = self.inception4a(x)
          # N x 512 x 14 x 14
          
          # 注意辅助分类器的调用位置 训练阶段且需要使用辅助分类器
          if self.training and self.aux_logits:    # eval model lose this layer
              aux1 = self.aux1(x)
  
          x = self.inception4b(x)
          # N x 512 x 14 x 14
          x = self.inception4c(x)
          # N x 512 x 14 x 14
          x = self.inception4d(x)
          # N x 528 x 14 x 14
          
          # 注意辅助分类器的调用位置 训练阶段且需要使用辅助分类器
          if self.training and self.aux_logits:    # eval model lose this layer
              aux2 = self.aux2(x)
  
          x = self.inception4e(x)
          # N x 832 x 14 x 14
          x = self.maxpool4(x)
          # N x 832 x 7 x 7
          x = self.inception5a(x)
          # N x 832 x 7 x 7
          x = self.inception5b(x)
          # N x 1024 x 7 x 7
  
          x = self.avgpool(x)
          # N x 1024 x 1 x 1
          x = torch.flatten(x, 1)
          # N x 1024
          x = self.dropout(x)
          x = self.fc(x)
          # N x 1000 (num_classes)
          if self.training and self.aux_logits:   # eval model lose this layer
              return x, aux2, aux1
          return x
  
      def _initialize_weights(self):
          for m in self.modules():
              if isinstance(m, nn.Conv2d):
                  nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                  if m.bias is not None:
                      nn.init.constant_(m.bias, 0)
              elif isinstance(m, nn.Linear):
                  nn.init.normal_(m.weight, 0, 0.01)
                  nn.init.constant_(m.bias, 0)
  
  
  class Inception(nn.Module):
      def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
          super(Inception, self).__init__()
  
          self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
  
          self.branch2 = nn.Sequential(
              BasicConv2d(in_channels, ch3x3red, kernel_size=1),
              BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)   # 保证输出大小等于输入大小
          )
  
          self.branch3 = nn.Sequential(
              BasicConv2d(in_channels, ch5x5red, kernel_size=1),
              BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)   # 保证输出大小等于输入大小
          )
  
          self.branch4 = nn.Sequential(
              nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
              BasicConv2d(in_channels, pool_proj, kernel_size=1)
          )
  
      def forward(self, x):
          branch1 = self.branch1(x)
          branch2 = self.branch2(x)
          branch3 = self.branch3(x)
          branch4 = self.branch4(x)
  
          outputs = [branch1, branch2, branch3, branch4]
          return torch.cat(outputs, 1)
  
  
  class InceptionAux(nn.Module):
      def __init__(self, in_channels, num_classes):
          super(InceptionAux, self).__init__()
          self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
          self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # output[batch, 128, 4, 4]
  
          self.fc1 = nn.Linear(2048, 1024)
          self.fc2 = nn.Linear(1024, num_classes)
  
      def forward(self, x):
          # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
          x = self.averagePool(x)
          # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
          x = self.conv(x)
          # N x 128 x 4 x 4
          x = torch.flatten(x, 1)
          x = F.dropout(x, 0.5, training=self.training)
          # N x 2048
          x = F.relu(self.fc1(x), inplace=True)
          x = F.dropout(x, 0.5, training=self.training)
          # N x 1024
          x = self.fc2(x)
          # N x num_classes
          return x
  
  
  class BasicConv2d(nn.Module):
      def __init__(self, in_channels, out_channels, **kwargs):
          super(BasicConv2d, self).__init__()
          self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
          self.relu = nn.ReLU(inplace=True)
  
      def forward(self, x):
          x = self.conv(x)
          x = self.relu(x)
          return x
  ```

  

- tensorflow2 版本的 GoogLeNet 网络搭建

  ```python
  from tensorflow.keras import layers, models, Model, Sequential
  
  
  def GoogLeNet(im_height=224, im_width=224, class_num=1000, aux_logits=False):
      # tensorflow 中的 tensor 通道排序是 NHWC
      input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")
      # (None, 224, 224, 3)
      x = layers.Conv2D(64, kernel_size=7, strides=2, padding="SAME", activation="relu", name="conv2d_1")(input_image)
      # (None, 112, 112, 64)
      x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_1")(x)
      # (None, 56, 56, 64)
      x = layers.Conv2D(64, kernel_size=1, activation="relu", name="conv2d_2")(x)
      # (None, 56, 56, 64)
      x = layers.Conv2D(192, kernel_size=3, padding="SAME", activation="relu", name="conv2d_3")(x)
      # (None, 56, 56, 192)
      x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_2")(x)
  
      # (None, 28, 28, 192)
      x = Inception(64, 96, 128, 16, 32, 32, name="inception_3a")(x)
      # (None, 28, 28, 256)
      x = Inception(128, 128, 192, 32, 96, 64, name="inception_3b")(x)
  
      # (None, 28, 28, 480)
      x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_3")(x)
      # (None, 14, 14, 480)
      x = Inception(192, 96, 208, 16, 48, 64, name="inception_4a")(x)
      if aux_logits:
          aux1 = InceptionAux(class_num, name="aux_1")(x)
  
      # (None, 14, 14, 512)
      x = Inception(160, 112, 224, 24, 64, 64, name="inception_4b")(x)
      # (None, 14, 14, 512)
      x = Inception(128, 128, 256, 24, 64, 64, name="inception_4c")(x)
      # (None, 14, 14, 512)
      x = Inception(112, 144, 288, 32, 64, 64, name="inception_4d")(x)
      if aux_logits:
          aux2 = InceptionAux(class_num, name="aux_2")(x)
  
      # (None, 14, 14, 528)
      x = Inception(256, 160, 320, 32, 128, 128, name="inception_4e")(x)
      # (None, 14, 14, 532)
      x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_4")(x)
  
      # (None, 7, 7, 832)
      x = Inception(256, 160, 320, 32, 128, 128, name="inception_5a")(x)
      # (None, 7, 7, 832)
      x = Inception(384, 192, 384, 48, 128, 128, name="inception_5b")(x)
      # (None, 7, 7, 1024)
      x = layers.AvgPool2D(pool_size=7, strides=1, name="avgpool_1")(x)
  
      # (None, 1, 1, 1024)
      x = layers.Flatten(name="output_flatten")(x)
      # (None, 1024)
      x = layers.Dropout(rate=0.4, name="output_dropout")(x)
      x = layers.Dense(class_num, name="output_dense")(x)
      # (None, class_num)
      aux3 = layers.Softmax(name="aux_3")(x)
  
      if aux_logits:
          model = models.Model(inputs=input_image, outputs=[aux1, aux2, aux3])
      else:
          model = models.Model(inputs=input_image, outputs=aux3)
      return model
  
  
  class Inception(layers.Layer):
      def __init__(self, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj, **kwargs):
          super(Inception, self).__init__(**kwargs)
          self.branch1 = layers.Conv2D(ch1x1, kernel_size=1, activation="relu")
  
          self.branch2 = Sequential([
              layers.Conv2D(ch3x3red, kernel_size=1, activation="relu"),
              # output_size = input_size
              layers.Conv2D(ch3x3, kernel_size=3, padding="SAME", activation="relu")])      
  
          self.branch3 = Sequential([
              layers.Conv2D(ch5x5red, kernel_size=1, activation="relu"),
              # output_size = input_size
              layers.Conv2D(ch5x5, kernel_size=5, padding="SAME", activation="relu")])      
  
          self.branch4 = Sequential([
              # caution: default strides == pool_size
              layers.MaxPool2D(pool_size=3, strides=1, padding="SAME"),  
              # output_size= input_size
              layers.Conv2D(pool_proj, kernel_size=1, activation="relu")])                  
  
      def call(self, inputs, **kwargs):
          branch1 = self.branch1(inputs)
          branch2 = self.branch2(inputs)
          branch3 = self.branch3(inputs)
          branch4 = self.branch4(inputs)
          outputs = layers.concatenate([branch1, branch2, branch3, branch4])
          return outputs
  
  
  class InceptionAux(layers.Layer):
      def __init__(self, num_classes, **kwargs):
          super(InceptionAux, self).__init__(**kwargs)
          self.averagePool = layers.AvgPool2D(pool_size=5, strides=3)
          self.conv = layers.Conv2D(128, kernel_size=1, activation="relu")
  
          self.fc1 = layers.Dense(1024, activation="relu")
          self.fc2 = layers.Dense(num_classes)
          self.softmax = layers.Softmax()
  
      def call(self, inputs, **kwargs):
          # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
          x = self.averagePool(inputs)
          # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
          x = self.conv(x)
          # N x 128 x 4 x 4
          x = layers.Flatten()(x)
          x = layers.Dropout(rate=0.5)(x)
          # N x 2048
          x = self.fc1(x)
          x = layers.Dropout(rate=0.5)(x)
          # N x 1024
          x = self.fc2(x)
          # N x num_classes
          x = self.softmax(x)
  
          return x
  ```

  