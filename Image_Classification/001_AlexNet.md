### AlexNet

- 网络亮点

  - 首次利用 GPU 对网络进行加速训练

  - 使用了 ReLU 激活函数，而不是使用传统的 Sigmoid 激活函数以及 Tanh 激活函数 （饱和型激活函数）

  - 使用 LRN (局部响应归一化)
    - LRN: Local Response Normalization
    - 符号：$b_{x, y}^i$ 表示归一化之后的值, $i$ 表示通道的位置，表示更新第几个通道的值，$x$ 和 $y$ 表示更新像素的位置， $a_{x, y}^i$ 表示输入值，即激活函数 ReLU 的输出值，$k, \alpha, \beta, \frac n2$ 都是自定义系数- 
    - LRN 表达公式

    $$
    b_{x, y}^i = \frac {a_{x, y}^i}{(k+\alpha\sum_{j=max(0, i-\frac n2)}^{min(N-1, i+\frac n2)}} (a_{x, y}^j)^2)^\beta
    $$

  - 在全连接层的前两层使用了 Dropout 随机失活神经元，从而避免过拟合

- 过拟合和欠拟合

  - 过拟合：在训练集上表现良好，在测试集上表现糟糕，即泛化能力不强
  - 欠拟合：在训练集和测试集上的表现都很糟糕

- 卷积输出尺寸的计算公式

  - 参数解释：
    - W 表示输入的尺寸大小
    - F 表示卷积核的大小
    - P 表示 padding 的大小
    - S 表示 stride 的值

  $$
  N = \frac {W - F + 2P}{S} + 1
  $$

- 模型网络架构

  - 结构说明：
    - 特征提取部分
      - 卷积核大小为 11，填充为 2，卷积核个数为 96， 步距为 4 [$224 \times 224 \rightarrow 55 \times 55$]
      - 池化核大小为 3，步距为 2 [$55 \times 55 \rightarrow 27 \times 27$]
      - 卷积核大小为 5，填充为 2， 卷积核个数为 256 [$27 \times 27 \rightarrow 27\times 27$]
      - 池化核大小为 3，步距为 2 [$27 \times 27 \rightarrow 13 \times 13$]
      - 卷积核大小为 3，填充为 1，卷积核个数为 384 [$13 \times 13 \rightarrow 13 \times 13$]
      - 卷积核大小为 3，填充为 1，卷积核个数为 384 [$13 \times 13 \rightarrow 13 \times 13$]
      - 卷积核大小为 3，填充为 1，卷积核个数为 256 [$13 \times 13 \rightarrow 13 \times 13$]
      - 池化核大小为 3， 步距为 2 [$13 \times13 \rightarrow 6 \times 6$]
    - 分类器部分
      - Dropout 层，rate = 0.5
      - 全连接层，神经元个数为 2048 [$256 \times 6 \times 6 \rightarrow 2048$]
      - Dropout 层，rate = 0.5
      - 全连接层，神经元个数为 2048 [$2048 \rightarrow 2048$]
      - 全连接层，神经元个数为 num_classes [数据集类别数]

  ![AlexNet 网络架构](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/AlexNet.png)



- pytorch 版本的 AlexNet 网络搭建

  ```python
  import torch.nn as nn
  import torch
  
  
  class AlexNet(nn.Module):
      def __init__(self, num_classes=1000, init_weights=False):
          super(AlexNet, self).__init__()
          self.features = nn.Sequential(
              # input [3, 224, 224]  output [96, 55, 55]
              nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),  
              nn.ReLU(inplace=True),
              # input [96, 55, 55]  output [96, 27, 27]
              nn.MaxPool2d(kernel_size=3, stride=2),     
              # input [96, 27, 27]  output [256, 27, 27]
              nn.Conv2d(96, 256, kernel_size=5, padding=2),           
              nn.ReLU(inplace=True),
              # input [256, 27, 27]  output [256, 13, 13]
              nn.MaxPool2d(kernel_size=3, stride=2),       
              # input [256, 13, 13]  output [384, 13, 13]
              nn.Conv2d(256, 384, kernel_size=3, padding=1),          
              nn.ReLU(inplace=True),
              # input [384, 13, 13]  output [384, 13, 13]
              nn.Conv2d(384, 384, kernel_size=3, padding=1),          
              nn.ReLU(inplace=True),
              # input [384, 13, 13]  output [256, 13, 13]
              nn.Conv2d(384, 256, kernel_size=3, padding=1),          
              nn.ReLU(inplace=True),
              # input [384, 13, 13]  output [256, 6, 6]
              nn.MaxPool2d(kernel_size=3, stride=2),                  
          )
          self.classifier = nn.Sequential(
              nn.Dropout(p=0.5),
              # input 256*6*6  output 2048
              nn.Linear(256 * 6 * 6, 2048),
              nn.ReLU(inplace=True),
              nn.Dropout(p=0.5),
              # input 2048  output 2048
              nn.Linear(2048, 2048),
              nn.ReLU(inplace=True),
              # input 2048  output num_classes
              nn.Linear(2048, num_classes),
          )
          if init_weights:
              self._initialize_weights()
  
      def forward(self, x):
          x = self.features(x)
          x = torch.flatten(x, start_dim=1)
          x = self.classifier(x)
          return x
  
      def _initialize_weights(self):
          for m in self.modules():
              if isinstance(m, nn.Conv2d):
                  nn.init.kaiming_normal_(
                      m.weight, 
                      mode='fan_out', 
                      nonlinearity='relu'
                  )
                  if m.bias is not None:
                      nn.init.constant_(m.bias, 0)
              elif isinstance(m, nn.Linear):
                  nn.init.normal_(m.weight, 0, 0.01)
                  nn.init.constant_(m.bias, 0)
  ```

  

- tensorflow2 版本的 AlexNet 网络搭建

  ```python
  from tensorflow.keras import layers, models, Model, Sequential
  
  # 函数式实现，定义的同时直接使用
  def AlexNet_v1(im_height=224, im_width=224, num_classes=1000):
      # tensorflow 中的 tensor 通道排序 是 NHWC
      # 定义输入  
      # output(None, 224, 224, 3)
      input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")  
      # output(None, 227, 227, 3)
      x = layers.ZeroPadding2D(((1, 2), (1, 2)))(input_image)                     # output(None, 55, 55, 96)
      x = layers.Conv2D(96, kernel_size=11, strides=4, activation="relu")(x)       # output(None, 27, 27, 96)
      x = layers.MaxPool2D(pool_size=3, strides=2)(x)                             # output(None, 27, 27, 256)
      x = layers.Conv2D(256, kernel_size=5, padding="same", activation="relu")(x) 
      # output(None, 13, 13, 256)
      x = layers.MaxPool2D(pool_size=3, strides=2)(x)                             # output(None, 13, 13, 384)
      x = layers.Conv2D(384, kernel_size=3, padding="same", activation="relu")(x) 
      # output(None, 13, 13, 384)
      x = layers.Conv2D(384, kernel_size=3, padding="same", activation="relu")(x) 
      # output(None, 13, 13, 256)
      x = layers.Conv2D(256, kernel_size=3, padding="same", activation="relu")(x) 
      # output(None, 6, 6, 256)
      x = layers.MaxPool2D(pool_size=3, strides=2)(x)                             
      # output(None, 6*6*256)
      x = layers.Flatten()(x)                         
      x = layers.Dropout(0.2)(x)
      # output(None, 2048)
      x = layers.Dense(2048, activation="relu")(x)    
      x = layers.Dropout(0.2)(x)
      # output(None, 2048)
      x = layers.Dense(2048, activation="relu")(x)
      # output(None, num_classes)
      x = layers.Dense(num_classes)(x)                  
      predict = layers.Softmax()(x)
  
      # 将输入输出格式传入到模型中。得到最终的模型
      model = models.Model(inputs=input_image, outputs=predict)
      return model
  
  
  # 模块式实现，和 torch 的实现方式类似，注意不同的函数调用
  class AlexNet_v2(Model):
      def __init__(self, num_classes=1000):
          super(AlexNet_v2, self).__init__()
          self.features = Sequential([
              # output(None, 227, 227, 3)
              layers.ZeroPadding2D(((1, 2), (1, 2))),                                 	# output(None, 55, 55, 96)
              layers.Conv2D(96, kernel_size=11, strides=4, activation="relu"),        	 # output(None, 27, 27, 96)
              layers.MaxPool2D(pool_size=3, strides=2),                               	# output(None, 27, 27, 256)
              layers.Conv2D(256, kernel_size=5, padding="same", activation="relu"),   
              # output(None, 13, 13, 256)
              layers.MaxPool2D(pool_size=3, strides=2),                               	# output(None, 13, 13, 384)
              layers.Conv2D(384, kernel_size=3, padding="same", activation="relu"), 
              # output(None, 13, 13, 384)
              layers.Conv2D(384, kernel_size=3, padding="same", activation="relu"), 
              # output(None, 13, 13, 256)
              layers.Conv2D(256, kernel_size=3, padding="same", activation="relu"),  
              # output(None, 6, 6, 256)
              layers.MaxPool2D(pool_size=3, strides=2)])                              
  
          self.flatten = layers.Flatten()
          self.classifier = Sequential([
              layers.Dropout(0.2),
              # output(None, 2048)
              layers.Dense(2048, activation="relu"),                                  
              layers.Dropout(0.2),
              # output(None, 2048)
              layers.Dense(2048, activation="relu"),                                   	 # output(None, num_classes)
              layers.Dense(num_classes),                                                
              layers.Softmax()
          ])
  
      def call(self, inputs, **kwargs):
          x = self.features(inputs)
          x = self.flatten(x)
          x = self.classifier(x)
          return x
  ```

  