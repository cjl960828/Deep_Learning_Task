### MobileNet v1

- 网络亮点：
  - 提出了 DW (DepthWise) 卷积，大大减少了运算量和参数数量
  - 添加了超参数 $\alpha, \beta$ 
    - $\alpha$ 用来控制模型宽度
    - $\beta$ 用来控制分辨率大小
  
- 传统卷积 VS  DW 卷积

  - 传统卷积
    - 卷积核的通道数 = 输入特征的通道数
    - 卷积核的个数 = 输出特征的通道数
  - DW 卷积
    - 卷积核的通道数为 1，即一个卷积核对应于一个 feature map
    - 输入特征的通道数 = 卷积核个数 = 输出特征的通道数

- DW 卷积实现

  1. 对输入大小为 $[H, W, C]$ 的特征采用 $C$ 个 $H \times W \times 1$ 大小的卷积逐通道进行处理，得到的结果依旧是 $[H, W, C]$
  2. 将得到的 $[H, W, C]$ 的特征采用 $M$ 个 $1 \times 1 \times C$ 大小的卷积逐空间进行处理，得到大小为 $[H, W, M]$ 的特征

- DW 卷积图示：

  ![DW 卷积](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/MobileNet4DWConv.png)

- MobileNet v1 网络结构

  ![MobileNet V1](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/MobileNet_v1_Structure.png)

- MobileNet V1 存在的问题：DW 卷积在模型训练过程中容易废掉

- Pytorch 版本的 MobileNet V1

  ```python
  import torch
  import torch.nn as nn
  
  class MobileNet_V1(nn.Module):
      def __init__(self):
          super(Net, self).__init__()
          def conv_bn_1x1(inp, oup, stride):
              return nn.Sequential(
                  nn.Conv2d(inp, oup, 1, stride, 1, bias=False),
                  nn.BatchNorm2d(oup),
                  nn.ReLU(inplace=True)
              )
          
          def conv_bn_3x3(inp, oup, stride):
              return nn.Sequential(
                  nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                  nn.BatchNorm2d(oup),
                  nn.ReLU(inplace=True)
              )
          def conv_dw(inp, oup, stride):
              return nn.Sequential(
                  # 通过控制 组数即可用于逐通道处理，注意输出维度与输入维度一致
                  nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                  nn.BatchNorm2d(inp),
                  nn.ReLU(inplace=True),
      
                  # 采用 1x1 的卷积
                  nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                  nn.BatchNorm2d(oup),
                  nn.ReLU(inplace=True),
              )
          layers = []
          def dwConv_conv(inp, out, stide):
              for i in range(5):
                  layers.append(
                      conv_dw(512, 512, 1),
                      conv_bn_1x1(512, 512, 1),
                  )
  
  			return nn.Sequential(*layers)
          
         	# 简单地由普通卷积和 DW 卷积组成
          self.model = nn.Sequential(
              conv_bn_3x3(3, 32, 2), 
              conv_dw(32, 32, 1),
              conv_bn_1x1(32, 64, 1), 
              conv_dw( 64, 64, 2),
              conv_bn_1x1(64, 128, 1), 
              conv_dw(128, 128, 1),
              conv_bn_1x1(128, 128, 1), 
              conv_dw(128, 128, 2),
              conv_bn_1x1(128, 256, 1), 
              conv_dw(256, 256, 1),
              conv_bn_1x1(256, 256, 1), 
              conv_dw(256, 256, 2),
              conv_bn_1x1(256, 512, 1), 
  			dwConv_conv(512, 512, 1), 
              conv_dw(512, 512, 2),
              conv_bn_1x1(512, 1024, 1), 
              conv_dw(1024, 1024, 2),
              conv_bn_1x1(1024, 1024, 1), 
              nn.AvgPool2d(7),
          )
          self.fc = nn.Linear(1024, 1000)
      def forward(self, x):
          x = self.model(x)
          x = x.view(-1, 1024)
          x = self.fc(x)
          return x
  ```
  
  