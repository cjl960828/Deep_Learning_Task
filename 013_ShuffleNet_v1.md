### ShuffleNet V1

- 网络亮点
  - 逐点组卷积（pointwise group convolution） $\rightarrow$ 大幅度降低参数量
  - 通道打散（channel shuffle）$\rightarrow$ 促进通道间信息交流
  
- Channel Shuffle 结构

  - 图解

    - 图 (a) 是普通的组卷积，通道之间的信息只能在组内进行传递
    - 图 (b) 与图 (c) 表示允许不同组内通道间的信息进行交互

  - 图示

    ![Channel Shuffle](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/ShuffleNet_v1_shuffle_channel.png)

- Channel Shuffle 卷积设计

  - 图解

    - 在图 (a) 中，将 ResNet 中提及的残差结构的主分支中的 $3 \times 3 $ 的卷积替换为 DW 卷积
    - 在图 (b) 中，将图 (a) 中的 $1 \times 1$ 卷积采用逐点组卷积进行处理，随后进行 channel shuffle，并且将最后一个 $1\times 1$ 卷积采用逐点组卷积进行处理。
    - 在图 (c) 中，表示需要进行降采样的 Shuffle Channel Unit，即在辅分支上添加步长为 2 的 $3 \times 3$ 的平均池化层，另外在主分支上的 DW 卷积的步距大小为 2。最后将 Add 操作转换为 Concat 操作，通过增加少量的运算成本而增加了通道数。【一路进行池化，一路进行卷积】

  - Channel Shuffle 的实现流程 【划分为 g 个组，每一个组有 c 个通道，即总通道数为 $g \times c$】

    1. (a) $\rightarrow$ (b) 过程：将大小为 $(1, g\times c)$ 的特征进行形状变化，变为 $(g, c)$  
    2. (b) $\rightarrow$ (c) 过程：将大小为 $(g, c)$ 的特征再次进行形状变化，变为 $(c, g)$
    3. (c) $\rightarrow$ (d) 过程：将大小为 $(c, g)$ 的特征进行展开操作，变为 $(1, c \times g)$
    4. (d) 过程：最终得到 Channel Shuffle 后的结果

  - Channel Shuffle 图示 [ 变形 $\rightarrow$  转置 $\rightarrow$ 展平]

    ![Channel Shuffle 的简单实现流程](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/ShuffleNet_v1_shuffle_channel_implement.png)

  - 图示

    ![Channel Shuffle 不同卷积单元](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/ShuffleNet_v1_Union.png)

- ShuffleNet v1 的网络结构表示

  - FLOPs 与 FLOPS 的区别
    - FLOPS 表示每一秒浮点运算次数，可以理解为计算的速度，是衡量硬件性能的一个指标（硬件）
    - FLOPs 表示浮点运算数，可以理解为计算量，是衡量算法/模型的复杂度（模型），论文中使用该概念
  - 图示

  ![ShuffleNet V1 结构表](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/ShuffleNet_v1_Structure.png)

  - 结论：从图中可以看出，在 $g = 4$ 时，得到的参数最少
  - 图表说明
    1. 每个阶段的第一个 block 的步长为2，下一阶段的通道翻倍
    2. 每个阶段内的除步长其他超参数保持不变
    3. 每个 ShuffleNet unit 的 bottleneck 通道数为输出的 $\frac 14$ (和ResNet设置一致)

- Pytorch 版本的 ShuffleNet V1 网络搭建

  ```python
  import torch.nn as nn
  import torch
  import torch.nn.functional as F
  from collections import OrderedDict
  from torchsummary import summary
   
   
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
   
  # 基本的 卷积 -> BN -> relu
  class baseConv(nn.Module):
      def __init__(self, inchannels, outchannels, kernel_size, stride, groups, hasRelu=False):
          super(baseConv, self).__init__()
          if hasRelu:
              # 判断是否有 ReLU 激活函数
              activate = nn.ReLU
          else:
              activate = nn.Identity  # y = x
          pad = kernel_size // 2
          self.baseconv=nn.Sequential(
              nn.Conv2d(
                  in_channels=inchannels, 
                  out_channels=outchannels, 
                  kernel_size=kernel_size, 
                  stride=stride, 
                  padding=pad, 
                  groups=groups,
                  bias=False),
              nn.BatchNorm2d(outchannels),
              activate()
          )
   
      def forward(self,x):
          out=self.baseconv(x)
          return out
   
   
  #通道重排
  def ChannelShuffle(x,groups):
      # Pytorch 数据格式 [batch_size, channel, height, width]
      batch_size, channel, height, width = x.size()
      # 获得每组组卷积过程中的组内 channel
      inner_channel = channel // groups
      # [batch, groups, inner_channel, height, width]
      x = x.view(batch_size, groups, inner_channel, height, width)
      # [batch, inner_channel, groups, height, width]
      x = torch.transpose(x, 1, 2).contiguous()
      x = x.view(batch_size, -1, height, width)
      return x
   
   
  # stage 结构
  class Residual(nn.Module):
      def __init__(self, inchannels, outchannels, stride, groups):
          super(Residual, self).__init__()
          self.add_ = True      # shortcut 为相加操作
          self.groups = groups
   
          hidden_channel = inchannels//4
          # 当输入 channel 不等于 24 时候才有第一个 1*1 conv
          self.has_conv1 = True
          if inchannels! = 24:
              self.channel1_first1 = baseConv(
                  inchannels=inchannels,
                  outchannels=hidden_channel,
                  kernel_size=1,
                  stride=1,
                  groups=groups,
                  hasRelu=True)
          else:
              self.has_conv1 = False
              self.channel1_first1 = nn.Identity()
              hidden_channel = inchannels
   
          # channel1
          self.channel1=nn.Sequential(
              baseConv(
                  inchannels=hidden_channel,
                  outchannels=hidden_channel,
                  kernel_size=3,
                  stride=stride,
                  groups=hidden_channel),
              baseConv(
                  inchannels=hidden_channel,
                  outchannels=outchannels,
                  kernel_size=1,
                  stride=1,
                  groups=groups)
          )
   
          # channel2
          if stride==2:
              self.channel2=nn.AvgPool2d(
                  kernel_size=3,
                  stride=stride,
                  padding=1)
              self.add_=False
   
   
      def forward(self,x):
          if self.has_conv1:
              x1 = self.channel1_first1(x)
              x1 = ChannelShuffle(x1,groups=self.groups)
              out = self.channel1(x1)
          else:
              out = self.channel1(x)
          if self.add_:
              out += x
              return F.relu_(out)
          else:
              out2 = self.channel2(x)
              out = torch.cat((out,out2),dim=1)
              return F.relu_(out)
   
   
  # shuffleNet V1 g = 3
  class ShuffleNet(nn.Module):
      def __init__(self, groups, out_channel_list, num_classes, rate, init_weight=True):
          super(ShuffleNet, self).__init__()
   
          # 定义有序字典存放网络结构
          self.Module_List = OrderedDict()
   
          self.Module_List.update(
              {'Conv1':nn.Sequential(
                  nn.Conv2d(
                      3,
                      _make_divisible(24*rate,divisor=4*groups),
                      3,
                      2,
                      1,
                      bias=False),
                  nn.BatchNorm2d(
                      _make_divisible(24*rate,4*groups)),
                  nn.ReLU())}
          )
          self.Module_List.update({'MaxPool1':nn.MaxPool2d(3,2,1)})
   
          # net_config [inchannels,outchannels,stride]
          net_config = [[out_channel_list[0],out_channel_list[0],1],
                        [out_channel_list[0],out_channel_list[1],2],
                        [out_channel_list[1],out_channel_list[2],1],
                        [out_channel_list[2],out_channel_list[3],2],
                        [out_channel_list[3],out_channel_list[4],1]]
          repeat_num = [3, 1, 7, 1, 3]
   
          # 搭建stage部分
          self.Module_List.update(
              {'stage0_0':Residual(
                  _make_divisible(24*rate,4*groups),
                  _make_divisible((out_channel_list[0]-_make_divisible(24*rate,4*groups))*rate,4*groups),
                  stride=2,
                  groups=groups)})
          for idx,item in enumerate(repeat_num):
              config_item = net_config[idx]
              for j in range(item):
                  if j==0 and idx!=0 and config_item[-1]==2:
                      self.Module_List.update(
                          {'stage{}_{}'.format(idx,j+1):Residual(
                              _make_divisible(config_item[0]*rate,4*groups),
                              _make_divisible((config_item[1]-config_item[0])*rate,4*groups),
                              config_item[2],groups)}
                      )
                  else:
                      self.Module_List.update(
                          {'stage{}_{}'.format(idx,j+1):Residual(
                              _make_divisible(config_item[0]*rate,4*groups),
                              _make_divisible(config_item[1]*rate,4*groups),
                              config_item[2],groups)})
                  config_item[-1] = 1       # 重复 stage 的 stride = 1
                  config_item[0] = config_item[1]
   
          self.Module_List.update(
              {'GlobalPool':nn.AvgPool2d(kernel_size=7,stride=1)}
          )
   
          self.Module_List = nn.Sequential(self.Module_List)
   
          self.linear = nn.Sequential(
              nn.Dropout(p=0.2),
              nn.Linear(
                  _make_divisible(out_channel_list[-1]*rate,4*groups),
                  num_classes)
          )
   
          if init_weight:
              self.init_weight()
              
      def forward(self,x):
          out = self.Module_List(x)
          out = out.view(out.size(0),-1)
          out = self.linear(out)
          return out
   
      def init_weight(self):
          for w in self.modules():
              if isinstance(w, nn.Conv2d):
                  nn.init.kaiming_normal_(w.weight, mode='fan_out')
                  if w.bias is not None:
                      nn.init.zeros_(w.bias)
              elif isinstance(w, nn.BatchNorm2d):
                  nn.init.ones_(w.weight)
                  nn.init.zeros_(w.bias)
              elif isinstance(w, nn.Linear):
                  nn.init.normal_(w.weight, 0, 0.01)
                  nn.init.zeros_(w.bias)
   
  # 定义 shufflenet
  def shuffleNet_g1_(num_classes,rate=1.0):
      config=[144, 288, 288, 576, 576]
      return ShuffleNet(groups=1,out_channel_list=config,num_classes=num_classes,rate=rate)
   
  def shuffleNet_g2_(num_classes,rate=1.0):       #
      config=[200, 400, 400, 800, 800]
      return ShuffleNet(groups=2,out_channel_list=config,num_classes=num_classes,rate=rate)
   
  def shuffleNet_g3_(num_classes,rate=1.0):
      config=[240, 480, 480, 960, 960]
      return ShuffleNet(groups=3,out_channel_list=config,num_classes=num_classes,rate=rate)
   
  def shuffleNet_g4_(num_classes,rate=1.0):
      config=[272, 544, 544, 1088, 1088]
      return ShuffleNet(groups=4,out_channel_list=config,num_classes=num_classes,rate=rate)
   
  def shuffleNet_g8_(num_classes,rate=1.0):
      config=[384, 768, 768, 1536, 1536]
      return ShuffleNet(groups=8,out_channel_list=config,num_classes=num_classes,rate=rate)
   
  if __name__ == '__main__':
      net=shuffleNet_g3_(10,rate=1.0).to('cuda')
      print(net)
      summary(net,(3,224,224))  # 显示网络结构
  ```

  