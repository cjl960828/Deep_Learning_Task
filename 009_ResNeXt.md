### ResNeXt

- 网络亮点：更新了残差模块，从而进行分组计算

- 使用情况：一般用于深层的 ResNet 网络，即 50、101 或者 152 层

- 分组卷积

  - 图解 

    - 左图
    - 将通道数为 256 的输入特征输入到 64 个 $1 \times 1$ 的卷积，得到通道数为 64 的特征
      - 将通道数为 64 的特征输入到 64 个 $3 \times 3$ 的卷积，得到通道数为 64 的特征
    - 将通道数为 64 的特征输入到 256 个 $1 \times 1$ 的卷积，得到通道数为 256 的特征
      - 将通道数为 256 的特征与原始的通道数为 256 的特征进行相加处理，得到通道数为 256 的特征
    - 右图
      - 将原始的 128 个 $1 \times 1$ 卷积分为 32 个组，即每组有 4 个卷积核
      - 将通道数为 256 的输入特征输入到 32 组输出维度为 4 的 $1 \times 1$ 的卷积，得到 32 组通道数为 4 的特征
      - 将每组进行以下相同的操作 【32 组】
        - 将通道数为 4 的特征输入到 4 个 $3 \times 3$ 的卷积，得到通道数为 4 的特征
        - 将通道数为 4 的特征输入到 256 个 $1 \times 1$ 的卷积，得到通道数为 256 的特征
      - 将上述得到的 32 组通道数为 256 的输出特征进行相加处理，得到通道数为 256 的特征
      - 将通道数为 256 的特征与原始的通道数为 256 的特征进行相加处理，得到通道数为 256 的特征
  
  - 图示
  
    ![分组卷积结构](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/ResNeXt4Residual.png)

- ResNeXt 与 ResNet 结构比较

  - 32 表示总共划分为 32 个组
  - 在每一个残差块中，即 layer2-layer5 中，ResNeXt 中第一层和第二层卷积核数目为 ResNet 的 2 倍

  ![ResNeXt VS ResNet](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/ResNeXt_VS_ResNet.png)

- 代码实现：在调用 ResNet 的过程中，设置相应的组数以及 width_per_group 数，即
  $$
  num\_of\_channel = group \times width\_per\_group
  $$

  ```python
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

  

  