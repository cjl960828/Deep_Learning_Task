### Inception V2

- Inception V1 结构回顾

  ![Inception V1](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/GoogLeNet4Inception.png)

- 网络设计准则：

  - 避免表达瓶颈，尤其是网络早期 $\rightarrow$  **尺寸逐步减小【缓慢】，深度逐步增加【缓慢】**
  -  高维特征更易处理 $\rightarrow$ **高维特征具有更多的信息，对于模型学习有帮助**
  - 可以在低维特征时进行空间聚合，而不必担心会损失太多信息 $\rightarrow$ **低维特征的 $3 \times 3 $ 卷积前采用 *池化层* **
  - 平衡网络的宽度与深度 $\rightarrow$ **合适的深度和宽度对模型性能有帮助** 【并不是越宽越深性能越好】

- 三种改进的 Inception 网络结构

  - A 方案

    - 结构说明

      - 将带有降维功能的 Inception V1 结构中的 $5 \times 5$ 结构采用 2 个 $3 \times 3$ 的卷积代替 【见 VGG】

    - 图示

      ![Inception V2 结构中改进的 A 种方案](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/GoogLeNet4Inception_V2_A.png)

  - B 方案

    - 结构说明

      - 在 Inception V2 的 A 方案的基础上将 $3 \times 3$ 卷积改为 $1 \times 3$ 和 $3 \times 1$ 卷积 【串行】

    - 图示

      ![Inception V2 结构中改进的 B 种方案](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/GoogLeNet4Inception_V2_B.png)

  - C 方案

    - 结构说明

      - 在 Inception V2 的 A 方案基础上将最后的 $3 \times 3$ 卷积替换为 $1 \times 3$ 和 $3 \times 1$ 卷积 【并行】

    - 图示

      ![Inception V2 结构中改进的 C 种方案](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/GoogLeNet4Inception_V2_C.png)

- 下采样操作 [降低特征尺寸大小]

  - 普通的下采样操作

    - 普通的池化：先卷积，再池化 $\rightarrow$ 先增加维度，再池化$\rightarrow$ 计算量大
    - 使用 stride=2 的卷积进行下采样 $\rightarrow$ 会丢失一些信息

  - Inception V2 网络中的下采样操作

    - 结构说明

      - 一路进行池化，下采样
      - 一路进行卷积，增维
      - 进行 concat

    - 图示

      ![采用 Inception V2 结构的 GoogLeNet 的池化方式](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/GoogLeNet4Inception_V2_MP.png)

- Inception V2 网络结构

  - 结构说明 【尺寸缓慢减小，深度逐渐增加】

    - 前三层卷积，$k \times k / s$ 表示采用的卷积核大小为 $k$，步距为 $s$，填充为 0，可以根据公式求得输出维度
    - 最大池化，一路进行卷积，一路进行池化的下采样
    - 中间的三层卷积，与首三层卷积效果相同
    - $3 \times Inception$，采用 Inception V2 中的 A 方案
    - $5 \times Inception$，采用 Inception V2 中的 B 方案
    - $2 \times Inception$，采用 Inception V2 中的 C 方案

  - 图示

    ![GoogLeNet 网络采用 Iception V2 的结构表](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/GoogLeNet4Inception_V2.png)

- 总结 【Inception V2 改进点】

  - 移除浅层的辅助分类器，即仅保留高层的辅助分类器和主分类器
  - 修改 Inception 网络结构，一共有 3 种修改方案
  - 修改下采样的方法，一路进行卷积，一路进行池化，然后拼接起来