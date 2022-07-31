### Inception V4

- Inception V4 中使用的三种结构

  - A 方案：将 $5 \times 5$ 的卷积修改为两个串联的 $3 \times 3$ 的卷积；将 Max Pooling 替换为 Avg Pooling + $1 \times 1$ 的卷积

    ![Inception V4 中的 A 方案](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/GoogLeNet4Inception_V4_A.png)

  - B 方案：将 $3 \times 3$ 的卷积替换为 $1 \times 7 + 7 \times 1$ 的卷积

    ![Inception V4 中的 B 方案](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/GoogLeNet4Inception_V4_B.png)

  - C 方案：将最后一个 $3 \times 3$ 的卷积修改为并行结构

    ![Inception V4 中的 C 方案](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/GoogLeNet4Inception_V4_C.png)

- Inception V4 结构：由 Inception 结构组成，并修改了降采样方法

  ![Inception V4 结构](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/GoogLeNet4Inception_V4.png)

- Stem 结构 【特征尺寸变为输入的 $\frac 18$】

  - 首先将输入经过 3 个卷积操作，缓慢降低特征尺寸，逐渐增加特征深度
  - 一路池化，一路卷积，最后在通道维度进行拼接 $\rightarrow$ 尺寸减半
  - 分两路
    - 一路为原始的 $1 \times 1 + 3 \times 3$ 结构
    - 一路将原始的 $5 \times 5$ 修改为两个 $3 \times 3 $ 的串联，随后将第一个 $3 \times 3$ 修改为 $1 \times 7 + 7 \times 1$ 的卷积
  - 将两路结果进行合并
  - 一路池化，一路卷积，最后再通道维度进行拼接 $\rightarrow$ 尺寸减半

  ![Inception V4 降采样方法](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/GoogLeNet4Inception_V4_Stem.png)