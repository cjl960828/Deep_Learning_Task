### GoogLeNet 启发

- 网络处理的一些小技巧
  1. 使用多个小的卷积核堆叠替代一个大的卷积核
  2. 使用 1x1 卷积核进行升维和降维操作
  3. 将 nxn 修改成为 1xn 和 nx1 的卷积核 【串行或者并行】
  4. 在池化部分，采用分路设计（一路卷积 + 一路池化）