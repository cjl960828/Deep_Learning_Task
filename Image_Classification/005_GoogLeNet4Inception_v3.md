### Inception V3

- 标签编码方式求损失
  - one-hot 编码计算交叉熵损失：
    - 损失函数目标：
      - 最小化交叉熵损失等效于最大化正确类别的对数似然估计
      - 换句话说就是正确类别对应的预测分数会一味增大，直到正无穷
    - 存在的问题：
      - 导致过拟合，因为模型会对结果进行“死记硬背”
      - 鼓励模型过于自信，即不计其余因素一位增大 logit 的值
  - Inception V3 中的标签编码方式 $\rightarrow$ label smooth
    - 案例：one-hot (1, 0, 0, 0)，此时取 $\epsilon=0.1$，那么得到缓和后的结果为 $(0.9, 0.333, 0.333, 0.333) [0.9 = 1-0.1， 0.333=0.1/(4-1)]$

- 主要的 V3 模型：在辅助分类器中添加了 BN

- 网络结构

  - 结构说明

    - Inception-v3-xxx 表明在 V3 网络中采用的新策略，比如 RMSProp  表示采用 RMSProp 优化器

  - 图示

    ![Inception V3 与其余网络性能比较](https://cdn.jsdelivr.net/gh/cjl960828/Deep_Learning_Task/Image_Classification/img/GoogLeNet4Inception_V3.png)

- Inception V3 总结

  - 修改标签编码方式【label smooth】
  - 在辅助分类器中添加 BN 【因为辅助分类器实际上起正则化效果，而 BN 也是起正则化效果，因此添加 BN】