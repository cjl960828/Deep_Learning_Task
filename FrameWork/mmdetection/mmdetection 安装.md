### MMDetection 的安装

1. 创建虚拟环境

```shell
conda create -n mmdetection python=3.8 -y
```

2. 安装 torch [需要对应本地的 nvcc 版本]、

```shell
# https://pytorch.org/get-started/previous-versions/
# 以 cuda11.1 的 torch 1.9.1 为例子
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

3. 1 使用 openmim 安装 mmcv

```shell
pip install -U openmim
mim install mmcv-full
```

3. 2 手动安装 mmcv [需要与安装的 torch 版本对应]

```shell
# 切记不要直接使用 pip install mmcv-full
# 建议安装方式 pip install mmcv -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu_111/torch1.9.1/index.html
```

4. 安装 mmdetection

```shell
# down 下来 mmdetection 的仓库
git clone https://github.com/open-mmlab/mmdetection.git
# 进入目录
cd mmdetection
# 安装对应的依赖包
pip install -r requirements/build.txt
# 安装 mmdetection
pip install -e . # 或者使用 python setup.py develop
```

5. 测试是否安装成功

```shell
# for test_rnv,获取模型
wget https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_2020130-047c8118.pth
```

```python
from mmdet.apis import init_detector, inference_detector, show_result_pyplot

# 获取配置文件路径
config_file = "configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py"

# 获取模型文件路径
checkpoint_file = "checkpoints/faster_rcnn_r50_fpn_1x_coco_2020130-047c8118.pth"

# 加载设备
device = "cuda:0"  # or device = "cpu"

# 初始化 detecotor
model = init_detector(config_file, checkpoint_file, device=device)

# 加载图片并使用模型对其进行推理
img = "./demo/demo.jpg"
result = inference_detector(model, img)  # 获取模型对图片的推理结果

# 可视化推理结果
show_result_pyplot(model, img, result)  # 可视化模型对图片推理的结果
```

