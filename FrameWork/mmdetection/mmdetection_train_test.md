### train

- 单 GPU 训练

```shell
CONFIG = _  # 配置文件路径
WORKDIR = _  # 结果保存目录
python ./tools/train.py $CONFIG --work-dir $WORKDIR
# 其他参数详细见 train.py 文件，训练配置文件详细见 config 文件
```



- 多 GPU 训练

```shell
CONFIG = _  # 配置文件路径
GPU_NUM = _  # 使用的 GPU 的数量
WORKDIR = _  # 结果保存目录
CUDA_VISIBLE_DEVICES=_ bash ./tools/dist_train.sh $CONFIG $GPU_NUM --work-dir $WORKDIR
# 其他参数详细见 train.py 文件，训练配置文件详细见 config 文件
```

> 端口设置：PORT=${PORT:-$((29500 + $RANDOM % 10))}

### test

- 单 GPU 测试

```shell
CONFIG = _  # 配置文件路径
CHECKPOINT = _  # 结果保存目录
python ./tools/test.py $CONFIG $CHECKPOINT --out $OUTPUTDIR --eval segm
# 其他参数详细见 train.py 文件，训练配置文件详细见 config 文件
```

- 多 GPU 测试

```shell
CONFIG = _  # 配置文件路径
CHECKPOINT = _  # 结果保存目录
GPU_NUM = _  # GPU 数量
CUDA_VISIBLE_DEVICES=_ bash ./tools/dist_test.sh $CONFIG $CHECKPOINT $GPU_NUM --out $OUTPUTDIR --eval segm
# 其他参数详细见 train.py 文件，训练配置文件详细见 config 文件
```



### 分析工具

- 日志分析

```shell
# 安装可视化工具 seaborn
LOGFILE = _  # log 文件所在路径  log.json
OUTFILE = _  # 图片输出地址
KEYS = _  # 打印的键值
TITLE = _  # 输出图片 title
python tools/analysis_tools/analyze_logs.py plot_curve $LOGFILE [--key ${KEYS}] [--title ${TITLE}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUTFILE}]

# e.g
# python tools/analysis_tools/analyze_logs.py plot_curve logo_train/20220723_033839.log.json --keys bbox_mAP --legend bbox_mAP
```

- 计算平均训练时长

```shell
LOGFILE = _  # log 文件所在路径  log.json
python tools/analysis_tools/analyze_logs.py cal_train_time $LOGFILE
```

- 可视化工具 DetVisGUI 使用

```shell
# https://github.com/Chien-Hung/DetVisGUI/tree/mmdetection
# 路径位于 mmdetection 根目录下
# 注意配置文件的路径问题，是 mmdetection 目录下，不是 DetVisGUI 目录下
CONFIG_FILE = _  # mmdetection 的配置文件
RESULT_FILE = _  # pickle / json 文件
STAGE = _  # train val or test, default is 'val'
SAVE_DIRECTORY = _  # default is 'output'
python DetVisGUI.py ${CONFIG_FILE} [--det_file ${RESULT_FILE}] [--stage ${STAGE}] [--output ${SAVE_DIRECTORY}]
```



- 测试预测结果展示

```shell
CONFIG = _  # 配置文件
PREDICTION_PATH = _  # test 预测结果的 pkl 文件
SHOW_DIR = _  # 保存结果的目录
# --show  是否直接显示结果，默认为 False
WAIT_TIME = _  # 直接展示结果的等待时长
TOPK = _  # 展示前几个结果
SHOW_SCORE_THR = _  # 展示结果的阈值
CFG_OPTIONS = _  # 配置文件的选项，默认为 config 文件
python tools/analysis_tools/analyze_results.py ${CONFIG} ${PREDICTION_PATH} ${SHOW_DIR}  [--show] [--wait-time ${WAIT_TIME}] [--topk ${TOPK}] [--show-score-thr ${SHOW_SCORE_THR}] [--cfg-options ${CFG_OPTIONS}]
```

- coco_error_analysis 结果分析，每个类的分数表示

```shell
# 获取 json 格式的结果文件
# out: results.bbox.json and results.segm.json
CONFIG = _  # 配置文件
CHECKPOINT = _  # 模型文件领
RESULT_DIR = _  # 保存结果的目录
ANN_FILE = _  # 分析文件保存目录
# --show  是否直接显示结果，默认为 False
WAIT_TIME = _  # 直接展示结果的等待时长
TOPK = _  # 展示前几个结果
SHOW_SCORE_THR = _  # 展示结果的阈值
CFG_OPTIONS = _  # 配置文件的选项，默认为 config 文件
```

- 模型复杂度分析

```shell
CONFIG_FILE = _
INPUT_SHAPE = _
# FLOPs 与输入的大小有段，parameters 与输入的大小无关
python tools/analysis_tools/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]
```

