### 查看配置文件

- 说明：有些配置文件可能索引了之前的配置文件，避免跳转太多次，直接打印完成配置文件即可

```shell
python tools/misc/print_config.py configs/faster_rcnn/faster_rcnn_r50_fpn_iou_1x_coco.py > config.txt
```



### 配置文件举例

- 模型设置

```python
# model settings
model = dict(
    type='FastRCNN',
    backbone=dict(
        # type 在 mmdet/models/backbones/__init__.py 中有声明
        # ResNet 在 mmdet/models/backbones/resnet.py 中有定义
        type='ResNet',  
        # 以下值表示 mmdet/models/backbones/resnet.py 中 ResNet 的 __init__ 的参数
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    
    neck=dict(
        # type 在 mmdet/models/necks/__init__.py 中有声明
        # FPN 在 mmdet/models/necks/fpn.py 中有定义
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    
    roi_head=dict(
        # type 在 mmdet/models/roi_heads/__init__.py 中有声明
        # StandardRoIHead 在 mmdet/models/roi_heads/standard_roi_head.py 中有定义
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            # type 在 mmdet/models/roi_heads/roi_extractors/__init__.py 中有声明
        	# SingleRoIExtractor 在 mmdet/models/roi_heads/roi_extractors/single_level_roi_extractor.py 中有定义
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        
        bbox_head=dict(
            # type 在 mmdet/models/roi_heads/bbox_heads/__init__.py 中有声明
        	# Shared2FCBBoxHead 在 mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py 中有定义
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,  # 此类别数可以根据实际数据进行调整
            
            bbox_coder=dict(
            # type 在 mmdet/core/bbox/coder/__init__.py 中有声明
        	# DeltaXYWHBBoxCoder 在 mmdet/core/bbox/coder/delta_xywh_bbox_coder 中有定义
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    
    
    # model training and testing settings
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))
```



- 数据设置

```python
# dataset settings
dataset_type = 'CocoDataset'  # mmdet/datasets/__init__ 中声明，同目录 coco.py 中定义
data_root = 'data/coco/'

# 多尺度
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,  # samples_per_gpu * gpu_nums = batch_size
    workers_per_gpu=2,  # workers_per_gpu * gpu_nums = num_workers
    
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
```



- 优化器设定

```python
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])

# 训练总周期
runner = dict(type='EpochBasedRunner', max_epochs=12)

# 设置 checkpoint 间隔，每多少个 epoch 保存一次模型
checkpoint_config = dict(interval=1)

# 打印 log 间隔，没多少个 step 打印一次
log_config = dict(interval=50, hooks=[dict(type="TextLoggerHook")])
custom_hooks = [dict(type="NumClassCheckHook")]
dist_params = dict(backend="nccl")

# 加载参数 weight
load_from = None

# 重新加载 resume (包括 epoch 等信息，会覆盖 load_from)
resume_from = None

# 工作流 train val test
workflow = [('train', 1)]
# train 12 个 epoch， val 一个 epoch, test 一个 epoch
# 然后会出问题，因为在 train.py 中的 train_detector 中有 validate 选项永远为 True，即每一个 epoch 就验证一次。修改方法：将其设置为 False，跑完后自己再验证
# workflow = [('train', 12), ('val', 1), ('test', 1)]

# 保存目录
work_dir = '_'
```

