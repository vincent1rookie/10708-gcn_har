num_person=2
model = dict(
    type='SkeletonGCN',
    backbone=dict(
        type='LOCGCN',
        in_channels=3,
        edge_importance_weighting=True,
        graph_cfg=dict(layout='coco', strategy='spatial')),
    cls_head=dict(
        type='STGCNHead',
        num_classes=60,
        in_channels=256,
        num_person=num_person,
        spatial_type='avg',
        loss_cls=dict(type='CrossEntropyLoss')),
    train_cfg=None,
    test_cfg=None)

dataset_type = 'PoseDataset'
ann_file_train = '/home/ubuntu/code/mmaction2/tools/data/skeleton/mini_train.json'
ann_file_val = '/home/ubuntu/code/mmaction2/tools/data/skeleton/mini_test.json'
train_pipeline = [
    dict(type='PaddingWithLoop', clip_len=30),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput',num_person=num_person, input_format='NCTVM', use_node_feature=True),
    dict(type='PoseNormalize'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PaddingWithLoop', clip_len=30),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput',num_person=num_person, input_format='NCTVM', use_node_feature=True),
    dict(type='PoseNormalize'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PaddingWithLoop', clip_len=30),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput',num_person=num_person, input_format='NCTVM', use_node_feature=True),
    dict(type='PoseNormalize'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=64,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix='',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix='',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix='',
        pipeline=test_pipeline))

# optimizer
optimizer = dict(type='Adam', lr=0.0005)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[10, 40, 70])
total_epochs = 80
checkpoint_config = dict(interval=5)
evaluation = dict(interval=5, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/home/ubuntu/test_run'
load_from = None
resume_from = None
workflow = [('train', 1)]
