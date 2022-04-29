num_person=2
model = dict(
    type='SkeletonGCN',
    backbone=dict(
        type='STGCN',
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
ann_file_train = '/home/tong/10708/skeleton_val/'
ann_file_val = '/home/tong/10708/skeleton_val/'
train_pipeline = [
    dict(type='PaddingWithLoop', clip_len=30),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput',num_person=num_person, input_format='NCTVM'),
    dict(type='PoseNormalize'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PaddingWithLoop', clip_len=30),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput',num_person=num_person, input_format='NCTVM'),
    dict(type='PoseNormalize'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PaddingWithLoop', clip_len=30),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput',num_person=num_person, input_format='NCTVM'),
    dict(type='PoseNormalize'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=64,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=4),
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
checkpoint_config = dict(interval=5, create_symlink=False)
evaluation = dict(interval=5, metrics=['top_k_accuracy'])
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
# work_dir = './stgcn_test_modify_dataset/'
load_from = None
resume_from = None
workflow = [('train', 1)]
