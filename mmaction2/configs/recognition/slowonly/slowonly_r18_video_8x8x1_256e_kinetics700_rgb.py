_base_ = [
    '../../_base_/models/slowonly_r18.py', '../../_base_/default_runtime.py'
]

# model settings
# model = dict(backbone=dict(pretrained=None), cls_head=dict(num_classes=700))

dataset_type = 'VideoDataset'
data_root = '/home/tong/10708/course_project/mmaction2/'
data_root_val = '/home/tong/10708/course_project/mmaction2/'
data_root_test = '/home/tong/10708/course_project/mmaction2/'
ann_file_train = 'data/kinetics_700_new/train_list_videos.txt'
ann_file_val = 'data/kinetics_700_new/val_list_videos.txt'
ann_file_test = 'data/kinetics_700_new/test_list_videos.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=8, frame_interval=8, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=10,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=12,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy', 'confusion_matrix', 'binary_precision_recall_curve'])

# # optimizer
# optimizer = dict(
#     type='SGD', lr=0.08, momentum=0.9,
#     weight_decay=0.0001)  # this lr is used for 8 gpus
# optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# # learning policy
# lr_config = dict(policy='CosineAnnealing', min_lr=0)
# total_epochs = 50
optimizer = dict(
    type='SGD', lr=0.01, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    step=[60, 85],
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=10)
total_epochs = 100

# runtime settings
checkpoint_config = dict(interval=5)
work_dir = './work_dirs/slowonly_r18_video_8x8x1_256e_kinetics700_rgb_100epoch'
find_unused_parameters = False
