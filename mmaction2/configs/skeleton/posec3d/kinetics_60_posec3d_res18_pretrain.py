model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowOnly',
        depth=18,
        pretrained='torchvision://resnet18',
        in_channels=17,
        base_channels=64,
        # lateral=False,
        # conv1_kernel=(1, 7, 7),
        # num_stages=4,
        # out_indices=(3, ),
        # stage_blocks=(3, 4, 6),
        conv1_stride_s=1,
        pool1_stride_s=1,
        inflate=(1, 1, 1, 1)),
        # spatial_strides=(1, 2, 2, 2),
        # temporal_strides=(1, 1, 1, 1),
        # dilations=(1, 1, 1, 1)),
    cls_head=dict(
        type='I3DHead',
        in_channels=512,
        num_classes=60,
        spatial_type='avg',
        dropout_ratio=0.5),
    train_cfg=dict(),
    test_cfg=dict(average_clips='prob'))

dataset_type = 'PoseDataset'
ann_file_train = '/home/tong/10708/skeleton_processed/skeleton_train_processed/'
ann_file_val = '/home/tong/10708/skeleton_processed/skeleton_val_processed/'
data_root = '/home/tong/10708/kinetics/kinetics60/'
video_path_files = ['./train_list.txt', './val_list.txt']
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(48, 48), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(
        type='GeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 56)),
    dict(type='CenterCrop', crop_size=56),
    dict(
        type='GeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='UniformSampleFrames', clip_len=48, num_clips=10, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 56)),
    dict(type='CenterCrop', crop_size=56),
    dict(
        type='GeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False,
        double=True,
        left_kp=left_kp,
        right_kp=right_kp),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
            type=dataset_type,
            ann_file=ann_file_train,
            data_prefix='',
            pipeline=train_pipeline,
            data_root=data_root,
            video_path_files=video_path_files),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix='',
        pipeline=val_pipeline,
        data_root=data_root,
        video_path_files=video_path_files),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix='',
        pipeline=test_pipeline,
        data_root=data_root,
        video_path_files=video_path_files))
# optimizer
optimizer = dict(
    type='SGD', lr=0.005, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[35, 45])
total_epochs = 50
checkpoint_config = dict(interval=4, create_symlink=False)
workflow = [('train', 1)]
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))
log_config = dict(
    interval=20, hooks=[
        dict(type='TextLoggerHook'),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/posec3d_iclr/slowonly_kinetics400_pretrained_r50_u48_120e_hmdb51_split1_keypoint'  # noqa: E501
load_from = None  # noqa: E501
resume_from = None
find_unused_parameters = True
