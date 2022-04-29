# Copyright (c) OpenMMLab. All rights reserved.
import copy as cp
import os
import os.path as osp
import time
import glob
import numpy as np
import torch
import torch.distributed as dist
import mmcv
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner, OptimizerHook,
                         build_optimizer, get_dist_info)
from mmcv.runner.hooks import Fp16OptimizerHook

from ..core import (DistEvalHook, EvalHook, OmniSourceDistSamplerSeedHook,
                    OmniSourceRunner)
from ..datasets import build_dataloader, build_dataset
from ..utils import PreciseBNHook, get_root_logger
from .test import multi_gpu_test


def init_random_seed(seed=None, device='cuda', distributed=True):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
        distributed (bool): Whether to use distributed training.
            Default: True.
    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)

    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)

    if distributed:
        dist.broadcast(random_num, src=0)
    return random_num.item()


def test_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                test=dict(test_best=False, test_last=False),
                timestamp=None,
                meta=None):
    """Train model entry function.

    Args:
        model (nn.Module): The model to be trained.
        dataset (:obj:`Dataset`): Train dataset.
        cfg (dict): The config dict for training.
        distributed (bool): Whether to use distributed training.
            Default: False.
        validate (bool): Whether to do evaluation. Default: False.
        test (dict): The testing option, with two keys: test_last & test_best.
            The value is True or False, indicating whether to test the
            corresponding checkpoint.
            Default: dict(test_best=False, test_last=False).
        timestamp (str | None): Local time for runner. Default: None.
        meta (dict | None): Meta dict to record some important information.
            Default: None
    """
    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        persistent_workers=cfg.data.get('persistent_workers', False),
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('train_dataloader', {}))

    if cfg.omnisource:
        # The option can override videos_per_gpu
        train_ratio = cfg.data.get('train_ratio', [1] * len(dataset))
        omni_videos_per_gpu = cfg.data.get('omni_videos_per_gpu', None)
        if omni_videos_per_gpu is None:
            dataloader_settings = [dataloader_setting] * len(dataset)
        else:
            dataloader_settings = []
            for videos_per_gpu in omni_videos_per_gpu:
                this_setting = cp.deepcopy(dataloader_setting)
                this_setting['videos_per_gpu'] = videos_per_gpu
                dataloader_settings.append(this_setting)
        data_loaders = [
            build_dataloader(ds, **setting)
            for ds, setting in zip(dataset, dataloader_settings)
        ]

    else:
        data_loaders = [
            build_dataloader(ds, **dataloader_setting) for ds in dataset
        ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    Runner = OmniSourceRunner if cfg.omnisource else EpochBasedRunner
    runner = Runner(
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)
    
    eval_cfg = cfg.get('evaluation', {})
    val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
#         dataloader_setting = dict(
#             videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
#             workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
#             persistent_workers=cfg.data.get('persistent_workers', False),
#             # cfg.gpus will be ignored if distributed
#             num_gpus=len(cfg.gpu_ids),
#             dist=distributed,
#             shuffle=False)
    dataloader_setting = dict(
            videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
            workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
            persistent_workers=cfg.data.get('persistent_workers', False),
            # cfg.gpus will be ignored if distributed
            num_gpus=1,
            dist=False,
            shuffle=False)
    dataloader_setting = dict(dataloader_setting,
                                  **cfg.data.get('val_dataloader', {}))
    print("dataloader_setting: ", dataloader_setting)
    val_dataloader = build_dataloader(val_dataset, **dataloader_setting)
    
    cfg.resume_from = '/home/tong/10708/gcn/10708-gcn_har/mmaction2/stgcn_loc_5_modify_dataset/epoch_80.pth'
    # checkpoint_files = sorted(glob.glob(cfg.resume_from + '/epoch_*.pth'))
    # validation_results = []
    # for ckpt in checkpoint_files:
        runner.resume(ckpt)
#     if cfg.resume_from:
#         runner.resume(cfg.resume_from)
#     elif cfg.load_from:
#         runner.load_checkpoint(cfg.load_from)
        
        runner.model.eval()
        results = []
        gts = []

        with torch.no_grad():
            prog_bar = mmcv.ProgressBar(len(val_dataloader))
            for i, data in enumerate(val_dataloader):

                res = model(return_loss=False, **data)
                results.append(res)
                label = data['label']
                # print(res.shape, label.shape)
                gts.append(label.numpy())
                prog_bar.update()

        preds = np.concatenate(results, axis=0)
        labels = np.concatenate(gts, axis=0)
        acc = top_k_accuracy(preds, labels, topk=(1, 5,))
        print("Validation Acc: ", acc)
        
        validation_results.append([os.path.basename(ckpt), acc])
        
    for res in validation_results:
        print('Model: ', res[0], f"Top 1 Acc: {res[1][0]:.3f}, Top 5 Acc: {res[1][1]:.3f}")
    
def top_k_accuracy(scores, labels, topk=(1, )):
    """Calculate top k accuracy score.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.
        topk (tuple[int]): K value for top_k_accuracy. Default: (1, ).

    Returns:
        list[float]: Top k accuracy score for each k.
    """
    res = []
    labels = np.array(labels)[:, np.newaxis]
    for k in topk:
        max_k_preds = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
        match_array = np.logical_or.reduce(max_k_preds == labels, axis=1)
        topk_acc_score = match_array.sum() / match_array.shape[0]
        res.append(topk_acc_score)

    return res
#     if test['test_last'] or test['test_best']:
#         best_ckpt_path = None
#         if test['test_best']:
#             ckpt_paths = [x for x in os.listdir(cfg.work_dir) if 'best' in x]
#             ckpt_paths = [x for x in ckpt_paths if x.endswith('.pth')]
#             if len(ckpt_paths) == 0:
#                 runner.logger.info('Warning: test_best set, but no ckpt found')
#                 test['test_best'] = False
#                 if not test['test_last']:
#                     return
#             elif len(ckpt_paths) > 1:
#                 epoch_ids = [
#                     int(x.split('epoch_')[-1][:-4]) for x in ckpt_paths
#                 ]
#                 best_ckpt_path = ckpt_paths[np.argmax(epoch_ids)]
#             else:
#                 best_ckpt_path = ckpt_paths[0]
#             if best_ckpt_path:
#                 best_ckpt_path = osp.join(cfg.work_dir, best_ckpt_path)

#         test_dataset = build_dataset(cfg.data.test, dict(test_mode=True))
#         gpu_collect = cfg.get('evaluation', {}).get('gpu_collect', False)
#         tmpdir = cfg.get('evaluation', {}).get('tmpdir',
#                                                osp.join(cfg.work_dir, 'tmp'))
#         dataloader_setting = dict(
#             videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
#             workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
#             persistent_workers=cfg.data.get('persistent_workers', False),
#             num_gpus=len(cfg.gpu_ids),
#             dist=distributed,
#             shuffle=False)
#         dataloader_setting = dict(dataloader_setting,
#                                   **cfg.data.get('test_dataloader', {}))

#         test_dataloader = build_dataloader(test_dataset, **dataloader_setting)

#         names, ckpts = [], []

#         if test['test_last']:
#             names.append('last')
#             ckpts.append(None)
#         if test['test_best'] and best_ckpt_path is not None:
#             names.append('best')
#             ckpts.append(best_ckpt_path)

#         for name, ckpt in zip(names, ckpts):
#             if ckpt is not None:
#                 runner.load_checkpoint(ckpt)

#             outputs = multi_gpu_test(runner.model, test_dataloader, tmpdir,
#                                      gpu_collect)
#             rank, _ = get_dist_info()
#             if rank == 0:
#                 out = osp.join(cfg.work_dir, f'{name}_pred.pkl')
#                 test_dataset.dump_results(outputs, out)

#                 eval_cfg = cfg.get('evaluation', {})
#                 for key in [
#                         'interval', 'tmpdir', 'start', 'gpu_collect',
#                         'save_best', 'rule', 'by_epoch', 'broadcast_bn_buffers'
#                 ]:
#                     eval_cfg.pop(key, None)

#                 eval_res = test_dataset.evaluate(outputs, **eval_cfg)
#                 runner.logger.info(f'Testing results of the {name} checkpoint')
#                 for metric_name, val in eval_res.items():
#                     runner.logger.info(f'{metric_name}: {val:.04f}')
