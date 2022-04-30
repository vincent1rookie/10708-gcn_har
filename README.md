# gcn_har

10708 project

## Installation

You should follow the installation instruction of `mmaction2`, `mmdetection` and `mmpose`, but you should build the three pachages from our source.

### requirement

- Linux
- Python 3.6+
- Pytorch 1.3+
- CUDA 9.2+

```
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
conda install pytorch torchvision -c pytorch
# pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html
cd mmaction2
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
cd ..
cd mmpose
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
cd ..
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
cd ..
```

## configuration

##TODO: how to download data

- The unzipped data should be placed at /home/ubuntu/data

- You can visualized the skeleton result of each video using our code at mmaction2/demo/visualization.ipynb

## How to obtian data

To use `merge_pkl.py`, you need to change the path at line 5 and line 23 according to your system

```
conda activate open-mmlab
cd [your folder]/mmaction2/tools/data/skeleton
python ntu_pose_extraction.py
cd [your folder]/mmaction2/configs/skeleton/10708
python merge_pkl.py
```

## How to run the code

You may need to change the `ann_file_train` and `ann_file_val` based on your datafile name before executing the code
You may also need to change `gpu-ids` based on your system.

```
cd [your folder]/mmaction2/tools
python train.py [your folder]/mmaction2/configs/skeleton/10708/test_config_gbl.py --gpu-ids 0 --validate # global feature gcn
python train.py [your folder]/mmaction2/configs/skeleton/10708/test_config_loc.py --gpu-ids 0 --validate # local feature gcn
python train.py [your folder]/mmaction2/configs/skeleton/10708/test_config.py --gpu-ids 0 --validate # gcn baseline
```

=======

- You can visualized the skeleton result of each video using our code at mmaction2/demo/visualization.ipynb
