# gcn_har
10708 project


ssh ubuntu@3.137.60.169 -i .ssh/22spring.pem
source anaconda3/bin/activate
conda activate open-mmlab
cd code/mmaction2/tools/data/skeleton/
python ntu_pose_extraction.py

注意事项：
尽量每个在server git clone 自己的code base，避免大家的代码重复修改


open-mmlab 的mmdet mmaction mmpose 的路径都在本地，所以如果复制了code base的话，也最好复制环境并且把三个mm库指向自己的code base

数据位置：
/home/ubuntu/data