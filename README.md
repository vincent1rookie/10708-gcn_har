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

skeleton流程：
1. fasterrcnn for person detection进行目标检测
2.  每个bbox内选取17个点（17个headmap，最高的score作为点)，score之间的连接是固定的,由mmpose model决定（见下面例子）。但是可以根据每个点的score决定这个点是否存在  
e.g    
skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                        [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                        [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
                        [3, 5], [4, 6]]
    
3. 可视化可以使用mmaction2/demo/visualization.ipynb 的 例子进行