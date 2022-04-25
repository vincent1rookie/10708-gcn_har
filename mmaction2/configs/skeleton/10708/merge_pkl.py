from mmcv import load, dump
import os
import numpy as np
import json
base_path = '/home/ubuntu/code/mmaction2/tools/data/skeleton/keypoints'
files = os.listdir(base_path)
b = []
for f in files[1000:1100]:
    p = os.path.join(base_path,f)
    temp = load(p)
    if(temp[0]['keypoint'].shape[0] == 0):
        continue
    if isinstance(temp[0], list):
        print(f)
    else:
        data = {}
        for k, v in temp[0].items():
            if isinstance(v, np.ndarray):
                data[k] = v.tolist()
            else:
                data[k] = v
        b.append(data)
with open('/home/ubuntu/code/mmaction2/tools/data/skeleton/mini_test.json', 'w') as f:
    json.dump(b, f)