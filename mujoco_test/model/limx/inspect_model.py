import torch
from collections import OrderedDict
# 将路径替换为你的 model_7500.pt 文件路径
path = "/home/yd/program/legged_ball_catching-1/legged_gym/logs/rough_go2/12/model_7500.pt"
data = torch.load(path, map_location="cpu")

def print_shapes(obj, prefix=""):
    if isinstance(obj, torch.Tensor):
        print(f"{prefix}: {list(obj.shape)}")
    elif isinstance(obj, dict) or isinstance(obj, OrderedDict):
        for k, v in obj.items():
            # 把 k 转为 str，避免拼接出错
            key = str(k)
            new_prefix = prefix + "." + key if prefix else key
            print_shapes(v, new_prefix)
    else:
        print(f"{prefix}: <{type(obj)}>")

print_shapes(data)