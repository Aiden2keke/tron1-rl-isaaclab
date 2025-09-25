import onnx
import numpy as np
import torch
from onnx2torch import convert

onnx_path = '/home/yd/program/tron1-rl-isaaclab-pose/mujoco_test/model/limx/encoder.onnx'
pt_path   = '/home/yd/program/tron1-rl-isaaclab-pose/mujoco_test/model/limx/encoder.pt'
pth_path   = '/home/yd/program/tron1-rl-isaaclab-pose/mujoco_test/model/limx/encoder.pth'

# onnx_path = '/home/yd/program/tron1-rl-isaaclab-pose/mujoco_test/model/limx/policy.onnx'
# pt_path   = '/home/yd/program/tron1-rl-isaaclab-pose/mujoco_test/model/limx/policy.pt'
# pth_path   = '/home/yd/program/tron1-rl-isaaclab-pose/mujoco_test/model/limx/policy.pth'

# get .pth
# 1. ONNX -> PyTorch nn.Module
torch_model = convert(onnx_path)

# 2. 直接保存 state_dict（普通 PyTorch 权重文件）
torch.save(torch_model.state_dict(), pth_path)
print('PyTorch state_dict saved ->', pth_path)



# get .pt
# # 1. 读取 ONNX 图
# model_onnx = onnx.load(onnx_path)
# g = model_onnx.graph

# # 2. 根据 ONNX 输入信息自动生成 dummy 张量
# dummy_inputs = []
# for inp in g.input:
#     name = inp.name
#     tensor_type = inp.type.tensor_type
#     shape = []
#     for d in tensor_type.shape.dim:
#         if d.HasField('dim_value'):          # 静态维度
#             shape.append(d.dim_value)
#         elif d.HasField('dim_param'):        # 动态维度（N, ?, batch...）
#             # 随意给个常数，通常 1 最不容易炸
#             shape.append(1)
#         else:                                # 罕见情况
#             shape.append(1)
#     dtype_map = {1: torch.float32, 3: torch.int32, 7: torch.int64}
#     dtype = dtype_map.get(tensor_type.elem_type, torch.float32)
#     dummy = torch.randn(*shape, dtype=dtype)   # 默认 float32
#     dummy_inputs.append(dummy)

# # 3. ONNX -> PyTorch nn.Module
# torch_model = convert(onnx_path)

# # 4. 一键 trace（若模型有多个输入，把 tuple 传进去）
# if len(dummy_inputs) == 1:
#     traced = torch.jit.trace(torch_model, dummy_inputs[0])
# else:
#     traced = torch.jit.trace(torch_model, tuple(dummy_inputs))

# # 5. 保存
# traced.save(pt_path)
# print('TorchScript saved ->', pt_path)