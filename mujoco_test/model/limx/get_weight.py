import os
import torch

# 可配置的变量
log_dir = "/home/yd/program/tron1-rl-isaaclab-pose/mujoco_test/model/limx/encoder.pt"
# log_dir = "/home/yd/program/tron1-rl-isaaclab-pose/mujoco_test/model/limx/policy.pt"
ckpt = torch.load(log_dir, map_location="cpu")
model_dict = ckpt["model_state_dict"]

# 2. 提取 proprioceptive_encoder
prop_keys = [k for k in model_dict if k.startswith("proprioceptive_encoder.")]
prop_dict = {
    k.replace("proprioceptive_encoder.", ""): model_dict[k]  # 移除前缀
    for k in prop_keys
}

# 3. 提取 actor
# actor_keys = [k for k in model_dict if k.startswith("actor.")]
# actor_dict = {
#     k: model_dict[k]  # 保留原始键名称
#     for k in actor_keys
# }
##### 换一种方式提取 #####
actor_dict = {k: v for k, v in model_dict.items() if k.startswith("actor.")}

torch.save(prop_dict, "encoder.pth")
print(f"✅ 已保存 encoder.pth")

torch.save(actor_dict, "policy.pth")
print(f"✅ 已保存 policy.pth")
