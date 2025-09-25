import torch
import os

# 定义要转换的文件路径
directory = "/home/yd/program/tron1-rl-isaaclab-pose/mujoco_test/model/limx"
actor_file = os.path.join(directory, "policy.pth")
encoder_file = os.path.join(directory, "encoder.pth")

# 转换Actor模型的权重文件
def convert_actor_pth():
    # 加载原pth文件
    state_dict = torch.load(actor_file, map_location=torch.device('cpu'))
    
    # 创建新的state_dict，转换键名格式
    new_state_dict = {}
    for key, value in state_dict.items():
        # 将 "0/Gemm.weight" 转换为 "actor.0.weight"
        new_key = key.replace("/Gemm", "")
        new_key = "actor." + new_key
        new_state_dict[new_key] = value
    
    # 保存转换后的pth文件
    new_file = os.path.join(directory, "policy_converted.pth")
    torch.save(new_state_dict, new_file)
    print(f"Actor模型权重转换完成，已保存至：{new_file}")
    print(f"原键名: {list(state_dict.keys())[:2]}...")
    print(f"新键名: {list(new_state_dict.keys())[:2]}...")

# 转换MLPEncoder模型的权重文件
def convert_encoder_pth():
    # 加载原pth文件
    state_dict = torch.load(encoder_file, map_location=torch.device('cpu'))
    
    # 创建新的state_dict，转换键名格式
    new_state_dict = {}
    for key, value in state_dict.items():
        # 处理两种可能的键名格式
        if "encoder/encoder/" in key:
            # 对于 "encoder/encoder/0/Gemm.weight" 转换为 "encoder.0.weight"
            new_key = key.replace("encoder/encoder/", "").replace("/Gemm", "")
            new_key = "encoder." + new_key
        else:
            # 对于 "0/Gemm.weight" 转换为 "encoder.0.weight"
            new_key = key.replace("/Gemm", "")
            new_key = "encoder." + new_key
        new_state_dict[new_key] = value
    
    # 保存转换后的pth文件
    new_file = os.path.join(directory, "encoder_converted.pth")
    torch.save(new_state_dict, new_file)
    print(f"Encoder模型权重转换完成，已保存至：{new_file}")
    print(f"原键名: {list(state_dict.keys())[:2]}...")
    print(f"新键名: {list(new_state_dict.keys())[:2]}...")

if __name__ == "__main__":
    print("开始转换pth文件中的键名格式...")
    convert_actor_pth()
    convert_encoder_pth()
    print("所有转换已完成！")
    print("\n请在blind_locomotion.py文件中修改加载权重的代码，使用转换后的文件")