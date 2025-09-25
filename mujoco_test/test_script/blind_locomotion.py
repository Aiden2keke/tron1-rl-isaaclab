#!/usr/bin/env python3

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
unitree_mujoco_test_dir = os.path.join(current_dir, "..")
sys.path.append(unitree_mujoco_test_dir)
actor_dir = os.path.join(unitree_mujoco_test_dir, "model/limx")
proprio_encoder_dir = os.path.join(unitree_mujoco_test_dir, "model/limx")
robot_data_dir = os.path.join(unitree_mujoco_test_dir, "data")

import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn
from module.modules import Actor, MLPEncoder
from pynput import keyboard
import math
import random


def quat_rotate_inverse(q, v):
    # 假设 q 的形状为 (4,)，v 的形状为 (3,)
    q_w = q[0]
    q_vec = q[1:]
    # 计算部分 a
    a = v * (2.0 * q_w**2 - 1.0)
    # 计算部分 b
    b = np.cross(q_vec, v) * q_w * 2.0
    # 计算部分 c
    c = q_vec * (np.dot(q_vec, v) * 2.0)

    return a - b + c

##############
# 初始化键盘监听
##############
# 全局变量，保存键盘状态
key_state = {
    "up": False,
    "down": False,
    "left": False,
    "right": False,
    "l": False,
    "m": False,
}
# 最大和最小速度限制
max_speed = 0.5
min_speed = -0.5
# 阻尼系数，用于速度平滑下降
damping_factor = 0.2  # 调整该值以改变速度衰减快慢


def set_target_by_key(key):
    target_pos = np.zeros(3)
    a = 0.5 / 10
    # 获取机器人朝向（世界坐标系下）
    # fw = quat_rotate_inverse(body_quat, np.array([1, 0, 0]))  # 前方
    # left = quat_rotate_inverse(body_quat, np.array([0, 1, 0]))  # 左方
    fw=np.array([1, 0, 0])
    left=np.array([0, 1, 0])
    if key == "up":
        return fw * a
    elif key == "down":
        return -fw * a
    elif key == "left":
        return left * a
    elif key == "right":
        return -left * a
    else:
        return target_pos

def on_press(key):
    """按键按下时的回调函数"""
    if key == keyboard.Key.up:
        key_state["up"] = True
    elif key == keyboard.Key.down:
        key_state["down"] = True
    elif key == keyboard.Key.left:
        key_state["left"] = True
    elif key == keyboard.Key.right:
        key_state["right"] = True
    elif key == keyboard.KeyCode.from_char("l"):
        key_state["l"] = True
    elif key == keyboard.KeyCode.from_char("l"):
        key_state["r"] = True
    elif key == keyboard.KeyCode.from_char("m"):
        key_state["m"] = True


def on_release(key):
    """按键释放时的回调函数"""
    if key == keyboard.Key.up:
        key_state["up"] = False
    elif key == keyboard.Key.down:
        key_state["down"] = False
    elif key == keyboard.Key.left:
        key_state["left"] = False
    elif key == keyboard.Key.right:
        key_state["right"] = False
    elif key == keyboard.KeyCode.from_char("l"):
        key_state["l"] = False
    elif key == keyboard.KeyCode.from_char("l"):
        key_state["r"] = False
    elif key == keyboard.KeyCode.from_char("m"):
        key_state["m"] = False


##############
# 加载机器人模型
##############
model = mujoco.MjModel.from_xml_path(robot_data_dir + "/WF_TRON1A/xml/robot.xml")
data = mujoco.MjData(model)
# for i in range(model.njnt):
#     print(f"Joint index {i}: {model.joint(i).name}")
#     print(f"  qpos index: {model.joint(i).qposadr}")
#     print(f"  qvel index: {model.joint(i).dofadr}")
############################
# 加载 policy 和 encoder 模型
############################
device = torch.device("cuda")
actor = Actor(num_obs=34+3, num_actions=8, hidden_dims=[512, 256, 128])
actor.load_state_dict(torch.load(actor_dir + "/policy_converted.pth"))
actor = actor.to(device)
actor.eval()
proprio_encoder = MLPEncoder(input_dim=28*10)
proprio_encoder.load_state_dict(torch.load(proprio_encoder_dir + f"/encoder_converted.pth"))
proprio_encoder = proprio_encoder.to(device)
proprio_encoder.eval()

###################
# 初始化储存状态的变量
###################
global command
body_pos = np.zeros(3)  # 机器人的位置 在世界坐标系下
body_quat = np.zeros(4)  # 机器人的orientation 在世界坐标系下
body_lin_vel = np.zeros(3)
body_ang_vel = np.zeros(3)
gravity_projection = np.zeros(3)
command = np.zeros(3)
joint_pos = np.zeros(6)
joint_vel = np.zeros(8)
last_action = np.zeros(8)
torques = np.zeros(8)
default_joint_angles = {  # target angles when action = 0.0
            "abad_L_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "foot_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_R_Joint": 0.0,
            "foot_R_Joint": 0.0,
        }
default_dof_pos = np.array(list(default_joint_angles.values()))
data.qpos[0:3] = np.array([0.0, 0.0, 0.6])
target_pos = np.array([0.0, 0.0, 0.0])
target_time = 0.0
dt = 0.005  # 仿真步长（可根据实际设置）
target_dist_scaled = 0.0
target_direction = np.zeros(2)
theta = random.uniform(-math.pi, math.pi)
target_orientation_x = np.array([math.cos(theta), math.sin(theta)])
base_pose_commands = np.concatenate([
    np.array([target_dist_scaled]),
    target_direction,
    target_orientation_x,
], axis=-1)
rate = random.uniform(0.5, 1.4)
base_se3_decrease_rate = np.array([rate])  # 0.5-1.4

#####################################
# 力矩计算相关增益以及 observation scale
#####################################
# 根据wheelfoot_cfg.py中的配置更新参数
# 腿部关节参数 (abad, hip, knee)
leg_p_gains = np.array([40.0, 40.0, 40.0, 40.0, 40.0, 40.0])  # 位置项增益 (stiffness)
leg_d_gains = np.array([1.8, 1.8, 1.8, 1.8, 1.8, 1.8])  # 速度项增益 (damping)
leg_torque_limits = np.array([80.0, 80.0, 80.0, 80.0, 80.0, 80.0])  # 力矩限制 (effort_limit)

# 轮关节参数
wheel_p_gains = np.array([0.0, 0.0])  # 位置项增益 (stiffness)
wheel_d_gains = np.array([0.5, 0.5])  # 速度项增益 (damping)
wheel_torque_limits = np.array([40.0, 40.0])  # 力矩限制 (effort_limit)

# 合并腿部和轮关节的参数
p_gains = np.concatenate([leg_p_gains[:3], wheel_p_gains[0:1], leg_p_gains[3:], wheel_p_gains[1:2]])
d_gains = np.concatenate([leg_d_gains[:3], wheel_d_gains[0:1], leg_d_gains[3:], wheel_d_gains[1:2]])
torque_limits = np.concatenate([leg_torque_limits[:3], wheel_torque_limits[0:1], leg_torque_limits[3:], wheel_torque_limits[1:2]])
# print("p_gains:", p_gains)
# print("d_gains:", d_gains)
# print("torque_limits:", torque_limits)
# 根据limx_base_env_cfg.py中的配置更新缩放因子
actions_scale = 0.5  # 从ActionsCfg.joint_pos.scale获取
body_lin_vel_scale = 1.0  # 从ObservationsCfg.CriticCfg.base_lin_vel.scale获取
body_ang_vel_scale = 0.25  # 从ObservationsCfg.PolicyCfg.base_ang_vel.scale获取
command_scale = np.array([1.0, 1.0, 1.0])  # 根据CommandsCfg中的范围设置
joint_vel_scale = 0.05  # 从ObservationsCfg.PolicyCfg.joint_vel.scale获取
joint_pos_scale = 1.0  # 从ObservationsCfg.PolicyCfg.joint_pos(无显式scale但使用默认)
episode_length_s = 20.0  # 从WFEnvCfg.__post_init__获取
arget_active = False

##############
# 关节状态初始化
##############
data.qpos[7:15] = default_dof_pos

#########################
# 监听键盘输入以发送 command
#########################
command = np.array([0.0, 0.0, 0.0])
prev_command = np.array([0.0, 0.0, 0.0])
# print(
#     "Use arrow keys to move the robot.\n",
#     # "'l': turn left\n",
#     # "'r': turn left\n",
#     "UP: move forward,\n",
#     "DOWN: move backward,\n",
#     "LEFT: move left,\n" "RIGHT: move right.\n",
# )
# 监听键盘输入
command_scale_factor = 0.005
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()
proprio_obs_history = torch.zeros((1, 28*10)).to(device)

###########
# 仿真主循环
###########
m = model
d = data
velocity_scale_factor = 2.0  # 调整此值以使箭头长度在视觉上更明显
min_arrow_length = 0.05      # 即使速度为零，也确保箭头有一个最小可见长度
max_arrow_length = 1.0       # 限制箭头的最大长度，防止在高速时过长导致画面混乱
dist_history = torch.zeros(50).to(device)

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat = [-7., 4.5, 3.]  # Example: move camera to (0, 0, 0.5)
    viewer.cam.azimuth = -40 # Example: rotate camera by 180 degree
    viewer.cam.elevation = -30  # Example: tilt camera by 30 degree
    start = time.time()
    last_update_time = start
    t = 0.0    
    while viewer.is_running() and time.time() - start < 5000:
        # 获取当前机器人位置和朝向
        body_pos = np.array(data.qpos[0:3])
        body_quat = np.array(data.qpos[3:7])
        # target_pos = np.array([-3, 3, 0])/2
        # arget_active = True
        t += dt
        # last_timer_active = timer_active
        # timer_active = False

    # 检查按键并设置目标点和时间
        for direction in ["up", "down", "left", "right"]:
            if key_state[direction]:
                target_pos_add = set_target_by_key(direction)
                # print("target_pos_add:", target_pos_add)
                target_pos = target_pos + target_pos_add
                arget_active = True
                timer_active = True
                break  # 只响应一个方向
    # 计算 command（期望速度），根据目标点和当前位置
        # print("target_pos:", target_pos)
        # print("body_pos:", body_pos)
        pos_diff = target_pos - body_pos
        # print("pos_diff:", pos_diff)
        distance = np.linalg.norm(pos_diff[:2])
        # print("distance:", distance)
        dist_history = torch.cat((dist_history[1:], torch.tensor([distance], device=dist_history.device)), dim=-1)
        if (dist_history < 0.1 / 10).all():
            arget_active = False
            distance = 0.0
        if arget_active and key_state["m"]:
            target_dist_scaled = 0.5 * np.log(1. + 3. * distance)
            target_direction = pos_diff / distance
            
            base_pose_commands = np.concatenate([
                        [target_dist_scaled],
                        target_direction[:2],
                        target_orientation_x[:2],
                    ], axis=-1)
        else:
            base_pose_commands = np.zeros(5)


        ############
        # 更新模型输入
        ############
        current_time = time.time()

        body_lin_vel = quat_rotate_inverse(np.array(data.qpos[3:7]), np.array(data.qvel[0:3]))
        body_ang_vel = np.array(data.qvel[3:6])  # mujoco 中的角速度是在局部坐标系下的，所以不需要转换
        gravity_projection = quat_rotate_inverse(np.array(data.qpos[3:7]), np.array([0, 0, -1.0]))
        command_input = command
        # command_input=np.array([1.0,0.0,0.0]) # 直接一直让vx=1
        joint_pos = np.concatenate([np.array(data.qpos[7:8]), np.array(data.qpos[11:12]),
                                    np.array(data.qpos[8:9]), np.array(data.qpos[12:13]),
                                    np.array(data.qpos[9:10]), np.array(data.qpos[13:14]),
                                    ])  # 去掉轮子的关节位置
        joint_vel = np.array(data.qvel[6:14])
        joint_vel = np.concatenate([np.array(data.qvel[6:7]), np.array(data.qvel[10:11]),
                                    np.array(data.qvel[7:8]), np.array(data.qvel[11:12]),
                                    np.array(data.qvel[8:9]), np.array(data.qvel[12:13]),
                                    np.array(data.qvel[9:10]), np.array(data.qvel[13:14]),
                                    ])


        proprio_observation = np.concatenate(
            [
                body_ang_vel * body_ang_vel_scale, #3
                gravity_projection, #3
                (joint_pos - default_dof_pos[:6]) * joint_pos_scale, #6
                joint_vel * joint_vel_scale, #8
                last_action, #8
            ]
        )   # shape (28,)

        command_obs = np.concatenate(
            [
                base_pose_commands, #5
                base_se3_decrease_rate, #1
            ]
        )    # shape (6,)

        # noise_strength = 0.02  # 噪声的强度，值越大，噪声越强
        # noise = np.random.randn(*proprio_observation.shape) * noise_strength  # 生成与 proprio_observation 形状相同的噪声
        # proprio_observation += noise  # 将噪声添加到 proprio_observation# 加入噪声

        proprio_observation = torch.from_numpy(proprio_observation).float().to(device).unsqueeze(0) # shape (1,28)
        proprio_obs_history = torch.cat((proprio_obs_history[:,28:], proprio_observation),dim=-1) # shape (1,280)
        
        # Convert command_obs to PyTorch tensor and move to device
        command_obs = torch.from_numpy(command_obs).float().to(device).unsqueeze(0) # shape (1,6)

        #####################################
        # 将 observation 输入模型得到输出 action
        #####################################
        proprio_latent = proprio_encoder(proprio_obs_history) #3
        proprio_latent = torch.nn.functional.normalize(proprio_latent, p=2, dim=-1)
        # print(f"proprio_latent shape: {proprio_latent.shape}")
        # print(f"proprio_observation shape: {proprio_observation.shape}")
        # print(f"command_obs shape: {command_obs.shape}")
        actor_input = torch.cat((proprio_latent, proprio_observation, command_obs), dim=-1)
        # print(f"actor_input shape: {actor_input.shape}")

        action = actor(actor_input)
        action = action.detach().cpu().numpy()
        action = np.clip(action, -6.0, 6.0)
        last_action[:] = action[:]

        ####################################
        # 将 action 映射为 torque 作为控制输出
        ####################################
        decimation = 4
        for i in range(decimation):
            # 计算力矩
            dof_pos = np.array(data.qpos[7:15])
            dof_vel = np.array(data.qvel[6:14])

            # 分离腿部和轮子的action - 首先将action展平为一维数组
            action_flat = action.flatten()  # 将(1,8)的二维数组展平为(8,)的一维数组
            # 重新计算腿部和轮子的action
            leg_action = action_flat[:6]  # 去掉轮子的action
            leg_action = np.concatenate((leg_action[0:1], leg_action[3:4],
                                         leg_action[1:2], leg_action[4:5],
                                         leg_action[2:3], leg_action[5:6],
                                         ))  # 确保顺序正确
            wheel_action = action_flat[6:8]  # 轮子的action
            # print(f"action_flat: {action_flat}")
            # print(f"leg_action: {leg_action}")
            # print(f"wheel_action: {wheel_action}")
            # 腿部关节使用位置控制
            leg_pos = np.concatenate((dof_pos[0:1], dof_pos[4:5],
                                      dof_pos[1:2], dof_pos[5:6],
                                      dof_pos[2:3], dof_pos[6:7]))  # 去掉轮子的关节位置

            leg_vel = np.concatenate((dof_vel[0:1], dof_vel[4:5],
                                      dof_vel[1:2], dof_vel[5:6],
                                      dof_vel[2:3], dof_vel[6:7]))  # 去掉轮子的关节速度
            # print(f"leg_pos: {leg_pos}")
            # print(f"leg_vel: {leg_vel}")

            leg_torques = leg_p_gains * (leg_action * actions_scale + default_dof_pos[:6] - leg_pos) - leg_d_gains * leg_vel
            
            # 轮子关节使用速度控制
            wheel_vel = np.array([dof_vel[3], dof_vel[7]])  # 轮子的关节速度
            # print(f"wheel_vel: {wheel_vel}")
            wheel_torques = wheel_d_gains * (wheel_action * actions_scale - wheel_vel)
            
            # 合并力矩并裁剪
            torques = np.concatenate((leg_torques[0:1],leg_torques[2:3],leg_torques[4:5], wheel_torques[0:1],
                                      leg_torques[1:2],leg_torques[3:4],leg_torques[5:6], wheel_torques[1:2]))
            print(f"torques: {torques}")
            torques = np.clip(torques, -torque_limits, torque_limits)
            
            data.ctrl[0:12] = torques
            # 执行仿真
            mujoco.mj_kinematics(m, d)
            mujoco.mj_step(model, data)
            time.sleep(0.003)

        ###################
        # 实时输出command命令
        ###################
        if not np.array_equal(command, prev_command):
            print(
                "Use arrow keys to move the robot.\n",
                # "'l': turn left\n",
                # "'r': turn left\n",
                "UP: move forward,\n",
                "DOWN: move backward,\n",
                "LEFT: move left,\n" "RIGHT: move right.\n",
            )
            # print(
            #     "x_velocity: {:.2f},\ny_velocity: {:.2f},\nangular_velocity: {:.2f}\n".format(
            #         command[0], command[1], command[2]
            #     )
            # )
            print(
                "x: {:.2f},\ny: {:.2f},\nangular_velocity: {:.2f}\n".format(
                    command[0], command[1], command[2]
                )
            )
            prev_command = command.copy()

        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 2
            # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
        viewer.sync()
        viewer.user_scn.ngeom = 0
        # 1. 绘制目标位置的蓝色球体
        sphere_radius = 0.05
        sphere_size = np.array([sphere_radius, 0, 0]) # 对于球体，只有第一个元素代表半径
        sphere_rgba = np.array([0.0, 0.0, 1.0, 1.0])         # 蓝色 (R, G, B, Alpha)
        target_pos_ball = target_pos + np.array([0.0, 0.0, 0.3]) # 球体中心位置
        # 获取当前可用的几何体ID，并将其用于新几何体
        geom_id = viewer.user_scn.ngeom
        # 球体不需要特定的旋转，所以使用单位矩阵
        identity_mat = np.eye(3).flatten() # 展平的3x3单位矩阵
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[geom_id], # 传入要初始化的 mjvGeom 对象
            mujoco.mjtGeom.mjGEOM_SPHERE,   # 几何体类型：球体
            sphere_size,
            target_pos_ball,                # 球体的中心位置
            identity_mat,
            sphere_rgba
        )
        viewer.user_scn.ngeom += 1 # 增加自定义几何体计数