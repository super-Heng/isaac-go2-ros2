import os
import hydra
import rclpy
import torch
import time
import math
import argparse
import gymnasium as gym  # [修改点 1] 引入 gymnasium 用于视频录制
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# Disable Kit's hang detector — long env.step() ticks under heavy sensors
# would otherwise trigger the "Kit appears to be hanging" zenity dialog.
import sys
sys.argv += ["--/app/hangDetector/enabled=false"]

# === 新增这一行：强行让无头模式加载合成数据引擎，防止 import 报错 ===
sys.argv += ["--enable", "omni.replicator.core"]

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.argv = [sys.argv[0]]
"""Rest everything follows."""

import torch

from go2.go2_env import Go2RSLEnvCfg, camera_follow
import env.sim_env as sim_env
#import go2.go2_sensors as go2_sensors
import omni
import carb
import go2.go2_ctrl as go2_ctrl
import ros2.go2_ros2_bridge as go2_ros2_bridge

FILE_PATH = os.path.join(os.path.dirname(__file__), "cfg")
@hydra.main(config_path=FILE_PATH, config_name="sim", version_base=None)
def run_simulator(cfg):

    # Go2 Environment setup
    go2_env_cfg = Go2RSLEnvCfg()
    go2_env_cfg.scene.num_envs = cfg.num_envs
    go2_env_cfg.decimation = math.ceil(1./go2_env_cfg.sim.dt/cfg.freq)
    go2_env_cfg.sim.render_interval = go2_env_cfg.decimation
    go2_ctrl.init_base_vel_cmd(cfg.num_envs)
    
    # 强制使用平地策略，保证基准测试不摔倒
    env, policy = go2_ctrl.get_rsl_flat_policy(go2_env_cfg)
    #env, policy = go2_ctrl.get_rsl_rough_policy(go2_env_cfg)

    # Simulation environment - 建议在这里也暂时强制为 flat 以避免地形消耗
    if (cfg.env_name == "obstacle-dense"):
        sim_env.create_obstacle_dense_env() # obstacles dense
    elif (cfg.env_name == "obstacle-medium"):
        sim_env.create_obstacle_medium_env() # obstacles medium
    elif (cfg.env_name == "obstacle-sparse"):
        sim_env.create_obstacle_sparse_env() # obstacles sparse
    elif (cfg.env_name == "warehouse"):
        sim_env.create_warehouse_env() # warehouse
    elif (cfg.env_name == "warehouse-forklifts"):
        sim_env.create_warehouse_forklifts_env() # warehouse forklifts
    elif (cfg.env_name == "warehouse-shelves"):
        sim_env.create_warehouse_shelves_env() # warehouse shelves
    elif (cfg.env_name == "full-warehouse"):
        sim_env.create_full_warehouse_env() # full warehouse

    # [修改点 2] 为了最佳内录性能，强制注释掉所有复杂的传感器！
    # 如果你想在视频里看到相机画面，你可以在 yaml 里只开 enable_camera，关掉 LiDAR 和 Segmentation
    #sm = go2_sensors.SensorManager(cfg.num_envs)
    # lidar_annotators = sm.add_rtx_lidar()
    # cameras = sm.add_camera(cfg.freq)
    lidar_annotators = None
    cameras = None

    # Keyboard control
    system_input = carb.input.acquire_input_interface()
    system_input.subscribe_to_keyboard_events(
        omni.appwindow.get_default_app_window().get_keyboard(), go2_ctrl.sub_keyboard_event)

    # ROS2 Bridge
    rclpy.init()
    dm = go2_ros2_bridge.RobotDataManager(env, lidar_annotators, cameras, cfg)

    # ================= [修改点 3] 挂载录像机 =================
    print("[INFO] 正在挂载自动录像机 (底层注入)...")
    try:
        # 1. 剥离 rsl_rl 的外壳，取出里面的原生 IsaacLab Gym 环境
        base_env = env.unwrapped

        # 2. 强制设为 rgb_array 模式，以支持后台抓取画面
        base_env.render_mode = "rgb_array"

        # 3. 将录像机套在底层的原生环境上
        recorded_base_env = gym.wrappers.RecordVideo(
            base_env,
            video_folder=os.path.join(os.path.dirname(__file__), "videos"), # 绝对路径
            episode_trigger=lambda x: True,
            name_prefix="go2_headless_test"
        )

        # 4. 偷梁换柱：把装好录像机的底层环境，塞回 rsl_rl 的外壳里
        if hasattr(env, 'env'):
            env.env = recorded_base_env
        else:
            env._env = recorded_base_env # 兼容部分版本的隐藏属性

    except Exception as e:
        print(f"[Error] 录像机挂载失败: {e}")
    # =========================================================

    # Run simulation
    sim_step_dt = float(go2_env_cfg.sim.dt * go2_env_cfg.decimation)
    
    # 包装完之后必须 reset 一次，录像机才会准备就绪
    obs, _ = env.reset()
    
    # [修改点 4] 设置最大录制步数，因为纯无头跑得很快，我们录 500 步 (约 10 秒)
    max_record_steps = 500
    current_step = 0

    print(f"[INFO] 开始录制，预计录制 {max_record_steps} 步 (10秒)...")

    # 修改循环条件：不仅程序要 running，而且步数不能超
    while simulation_app.is_running() and current_step < max_record_steps:
        start_time = time.time()
        with torch.inference_mode():
            # control joints
            actions = policy(obs)

            # step the environment
            obs, _, _, _ = env.step(actions)

            # # ROS2 data (因为没有传感器，这里暂时不发，或者你只发送基础关节数据)
            # dm.pub_ros2_data()
            rclpy.spin_once(dm, timeout_sec=0.0)

            # Camera follow
            if (cfg.camera_follow):
                camera_follow(env)

            # Pump the Kit UI
            simulation_app.update()

            # limit loop time
            elapsed_time = time.time() - start_time
            if elapsed_time < sim_step_dt:
                sleep_duration = sim_step_dt - elapsed_time
                time.sleep(sleep_duration)
        
        # 计算步数和打印信息
        current_step += 1
        actual_loop_time = time.time() - start_time
        rtf = min(1.0, sim_step_dt/elapsed_time)
        print(f"\rStep: {current_step}/{max_record_steps} | Step time: {actual_loop_time*1000:.2f}ms, Real Time Factor: {rtf:.2f}", end='', flush=True)

    print("\n[INFO] 达到指定步数，录制结束，正在生成 MP4 文件...")
    
    # ================= [修改点 5] 必须执行 close 才能合成视频 =================
    env.close()  
    
    dm.destroy_node()
    rclpy.shutdown()
    simulation_app.close()

if __name__ == "__main__":
    run_simulator()
