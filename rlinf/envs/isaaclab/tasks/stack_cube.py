# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""中文学习注释：IsaacLab Stack-Cube 任务适配器。

它定义如何创建 Franka stack-cube 仿真环境，以及如何把 IsaacLab 原始观测整理成 OpenPI 需要的图像、状态和语言指令。
"""

import gymnasium as gym
import torch

from rlinf.envs.isaaclab.utils import quat2axisangle_torch

from ..isaaclab_env import IsaaclabBaseEnv


# 中文学习注释：StackCube 任务的具体环境类，继承通用 IsaaclabBaseEnv。
class IsaaclabStackCubeEnv(IsaaclabBaseEnv):
    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
    ):
        super().__init__(
            cfg,
            num_envs,
            seed_offset,
            total_num_processes,
            worker_info,
        )

    def _make_env_function(self):
        # 中文学习注释：返回子进程里真正创建 IsaacLab sim_app/env 的函数。
        """
        function for make isaaclab
        """

        def make_env_isaaclab():
            # 中文学习注释：这个内部函数会被 SubProcIsaacLabEnv 发送到子进程执行。
            import os

            # Remove DISPLAY variable to force headless mode and avoid GLX errors
            os.environ.pop("DISPLAY", None)  # 中文学习注释：强制 headless，避免远程服务器没有 X11 显示导致 GLX 错误。

            from isaaclab.app import AppLauncher

            sim_app = AppLauncher(headless=True, enable_cameras=True).app  # 中文学习注释：启动 Isaac Sim 应用，并开启相机渲染。
            from isaaclab_tasks.utils import load_cfg_from_registry

            isaac_env_cfg = load_cfg_from_registry(
                self.isaaclab_env_id, "env_cfg_entry_point"
            )
            # Seed the IsaacLab env config before construction so the simulator's
            # initial reset path is deterministic and doesn't warn about an unset seed.
            isaac_env_cfg.seed = self.seed  # 中文学习注释：把 RLinf seed 传给 IsaacLab，保证 reset 可复现。
            isaac_env_cfg.scene.num_envs = (  # 中文学习注释：设置 IsaacLab 内部并行环境数量。
                self.cfg.init_params.num_envs
            )  # default 4096 ant_env_spaces.pkl

            isaac_env_cfg.scene.wrist_cam.height = self.cfg.init_params.wrist_cam.height
            isaac_env_cfg.scene.wrist_cam.width = self.cfg.init_params.wrist_cam.width
            isaac_env_cfg.scene.table_cam.height = self.cfg.init_params.table_cam.height
            isaac_env_cfg.scene.table_cam.width = self.cfg.init_params.table_cam.width

            # 中文学习注释：创建 Gymnasium 环境，render_mode=rgb_array 让相机图像以数组形式返回。
            env = gym.make(
                self.isaaclab_env_id, cfg=isaac_env_cfg, render_mode="rgb_array"
            ).unwrapped
            return env, sim_app

        return make_env_isaaclab

    def _wrap_obs(self, obs):
        # 中文学习注释：把 IsaacLab policy obs 映射成 RLinf/OpenPI 的统一 observation 字段。
        instruction = [self.task_description] * self.num_envs  # 中文学习注释：每个并行环境共享同一条语言任务指令。
        wrist_image = obs["policy"]["wrist_cam"]  # 中文学习注释：腕部相机图像，后续喂给 OpenPI left_wrist_0_rgb。
        table_image = obs["policy"]["table_cam"]  # 中文学习注释：桌面/第三人称相机图像，后续喂给 OpenPI base_0_rgb。
        quat = obs["policy"]["eef_quat"][  # 中文学习注释：末端执行器四元数，IsaacLab 原始顺序是 wxyz。
            :, [1, 2, 3, 0]
        ]  # In isaaclab, quat is wxyz not like libero
        states = torch.concatenate(  # 中文学习注释：拼出 7 维状态：eef 位置 + 旋转 axis-angle + gripper。
            [
                obs["policy"]["eef_pos"],
                quat2axisangle_torch(quat),
                obs["policy"]["gripper_pos"],
            ],
            dim=1,
        )

        env_obs = {  # 中文学习注释：EnvWorker/rollout 后续只认这一套统一字段名。
            "main_images": table_image,
            "task_descriptions": instruction,
            "states": states,
            "wrist_images": wrist_image,
        }
        return env_obs
