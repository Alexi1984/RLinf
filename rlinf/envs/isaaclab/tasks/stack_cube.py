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

"""中文学习注释：RLinf 侧的 IsaacLab Stack-Cube 任务适配器。

这个文件不定义 cube 怎么摆、奖励怎么算、成功条件是什么；那些在 IsaacLab 的 stack 任务配置里。
它负责两件更“接口层”的事：
1. 按 env/isaaclab_stack_cube.yaml 里的任务 id 创建真实 IsaacLab/Gym 环境。
2. 把 IsaacLab 返回的 obs["policy"] 改包成 RLinf rollout 和 OpenPI policy 认识的 observation 字段。
"""

import gymnasium as gym
import torch

from rlinf.envs.isaaclab.utils import quat2axisangle_torch

from ..isaaclab_env import IsaaclabBaseEnv


# 中文学习注释：StackCube 的 RLinf wrapper；REGISTER_ISAACLAB_ENVS 会把任务 id 映射到这个类。
class IsaaclabStackCubeEnv(IsaaclabBaseEnv):
    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
    ):
        # super().__init__ 会保存 cfg/seed/num_envs，并调用本类的 _make_env_function 去启动 IsaacLab 子进程环境。
        super().__init__(
            cfg,
            num_envs,
            seed_offset,
            total_num_processes,
            worker_info,
        )

    def _make_env_function(self):
        # 中文学习注释：返回“环境工厂函数”；IsaaclabBaseEnv 会把它交给 SubProcIsaacLabEnv，让真实 Isaac Sim 在子进程里启动。
        """
        function for make isaaclab
        """

        def make_env_isaaclab():
            # 中文学习注释：这个内部函数不是立刻在当前进程执行，而是在 IsaacLab 子进程里执行，所以 import 放在函数内更安全。
            import os

            # Remove DISPLAY variable to force headless mode and avoid GLX errors
            os.environ.pop("DISPLAY", None)  # 中文学习注释：删掉 DISPLAY，明确走无界面/headless 模式，避免服务器没有 X11 时触发窗口初始化错误。

            from isaaclab.app import AppLauncher

            sim_app = AppLauncher(headless=True, enable_cameras=True).app  # 中文学习注释：启动 Isaac Sim/Kit 应用；enable_cameras=True 是为了让 table/wrist RGB 观测能渲染出来。
            from isaaclab_tasks.utils import load_cfg_from_registry

            isaac_env_cfg = load_cfg_from_registry(
                self.isaaclab_env_id, "env_cfg_entry_point"#“根据当前任务 ID，去 IsaacLab 注册表里找到 env_cfg_entry_point 这一项，然后加载它指向的环境配置类。”
            )  # 中文学习注释：根据 IsaacLab 注册任务 id 取出底层 EnvCfg；当前 id 来自 init_params.id。
            # Seed the IsaacLab env config before construction so the simulator's
            # initial reset path is deterministic and doesn't warn about an unset seed.
            isaac_env_cfg.seed = self.seed  # 中文学习注释：把 RLinf 运行时 seed 传给 IsaacLab，保证同一配置下 reset/随机初始化尽量可复现。
            isaac_env_cfg.scene.num_envs = (  # 中文学习注释：设置这个 IsaacLab env 实例内部并行仿真的环境数；值由 EnvWorker 根据 total_num_envs 分配后写入 cfg.init_params.num_envs。
                self.cfg.init_params.num_envs
            )  # default 4096 ant_env_spaces.pkl

            isaac_env_cfg.scene.wrist_cam.height = self.cfg.init_params.wrist_cam.height  # 中文学习注释：用 RLinf YAML 覆盖腕部相机输出高度；这是环境渲染尺寸，不等同于视觉塔最终输入尺寸。
            isaac_env_cfg.scene.wrist_cam.width = self.cfg.init_params.wrist_cam.width  # 中文学习注释：用 RLinf YAML 覆盖腕部相机输出宽度，保持和 SFT 数据/归一化统计的观测格式一致。
            isaac_env_cfg.scene.table_cam.height = self.cfg.init_params.table_cam.height  # 中文学习注释：覆盖桌面/第三人称相机输出高度；后面会进入 obs["policy"]["table_cam"]。
            isaac_env_cfg.scene.table_cam.width = self.cfg.init_params.table_cam.width  # 中文学习注释：覆盖桌面/第三人称相机输出宽度；OpenPI policy 后续会把它当 base 相机图像。

            # 中文学习注释：创建真实 Gymnasium/IsaacLab 环境；unwrapped 拿到底层 ManagerBasedRLEnv，方便 SubProcIsaacLabEnv 直接 reset/step/device。
            env = gym.make(
                self.isaaclab_env_id, cfg=isaac_env_cfg, render_mode="rgb_array"#如果这个环境需要 render，不要弹窗口给人看，而是把画面作为 RGB 图像数组返回
            ).unwrapped
            return env, sim_app  # 中文学习注释：同时返回 env 和 sim_app；关闭环境时需要一起释放 Isaac Sim 应用资源。

        return make_env_isaaclab  # 中文学习注释：返回函数本身，而不是返回 env；真正创建动作留给 SubProcIsaacLabEnv 子进程完成。

    def _wrap_obs(self, obs):
        # 中文学习注释：把 IsaacLab 的 obs["policy"] 改包成 RLinf/OpenPI 统一字段；reset、step、chunk_step 都会走这里。
        instruction = [self.task_description] * self.num_envs  # 中文学习注释：同一个 stack-cube 任务在每个并行 env 上使用同一句 prompt，后续会变成 OpenPI 的 prompt。
        wrist_image = obs["policy"]["wrist_cam"]  # 中文学习注释：腕部相机 RGB；字段由 IsaacLab stack visuomotor PolicyCfg 定义，OpenPI 输入侧会映射成 left_wrist_0_rgb。
        table_image = obs["policy"]["table_cam"]  # 中文学习注释：桌面/第三人称 RGB；OpenPI 输入侧会映射成 base_0_rgb，也就是主视角图像。
        quat = obs["policy"]["eef_quat"][  # 中文学习注释：IsaacLab 给的末端姿态四元数是 wxyz；quat2axisangle_torch 期望 xyzw，所以这里重排维度。
            :, [1, 2, 3, 0]
        ]  # In isaaclab, quat is wxyz not like libero
        states = torch.concatenate(  # 中文学习注释：拼出 OpenPI 使用的低维 robot state：末端位置 + 末端旋转 axis-angle + 夹爪位置。
            [
                obs["policy"]["eef_pos"],  # 中文学习注释：shape [num_envs, 3]，末端执行器在每个环境局部坐标系下的位置。
                quat2axisangle_torch(quat),  # 中文学习注释：shape [num_envs, 3]，把四元数转成旋转向量，和 OpenPI/其它机器人数据配置更容易对齐。
                obs["policy"]["gripper_pos"],  # 中文学习注释：夹爪观测；Franka 并指夹爪通常包含两个 finger joint 的状态。
            ],
            dim=1,
        )

        env_obs = {  # 中文学习注释：这是 RLinf embodied rollout 传给 OpenPI policy 的统一 observation，不再暴露 IsaacLab 原始 obs["policy"] 命名。
            "main_images": table_image,  # 中文学习注释：主视角图像 batch；后续进入 IsaacLabInputs 时对应 observation/image。
            "task_descriptions": instruction,  # 中文学习注释：语言指令 batch；后续作为 prompt/token 条件，让 VLM 知道“把方块叠起来”。
            "states": states,  # 中文学习注释：低维机器人状态 batch；后续对应 observation/state，和图像一起条件化 OpenPI 动作生成。
            "wrist_images": wrist_image,  # 中文学习注释：腕部图像 batch；后续进入 IsaacLabInputs 时对应 observation/wrist_image。
        }
        return env_obs  # 中文学习注释：返回包装后的观测；上层 EnvWorker/RolloutWorker 不需要知道 IsaacLab 内部 obs 结构。
