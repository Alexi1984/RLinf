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

"""中文学习注释：RLinf 对 IsaacLab 环境的统一封装层。

这里把 IsaacLab/Gymnasium 环境包装成 RLinf EnvWorker 能消费的接口：reset、step、metrics、auto-reset、视频信息和 observation 字段格式。
"""

import copy
from typing import Optional

import gymnasium as gym
import torch
from omegaconf import open_dict

from rlinf.envs.isaaclab.venv import SubProcIsaacLabEnv


# 中文学习注释：所有 IsaacLab 任务的基类；具体任务只需实现如何创建底层 IsaacLab env 和如何包装 observation。
class IsaaclabBaseEnv(gym.Env):
    """
    Class for isaaclab in rlinf. Different from other lab enviromnent, the output of isaaclab is all tensor on
    cuda.
    """

    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
    ):
        self.cfg = cfg  # 中文学习注释：当前 train/eval 环境配置，来自 env/isaaclab_stack_cube.yaml 和顶层覆盖。
        self.isaaclab_env_id = self.cfg.init_params.id  # 中文学习注释：Gymnasium/IsaacLab 注册的任务 ID。
        self.num_envs = num_envs  # 中文学习注释：该 worker 内部实际承载的并行 IsaacLab 子环境数量。

        with open_dict(cfg):
            cfg.init_params.num_envs = num_envs  # 中文学习注释：运行时把并行数写回 IsaacLab 初始化参数。
        self.seed = self.cfg.seed + seed_offset  # 中文学习注释：不同 worker 用 seed_offset 错开随机种子。
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.video_cfg = cfg.video_cfg
        self._init_isaaclab_env()  # 中文学习注释：启动子进程中的 IsaacLab 仿真器，并完成首次 reset。
        self.device = self.env.device()  # 中文学习注释：IsaacLab observation/reward 通常直接在 CUDA device 上。

        self.task_description = cfg.init_params.task_description
        self._is_start = True  # if this is first time for simulator
        self.auto_reset = cfg.auto_reset
        self.prev_step_reward = torch.zeros(self.num_envs).to(self.device)  # 中文学习注释：记录上一步 reward，支持相对奖励/指标计算。
        self.use_rel_reward = cfg.use_rel_reward

        self._init_metrics()
        self._elapsed_steps = torch.zeros(self.num_envs, dtype=torch.int32).to(
            self.device
        )
        self.ignore_terminations = cfg.ignore_terminations

    def _make_env_function(self):
        # 中文学习注释：子类返回一个可被 cloudpickle 序列化的工厂函数，用于在子进程创建 IsaacLab env。
        raise NotImplementedError

    def _init_isaaclab_env(self):
        # 中文学习注释：把底层 IsaacLab 放到单独进程里，主进程通过 Pipe/Queue 发送 reset/step 命令。
        env_fn = self._make_env_function()
        self.env = SubProcIsaacLabEnv(env_fn)
        self.env.reset(seed=self.seed)

    def _init_metrics(self):
        # 中文学习注释：初始化 episode 级指标缓存，所有张量和环境并行数对齐。
        self.success_once = torch.zeros(self.num_envs, dtype=bool).to(self.device)
        self.fail_once = torch.zeros(self.num_envs, dtype=bool).to(self.device)
        self.returns = torch.zeros(self.num_envs).to(self.device)

    def _reset_metrics(self, env_idx=None):
        # 中文学习注释：重置全部或指定 env 的 return/success/长度统计。
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool).to(self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0
            self._elapsed_steps[env_idx] = 0
        else:
            self.prev_step_reward[:] = 0
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self._elapsed_steps[:] = 0

    def _record_metrics(self, step_reward, terminations, infos):
        # 中文学习注释：把当前 step reward 累加成 episode 统计，并写入 infos['episode'] 供 runner/logger 使用。
        episode_info = {}
        self.returns += step_reward
        self.success_once = self.success_once | (step_reward > 0)
        # batch level
        episode_info["success_once"] = self.success_once.clone()
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = self.elapsed_steps.clone()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        infos["episode"] = episode_info
        return infos

    def reset(
        self,
        seed: Optional[int] = None,
        env_ids: Optional[torch.Tensor] = None,
    ):
        # 中文学习注释：env_ids=None 表示重置全部并行环境；否则只重置结束的子环境。
        if env_ids is None:
            obs, _ = self.env.reset(seed=seed)
        else:
            obs, _ = self.env.reset(seed=seed, env_ids=env_ids)
        infos = {}
        obs = self._wrap_obs(obs)  # 中文学习注释：把 IsaacLab 原始 obs 转成 OpenPI/RLinf 统一字段。
        self._reset_metrics(env_ids)
        return obs, infos

    def step(self, actions=None, auto_reset=True):
        # 中文学习注释：执行一批并行动作，返回包装后的 obs、reward、done、infos。
        obs, step_reward, terminations, truncations, infos = self.env.step(actions)  # 中文学习注释：真正调用子进程 IsaacLab env.step。

        step_reward = step_reward.clone()
        terminations = terminations.clone()
        truncations = truncations.clone()

        obs = self._wrap_obs(obs)  # 中文学习注释：把 IsaacLab 原始 obs 转成 OpenPI/RLinf 统一字段。

        self._elapsed_steps += 1

        truncations = (self.elapsed_steps >= self.cfg.max_episode_steps) | truncations  # 中文学习注释：到达配置步数也算截断结束。

        dones = terminations | truncations  # 中文学习注释：RL 语义中的 done = 成功/失败终止 或 时间截断。

        infos = self._record_metrics(
            step_reward, terminations, {}
        )  # return infos is useless
        if self.ignore_terminations:  # 中文学习注释：评估时可忽略成功 termination，让视频/轨迹跑满固定长度。
            infos["episode"]["success_at_end"] = terminations
            terminations[:] = False

        _auto_reset = auto_reset and self.auto_reset  # always False
        if dones.any() and _auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)

        return (
            obs,
            step_reward,
            terminations,
            truncations,
            infos,
        )

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = chunk_actions.shape[1]
        obs_list = []
        infos_list = []

        chunk_rewards = []

        raw_chunk_terminations = []
        raw_chunk_truncations = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=False
            )
            obs_list.append(extracted_obs)
            infos_list.append(infos)

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_terminations = torch.stack(
            raw_chunk_terminations, dim=1
        )  # [num_envs, chunk_steps]
        raw_chunk_truncations = torch.stack(
            raw_chunk_truncations, dim=1
        )  # [num_envs, chunk_steps]

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones, obs_list[-1], infos_list[-1]
            )

        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_chunk_terminations).to(
                self.device
            )
            chunk_terminations[:, -1] = past_terminations

            chunk_truncations = torch.zeros_like(raw_chunk_truncations).to(self.device)
            chunk_truncations[:, -1] = past_truncations
        else:
            chunk_terminations = raw_chunk_terminations.clone()
            chunk_truncations = raw_chunk_truncations.clone()
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _handle_auto_reset(self, dones, _final_obs, infos):
        final_obs = copy.deepcopy(_final_obs)
        env_idx = torch.arange(0, self.num_envs).to(dones.device)
        env_idx = env_idx[dones]
        final_info = copy.deepcopy(infos)
        obs, infos = self.reset(
            env_ids=env_idx,
        )

        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    def _wrap_obs(self, obs):
        raise NotImplementedError

    def close(self):
        self.env.close()

    def update_reset_state_ids(self):
        """
        No muti task.
        """
        pass

    """
    Below codes are all copied from libero, thanks to the author of libero!
    """

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    @property
    def elapsed_steps(self):
        return self._elapsed_steps.to(self.device)

    def _calc_step_reward(self, terminations):
        reward = self.cfg.reward_coef * terminations
        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward

        if self.use_rel_reward:
            return reward_diff
        else:
            return reward
