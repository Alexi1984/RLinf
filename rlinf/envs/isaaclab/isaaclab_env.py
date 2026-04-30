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

"""中文学习注释：RLinf 的 IsaacLab 环境基类。

执行链路是：Hydra 合并 env/isaaclab_stack_cube.yaml 和顶层实验 YAML -> EnvWorker 按 env_type/id 选中
IsaaclabStackCubeEnv -> 这个基类启动 SubProcIsaacLabEnv 子进程 -> 子类把 IsaacLab 原始观测包装成
RLinf/OpenPI 统一字段。这个文件本身不定义具体任务几何，也不定义 OpenPI 图像预处理；具体任务在
tasks/stack_cube.py，OpenPI 输入字段消费在 policies/isaaclab_policy.py。
"""

import copy
from typing import Optional

import gymnasium as gym
import torch
from omegaconf import open_dict

from rlinf.envs.isaaclab.venv import SubProcIsaacLabEnv  # 子进程代理；真正的 Isaac Sim/Gym env 在另一个进程里创建。


# 中文学习注释：所有 RLinf IsaacLab 任务共享的 Gym 接口基类；StackCube 等子类只补“怎么创建底层环境”和“怎么整理 obs”。
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
        # cfg：EnvWorker 传进来的 env.train 或 env.eval 配置；来源是 env/isaaclab_stack_cube.yaml 再叠加顶层实验 YAML 覆盖。
        self.cfg = cfg
        # init_params.id：IsaacLab/Gymnasium 注册任务名；rlinf/envs/__init__.py 用它选中具体 wrapper，StackCube 子类再用它 gym.make。
        self.isaaclab_env_id = self.cfg.init_params.id
        # num_envs：当前这个 EnvWorker stage 内部实际承载的并行环境数；EnvWorker 会由 total_num_envs / worker数 / stage数 算出来。
        self.num_envs = num_envs

        with open_dict(cfg):
            # cfg 默认可能是只读 OmegaConf；这里临时打开写权限，把运行时算出的 num_envs 写回给 StackCube 的 IsaacLab cfg。
            cfg.init_params.num_envs = num_envs
        # seed_offset：同一个 Ray EnvWorker 组里不同 rank/stage 的偏移量，避免所有 IsaacLab 子环境 reset 到完全相同随机流。
        self.seed = self.cfg.seed + seed_offset
        # total_num_processes：EnvWorker 传入的全局环境进程/stage 数量；本基类保存它，方便子类/包装器需要全局进程信息时使用。当前环境组被拆成了多少个 EnvWorker/stage 实例；不是并行环境数量。
        self.total_num_processes = total_num_processes
        # worker_info：RLinf scheduler 里的 worker 元信息；这里保存但当前 IsaacLab 基类不直接读取。
        self.worker_info = worker_info
        # video_cfg：来自 env.*.video_cfg；RecordVideo wrapper 在 EnvWorker 里读取同一配置决定是否包视频记录器。
        self.video_cfg = cfg.video_cfg
        # 启动 IsaacLab 子进程并做一次 reset；这一步之后 self.env 是主进程代理，真实仿真器在 SubProcIsaacLabEnv 的子进程里。
        self._init_isaaclab_env()
        # device：从子进程里的 isaac_env.device 查询得到；IsaacLab 返回的 obs/reward/done 通常已经在 CUDA 上。
        self.device = self.env.device()

        # task_description：语言任务说明；StackCube._wrap_obs 会复制成每个并行 env 一条 prompt，后续 OpenPI policy 消费。
        self.task_description = cfg.init_params.task_description
        self._is_start = True  # _is_start：兼容其它 embodied env 的“第一次模拟器启动/首帧”状态标记。
        self.auto_reset = cfg.auto_reset  # auto_reset：done 后是否立即 reset；训练配置通常为 False，由 rollout 逻辑管理 episode 边界。
        # prev_step_reward：_calc_step_reward 的缓存；当前 IsaacLab stack-cube step 直接使用底层 env.step reward，未调用该 helper。
        self.prev_step_reward = torch.zeros(self.num_envs).to(self.device)
        self.use_rel_reward = cfg.use_rel_reward  # use_rel_reward：兼容字段；当前 IsaacLab stack-cube 路径不实际改写训练 reward。

        self._init_metrics()#为每个并行环境准备 return / success / fail 这些统计表。
        # _elapsed_steps：每个并行 env 当前 episode 已执行步数；step 里递增，并用于 max_episode_steps 截断。
        self._elapsed_steps = torch.zeros(self.num_envs, dtype=torch.int32).to(#给每个并行环境记录当前 episode 已经执行了多少步
            self.device
        )
        # ignore_terminations：评估可设 True，让成功 termination 不马上打断轨迹；真正成功信号会保存在 infos["episode"]。
        self.ignore_terminations = cfg.ignore_terminations

    def _make_env_function(self):
        # 中文学习注释：子类必须返回一个可被 CloudpickleWrapper 序列化的工厂函数；SubProcIsaacLabEnv 会在子进程执行它。
        raise NotImplementedError

    def _init_isaaclab_env(self):#真正把 IsaacLab 仿真环境启动起来，但不是直接在当前进程里启动，而是放到一个子进程里启动。
        # env_fn：具体任务子类提供的“创建 Isaac Sim app + gym env”的函数；StackCube 在 tasks/stack_cube.py 里实现。
        env_fn = self._make_env_function()#拿到 stack-cube 子类提供的环境创建函数
        # self.env：主进程侧代理；reset/step/device 会通过 Pipe/Queue 转发到子进程，避免主进程直接持有 Isaac Sim app。
        self.env = SubProcIsaacLabEnv(env_fn)#用 SubProcIsaacLabEnv 在子进程里启动 Isaac Sim / IsaacLab
        self.env.reset(seed=self.seed)  # 首次 reset 用于完成 IsaacLab 初始化并让后续 device/step 有有效环境状态。

    def _init_metrics(self):
        # success_once：每个并行 env 在当前 episode 内是否曾经拿到正 reward；logger 用它观察“是否成功过”。
        self.success_once = torch.zeros(self.num_envs, dtype=bool).to(self.device)
        # fail_once：预留的失败统计位；本文件目前没有更新它，但保持字段能和其它 embodied env 的 episode_info 对齐。
        self.fail_once = torch.zeros(self.num_envs, dtype=bool).to(self.device)
        # returns：当前 episode 内逐步累积的 reward；_record_metrics 每步更新，并放进 infos["episode"]["return"]。
        self.returns = torch.zeros(self.num_envs).to(self.device)

    def _reset_metrics(self, env_idx=None):
        # env_idx=None：全量 reset；env_idx 是 tensor 时只重置 done 的那几个并行 env，对应 _handle_auto_reset 的局部 reset。
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool).to(self.device)  # mask：把 env 索引转成布尔选择，便于批量清理指标张量。
            mask[env_idx] = True  # env_idx 来自 dones.nonzero 的同类索引，表示本次需要重新开始 episode 的子环境。
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0
            self._elapsed_steps[env_idx] = 0
        else:
            # 全部 reset 时，把所有并行 env 的 reward、成功标记和步数都归零，避免新 episode 继承旧统计。
            self.prev_step_reward[:] = 0
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self._elapsed_steps[:] = 0

    def _record_metrics(self, step_reward, terminations, infos):
        # step_reward：当前一步每个并行 env 的 reward；IsaacLab stack-cube 原始 step 已经返回它，本函数只做统计不重算。
        episode_info = {}  # episode_info：按 Gym/RLinf 约定塞进 infos["episode"]，供 EnvWorker/runner/logger 聚合。
        self.returns += step_reward  # returns：episode 累计回报，shape 与 num_envs 对齐。
        self.success_once = self.success_once | (step_reward > 0)  # 只要当前 episode 任意一步 reward>0，就把该 env 标为曾成功。
        # batch level：下面每个字段都是长度 num_envs 的 batch 统计，而不是单个环境标量。
        episode_info["success_once"] = self.success_once.clone()  # clone 防止后续原地更新污染已经交给 logger 的信息。
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = self.elapsed_steps.clone()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]  # 平均每步 reward，便于不同长度轨迹比较。
        infos["episode"] = episode_info  # EnvWorker 后续从 infos["episode"] 里取训练/评估指标。
        return infos

    def reset(
        self,
        seed: Optional[int] = None,
        env_ids: Optional[torch.Tensor] = None,
    ):
        # reset 是 RLinf EnvWorker 调用的标准接口；env_ids=None 表示全量 reset，否则只 reset 指定并行子环境。
        if env_ids is None:
            obs, _ = self.env.reset(seed=seed)  # 通过 SubProcIsaacLabEnv 请求子进程 reset 全部 IsaacLab env。
        else:
            obs, _ = self.env.reset(seed=seed, env_ids=env_ids)  # 局部 reset：只重置已经 done 的 env，保留其它 env 连续交互。
        infos = {}  # reset 当前不额外返回 IsaacLab info；后续 step 才写 episode 指标。
        obs = self._wrap_obs(obs)  # 这句是在把 IsaacLab 原始 observation 转成 RLinf / OpenPI 能识别的 observation 格式
        self._reset_metrics(env_ids)  # reset 之后同步清理对应 env 的 return、success 和 elapsed_steps。
        return obs, infos

    def step(self, actions=None, auto_reset=True):
        # actions：一批并行环境动作，通常 shape 为 [num_envs, action_dim]；来自 OpenPI 输出 transform 后的连续控制量。
        obs, step_reward, terminations, truncations, infos = self.env.step(actions)  # 子进程真正调用 isaac_env.step(actions)。

        step_reward = step_reward.clone()  # clone：后面可能记录/修改张量，避免直接操作子进程返回对象的共享引用。
        terminations = terminations.clone()  # terminations：任务自然终止，如成功/失败；和时间截断 truncations 分开。
        truncations = truncations.clone()  # truncations：环境自身报告的截断，下面还会叠加 RLinf 配置的 max_episode_steps。

        obs = self._wrap_obs(obs)  # 将 IsaacLab 原始 obs 包装成 rollout/policy 认识的统一 observation 字典。

        self._elapsed_steps += 1  # 每个并行 env 都推进了一步，所以 episode 步数整体 +1。

        # max_episode_steps 来自 env.* 配置；达到上限时即使任务未 termination，也强制标记为时间截断。
        truncations = (self.elapsed_steps >= self.cfg.max_episode_steps) | truncations

        dones = terminations | truncations  # dones：内部控制 auto-reset 的总结束信号；返回时仍拆成 termination/truncation 两个张量。

        infos = self._record_metrics(
            step_reward, terminations, {}
        )  # 当前没有使用 IsaacLab 原始 infos；这里重新构造 RLinf 需要的 episode 指标。
        if self.ignore_terminations:
            # ignore_terminations 常用于固定长度评估/视频：保留成功信号到 success_at_end，但对外把 termination 清零。
            infos["episode"]["success_at_end"] = terminations
            terminations[:] = False

        _auto_reset = auto_reset and self.auto_reset  # 两个开关同时为 True 才自动 reset；chunk_step 会显式传 auto_reset=False。
        if dones.any() and _auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)  # 只 reset 已 done 的 env，并把 final_observation 放入 infos。

        return (
            obs,  # 当前 step 之后的 observation；若触发 auto_reset，则 done env 的 obs 会变成 reset 后新初始 obs。
            step_reward,  # shape [num_envs]，当前一步 reward。
            terminations,  # shape [num_envs]，任务自然结束信号；ignore_terminations=True 时会被清零。
            truncations,  # shape [num_envs]，时间/步数截断信号。
            infos,  # 包含 infos["episode"] 指标，以及 auto-reset 时的 final_observation/final_info。
        )

    def chunk_step(self, chunk_actions):
        # chunk_actions：OpenPI 一次输出的动作块，shape [num_envs, chunk_steps, action_dim]；这里逐个时间步喂给 IsaacLab。
        chunk_size = chunk_actions.shape[1]  # chunk_size：动作块里包含多少个 step，不是并行环境数量。
        obs_list = []  # obs_list：保存动作块内每一个 step 后的 observation，供 rollout 按 chunk 组织轨迹。
        infos_list = []  # infos_list：保存动作块内每一步 episode 指标/auto-reset 信息。

        chunk_rewards = []  # chunk_rewards：长度会等于 chunk_steps；每个元素是一个 [num_envs] 的 step_reward。

        raw_chunk_terminations = []  # 长度会等于 chunk_steps；每个元素是一个 [num_envs] 的 termination 张量。
        raw_chunk_truncations = []  # 长度会等于 chunk_steps；每个元素是一个 [num_envs] 的 truncation 张量。
        for i in range(chunk_size):
            actions = chunk_actions[:, i]  # actions：取动作块第 i 个时间步，shape [num_envs, action_dim]。
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=False
            )  # chunk 内先不让单步 step 自动 reset，等整个 chunk 走完后统一处理 done。
            obs_list.append(extracted_obs)  # extracted_obs 是第 i 个低层 step 后的包装观测。
            infos_list.append(infos)  # infos 与该 step 对齐，包含 episode return/len/success 等。

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # list 长度是 chunk_steps；每项是 [num_envs]，堆叠后得到 [num_envs, chunk_steps]。
        raw_chunk_terminations = torch.stack(
            raw_chunk_terminations, dim=1
        )  # shape [num_envs, chunk_steps]，记录 chunk 内每个低层 step 是否自然结束。
        raw_chunk_truncations = torch.stack(
            raw_chunk_truncations, dim=1
        )  # shape [num_envs, chunk_steps]，记录 chunk 内每个低层 step 是否时间截断。

        past_terminations = raw_chunk_terminations.any(dim=1)  # 压缩 chunk 时间维：该 env 在本动作块内是否曾 termination。它把 [每个环境, chunk里的每一步是否结束]压缩成[每个环境在整个chunk里是否曾经结束]，以下类似
        past_truncations = raw_chunk_truncations.any(dim=1)  # 压缩 chunk 时间维：该 env 在本动作块内是否曾 truncation。
        past_dones = torch.logical_or(past_terminations, past_truncations)  # chunk 级 done，用于 chunk 末尾统一 reset。

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones, obs_list[-1], infos_list[-1]
            )  # 若配置允许 auto_reset，只在动作块最后一步把 done env reset，避免 chunk 中途改变时间对齐。

        if self.auto_reset or self.ignore_terminations:
            # 对 chunk 接口来说，如果自动 reset 或忽略 termination，就只在 chunk 最后一列暴露“本 chunk 曾结束过”。
            # 这样上层 rollout 看到的是 chunk 粒度结束，而不是低层 step 粒度每一列都结束。
            chunk_terminations = torch.zeros_like(raw_chunk_terminations).to(
                self.device
            )
            chunk_terminations[:, -1] = past_terminations

            chunk_truncations = torch.zeros_like(raw_chunk_truncations).to(self.device)
            chunk_truncations[:, -1] = past_truncations
        else:
            chunk_terminations = raw_chunk_terminations.clone()  # 不做压缩时，保留每个低层 step 的真实 termination。
            chunk_truncations = raw_chunk_truncations.clone()  # 不做压缩时，保留每个低层 step 的真实 truncation。
        return (
            obs_list,  # 长度 chunk_steps；每项是一个包装后的 observation batch。
            chunk_rewards,  # [num_envs, chunk_steps]。
            chunk_terminations,  # [num_envs, chunk_steps]，可能被压缩到最后一列。
            chunk_truncations,  # [num_envs, chunk_steps]，可能被压缩到最后一列。
            infos_list,  # 长度 chunk_steps；每项对应一个 step 的 infos。
        )

    def _handle_auto_reset(self, dones, _final_obs, infos):
        final_obs = copy.deepcopy(_final_obs)  # final_obs：reset 前真正的末端观测；Gymnasium 约定要放到 infos 里防止丢失。
        env_idx = torch.arange(0, self.num_envs).to(dones.device)  # 生成所有并行 env 的索引。
        env_idx = env_idx[dones]  # 只挑出本次 done=True 的 env，局部 reset 不影响其它还在跑的 env。
        final_info = copy.deepcopy(infos)  # final_info：reset 前的 episode 指标快照。
        obs, infos = self.reset(
            env_ids=env_idx,
        )  # 局部 reset 后返回新的初始 obs 和清空后的 infos。

        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs  # reset 前的最后观测，供上层需要 terminal obs 时读取。
        infos["final_info"] = final_info  # reset 前的指标信息，避免 reset 清空后丢失 episode 结果。
        infos["_final_info"] = dones  # Gymnasium vector env 风格 mask：哪些 env 的 final_info 有效。
        infos["_final_observation"] = dones  # mask：哪些 env 的 final_observation 有效。
        infos["_elapsed_steps"] = dones  # mask：哪些 env 的 elapsed_steps 刚被 reset。
        return obs, infos

    def _wrap_obs(self, obs):
        # 中文学习注释：抽象观测包装点；StackCube 子类把 IsaacLab obs["policy"] 转成 main_images/states/wrist_images/task_descriptions。
        raise NotImplementedError

    def close(self):
        self.env.close()  # 通知 SubProcIsaacLabEnv 子进程关闭 isaac_env 和 sim_app，释放 Isaac Sim 资源。

    def update_reset_state_ids(self):
        """
        No muti task.
        """
        # 中文学习注释：多任务/固定 reset 状态环境可能会动态更新 reset ids；IsaacLab stack-cube 当前没有多任务切换，所以留空。
        pass

    """
    Below codes are all copied from libero, thanks to the author of libero!
    """

    @property
    def is_start(self):
        return self._is_start  # 兼容 RLinf 其它环境的启动状态属性；当前 IsaacLab 代码没有额外逻辑使用它。

    @is_start.setter
    def is_start(self, value):
        self._is_start = value  # 允许外部 wrapper/runner 修改启动状态标记。

    @property
    def elapsed_steps(self):
        return self._elapsed_steps.to(self.device)  # 返回与 IsaacLab reward/done 同 device 的步数张量，便于直接张量运算。

    def _calc_step_reward(self, terminations):
        # terminations：通常是 bool 张量；这是兼容其它环境的 helper，当前 IsaacLab step() 没有调用它。
        reward = self.cfg.reward_coef * terminations
        reward_diff = reward - self.prev_step_reward  # reward_diff：本步相对上一步新增的成功奖励，避免成功后每步重复给满分。
        self.prev_step_reward = reward  # 更新缓存，下一步计算相对奖励时使用。

        if self.use_rel_reward:
            return reward_diff  # 若未来改为调用该 helper，True 会把成功信号 0,0,1,1 变成 reward 0,0,1,0。
        else:
            return reward  # 若未来改为调用该 helper，False 会直接返回缩放后的成功信号，例如 0,0,1,1。
