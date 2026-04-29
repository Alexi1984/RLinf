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

"""中文学习注释：IsaacLab 子进程环境封装。

Isaac Sim/IsaacLab 对进程和 CUDA 状态比较敏感，所以 RLinf 把仿真器放进 spawn 出来的子进程，主进程用 Pipe/Queue 发送 reset/step。
"""

from multiprocessing.connection import Connection

import torch
import torch.multiprocessing as mp

from .utils import CloudpickleWrapper


def _torch_worker(
    child_remote: Connection,
    parent_remote: Connection,
    env_fn_wrapper: CloudpickleWrapper,
    action_queue: mp.Queue,
    obs_queue: mp.Queue,
    reset_idx_queue: mp.Queue,
):
    parent_remote.close()  # 中文学习注释：子进程只保留 child_remote，关闭父端避免管道引用混乱。
    env_fn = env_fn_wrapper.x  # 中文学习注释：取出可序列化包装里的 IsaacLab env 工厂函数。
    isaac_env, sim_app = env_fn()  # 中文学习注释：在子进程内真正启动 Isaac Sim app 和 Gym env。
    device = isaac_env.device
    try:
        while True:
            try:
                cmd = child_remote.recv()  # 中文学习注释：等待主进程发送 reset/step/close/device 命令。
            except EOFError:
                child_remote.close()
                break
            if cmd == "reset":  # 中文学习注释：重置全部或部分并行子环境。
                reset_index, reset_seed = reset_idx_queue.get()
                if reset_index is None:
                    reset_result = isaac_env.reset(seed=reset_seed)
                else:
                    reset_result = isaac_env.reset(
                        seed=reset_seed, env_ids=reset_index.to(device)
                    )
                obs_queue.put(reset_result)
            elif cmd == "step":  # 中文学习注释：从 action_queue 取动作并推进仿真一步。
                input_action = action_queue.get()
                step_result = isaac_env.step(input_action)
                obs_queue.put(step_result)
            elif cmd == "close":  # 中文学习注释：关闭 IsaacLab env 和 Isaac Sim app。
                isaac_env.close()
                child_remote.close()
                sim_app.close()
                break
            elif cmd == "device":
                child_remote.send(isaac_env.device)
            else:
                child_remote.close()
                raise NotImplementedError
    except KeyboardInterrupt:
        child_remote.close()
    finally:
        try:
            isaac_env.close()
        except Exception as e:
            print(f"IsaacLab Env Closed with error: {e}")


# 中文学习注释：主进程侧的代理对象，向子进程发送命令并取回结果。
class SubProcIsaacLabEnv:
    def __init__(self, env_fn):
        mp.set_start_method("spawn", force=True)  # 中文学习注释：使用 spawn 避免 fork 复制 CUDA/IsaacSim 状态。
        ctx = mp.get_context("spawn")
        self.parent_remote, self.child_remote = ctx.Pipe(duplex=True)  # 中文学习注释：Pipe 用于发送轻量命令和 device 查询。
        self.action_queue = ctx.Queue()  # 中文学习注释：动作张量通过 Queue 传给子进程。
        self.obs_queue = ctx.Queue()  # 中文学习注释：reset/step 的 observation/reward 结果从这里传回。
        self.reset_idx = ctx.Queue()
        args = (
            self.child_remote,
            self.parent_remote,
            CloudpickleWrapper(env_fn),
            self.action_queue,
            self.obs_queue,
            self.reset_idx,
        )
        self.isaac_lab_process = ctx.Process(
            target=_torch_worker, args=args, daemon=True
        )
        self.isaac_lab_process.start()  # 中文学习注释：启动独立仿真子进程。
        self.child_remote.close()

    def reset(self, seed=None, env_ids=None):
        # 中文学习注释：主进程请求子进程 reset，并等待 obs_queue 返回结果。
        self.parent_remote.send("reset")
        self.reset_idx.put((env_ids, seed))
        obs, info = self.obs_queue.get()
        return obs, info

    def step(self, action: torch.Tensor):
        # 中文学习注释：主进程把 batch action 发送给子进程执行。
        """
        action : (bs, action_dim)
        """
        self.parent_remote.send("step")
        self.action_queue.put(action)
        env_step_result = self.obs_queue.get()
        return env_step_result

    def close(self):
        # 中文学习注释：通知子进程关闭仿真器并回收进程。
        self.parent_remote.send("close")
        self.isaac_lab_process.join()
        self.isaac_lab_process.terminate()

    def device(self):
        # 中文学习注释：查询底层 IsaacLab env 所在 device。
        self.parent_remote.send("device")
        return self.parent_remote.recv()
