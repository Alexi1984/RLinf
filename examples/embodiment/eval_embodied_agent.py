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

"""中文学习注释：具身评估入口；负责读取 Hydra 配置、创建 rollout/env worker，并运行只评估流程。

和训练入口相比，这里不会创建 actor 更新 worker，只用 checkpoint/模型策略与环境交互。
"""

import json

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

mp.set_start_method("spawn", force=True)  # 中文学习注释：IsaacLab/Ray/torch 多进程更适合 spawn，避免 fork 复制 CUDA/仿真状态导致崩溃。


@hydra.main(
    version_base="1.1", config_path="config", config_name="maniskill_ppo_openvlaoft"
)
def main(cfg) -> None:
    # 中文学习注释：Hydra 会把 YAML 合并后的配置对象传进来；后续所有 worker 都共享这份 cfg。
    cfg.runner.only_eval = True  # 中文学习注释：强制评估模式，避免误触发训练更新。
    cfg = validate_cfg(cfg)  # 中文学习注释：补齐/校验 RLinf 运行所需字段，尽早发现配置错误。
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))  # 中文学习注释：打印 resolve 后的完整配置，排查 Hydra 覆盖是否生效。

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)  # 中文学习注释：根据 cluster.component_placement 决定 actor/env/rollout 放在哪些节点/GPU。

    # Create rollout worker group
    rollout_placement = component_placement.get_strategy("rollout")  # 中文学习注释：rollout 负责采样动作和 logprob，通常和 actor 同机但职责不同。
    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )
    # Create env worker group
    env_placement = component_placement.get_strategy("env")  # 中文学习注释：env worker 负责并行仿真环境。
    env_group = EnvWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )

    runner = EmbodiedEvalRunner(
        cfg=cfg,
        rollout=rollout_group,
        env=env_group,
    )

    runner.init_workers()  # 中文学习注释：真正让各 worker 加载模型、创建环境、初始化同步状态。
    runner.run()  # 中文学习注释：进入训练/评估主循环；训练时会反复 rollout -> PPO update -> sync。


if __name__ == "__main__":
    main()
