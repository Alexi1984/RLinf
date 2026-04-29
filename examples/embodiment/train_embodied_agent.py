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

"""中文学习注释：具身训练入口；负责读取 Hydra 配置、创建 Ray/RLinf worker 组，并启动 embodied runner。

这个文件不定义算法细节，而是把 actor、rollout、env、reward 等组件按配置组装起来。
"""

import json

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

mp.set_start_method("spawn", force=True)  # 中文学习注释：IsaacLab/Ray/torch 多进程更适合 spawn，避免 fork 复制 CUDA/仿真状态导致崩溃。


@hydra.main(
    version_base="1.1", config_path="config", config_name="maniskill_ppo_openvlaoft"
)
def main(cfg) -> None:
    # 中文学习注释：Hydra 会把 YAML 合并后的配置对象传进来；后续所有 worker 都共享这份 cfg。
    cfg = validate_cfg(cfg)  # 中文学习注释：补齐/校验 RLinf 运行所需字段，尽早发现配置错误。
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))  # 中文学习注释：打印 resolve 后的完整配置，排查 Hydra 覆盖是否生效。

    # 中文学习注释：Cluster 抽象 Ray/分布式资源，后续 worker group 都会注册到这里。
    cluster = Cluster(
        cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path
    )
    component_placement = HybridComponentPlacement(cfg, cluster)  # 中文学习注释：根据 cluster.component_placement 决定 actor/env/rollout 放在哪些节点/GPU。

    # Create actor worker group
    actor_placement = component_placement.get_strategy("actor")  # 中文学习注释：actor 负责训练更新，因此单独取放置策略。

    if cfg.algorithm.loss_type == "embodied_sac":
        from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy

        actor_worker_cls = EmbodiedSACFSDPPolicy
    elif cfg.algorithm.loss_type == "embodied_dagger":
        from rlinf.workers.actor.fsdp_dagger_policy_worker import (
            EmbodiedDAGGERFSDPPolicy,
        )

        actor_worker_cls = EmbodiedDAGGERFSDPPolicy
    elif cfg.algorithm.loss_type == "embodied_nft":
        from rlinf.workers.actor.fsdp_nft_policy_worker import EmbodiedNFTFSDPPolicy

        actor_worker_cls = EmbodiedNFTFSDPPolicy
    else:
        from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

        actor_worker_cls = EmbodiedFSDPActor
    actor_group = actor_worker_cls.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )

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

    reward_group = None
    if cfg.get("reward", {}).get("use_reward_model", False) and not cfg.get(
        "reward", {}
    ).get("standalone_realworld", False):
        # Create reward worker group
        reward_placement = component_placement.get_strategy("reward")
        reward_group = EmbodiedRewardWorker.create_group(cfg).launch(
            cluster, name=cfg.reward.group_name, placement_strategy=reward_placement
        )

    runner = EmbodiedRunner(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
        reward=reward_group,
    )

    runner.init_workers()  # 中文学习注释：真正让各 worker 加载模型、创建环境、初始化同步状态。
    runner.run()  # 中文学习注释：进入训练/评估主循环；训练时会反复 rollout -> PPO update -> sync。


if __name__ == "__main__":
    main()
