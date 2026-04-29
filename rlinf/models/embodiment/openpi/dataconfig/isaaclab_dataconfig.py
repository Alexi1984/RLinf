# Copyright 2026 The RLinf Authors.
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

"""中文学习注释：OpenPI 在 IsaacLab stack-cube 数据/观测上的 DataConfig。

它把 IsaacLab 的字段名、图像/状态格式、默认 prompt 和 OpenPI 模型预处理链路连接起来。
"""
import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import isaaclab_policy


@dataclasses.dataclass(frozen=True)
# 中文学习注释：DataConfigFactory 子类，负责生成 OpenPI 训练/推理所需的数据转换配置。
class LeRobotIsaacLabStackCubeDataConfig(DataConfigFactory):
    """OpenPI data config aligned with stack-cube fine-tuning recipe."""

    default_prompt: str | None = (  # 中文学习注释：如果样本没有 prompt，就使用 stack-cube 的默认语言指令。
        "Stack the red block on the blue block, then stack the green block on the red block"
    )

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = _transforms.Group(  # 中文学习注释：把 LeRobot/IsaacLab 数据字段重命名成 OpenPI 标准输入字段。
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.front",
                        "observation/wrist_image": "observation.images.wrist",
                        "observation/state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(  # 中文学习注释：把图像/state/action 进一步转成 OpenPI 模型可直接消费的结构。
            inputs=[isaaclab_policy.IsaacLabInputs(model_type=model_config.model_type)],
            outputs=[isaaclab_policy.IsaacLabOutputs()],
        )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(  # 中文学习注释：创建文本 prompt、token、action 归一化等模型级 transform。
            model_config
        )

        return dataclasses.replace(  # 中文学习注释：在基础 DataConfig 上替换 IsaacLab 专用 transform 和 action key。
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),  # 中文学习注释：声明哪些字段是动作序列，用于 OpenPI action horizon 处理。
        )
