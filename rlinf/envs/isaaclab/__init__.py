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

"""中文学习注释：IsaacLab 环境注册表。

EnvWorker 会根据配置中的 init_params.id 找到这里注册的环境类，从而把字符串任务 ID 映射到具体 Python wrapper。
"""

from .tasks.stack_cube import IsaaclabStackCubeEnv

REGISTER_ISAACLAB_ENVS = {  # 中文学习注释：任务 ID -> RLinf 环境 wrapper 类 的映射表。
    "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0": IsaaclabStackCubeEnv,
}

__all__ = [list(REGISTER_ISAACLAB_ENVS.keys())]  # 中文学习注释：导出已注册任务 ID，便于外部发现支持哪些 IsaacLab 环境。
