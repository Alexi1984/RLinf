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

"""中文学习注释：IsaacLab observation/action 与 OpenPI policy 接口之间的转换。

IsaacLab 给的是 table_cam、wrist_cam、state 等字段；OpenPI 需要 base_0_rgb、left_wrist_0_rgb、image_mask、state 和 prompt。
"""
import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model


def make_isaaclab_example() -> dict:
    # 中文学习注释：构造一个假样本，主要用于调试 transform/schema，而不是训练数据。
    """Creates a random input example for the IsaacLab policy."""
    return {
        "observation/state": np.random.rand(7),
        "observation/image": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(
            256, size=(256, 256, 3), dtype=np.uint8
        ),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    # 中文学习注释：统一图像 dtype 和布局，保证 OpenPI 收到 uint8 HWC 图像。
    image = np.asarray(image)  # 中文学习注释：把 torch/JAX/list 等输入转成 numpy array。
    if np.issubdtype(image.dtype, np.floating):  # 中文学习注释：浮点图像通常是 0-1，转成 OpenPI 常用的 0-255 uint8。
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:  # 中文学习注释：如果是 CHW，则转成 HWC。
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
# 中文学习注释：输入 transform：IsaacLab/RLinf obs -> OpenPI model input。
class IsaacLabInputs(transforms.DataTransformFn):
    """Convert IsaacLab observations into OpenPI model inputs."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])  # 中文学习注释：table/base camera 图像。
        wrist_image = _parse_image(data["observation/wrist_image"])  # 中文学习注释：wrist camera 图像。

        inputs = {  # 中文学习注释：OpenPI 统一输入字典，后面会进入 tokenizer/model transform。
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),  # 中文学习注释：IsaacLab 只有一路 wrist，这里补零占位以兼容 OpenPI 多相机 schema。
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_
                if self.model_type == _model.ModelType.PI0_FAST
                else np.False_,
            },
        }

        if "actions" in data:  # 中文学习注释：训练/SFT 数据有专家动作；在线 rollout 观测可能没有。
            inputs["actions"] = data["actions"]
        if "prompt" in data:  # 中文学习注释：有 prompt 时传给 OpenPI；否则 DataConfig 会使用 default_prompt。
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
# 中文学习注释：输出 transform：OpenPI action sequence -> IsaacLab 可执行 action。
class IsaacLabOutputs(transforms.DataTransformFn):
    """Convert OpenPI outputs to IsaacLab action format."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :7])  # 中文学习注释：只取 IsaacLab Franka 环境需要的前 7 维动作。
        # IsaacLab stack-cube expects binary gripper command in {-1, +1}.
        actions[..., -1] = np.sign(actions[..., -1])  # 中文学习注释：夹爪动作离散成 -1/+1，匹配环境控制约定。
        return {"actions": actions}
