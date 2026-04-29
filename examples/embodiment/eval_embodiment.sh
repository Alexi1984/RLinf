#! /bin/bash

# 中文学习注释：评估启动脚本，默认进入 eval_embodied_agent.py。
# 这个脚本主要负责设置路径/渲染后端/机器人平台，然后把控制权交给对应的 Python 入口。

# 中文学习注释：EMBODIED_PATH 指向 examples/embodiment，Hydra 配置目录也从这里找。
export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
# 中文学习注释：REPO_PATH 指向 RLinf 仓库根目录，用于拼 PYTHONPATH 和日志路径。
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
# 中文学习注释：SRC_FILE 是真正执行的 Python 入口文件。
export SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"

# 中文学习注释：设置离屏渲染后端，避免远程服务器没有显示器时 OpenGL 报错。
export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

# Base path to the BEHAVIOR dataset, which is the BEHAVIOR-1k repo's dataset folder
# Only required when running the behavior experiment.
export OMNIGIBSON_DATA_PATH=$OMNIGIBSON_DATA_PATH
export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATASET_PATH:-$OMNIGIBSON_DATA_PATH/behavior-1k-assets/}
export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_KEY_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson.key}
export OMNIGIBSON_ASSET_PATH=${OMNIGIBSON_ASSET_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson-robot-assets/}
export OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS:-1}
# Base path to Isaac Sim, only required when running the behavior experiment.
export ISAAC_PATH=${ISAAC_PATH:-/path/to/isaac-sim}
export EXP_PATH=${EXP_PATH:-$ISAAC_PATH/apps}
export CARB_APP_PATH=${CARB_APP_PATH:-$ISAAC_PATH/kit}

export ROBOTWIN_PATH=${ROBOTWIN_PATH:-"/path/to/RoboTwin"}
export PYTHONPATH=${REPO_PATH}:${ROBOTWIN_PATH}:$PYTHONPATH

export DREAMZERO_PATH=${DREAMZERO_PATH:-"/path/to/DreamZero"}
export PYTHONPATH=${DREAMZERO_PATH}:$PYTHONPATH

export HYDRA_FULL_ERROR=1

# 中文学习注释：第一个命令行参数是 Hydra config 名；不传时使用默认示例配置。
if [ -z "$1" ]; then
    CONFIG_NAME="maniskill_ppo_openvlaoft"
else
    CONFIG_NAME=$1
fi

# NOTE: Set the active robot platform (required for correct action dimension and normalization), supported platforms are LIBERO, ALOHA, BRIDGE, default is LIBERO
# 中文学习注释：第二个参数或环境变量指定机器人平台，影响动作维度和归一化约定。
ROBOT_PLATFORM=${2:-${ROBOT_PLATFORM:-"LIBERO"}}

export ROBOT_PLATFORM

# Libero variant: standard, pro, plus
export LIBERO_TYPE=${LIBERO_TYPE:-"standard"}
if [ "$LIBERO_TYPE" == "pro" ]; then
    export LIBERO_PERTURBATION="all"  # all,swap,object,lan
    echo "Evaluation Mode: LIBERO-PRO | Perturbation: $LIBERO_PERTURBATION"
elif [ "$LIBERO_TYPE" == "plus" ]; then
    export LIBERO_SUFFIX="all"
    echo "Evaluation Mode: LIBERO-PLUS | Suffix: $LIBERO_SUFFIX"
else
    echo "Evaluation Mode: Standard LIBERO"
fi

echo "Using ROBOT_PLATFORM=$ROBOT_PLATFORM"

# 中文学习注释：每次运行创建独立日志目录，避免覆盖上一次实验输出。
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}" #/$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/eval_embodiment.log"
mkdir -p "${LOG_DIR}"
# 中文学习注释：最终执行的训练/评估命令；run_embodiment.sh 不透传第三个及之后的 Hydra 覆盖参数。
CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR}"
echo ${CMD}
# 中文学习注释：tee 同时把输出打印到终端并写入日志文件，方便断线后排查。
${CMD} 2>&1 | tee ${MEGA_LOG_FILE}
