# GS-SLAM 使用说明

本项目在线接收 ROS 2 双目图像、彩色图像和里程计，前端将同步后的
RGB-D 帧落盘，后端持续构建 3D Gaussian 地图，并在退出时保存 PLY。

## 1. 输入话题

默认订阅以下话题：

| 话题 | 消息类型 | 用途 |
|---|---|---|
| `/camera/infra1/image_raw` | `sensor_msgs/msg/Image` | 左目灰度图 |
| `/camera/infra2/image_raw` | `sensor_msgs/msg/Image` | 右目灰度图 |
| `/camera/color/image_raw` | `sensor_msgs/msg/Image` | 彩色图 |
| `/odometry` | `nav_msgs/msg/Odometry` | 相机载体位姿 |

话题名称、相机内外参、同步容差和采样频率位于
[`src/gs_slam/config/stereo_camera.yaml`](src/gs_slam/config/stereo_camera.yaml)。
启动建图前，先确认上游节点正在发布：

```bash
source /opt/ros/humble/setup.bash

ros2 topic list
ros2 topic hz /camera/infra1/image_raw
ros2 topic hz /camera/infra2/image_raw
ros2 topic hz /camera/color/image_raw
ros2 topic hz /odometry
```

默认要求左右目时间差不超过 5 ms，四路消息的近似同步误差不超过
20 ms。前端以 1 Hz 接收并保存同步帧。

## 2. 首次安装

### 2.1 构建 ROS 前端

```bash
cd /home/ubuntu/GS-SLAM
source /opt/ros/humble/setup.bash

colcon build --packages-select gs_slam --symlink-install
source /home/ubuntu/GS-SLAM/install/setup.bash
```

### 2.2 安装 Gaussian 后端

后端必须运行在已经安装 Torch、CUDA rasterizer、`plyfile` 等依赖的
Gaussian Conda 环境中：

```bash
/home/ubuntu/miniconda3/envs/3dgs/bin/python -m pip install \
  --no-deps -e /home/ubuntu/GS-SLAM/src/gs_slam_backend
```

可用以下命令检查后端入口：

```bash
/home/ubuntu/miniconda3/envs/3dgs/bin/python \
  -m gs_slam_backend.runner --help
```

### 2.3 可选：安装 Sparse Gaussian Adam

普通 Adam 由 PyTorch 自带，不需要额外安装。使用 `sparse_adam` 时，需要把
普通 `diff_gaussian_rasterization` 替换为官方 `3dgs_accel` 分支；该分支同时
提供加速光栅器和 CUDA `SparseGaussianAdam`：

```bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate 3dgs

cd /home/ubuntu/gaussian-splatting/submodules/diff-gaussian-rasterization
git fetch origin
git checkout 3dgs_accel
git submodule update --init --recursive

CUDA_HOME="$CONDA_PREFIX" \
PATH="$CONDA_PREFIX/bin:$PATH" \
TORCH_CUDA_ARCH_LIST=8.9 \
python -m pip install --no-deps --no-build-isolation --force-reinstall .
```

这里的 `8.9` 对应本机 RTX 4060 Laptop；换用其他 GPU 时应填写相应计算能力，
并确保编译用的 CUDA 与 PyTorch CUDA 版本一致。当前环境为 PyTorch CUDA 11.8，
因此必须使用 Conda 环境中的 CUDA 11.8，不能误用系统 CUDA 12.8。

安装后验证：

```bash
python -c "from diff_gaussian_rasterization import SparseGaussianAdam; print('Sparse Adam ready')"
```

然后在后端配置的 `optimization` 中启用：

```json
"optimizer_type": "sparse_adam"
```

切回普通 Adam 只需改为 `"adam"`，不需要重新安装或修改模型代码。优化器在
后端启动时创建，修改配置后需要重启进程。

## 3. 三种运行模式

| 模式 | Launch | 用途 |
|---|---|---|
| 只采集 | `frame_archiver.launch.py` | 订阅实时话题并保存 RGB-D 数据 |
| 数据回放 | `replay_mapping.launch.py` | 对已有会话执行一次完整离线建图 |
| 在线建图 | `online_mapping.launch.py` | 一边接收话题，一边更新 Gaussian 地图 |

### 3.1 在线建图：推荐用法

每次采集建议使用新的 `session` 目录，避免新旧 manifest 混合：

```bash
cd /home/ubuntu/GS-SLAM
source /opt/ros/humble/setup.bash
source /home/ubuntu/GS-SLAM/install/setup.bash

ros2 launch gs_slam online_mapping.launch.py \
  session:=/home/ubuntu/GS-SLAM/data/session_01 \
  frontend_config:=/home/ubuntu/GS-SLAM/src/gs_slam/config/stereo_camera.yaml \
  backend_config:=/home/ubuntu/GS-SLAM/src/gs_slam/config/online_backend.json \
  preview:=true \
  preview_depth_min:=0.2 \
  preview_depth_max:=5.0
```

`preview:=true` 会以当前处理帧的固定相机视角并排显示彩色图、米制深度和
Gaussian 渲染。预览复用优化阶段已有的渲染结果，不会额外执行 rasterization；
窗口事件和图像合成在独立线程中运行，并且只保留最新待显示帧，避免建图变慢时
预览事件阻塞或旧帧积压。
按 `q` 或 `Esc` 可以结束在线处理。没有图形桌面时保持默认的 `false`。

该 launch 同时启动：

1. `frame_archiver`：同步话题、计算深度并将 RGB-D 帧原子落盘；
2. `gs_slam_backend.runner live`：读取新 manifest 并在线更新 Gaussian 地图。

如果 Python 环境或源码路径不同，可以显式指定：

```bash
ros2 launch gs_slam online_mapping.launch.py \
  session:=/path/to/session \
  frontend_config:=/path/to/stereo_camera.yaml \
  backend_config:=/path/to/online_backend.json \
  backend_python:=/path/to/3dgs/bin/python \
  backend_module_path:=/path/to/gs_slam_backend
```

## 4. 查看运行状态

确认前端节点存在：

```bash
ros2 node list
```

正常情况下应看到：

```text
/frame_archiver
```

查看后端实时状态：

```bash
watch -n 1 cat \
  /home/ubuntu/GS-SLAM/data/session_01/online_output/status.json
```

重点字段包括：

| 字段 | 含义 |
|---|---|
| `last_processed` | 已处理的最新帧编号 |
| `gaussians` | 当前 Gaussian 数量 |
| `keyframes` | 当前关键帧数量 |
| `added` / `pruned` | 本帧新增/删除数量 |
| `pending_prune` | 已满足在线剪枝条件、正在等待批量压缩的数量 |
| `coverage` | 当前优化帧的渲染覆盖率 |
| `mapping_ms` | 当前帧映射耗时 |
| `phase` | 当前阶段，例如 `final_refinement` 或 `complete` |

终端逐帧输出只保留帧号、关键帧标记、队列长度和 `timing_ms` 阶段耗时；完整的
Gaussian 数量、损失、覆盖率和累计统计仍写入 `status.json`。阶段耗时包括加载、
关键帧判断、Gaussian 增长、窗口选择、优化、深度一致性、剪枝、预览和 checkpoint。

在线剪枝采用延迟批量压缩：候选点达到 `pruning.prune_batch_min_points` 时统一删除；
少量候选最多等待 `pruning.prune_batch_max_keyframes` 个关键帧，程序结束前会强制清空。

会话目录结构如下：

```text
session_01/
├── images/          # 彩色 PNG
├── depth/           # uint16 逆深度 PNG
├── manifests/       # 后端消费的逐帧 JSON
├── sparse/0/        # COLMAP 兼容相机和位姿元数据
└── online_output/
    ├── status.json
    └── point_cloud_*.ply
```

## 5. 正确停止与保存

在运行 launch 的终端按 `Ctrl+C`。后端随后执行：

1. 配置的全帧最终 refinement；
2. 配置的最终清理；
3. 原子保存 `point_cloud_*.ply`；
4. 将 `status.json` 的 `phase` 更新为 `complete`。

当前默认配置会执行 2000 次最终 refinement，大地图可能需要数分钟。
看到 `phase: complete` 或终端明确退出前，不要连续发送 `Ctrl+C` 或强制杀死
进程，否则本次 PLY 可能不会生成。

最终结果位于：

```text
/home/ubuntu/GS-SLAM/data/session_01/online_output/point_cloud_*.ply
```

## 6. 修改配置

### 6.1 前端配置

编辑 [`stereo_camera.yaml`](src/gs_slam/config/stereo_camera.yaml) 修改：

- ROS 话题名称；
- 相机内参、双目基线和相机外参；
- `processing.rate_hz` 采样频率；
- 时间同步容差；
- 双目匹配和深度置信度参数。

启动时传入源码 YAML 路径，修改配置后无需重新构建。

### 6.2 后端配置

所有 Gaussian 算法参数集中在
[`src/gs_slam/config/online_backend.json`](src/gs_slam/config/online_backend.json)：

- 初始化密度和体素尺寸；
- 关键帧选择；
- RGB/深度损失与学习率；
- Adam/Sparse Adam 优化器选择；
- SH 阶数；
- 稠密化；
- 去虚影、透明度、陈旧点和尺度裁剪；
- opacity reset；
- 最终 refinement 和清理。

功能开关与对应参数放在同一配置块中：

```json
"ghost_pruning": {
  "enabled": false,
  "parameters": {
    "newborn_grace_keyframes": 4,
    "ghost_inconsistency_limit": 4
  }
}
```

CLI 不会覆盖这些算法参数，运行时以传入的 JSON 为准。当前配置以出图完整度
为优先，关闭了主要裁剪和最终清理，可能生成数百 MB 的 PLY。

在 `optimization` 中选择优化器：

```json
"optimizer_type": "adam"
```

可选值为 `adam` 和 `sparse_adam`。后者要求当前 Python 环境安装的
`diff_gaussian_rasterization` 提供 `SparseGaussianAdam`；不支持时后端会在
启动阶段明确报错。

## 7. 单独运行前端

只采集 RGB-D 数据，不启动 Gaussian 后端：

```bash
source /opt/ros/humble/setup.bash
source /home/ubuntu/GS-SLAM/install/setup.bash

ros2 launch gs_slam frame_archiver.launch.py \
  frontend_config:=/home/ubuntu/GS-SLAM/src/gs_slam/config/stereo_camera.yaml
```

此模式的输出目录由 `stereo_camera.yaml` 中的 `output.directory` 决定。

## 8. 回放已有数据

推荐通过 launch 对已采集会话重新建图：

```bash
source /opt/ros/humble/setup.bash
source /home/ubuntu/GS-SLAM/install/setup.bash

ros2 launch gs_slam replay_mapping.launch.py \
  source:=/home/ubuntu/GS-SLAM/data/session_01 \
  output:=/home/ubuntu/GS-SLAM/data/session_01/replay_output \
  backend_config:=/home/ubuntu/GS-SLAM/src/gs_slam/config/online_backend.json
```

`replay_mapping.launch.py` 只启动 Gaussian 后端，不订阅 ROS 话题。处理完已有
帧、最终 refinement 和 PLY 保存后会自动退出。

对应的底层命令是：

```bash
/home/ubuntu/miniconda3/envs/3dgs/bin/python \
  -m gs_slam_backend.runner replay \
  --source /home/ubuntu/GS-SLAM/data/session_01 \
  --config /home/ubuntu/GS-SLAM/src/gs_slam/config/online_backend.json
```

也可以直接使用底层命令指定输出目录：

```bash
/home/ubuntu/miniconda3/envs/3dgs/bin/python \
  -m gs_slam_backend.runner replay \
  --source /home/ubuntu/GS-SLAM/data/session_01 \
  --output /home/ubuntu/GS-SLAM/data/session_01/replay_output \
  --config /home/ubuntu/GS-SLAM/src/gs_slam/config/online_backend.json
```

限制回放帧数可用于快速实验：

```bash
/home/ubuntu/miniconda3/envs/3dgs/bin/python \
  -m gs_slam_backend.runner replay \
  --source /home/ubuntu/GS-SLAM/data/session_01 \
  --max-frames 50 \
  --config /home/ubuntu/GS-SLAM/src/gs_slam/config/online_backend.json
```

## 9. 从检查点恢复

```bash
/home/ubuntu/miniconda3/envs/3dgs/bin/python \
  -m gs_slam_backend.runner resume \
  --session /home/ubuntu/GS-SLAM/data/session_01 \
  --checkpoint /path/to/latest.pth \
  --config /home/ubuntu/GS-SLAM/src/gs_slam/config/online_backend.json
```

普通 `live` 模式只保证最终 PLY；需要恢复时应确认检查点文件真实存在。

## 10. 常见问题

### 没有生成 manifest

检查四路话题是否都在发布、时间戳是否接近，以及图像编码能否转换为
`mono8`/`bgr8`：

```bash
ros2 topic echo /odometry --once
ros2 topic hz /camera/infra1/image_raw
ros2 topic hz /camera/color/image_raw
```

### 后端提示 CUDA 不可用

确认 launch 使用的是 Gaussian 环境中的 Python：

```bash
/home/ubuntu/miniconda3/envs/3dgs/bin/python -c \
  "import torch; print(torch.cuda.is_available())"
```

必要时通过 `backend_python:=...` 指定正确解释器。

### 停止后没有新的 PLY

检查 `status.json`。如果停在 `final_refinement`，说明进程在保存前被再次中断；
如果是 `complete`，使用 `final_ply` 字段确认本次运行对应的文件名。

### 修改源码配置后没有生效

启动时显式传入源码配置路径。若使用 launch 的安装目录默认配置，则修改后需要
重新执行 `colcon build --packages-select gs_slam --symlink-install`。

## 项目结构与许可证

- `src/gs_slam`：ROS 2 前端，只负责同步、RGB-D 编码和落盘；
- `src/gs_slam_backend`：Gaussian 地图、优化、稠密化、裁剪和输出；
- [`docs/mapping_optimization_history.md`](docs/mapping_optimization_history.md)：
  质量优化历史。

ROS 包使用 Apache-2.0；Gaussian 后端遵循
[`src/gs_slam_backend/LICENSE.md`](src/gs_slam_backend/LICENSE.md) 中的
Graphdeco research-only license。
