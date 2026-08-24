# GS-SLAM 代码导读

本文面向需要阅读、调试或继续开发本项目的人员，说明 ROS 前端与 Gaussian
后端的职责边界、数据流、各源码文件用途以及主要类和函数的具体实现。

运行和安装方法见项目根目录的 [`README.md`](README.md)。

## 1. 总体结构

项目把实时建图拆成两个相互独立的进程：

```text
ROS 2 话题
  │
  │ 左/右灰度图、彩色图、里程计
  ▼
gs_slam 前端
  │  双目匹配、深度置信度过滤、坐标变换、原子落盘
  ▼
会话目录
  ├── images/*.png
  ├── depth/*.png
  ├── manifests/*.json
  └── sparse/0/*
  │
  │ 后端轮询 manifest，或者离线读取 COLMAP 数据
  ▼
gs_slam_backend 后端
  │  关键帧选择、Gaussian 增长、渲染、优化、稠密化和剪枝
  ▼
online_output/status.json、latest.pth、point_cloud*.ply
```

前端不维护 Gaussian 地图，只负责把同步传感器数据转换为稳定、可重放的
RGB-D 帧。后端不依赖 ROS，可在包含 PyTorch、CUDA rasterizer 的 Conda 环境中
独立运行。二者通过磁盘上的 `FramePacket` JSON 协议解耦。

### 1.1 关键坐标和数据约定

- ROS 四元数使用 `[x, y, z, w]`，COLMAP 四元数使用 `[w, x, y, z]`。
- `T_world_camera` 把相机坐标中的点变换到世界坐标。
- 深度 PNG 是单通道 `uint16` 逆深度，值 `0` 表示无效像素。
- `inverse_depth_scale` 和 `inverse_depth_offset` 用于从 PNG 恢复真实逆深度：
  `inverse_depth = encoded / 2**16 * scale + offset`。
- 彩色图落盘时沿用 OpenCV 的 BGR 排列，后端加载后转换为 RGB。
- 前端写文件时先写 `.tmp`，写完后再替换目标文件；后端因此不会读到半写入文件。

## 2. 前端：`src/gs_slam`

前端是 ROS 2 `ament_python` 包。入口节点名为 `frame_archiver`，Python
console script 名为 `map_generator`。

### 2.1 `gs_slam/map_generator.py`

这是 ROS 前端主入口，负责参数声明、四路消息同步、频率控制和调用归档器。

#### `FrameArchiver`

继承 `rclpy.node.Node`，代表运行中的 ROS 归档节点。

- `__init__(self)`
  - 声明话题、相机内外参、处理频率、同步容差、输出目录、SGBM 和深度置信度参数。
  - 用声明后的参数创建 `FrameArchive`。
  - 为左右灰度图、彩色图和里程计创建 `message_filters.Subscriber`。
  - 使用 `ApproximateTimeSynchronizer` 做四路近似时间同步，并把回调注册为
    `archive_frame()`。
  - `period_ns` 控制总体采样频率，`stereo_tolerance_ns` 额外约束左右目时间差。

- `archive_frame(self, left_message, right_message, color_message, odometry)`
  - 取得左目时间戳并拒绝左右目时间差过大的数据。
  - 根据 `next_stamp` 跳过采样周期内的高频帧。
  - 保证里程计的世界坐标系 `frame_id` 在一次会话中不发生变化。
  - 通过 `CvBridge` 将 ROS 图像转为 OpenCV 灰度图和 BGR 图。
  - 将里程计四元数转换为旋转矩阵，组装世界平移向量。
  - 调用 `FrameArchive.archive_frame()` 计算深度并落盘。
  - 成功后更新下一采样时间并记录有效深度样本数；可恢复的输入或写盘错误只记日志，
    不让节点崩溃。

- `save(self)`
  - 调用归档器的 `save()` 写最终 COLMAP 相机、位姿和深度参数。
  - 该方法可重复调用；底层归档器会避免重复保存。

#### 模块函数

- `main(args=None)`：初始化 `rclpy`，创建并 spin 节点；收到 `Ctrl+C` 或 ROS
  外部关闭信号后，先保存归档元数据，再销毁节点和关闭 ROS。

### 2.2 `gs_slam/colored_pointcloud_processor.py`

这是前端的数据处理核心。文件名保留了早期“彩色点云处理器”的命名，但当前实现
只生成 RGB-D 归档和 COLMAP 元数据，持久 Gaussian 地图由后端管理。

#### 原子写入和格式导出函数

- `atomic_binary_writer(path)`：创建父目录并返回同目录下的 `文件名.tmp` 路径，
  供其他写函数完成“临时文件 → 原子替换”。
- `write_colmap_cameras(path, width, height, intrinsics)`：按 COLMAP 二进制格式写一个
  `PINHOLE` 相机，字段包括 camera id、model id、图像宽高和 `fx/fy/cx/cy`。
- `write_colmap_images(path, images)`：写已注册图像的 COLMAP 位姿和文件名；本项目
  不保存特征观测，所以每张图的 `POINTS2D` 数量为零。
- `write_colmap_points3d(path, points, colors)`：写 XYZ、RGB、误差和空 track；实时
  前端通常写一个空点云，已有 COLMAP 数据回放时后端也能读取非空点云作为初始化。
- `write_png(path, image)`：用 OpenCV 在内存中编码 PNG，确认编码成功后原子写盘；
  可无损保存 `uint16` 深度。
- `write_depth_params(path, parameters)`：写 Graphdeco 风格的逐帧逆深度比例和偏移。
- `write_frame_manifest(path, packet)`：以格式化 JSON 原子发布一帧 `FramePacket`。
- `frame_filename(frame_index)`：生成从零开始的共享 RGB/深度文件名，如 `0.png`。

#### 深度和几何函数

- `render_depth_image(...)`
  - 输入左目坐标系中的可靠三维点，以及左右/彩色相机相对机体的外参。
  - 先把点从源相机变换到机体，再变换到目标彩色相机。
  - 用彩色相机内参投影到像素；通过 `np.minimum.at` 实现 Z-buffer，只保留同一像素
    最近的正深度点。
  - 输出和彩色图同尺寸的米制 Z 深度，未覆盖像素为零。

- `disparity_confidence(...)`
  - 把左视差对应位置映射到右图，并用 `cv2.remap` 取得右视差。
  - 检查有限值、左右视差符号、图像边界和左右一致性误差。
  - 把基础视差噪声和左右误差合成为估计标准差，再除以左视差得到相对不确定度。
  - 将不确定度线性映射到 `[0, 1]` 置信度；不可靠像素直接置零。

- `encode_graphdeco_inverse_depth(depth_image)`
  - 对正的有限深度求倒数，按当前帧最大逆深度归一化到 `uint16`。
  - 保留零值作为无效标记，并返回能准确恢复真实逆深度的 alignment scale。

- `downsample(points, colors, size)`
  - 按 `size` 划分体素，同一体素内的点坐标和颜色取平均。
  - 使用 `cKDTree` 做简单半径去噪，移除两倍体素半径内没有其他邻点的孤立点。
  - 这是兼容已有数据工具的辅助函数，不参与当前在线 Gaussian 主循环。

#### `FrameArchive`

负责一帧从双目输入到完整磁盘协议的转换。

- `__init__(self, parameters)`
  - 创建 `images`、`depth`、`sparse/0`、`manifests` 等目录。
  - 根据参数构造左目 `StereoSGBM` 和对应的右目 matcher。
  - 构造 OpenCV `reprojectImageTo3D` 使用的 Q 矩阵。
  - 解析左目和彩色相机相对机体的旋转、平移，并初始化帧计数和 COLMAP 元数据缓存。

- `archive_frame(...)`
  - 检查彩色图分辨率在会话内保持不变。
  - 计算左右视差，并用 Q 矩阵把左视差重投影为左相机三维点。
  - 调用 `disparity_confidence()` 去除不一致和高不确定度像素。
  - 调用 `render_depth_image()` 把可靠点投影到彩色相机，得到与 RGB 对齐的深度。
  - 调用 `encode_graphdeco_inverse_depth()` 生成 `uint16` 逆深度；无有效深度时丢弃该帧。
  - 原子写 RGB、深度、逐帧 manifest，并缓存 COLMAP 图像位姿和深度比例。
  - manifest 中写入时间戳、坐标系名、相对文件路径、内参和 `T_world_camera`。
  - 返回有效深度像素数；兼容别名 `process_single_colored_point_cloud` 指向此方法。

- `save(self)`
  - 只执行一次，写 `cameras.bin`、`images.bin`、空的 `points3D.bin` 和
    `depth_params.json`。
  - 返回会话累计有效深度样本数和帧数。

- `ColoredPointCloudProcessor`：指向 `FrameArchive` 的历史兼容别名。

### 2.3 `gs_slam/frame_archive.py`

窄接口兼容层。它从旧模块重新导出 `FrameArchive`，让 ROS 节点依赖清晰的归档
抽象，同时不破坏旧代码的 import 路径。

### 2.4 `gs_slam/utils.py`

- `stamp_ns(message)`：把 ROS 的秒和纳秒时间戳合成为单个纳秒整数。
- `rotation_matrix(quaternion)`：把 `[x, y, z, w]` 四元数转换为 `3×3` 旋转矩阵。
- `quaternion_wxyz(rotation)`：把旋转矩阵稳定转换为 COLMAP 使用的
  `[w, x, y, z]` 四元数；针对迹为正和三个主对角元素分别选择数值稳定分支，最后
  归一化并统一到 `w >= 0`。
- `colmap_image_pose(r_wb, t_wb, r_bc, t_bc)`：组合世界到机体、机体到相机的
  外参，计算 COLMAP 需要的 world-to-camera 旋转和平移。

### 2.5 前端启动文件

#### `launch/frame_archiver.launch.py`

- `generate_launch_description()`：声明 `frontend_config` 参数，加载 YAML 并启动单独的
  `map_generator` ROS 节点。适合只采集数据、不运行 GPU 后端。

#### `launch/online_mapping.launch.py`

- `generate_launch_description()`：声明会话目录、前后端配置、后端 Python 环境、模块
  路径和预览参数；同时启动 ROS `frame_archiver` 节点和
  `python -m gs_slam_backend.runner live` 进程。二者共享同一个 session 目录。

#### `launch/replay_mapping.launch.py`

- `generate_launch_description()`：声明输入会话和输出目录，启动后端 `replay` 子命令，
  对已有 manifest 或 COLMAP 数据做一次有限离线建图，不启动 ROS 订阅节点。

### 2.6 前端配置与打包文件

- `config/stereo_camera.yaml`：ROS 话题、相机内外参、同步和采样参数、目录结构、SGBM
  参数及深度置信度阈值。键最终会展平为 `camera_intrinsics.fx` 等 ROS 参数名。
- `config/online_backend.json`：后端默认运行配置。虽然放在 ROS 包中以便 launch
  查找，但由后端的 `MapperConfig` 读取。
- `setup.py`：定义 `gs_slam` Python 包、安装 YAML/launch/package 元数据，并注册
  `map_generator = gs_slam.map_generator:main`。
- `setup.cfg`：让 ROS 安装脚本进入 `$base/lib/gs_slam`，以便 `ros2 run` 找到入口。
- `package.xml`：声明 ament 构建类型、ROS 运行依赖和 lint/test 依赖。
- `resource/gs_slam`：ament resource index 标记文件，用于 ROS 包发现。
- `gs_slam/__init__.py`：Python 包标记，无运行逻辑。
- `LICENSE`：前端 Apache-2.0 许可证。

### 2.7 前端测试文件

- `test/test_colmap_export.py`：验证体素降采样、深度 Z-buffer、左右视差置信度、逆深度
  往返、PNG 精度、COLMAP 位姿约定和二进制布局。
- `test/test_frame_manifest.py`：验证 manifest 通过临时文件原子发布且内容正确。
- `test/test_copyright.py`：调用 ament copyright 检查。
- `test/test_flake8.py`：调用 ament Flake8 检查。
- `test/test_pep257.py`：调用 ament PEP 257 文档字符串检查。

## 3. 后端：`src/gs_slam_backend`

后端是普通 Python 包，不依赖 ROS。`runner.py` 负责应用编排，`online_mapper.py`
负责建图状态机，`model.py` 管理可训练 Gaussian 参数，`renderer.py` 对接 CUDA
rasterizer。

### 3.1 `gs_slam_backend/runner.py`

命令行入口和 live/replay/resume 三种模式的编排层。

- `_boolean(value)`：解析 launch 传入的 `true/false`、`1/0`、`yes/no`、`on/off`。
- `_create_mapper(...)`
  - 延迟导入 `OnlineMapper`，使没有 CUDA rasterizer 的环境仍可使用配置工具。
  - 推导默认输出目录，忽略不存在的可选 checkpoint。
  - 把 CLI 路径、配置、预览、初始化点云和最终 refinement 帧交给 mapper。
- `replay(args)`
  - 加载配置并判断输入是完整 COLMAP 数据集还是 manifest 会话。
  - 可用 `--max-frames` 截断帧数。
  - 若没有 checkpoint 且允许 bootstrap，则读取 `points3D.bin`，估计尺度后作为初始
    Gaussian 点云。
  - 顺序调用 `mapper.process()`，终端只打印紧凑的逐帧阶段耗时 JSON；完整状态保留在
    `status.json`，最后无条件 `close()` 保存结果。
- `live(args)`
  - 持续轮询原子发布的 manifest；`resume` 模式先恢复 checkpoint。
  - 队列积压时仅优化第一帧、队尾帧或短队列中的帧，其他帧保留在磁盘但不向地图
    添加未优化 Gaussian，从而追赶实时输入。
  - 维护 refinement 帧列表和 source 的 `start_after` 游标。
- `build_parser()`：建立 `replay`、`live`、`resume` 子命令和公共路径参数；算法参数
  只从 JSON 配置读取，CLI 不覆盖算法开关。
  - 内部 `common(command, checkpoint_required=False)` 给三个子命令添加配置、输出和
    checkpoint 公共参数。
  - 内部 `live_preview(command)` 给 live/resume 添加预览开关和深度显示范围参数。
- `main(argv=None)`：解析参数并调用对应 handler。

### 3.2 `gs_slam_backend/config.py`

#### `MapperConfig`

一个 dataclass，保存全部运行时参数。字段大致分为：运行设备、初始化和体素、深度
可靠性、关键帧、优化器和学习率、球谐阶数、稠密化、剪枝、最终 refinement。

- `_GROUPS`：定义普通参数在 JSON 中所属的分组。
- `_FEATURES`：定义带 `enabled` 开关的功能及其 `parameters` 字段。
- `load(cls, path=None)`
  - 无路径时返回默认配置。
  - 同时接受旧的平铺 JSON 和当前 schema 2 分组 JSON。
  - 验证每层都是对象，拒绝未知 section、feature、字段或参数，避免拼写错误被静默忽略。
  - `opacity_reset.enabled=false` 会把 interval 强制置零。
- `validate(self)`：在启动 CUDA 工作前检查像素步长、体素尺寸、迭代数、优化器类型
  和 opacity 上下界。
- `to_dict(self)`：把平铺 dataclass 重新组织成规范的 schema 2 分组结构。
- `write(self, path)`：把规范配置格式化写为 JSON。

### 3.3 `gs_slam_backend/frame_packet.py`

定义前后端磁盘协议，以及在线和离线两种帧源。

#### `FramePacket`

不可变 dataclass，描述一帧 RGB-D 观测：序号、时间戳、图像路径、逆深度还原参数、
分辨率、内参、坐标系和 `T_world_camera`。

- `validate(self)`：验证 schema 版本、非负 ID/时间、正分辨率和焦距、完整内参，以及
  有限且末行为 `[0, 0, 0, 1]` 的 `4×4` 齐次变换。
- `from_json(cls, path)`：从 manifest 构造并验证 packet。
- `write_atomic(self, path)`：把 packet 转为 dict，通过 `.tmp` 原子写 JSON。
- `resolve(self, session_directory)`：把 packet 中的相对 RGB/深度路径解析到会话根目录。
- `load_images(self, session_directory)`
  - 加载 BGR 和原始 `uint16` 深度，检查尺寸、通道和类型。
  - BGR 转 RGB，按 scale/offset 恢复逆深度，再求米制深度。
  - 返回 `rgb, inverse_depth, metric_depth, valid`。

#### COLMAP 读取函数

- `_rotation_from_qvec(qvec)`：把 COLMAP `[w, x, y, z]` 四元数转为旋转矩阵。
- `_read_colmap_camera(path)`：读取单相机 `cameras.bin`；支持 `SIMPLE_PINHOLE` 和
  `PINHOLE`，返回相机 ID、宽高和内参。
- `_read_colmap_images(path)`：读取 `images.bin`，跳过特征 track，把 COLMAP 的
  world-to-camera 位姿还原为 `T_world_camera`。
- `read_colmap_point_cloud(path)`：读取 `points3D.bin` 的 XYZ 和 RGB，跳过可变长度
  track，颜色归一化到 `[0, 1]`。

#### 帧源

- `ColmapReplaySource`
  - `__init__(source)` 保存数据集根目录。
  - `__iter__()` 读取相机、图像位姿和 `depth_params.json`，按纯数字文件名优先排序，
    将每张图适配成经过验证的 `FramePacket`。
  - `__iter__()` 内部的 `sort_key(record)` 让纯数字文件名按数值排序，其余文件名排在
    后面按字符串排序，避免 `10.png` 被错误排在 `2.png` 前面。
- `LiveManifestSource`
  - `__init__(session, start_after=-1, poll_seconds=0.1)` 保存 manifest 目录和消费游标。
  - `pending()` 读取所有编号大于游标的 JSON，按 `sequence_id` 排序返回。
  - `__iter__()` 在无帧时 sleep，有帧时逐个推进游标，形成无限在线迭代器。

### 3.4 `gs_slam_backend/camera.py`

- `projection_matrix(znear, zfar, fov_x, fov_y, device)`：创建 Graphdeco rasterizer
  使用的透视投影矩阵。
- `GaussianCamera.__init__(...)`
  - 把 RGB、逆深度和有效 mask 转为设备上的 PyTorch tensor。
  - 对 `T_world_camera` 求逆得到 world-view 矩阵并按 rasterizer 约定转置。
  - 从焦距和分辨率计算水平、垂直视场角。
  - 组合 view 与 projection 矩阵，并保存世界坐标中的相机中心。

### 3.5 `gs_slam_backend/geometry.py`

- `point_cloud_scales(points, fallback=0.03)`：用最多三个最近邻距离的均方根估计每个
  初始 splat 的各向同性尺度，点太少或结果异常时使用 fallback。
- `backproject(...)`
  - 取有效深度像素并按 `pixel_stride` 采样。
  - 用针孔模型反投影为相机坐标点，再经 `T_world_camera` 转到世界坐标。
  - 读取对应 RGB，按像素投影尺寸估计初始 scale。
  - 每个体素保留一个点，并通过均匀索引限制到 `max_points`。
- `pose_distance(first, second)`：返回两个相机位姿之间的平移距离和旋转角度（度）。
- `projected_overlap(...)`：采样当前有效深度点，投影到目标相机，统计落在带边界留白
  图像内部且位于相机前方的比例。
- `resize_depth_preserving_validity(...)`：线性缩放深度、最近邻缩放 mask，并确保无效零值
  不会通过插值重新变成有效深度。
- `reliable_depth_mask(...)`：腐蚀有效区域，并比较水平/垂直相邻像素的相对深度差，
  去掉无效边缘和深度突变边缘。
- `should_add_keyframe(...)`：先应用最小/最大帧间隔，再根据低重叠、相对场景深度的
  平移量或旋转角判断是否加入关键帧。

### 3.6 `gs_slam_backend/model.py`

定义可训练 Gaussian 地图及其动态增删。内部参数使用便于优化的未激活形式：opacity
保存 logit，scale 保存 log，rotation 保存待归一化四元数。

#### 模块函数

- `rgb_to_sh(rgb)`：把 RGB 转为零阶球谐 DC 系数。
- `inverse_sigmoid(value)`：把 `[0, 1]` opacity 转为 logit 参数。
- `quaternion_to_matrix(quaternion)`：归一化标量优先四元数并批量转换为旋转矩阵。

#### `GaussianMap`

- `__init__(...)`：创建空的 XYZ、SH、opacity、scale、rotation 参数和关键帧年龄、可见性、
  深度冲突、屏幕梯度等辅助数组。
- `xyz`、`features`、`opacity`、`scaling`、`rotation` 属性：分别返回坐标、拼接后的
  SH、sigmoid opacity、指数 scale 和归一化四元数。
- `render_subset(indices)`：只取视锥内或指定索引的可微渲染参数，同时提供合并 SH 和
  新 rasterizer 使用的 DC/rest 分离形式。
- `initialize(points, colors, scales, keyframe=0, opacity=None)`：用第一批点建立参数，
  初始化出生/最后可见关键帧和训练统计。
- `_new_tensors(...)`：把 NumPy 输入转到目标设备；生成 SH DC、零高阶 SH、单位旋转、
  log scale 和初始 opacity logit。
- `setup_optimizer(config)`：为不同属性建立独立学习率参数组；选择普通 Adam 或 CUDA
  `SparseGaussianAdam`，并检查后者的扩展和设备条件。
- `update_position_learning_rate(...)`：按训练进度在 log 空间从初始位置学习率插值到
  最终学习率。
- `constrain_parameters(...)`：限制颜色、opacity、scale 和最大各向异性，并重新归一化
  rotation，防止在线优化产生不可用参数。
- `_mutate_optimizer(extensions=None, keep=None)`：追加或剪除 Gaussian 时同步替换
  `nn.Parameter`，并同步扩展或切片 Adam 的一、二阶矩状态，保持行对齐。
- `append(points, colors, scales, keyframe)`：追加由新深度发现的 Gaussian，并扩展全部
  生命周期和梯度统计数组。
- `prune(prune_mask)`：删除 mask 选中的 Gaussian，同时裁剪参数、优化器状态和辅助数组。
- `add_densification_stats(indices, viewspace_gradient)`：累计可见 Gaussian 的屏幕空间
  XY 梯度模长及观测次数。
- `_append_internal(tensors, source_indices, keyframe)`：供 clone/split 使用的内部追加逻辑；
  可继承源 Gaussian 的元数据，或按新关键帧重置生命周期。
- `densify_and_prune(...)`
  - 用累计平均屏幕梯度选择候选，并限制单次候选数量。
  - 小于场景尺度阈值的 Gaussian 直接 clone；较大的沿自身旋转尺度随机采样并 split
    成两个更小 Gaussian。
  - 删除被 split 的原 Gaussian 和低 opacity 原点，最后重置梯度统计。
- `reset_opacity(maximum=0.01)`：把过高 opacity 降到指定值，并清零 opacity 参数对应的
  Adam 动量。
- `save_ply(path)`：按 Graphdeco 字段顺序写 PLY；运行时 SH 阶数较低时补零到三阶，
  保持与常见 viewer 兼容。
- `checkpoint_state()`：返回模型、优化器、SH 阶数、生命周期和稠密化统计。
- `restore_checkpoint_state(state, config)`：重建 Parameter 和优化器，恢复 checkpoint
  状态；对旧 checkpoint 缺少的梯度统计提供零值兼容。

### 3.7 `gs_slam_backend/renderer.py`

封装 `diff_gaussian_rasterization`，并让不含 CUDA 扩展的环境仍能导入配置模块。

- `require_rasterizer()`：在真正开始建图时检查扩展是否可用，否则给出明确环境错误。
- `frustum_indices(camera, model, margin, near, far)`：把 Gaussian 中心变换到相机坐标，
  按扩展后的水平/垂直视场和远近裁剪面返回全局索引。
- `_empty_render(camera, model, background)`：当前相机与地图不相交时返回尺寸正确且仍连接
  计算图的空 RGB、逆深度、silhouette 和空索引。
- `render(...)`
  - 选择显式 active indices、视锥裁剪结果或整个模型。
  - 构造 rasterizer settings 和可保留梯度的二维均值占位 tensor。
  - 根据扩展能力使用合并 SH 或分离的 DC/rest SH 接口。
  - 第一次 rasterization 输出 RGB、半径和累积逆深度。
  - 可选的第二次无梯度白色渲染生成 alpha/silhouette。
  - 返回渲染结果、可见 mask、屏幕空间点和 active 全局索引。

### 3.8 `gs_slam_backend/losses.py`

- `_ssim_window(...)`：生成归一化二维高斯核，并按通道扩展供 grouped convolution 使用。
- `ssim(first, second, size=11)`：计算局部均值、方差和协方差，返回整幅图平均 SSIM。
- `mapping_loss(...)`
  - 用 detached silhouette 产生高覆盖区域 mask。
  - RGB 损失由 mask 内 L1 和 SSIM 组合；mask 外用真实图替换渲染图，避免无覆盖区域
    影响 SSIM。
  - rasterizer 的逆深度是 alpha 累积量，因此先除以 detached alpha，再只在真实深度
    有效且覆盖充分的像素计算逆深度 L1。
  - 返回总损失及 detached 的 loss、RGB L1、depth L1、coverage 诊断值。

### 3.9 `gs_slam_backend/preview.py`

- `compose_preview(...)`：验证输入尺寸和深度范围，把米制深度反向归一化并应用 TURBO
  色图，最后横向拼接相机 RGB、深度和 Gaussian 渲染并写标签。
- `PreviewVisualizer.__init__(...)`：保存显示参数；Linux 没有 DISPLAY/Wayland 时自动
  禁用预览，不影响建图。
- `PreviewVisualizer` 在独立线程中持续处理 HighGUI 事件，并使用单槽最新帧缓冲；
  新帧会覆盖尚未显示的旧帧，避免长时间运行后预览延迟持续累积。
- `show(...)`：把画面发布到单槽缓冲，并在预览线程收到 `q` 或 `Esc` 后返回 `False`。
  HighGUI 错误会自动禁用预览而不是中止 mapper。
- `close()`：安全销毁已创建的窗口。

### 3.10 `gs_slam_backend/online_mapper.py`

`OnlineMapper` 是后端核心状态机。一次正常帧处理顺序为：加载观测 → 判断关键帧 →
补充新 Gaussian → 在关键帧窗口优化 → 更新深度一致性并剪枝 → 写状态/检查点。

#### 初始化、IO 和渲染辅助

- `__init__(...)`：检查 rasterizer 和 CUDA，建立输出目录、随机种子、空 `GaussianMap`、
  背景、关键帧库、计数器、体素占用索引和预览；可恢复 checkpoint，并注册 SIGINT/
  SIGTERM 停止标记。
- `_final_ply_path(unique)`：选择固定 `point_cloud.ply`，或带时间 run id 且不覆盖已有文件
  的唯一名称。
- `_stop(*_)`：信号处理器，只设置 `stopped`，让主循环走正常保存流程。
- `_load_observation(packet, load_camera=True)`：加载 packet 的 RGB/深度/mask，并按需创建
  GPU `GaussianCamera`。
- `_camera_for(observation)`：延迟创建和缓存 camera，控制长会话 GPU 内存占用。
- `_render(...)`：把 mapper 配置的视锥参数和 active indices 转发给 renderer。
- `_constrain_model()`：用配置边界调用模型参数约束。
- `_reliable_depth(observation)`：按配置生成去边缘后的可靠深度 mask。

#### 地图建立和新点发现

- `_initialize(observation)`：优先使用 COLMAP 初始点云，否则反投影第一帧；初始化模型和
  optimizer、注册体素、以有效深度 90 分位估计场景半径，并加入首个关键帧。
- `_voxel_keys(points)`：把世界点量化为整数体素 key。
- `_register_voxels(points)`：把点所在体素加入占用集合。
- `_rebuild_occupied_voxels()`：从当前已优化/剪枝后的模型重建体素集合，防止旧位置永久
  占坑。
- `_remove_occupied_candidates(points, colors, scales)`：去掉已被全局地图或当前候选批次
  占用的体素，每个新体素只保留一个候选。
- `_new_gaussians(observation, rendering)`
  - 将累积逆深度除以 silhouette，得到当前渲染表面逆深度。
  - 选择渲染覆盖不足的可靠像素，以及覆盖充分但观测表面明显更靠近相机的像素。
  - 反投影、体素去重、追加 Gaussian、约束参数并更新统计。

#### 关键帧和优化

- `_keyframe_overlap(observation, target)`：计算当前帧深度投影到目标关键帧的覆盖比例。
- `_maybe_add_keyframe(observation)`：结合重叠、深度尺度化平移、旋转和帧间隔决定是否
  加入关键帧库。
- `_mapping_window(current)`：窗口包含当前帧、最新关键帧、重叠最高的若干关键帧和随机
  历史帧，并按 sequence id 去重。
- `_optimizer_step(rendering)`
  - 把屏幕空间梯度、最后可见关键帧和最大二维半径写回全局 Gaussian 行。
  - 更新位置学习率，执行普通或 sparse Adam，约束参数并清梯度。
  - 按间隔提高 active SH degree。
  - 在配置区间内触发 clone/split densification 和 opacity reset；模型行变化时通知调用方
    清除 active-index 缓存。
- `_depth_weight()`：在 log 空间把初始深度权重平滑衰减到最终权重。
- `_optimize(observation)`：轮转 mapping window，受最大迭代数和单帧时间预算共同限制；
  每次渲染、计算 loss、反传和更新模型，并复用当前帧训练渲染作为预览。
- `_preview_bgr(rendered_rgb)`：在 GPU 上换通道、量化为 `uint8` 后只传输小尺寸结果到 CPU。

#### 深度一致性和剪枝

- `_depth_consistency_masks(observation)`：把 Gaussian 中心投影到当前深度图；明显位于观测
  表面之前的标为自由空间冲突，与表面深度一致的标为支持。表面之后的点视为被遮挡，
  不作为冲突。
- `_update_depth_consistency(observation)`：支持观测清零历史冲突计数，自由空间冲突累加。
- `_final_free_space_cleanup(frames)`：在最终 refinement 前汇总所有归档帧；只有冲突数达到
  阈值且大于支持数两倍的 Gaussian 才作为 ghost 删除。
- `_prune()`：只处理超过 newborn grace 的成熟 Gaussian；按配置组合 ghost、长期不可见、
  低 opacity、相对场景过大四类条件。少量候选只累计，达到批量点数或最长等待关键帧数
  后才统一压缩模型和重建体素索引；退出前强制清空。
- `_pruning_enabled()`：判断任一在线剪枝模式或 legacy 总开关是否启用。

#### 主流程、状态和持久化

- `process(packet, optimize=True, queue_length=0)`
  - 忽略已经处理过的 sequence id。
  - 空地图执行初始化；非空地图先判断关键帧，关键帧才根据渲染补充新 Gaussian。
  - `optimize=False` 用于跳过积压帧的建图更新。
  - 优化后可进行深度一致性更新和剪枝，并刷新预览。
  - 更新计数和诊断信息，原子写 `status.json`；达到关键帧间隔时保存 checkpoint。
  - 最后释放历史关键帧的 GPU camera，仅保留 CPU 图像和 packet。
- `_write_status(status)`：通过 `.tmp` 原子更新 `status.json`。
- `save_checkpoint(path=None)`：保存模型/优化器、最后帧、关键帧 packet、场景半径和累计
  统计到 `latest.pth` 或指定路径。
- `save_final_ply()`：先写临时 PLY，再原子替换最终路径。
- `_final_refinement()`
  - 根据配置选择关键帧或所有归档帧；可先执行全帧 ghost cleanup。
  - 支持随机帧顺序，或每帧连续优化五次后轮转。
  - 周期性写 refinement 状态，并继续复用标准 loss、optimizer step、SH 和 densification。
- `_final_cleanup()`：退出前按独立阈值删除低 opacity 或绝对尺度过大的 Gaussian。
- `restore(path)`：恢复模型/optimizer、游标、关键帧、场景和训练统计，并重建体素索引。
- `close()`：关闭预览；可执行最终 refinement 和 cleanup；保存 checkpoint、最终 PLY，
  最后把状态 phase 更新为 `complete`。即使 refinement/cleanup 异常，也会在 `finally`
  中尽力保存当前 PLY。

### 3.11 后端包级和工程文件

- `gs_slam_backend/__init__.py`：导出公共 `FramePacket` 类型。
- `pyproject.toml`：定义 Python 3.10+ 包、运行依赖和
  `gs-slam-backend = gs_slam_backend.runner:main` 命令行入口。
- `setup.cfg`：配置 99 字符 Flake8 行宽，并允许模型测试在导入前做环境准备。
- `COLCON_IGNORE`：阻止 ROS `colcon` 把独立 GPU 后端当作 ROS 包构建。
- `LICENSE.md`：后端主体及上游 Graphdeco 相关代码的许可条款。
- `NOTICE.md`：第三方来源和版权说明，同时作为后端包的 readme 元数据。

### 3.12 后端测试文件

- `tests/test_config.py`：验证分组配置往返、feature 禁用语义、未知字段拒绝、优化器校验，
  并确保 CLI 不提供绕过 JSON 的算法参数。
- `tests/test_frame_packet.py`：验证 packet 原子读写和深度解码、COLMAP 位姿恢复、点云尺度、
  关键帧规则、可靠深度 mask 及默认剪枝开关。
- `tests/test_model.py`：验证 sparse optimizer 能力检查、SH 接口、动态增删时 Adam 状态对齐、
  mask loss、alpha 归一化深度损失、PLY 兼容、参数约束、体素重建、densification 对齐、
  各向异性限制和自由空间 ghost 剪枝。
- `tests/test_preview.py`：验证三栏预览布局、无效深度黑色显示及有效深度色图。

## 4. 端到端调用链

### 4.1 在线模式

```text
online_mapping.launch.py
  ├── ROS Node: FrameArchiver.archive_frame()
  │     └── FrameArchive.archive_frame()
  │           ├── StereoSGBM.compute()
  │           ├── disparity_confidence()
  │           ├── render_depth_image()
  │           ├── encode_graphdeco_inverse_depth()
  │           └── write_frame_manifest()
  │
  └── Backend: runner.live()
        └── LiveManifestSource.pending()
              └── OnlineMapper.process()
                    ├── _initialize() / _maybe_add_keyframe()
                    ├── _new_gaussians()
                    ├── _optimize()
                    │     ├── renderer.render()
                    │     ├── mapping_loss()
                    │     └── _optimizer_step()
                    ├── _update_depth_consistency() + _prune()
                    └── _write_status() / save_checkpoint()
```

### 4.2 离线回放模式

```text
replay_mapping.launch.py
  └── runner.replay()
        ├── ColmapReplaySource 或 LiveManifestSource
        ├── 可选 read_colmap_point_cloud() bootstrap
        ├── 对每帧调用 OnlineMapper.process()
        └── OnlineMapper.close()
              ├── _final_refinement()
              ├── _final_cleanup()
              ├── save_checkpoint()
              └── save_final_ply()
```

## 5. 常见修改入口

| 需求 | 主要文件/函数 |
|---|---|
| 修改 ROS 话题、同步和双目参数 | `config/stereo_camera.yaml`、`FrameArchiver.__init__()` |
| 修改双目深度过滤 | `disparity_confidence()`、`FrameArchive.archive_frame()` |
| 修改前后端磁盘协议 | 前端 `write_frame_manifest()`、后端 `FramePacket` |
| 修改关键帧策略 | `geometry.should_add_keyframe()`、`OnlineMapper._maybe_add_keyframe()` |
| 修改新 Gaussian 生成规则 | `OnlineMapper._new_gaussians()`、`geometry.backproject()` |
| 修改损失函数 | `losses.mapping_loss()` |
| 修改渲染和视锥裁剪 | `renderer.render()`、`renderer.frustum_indices()` |
| 修改优化器或学习率 | `GaussianMap.setup_optimizer()`、`OnlineMapper._optimizer_step()` |
| 修改 densification | `GaussianMap.densify_and_prune()` |
| 修改在线/最终剪枝 | `OnlineMapper._prune()`、`_final_free_space_cleanup()`、`_final_cleanup()` |
| 修改输出 PLY 字段 | `GaussianMap.save_ply()` |
| 修改命令行模式 | `runner.build_parser()`、`runner.live()`、`runner.replay()` |
