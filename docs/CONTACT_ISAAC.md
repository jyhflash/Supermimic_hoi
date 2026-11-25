# Contact policy：MuJoCo → Isaac Lab / PhysX

训练使用 MuJoCo Warp；`contact_sim2sim.py` 是 CPU MuJoCo 移植检查。这里新增的 `contact_isaac.py` 才是不同物理引擎的闭环：Isaac Sim 5.1.0 / Isaac Lab 0.54.3 / PhysX，ONNX Runtime 推理。

已验证当前 `logs/hoi_contact_pose_all/latest.pt` 的 SHA256 与固定快照相同，step 112000。沿用 `sub7_largebox_016`，largebox 验证集第10段。无需重新训练或重新导出策略。

## 运行真实 PhysX 验证

```bash
cd /home/yihong/SuperMimic-Repro-Sim2Real-main
bash scripts/deploy/run_contact_isaac.sh \
  --headless --device cpu \
  --report /tmp/contact_isaac_validation.json
```

report 必须是新文件；`--steps 10` 可做短测。脚本默认使用 `/home/yihong/anaconda3/envs/env_isaaclab/bin/python`，可用 `ISAAC_PYTHON` 覆盖。已在该环境增加 onnxruntime 1.32.0，没有改变训练环境的 Torch/CUDA。

本机 Isaac 启动报 Vulkan `No device could be created`，因此本次使用 **CPU PhysX**，不是 GPU PhysX。无窗口物理仿真和报告输出已实际运行成功；原生 Isaac 图形窗口/GPU PhysX 尚未验证，不能把 PyTorch CUDA 可用当作 Isaac Vulkan 可用。本入口完成报告写入后采用 Isaac 官方立即关闭选项，避免无图形设备时 Replicator 清理阶段挂起。

## 网页查看 PhysX 结果

```bash
cd /home/yihong/SuperMimic-Repro-Sim2Real-main
/home/yihong/supermimic_ws/supermimic/.venv/bin/python \
  scripts/deploy/contact_isaac_replay.py --port 8085
```

打开 `http://localhost:8085`。远程使用服务器IP或转发8085端口。实色是 **PhysX 记录的机器人/物体轨迹**，青色是参考；支持暂停、拖动帧、调节速度。默认读取下方的 `isaac_aligned_report.json`，也可用 `--report` 指定新报告。

这是离线轨迹回放，使用 MuJoCo 模型做网格定位和 Viser 渲染，**不调用 mj_step，不重新计算 MuJoCo 物理**，也不支持施加推力。PhysX 闭环由前一个命令执行。

## 实际结果

报告：`deploy_artifacts/hoi_contact_pose_sub7_largebox_016/isaac_aligned_report.json`，包含完整状态轨迹。

| 指标 | CPU MuJoCo / ONNX | CPU PhysX / ONNX |
|---|---:|---:|
| 执行控制步数 | 320 / 320 | 320 / 320 |
| 停止原因 | reference_end | reference_end |
| 平均机器人根位置误差 | 5.41 cm | 5.56 cm |
| 平均物体位置误差 | 3.83 cm | 4.49 cm |
| 平均物体姿态误差 | 53.99° | 48.60° |
| 物体最大位移 | 0.943 m | 0.961 m |

两者均按 post-step 对下一参考帧比较。本次 PhysX 整段未触发高度偏差0.25m或投影重力z差0.8的机器人失败条件。**不代表 HOI success**：物体姿态误差仍超过原评估30°阈值；此入口尚未实现接触覆盖率/缺失接触终止，也未完成部署文档中的G2/G3/standby/DDS门禁。

`isaac_report.json` 是首次有效完整运行（默认材质组合），`isaac_aligned_report.json` 是显式设置摩擦参数后的最终对比。`isaac_invalid_zero_effort.json` 已标记无效：开发初期错误读取了未启用的 joint-level force range，而非 actuator force range；不可用来评价策略。该问题已修正并加入回归测试。

## 迁移内容和差异

- `prepare_contact_isaac.py` 从固定 `scene.mjb` 导出训练机器人的关节、局部变换、质量、惯性、几何和PD参数到 `isaac/robot.urdf`。没有依赖 SONIC 仓库中的 LFS 占位USD。
- Isaac 按关节名称重排，保留29DOF；导入总质量33.34114kg。10组关节姿态下，导出URDF与训练模型的FK位置差小于1e-8m。启动时检查根状态、关节状态和总质量。
- 每5ms执行一次显式PD和力矩限幅；策略20ms一次。使用训练 kp/kd、armature、actuator force range，目标 `offset + scale * action`。
- largebox 为动态刚体，保留导出的碰撞凸包、质量2kg、质心和主惯量。几何已经包含0.83缩放，不重复缩放。只在开始时写入物体参考状态。
- 摩擦设置 ground=1.0、robot=0.6、object=0.9，PhysX combine=max。机器人根/关节和物体均用参考首帧位置、姿态、速度初始化。
- actor 只读机器人状态、未来机器人参考和期望接触标签；物体状态用于物理、初始化、指标及回放。root速度采用 link-frame 原点的世界系速度，避免误用质心速度。
- MuJoCo capsule 在URDF中表示为 cylinder + 两个 sphere；PhysX凸包烘焙、接触算法、积分器与MuJoCo不同。机器人自碰撞目前关闭，未复现MuJoCo逐碰撞对排除规则。因此这也不是“仅求解器变化、其他数学细节完全相同”的实验。
- 使用仿真提供的完整机器人根部里程计；未验证真机状态估计、站立起步/结束过渡和硬件限幅状态机。

重新导出一个新策略后，先准备资产：

```bash
/home/yihong/supermimic_ws/supermimic/.venv/bin/python \
  scripts/deploy/prepare_contact_isaac.py --artifacts /path/to/new_artifacts
bash scripts/deploy/run_contact_isaac.sh \
  --artifacts /path/to/new_artifacts --headless --device cpu \
  --report /tmp/new_contact_isaac.json
```

测试：`tests/test_contact_native.py` 共5项通过，包括导出URDF的FK一致性、执行器力矩上限、世界/局部角速度、关节映射及参考与物理状态隔离。PhysX三步/完整片段验证和网页回放三帧启动检查均已完成；未进行浏览器像素检查。
