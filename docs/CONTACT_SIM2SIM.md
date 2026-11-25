# OMOMO contact-policy 的 ONNX / native MuJoCo 验证

不同物理引擎的闭环验证现已新增，见 [Isaac Lab / PhysX 验证](CONTACT_ISAAC.md)。本页描述的仍是 CPU MuJoCo 验证。

这一入口用于 `supermimic.hoi.contact.train` 的 checkpoint，不能替换原有单人 `DeployCore` / `g1_runner.py` 的 ONNX。代码在 `scripts/deploy/contact_sim2sim.py`，运行不导入 PyTorch、mjlab 或训练项目。

## 已导出的片段

`deploy_artifacts/hoi_contact_pose_sub7_largebox_016/` 保存：

- `source_checkpoint.pt`：导出时的固定快照，step 112000，避免正在训练的 latest.pt 变化影响验证；SHA256 在配置中。
- `policy.onnx`：actor、归一化、机器人状态预处理、参考动作和接触标签预处理。
- `reference.npz`：largebox / val / index 10，`omomo/unitree_g1/sub7_largebox_016`，321 帧、50 Hz。
- `scene.mjb`：训练场景的原生 MuJoCo 模型，包含动态 largebox、碰撞、惯性和训练 PD 执行器。
- `deploy_config.json`：关节顺序、位置/速度地址、执行器映射、action scale/offset、时间步长、输入窗口设置。
- `native_report.json` / `training_env_validation.json`：原生 MuJoCo 与训练环境的闭环结果。

## 打开网站

```bash
cd /home/yihong/SuperMimic-Repro-Sim2Real-main
bash scripts/deploy/run_contact_sim2sim.sh --port 8084
```

打开 `http://localhost:8084`；远程使用服务器 IP 或 SSH 转发端口。浏览器连接后开始；实色是物理仿真，青色半透明是参考机器人和物体。支持暂停、调整速度、Reset clip。结束或跌倒后停住，按 Reset 重试。参考物体不会写回物理世界。

```bash
# 无窗口整段验证；report 必须使用新文件名
bash scripts/deploy/run_contact_sim2sim.sh --headless --report /tmp/contact_native_validation.json
```

CPU ONNX + native MuJoCo 不需要 CUDA_VISIBLE_DEVICES。启动脚本默认使用训练项目已有 Python，仅使用其中的 numpy/scipy/mujoco/onnxruntime/viser/mjviser；可用 `CONTACT_PYTHON=/path/to/python` 覆盖。MJB 需要导出时的 MuJoCo 版本（当前 3.16.0），程序会检查。已在该环境安装 onnxruntime 1.32.0，不修改训练的 Torch/CUDA 或 uv.lock。新环境需另外安装这些运行依赖。

## 重新导出更新后的 checkpoint

在训练项目执行，output 必须是新目录：

```bash
cd /home/yihong/supermimic_ws/supermimic
CUDA_VISIBLE_DEVICES=1 TORCH_COMPILE_DISABLE=1 .venv/bin/python \
  -m supermimic.hoi.contact.export \
  --checkpoint logs/hoi_contact_pose_all/latest.pt \
  --object largebox --split val --clip sub7_largebox_016 \
  --output /home/yihong/SuperMimic-Repro-Sim2Real-main/deploy_artifacts/hoi_contact_pose_new
```

然后在启动命令中指定 `--artifacts /home/yihong/SuperMimic-Repro-Sim2Real-main/deploy_artifacts/hoi_contact_pose_new`。

## ONNX 接口

固定 batch=1，输入 float32：

| 输入 | 形状 | 内容 |
|---|---|---|
| robot_state | (1,71) | root pos3、wxyz4、q29、世界系线速度3/角速度3、dq29 |
| reference_robot | (1,10,71) | 当前帧开始、每隔5帧的机器人参考；末尾截到本片段最后一帧 |
| contact_labels | (1,10,10) | 同窗口的期望接触标签；5个机器人link × terrain/object |
| action（输出） | (1,29) | raw action；关节目标 = offset + scale × action |

参考输入仅使用 root pose / q / dq；不存在物体位置、姿态、BPS 或质量输入。物体参考仅供初始化、显示和指标使用。归一化统计量固化在 ONNX 中；不要再次归一化。MuJoCo 自由关节角速度 qvel 为局部坐标，适配器会转为世界坐标输入策略；不能直接拼接 qvel。

## 当前结果与局限

step 112000，指定验证片段：

| 指标 | 训练环境 | 原生 MuJoCo / ONNX |
|---|---:|---:|
| 平均机器人根位置误差 | 5.28 cm | 5.41 cm |
| 平均物体位置误差 | 3.79 cm | 3.83 cm |
| 平均物体姿态误差 | 53.34° | 53.99° |
| 完成整段 / 未触发机器人失败 | 是 | 是 |

训练环境期望物体接触覆盖率 96.30%，但正式 success=false：物体姿态误差超过30°阈值。ONNX 转换不是这项误差的主要来源。原生物体最大位移约0.943m，参考0.973m。

两列不是逐帧完全等价测量：训练 evaluate 比较 post-step 与当步目标且执行321步，native 比较下一帧参考、执行320个帧间隔。原生报告不伪装成完整 contact success 评估，也未加入训练的接触缺失提前终止。目前是训练场景从 GPU Warp 切到 CPU MuJoCo 的移植验证，不是 Unitree 官方硬件模型的鲁棒性门禁。

20组参考/扰动状态的训练 actor 对 ONNX 最大动作绝对误差4.1e-6；另有根角速度坐标、关节执行器映射、参考与物理状态隔离测试。

## 真机部署入口

已新增 contact 专用硬件适配器，复用原 G1 runner 的通讯和状态机。操作步骤见 [README_CONTACT_HARDWARE.md](../README_CONTACT_HARDWARE.md)。不要用单人策略的默认入口直接加载此 ONNX。

artifact 仍为 `real_robot_ready=false`：物体姿态指标未达标，真实 VIO 适配/标定和完整站立过渡尚未通过。当前只完成离线验证与部分本机 DDS 联调，没有发送真机命令。
