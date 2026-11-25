# G1 Contact Policy：从离线验证到真机验收

适用策略：`logs/hoi_contact_pose_all/latest.pt`，固定快照 step **112000**；参考 `sub7_largebox_016`。入口是 `scripts/deploy/contact_g1_runner.py`，不是单人 runner 的 ONNX 替换。

**当前还不能直接执行真机搬运。** 用户已确定使用 G1 + VIO，但腰部型号、VIO产品/驱动/坐标系未确定。当前物体姿态平均误差约49–54°，超过30°任务阈值。本文提供实际代码、命令和停止条件；尚未通过的检查不能靠把 JSON 改成 true 来代替。

本次没有连接真机网卡，没有发布 domain 0 电机命令。

## 0. 先看当前完成状态

| 项目 | 状态 |
|---|---|
| contact ONNX 与训练 actor 一致性 | 已通过 |
| 原生 MuJoCo、Isaac/PhysX 整段闭环 | 已运行；机器人未触发失败，物体姿态不达标 |
| 仿真/硬件共用 ContactCore（含目标限幅） | 已在 MuJoCo 跑完320步 |
| VIO缺失、失效、时间戳、乱序和非有限值测试 | 已实现并通过单元测试 |
| 实际 VIO驱动、外参、时间同步 | **待接入** |
| Unitree实机型号、状态与电机方向 | **待只读检查** |
| 完整站立过渡、DDS异常退出、真机接触搬运 | **不能视为已通过，按后续步骤实测** |

还应完成 [SIM2SIM.md](docs/SIM2SIM.md) 的官方G1模型、延迟/增益/载荷扰动和姿态裕量检查；当前contact DDS harness使用导出的训练模型，不等同于官方模型G2全网格通过。

## 1. 环境和文件

```bash
cd /home/yihong/SuperMimic-Repro-Sim2Real-main
source .venv-contact-deploy/bin/activate
export ART="$PWD/deploy_artifacts/hoi_contact_pose_sub7_largebox_016"
python scripts/deploy/contact_g1_runner.py --help
```

已建立独立 Python3.10 部署环境：numpy/scipy、MuJoCo3.16、onnxruntime、CycloneDDS0.10.2、Unitree SDK。训练 Torch/CUDA 环境不变。SDK安装在该venv的 `vendor/unitree_sdk2_python`，因原始ZIP中的CRC动态库是LFS占位文件，已从Unitree官方仓库取回并核验x86_64库 SHA256 `b136a91a7e99c5cf914f465e131f63c897a17107939184a6f85d1a57cf315ade`。不要把这个x86_64环境直接复制到机器人ARM电脑。

部署文件：

- `policy.onnx`、`reference.npz`：策略和固定参考。
- `hardware_config.json`：训练PD、力矩/速度限值、硬关节限位、scale/offset及Unitree电机顺序。
- `hardware_review.example.json`：本次artifact的hash和**全为未完成**的审核模板。
- `scene.mjb`：仅仿真用，真机runner不读取MuJoCo。

若换了checkpoint，先按 [CONTACT_SIM2SIM.md](docs/CONTACT_SIM2SIM.md) 重新导出到新目录，再生成该目录的硬件配置：

```bash
/home/yihong/supermimic_ws/supermimic/.venv/bin/python \
  scripts/deploy/prepare_contact_hardware.py --artifacts "$ART"
```

当前目录已生成，重复命令会拒绝覆盖。旧测试结论不能移用到新checkpoint。代码按关节名核对29电机顺序；暂不支持腰部roll/pitch锁定型号，不能跳过型号检查。

## 2. 离线检查：不接机器人

```bash
/home/yihong/supermimic_ws/supermimic/.venv/bin/python -m pytest \
  tests/test_contact_deploy.py tests/test_contact_native.py -q
python scripts/deploy/contact_sim2sim.py --headless --deploy-core \
  --report /tmp/contact_shared_core_new.json
```

后者使用和真机同一个 `ContactCore`，包含动作检查、姿态/速度检查、根位置检查、目标力矩限幅。要求跑完整段、无 `deploy_abort`，并记录物体误差。目前这一步完成了，但姿态任务指标不通过。

**注意**：原 `contact_sim2sim.py` 不带 `--deploy-core` 时仍是旧的直接ONNX验证，不能替代这个检查。

## 3. 本机 DDS 联调：只用 domain 1 / lo

以下脚本硬编码模拟器仅使用本机loopback和domain1；不接机器人网卡。

```bash
bash scripts/deploy/test_contact_dds.sh normal
bash scripts/deploy/test_contact_dds.sh vio-loss
bash scripts/deploy/test_contact_dds.sh nan
bash scripts/deploy/test_contact_dds.sh stall
```

每次输出一个新的 `deploy_artifacts/contact_dds_<scenario>_*` 日志目录。**进程退出码0不等于测试成功**，必须查看 `runner.log` 的 STATE、ABORT、DAMP，和命令记录：

| 场景 | 通过条件 |
|---|---|
| normal | PASSIVE→INTERP→HOLD→STANDBY_PRE→POLICY→STANDBY_POST，动作完整，无意外ABORT；之后主动DAMP |
| vio-loss | 进入POLICY后停止VIO更新；策略因VIO过期退出到DAMP，不能继续发送位置控制目标 |
| nan | 非有限策略输出导致DAMP，不能把NaN发送到电机 |
| stall | 主线程停顿时独立命令线程的heartbeat保护触发DAMP |

本次实际联调记录：`deploy_artifacts/contact_dds_normal_1uEH62/runner.log`。DDS低状态约197Hz、CRC错误0、指令约500Hz；在 `INTERP` 阶段倾角136°超过46°，进入 `DAMP` 并发送1秒阻尼后退出。**normal 未通过，尚未进入 POLICY，VIO-loss/NaN/stall 全链路故障场景也未验证。** 当前模拟吊带只约束根位置，不能据此认定真实吊具或起始姿态过渡可行。先解决仿真中姿态过渡，再重复上述场景；不得放宽倾角阈值来获得通过结果。

另需实测低状态断流、遥控L2+B、SIGINT、SIGTERM、重复控制器、型号改变及温度/速度异常。已有单人框架的相关逻辑保留，不能仅凭它以前的测试就把新contact链路标为通过。

DDS模拟器生成的VIO是**模拟真值**，校准JSON带 `simulation_only=true`，domain0会拒绝使用。测试时不触碰真实对象；动态largebox在MuJoCo中。

本步骤任何失败，停下排查，不转真机。静态contact standby是新的策略分布，尤其要验证起始/结束阶段有物体、无物体、物体摆放偏差三种情况。

## 4. 接入实际 VIO：当前需要你补齐的接口

需要先确定 VIO产品/算法、输出topic或API、输出频率、相机到pelvis外参、时间戳时钟。**程序没有假设机器人已提供这些数据，也没有零位置/零速度fallback。**

VIO适配进程在部署PC上持续发UDP到 `127.0.0.1:15151`，推荐至少50Hz。每包JSON：

| 字段 | 必需含义 |
|---|---|
| schema | 固定 `contact-vio-v1` |
| frame_id | 固定 `omomo/unitree_g1/sub7_largebox_016` |
| seq | 每个新测量递增；不因程序重启隐式清零继续跑 |
| acquisition_monotonic_ns | **采样时刻**映射到本PC `time.monotonic_ns()` 时钟；不是发送时刻 |
| tracking_valid | 实际跟踪状态，布尔值；失跟必须false |
| root_pos_w | pelvis/root link原点在参考世界坐标系中的xyz，米 |
| root_lin_vel_w | 同一原点、同一世界坐标系线速度，m/s |

不要把相机位置或机体坐标系速度直接填进去。若VIO输出相机位姿 `T_WC`，已标定相机在pelvis中的外参 `T_BC`：

- `T_WB = T_WC × inverse(T_BC)`，随后把VIO世界系变换到参考世界系。
- 相机原点线速度需扣除刚体转动引起的杆臂速度：`v_WB = v_WC − ω_W × (R_WB · t_BC)`。
- 世界z为竖直向上、长度米，floor z=0。不能通过强行设成参考高度掩盖高度估计错误。
- IMU姿态和gyro来自lowstate。策略姿态使用固定yaw校准后的IMU姿态，gyro转成相同世界系。VIO世界yaw和IMU yaw必须保持一致；这个版本没有自动融合VIO姿态、没有自动重定位。

实测坐标系后创建 `vio_calibration.json`：`frame_id`、测得的 `imu_yaw_offset_rad`、`verified:true`，以及 `simulation_only:false`。不要抄模拟文件中的0作为真机标定值。

先只读监测：

```bash
python scripts/deploy/contact_vio_check.py \
  --frame-id omomo/unitree_g1/sub7_largebox_016 --seconds 10
```

启动1秒内允许等待首包；之后必须无无效样本，测量年龄≤50ms，至少40个新样本/秒。通过后退出监测程序，再启动runner；两者使用同一个UDP端口，不能同时绑定。

手工检查：站立时位置稳定；向前/左/上移动时对应轴增大；原地转身不产生假平移；停止时速度回零；遮挡/失跟时有效标志变false。拔掉VIO、断流、时间戳冻结和VIO重定位后，控制器都应退出，不能继续运动。单元测试并不能替代这些设备测试。

## 5. 真机只读确认：仍不发电机命令

以下开始需要现场操作者。使用部署PC本地终端和有线网络；先按 [DEPLOY_G1.md](docs/DEPLOY_G1.md) 确认机器人调试模式、急停/阻尼方法及吊架支持。程序不会自动 ReleaseMode/SelectMode。

```bash
export NIC=实际连接机器人的有线网卡名
python scripts/deploy/check_robot.py --iface "$NIC" --domain 0 --check-mode
```

记录并确认：mode_machine=5、完整29DOF、腰部roll/pitch确实存在、IMU单位四元数有效、静止关节速度小、温度正常、没有另一rt/lowcmd发布者。若是mode_machine=6或腰部锁定，**此入口不支持，停止**。

核对左右腿/手、电机方向、IMU轴。没有该型号资料与现场确认时，不做“试一下看看方向”的带负载动作。

## 6. 首次带电：吊架支持，仅插值/HOLD，不运行policy

清空机器人周围空间，现场操作者持遥控，吊架承重；此阶段不让机器人自主承载largebox。

```bash
python scripts/deploy/contact_g1_runner.py \
  --iface "$NIC" --domain 0 --stage air --require-check-mode \
  --artifacts "$ART" --clip "$ART/reference.npz" \
  --no-policy --log-dir deploy_logs/contact_air_hold
```

程序先做状态、型号、其他控制器检查并显示确认提示。确认后进入PASSIVE阻尼。遥控器按钮必须先释放再按：

1. `L2+up`：从实测关节位置平滑插值到片段起始姿态，随后HOLD。
2. 观察方向、关节目标误差、抖动、温度和力矩需求；HOLD不能自主平衡，吊架持续承重。
3. `R1+A`在`--no-policy`下必须被拒绝。
4. `L2+B`进入不可恢复的DAMP；再启动需重启程序。
5. `Ctrl+C`触发退出阻尼。进程最终退出后不再发命令；机器人固件此后的行为需要现场确认，不能把进程停止当作机械急停。

任何异常立即停止。不得同时运行原单人runner、bridge runner或其他低层控制器。

## 7. 策略上机前的审核记录

复制 `hardware_review.example.json` 到实验记录目录。它绑定ONNX、参考、配置的SHA256。现场负责人逐项附上对应报告后填写：

- `robot_variant="g1_29dof"`、`reviewer`。
- `vio_frame_calibrated`：真实设备坐标/外参/时间同步验证。
- `shared_core_sim_pass`：共用控制核心仿真和模型扰动检查。
- `dds_fault_tests_pass`：通信/输入/线程故障均正确退出。
- `transitions_and_standby_pass`：起步、接触、结束、静态保持验证。
- `hoi_task_success`：物体任务指标确实达标；**当前checkpoint这一项不通过**。

domain0策略模式会检查以上记录及hash；`--no-policy`允许单独完成第6步。没有“忽略VIO”或“忽略腰部型号”开关。记录是人的实验结论，不是程序替你证明物理安全。

## 8. 条件全部满足后，逐阶段运行策略

这是后续步骤，不是当前即可跳过审核运行的命令。

```bash
python scripts/deploy/contact_g1_runner.py \
  --iface "$NIC" --domain 0 --stage feet --require-check-mode \
  --artifacts "$ART" --clip "$ART/reference.npz" \
  --vio-calibration /实际路径/vio_calibration.json \
  --contact-review /实际路径/contact_review.json \
  --log-dir deploy_logs/contact_feet
```

按顺序操作：

1. PASSIVE → `L2+up` → INTERP → HOLD；吊架支持，达到参考起始姿态。
2. `R1+A` → STANDBY_PRE。策略开始控制但参考时间不前进；不能假设这一步天然稳定。操作者按已验证的吊架流程逐步让足部承载。
3. 稳定至少3秒、近期tilt误差<8°、关节速度<1rad/s后，新的 `R1+X` 才能开始POLICY；不满足时程序拒绝。
4. 片段结束进入STANDBY_POST，先确认末端保持稳定，再按已验证流程重新由吊架承重，最后`L2+B`。
5. 异常随时`L2+B`；DAMP不是可以自动重试的状态。保存日志、排查后重新启动。

阶段依次为受支撑的检查 → `feet` → `slack` → `ground`，每次改变阶段都要重新记录结果。air/feet保护阈值更严，完整搬运可能被拒绝；**不应为了跑完而直接放宽阈值**。可先准备专用静态/短片段，但改片段后需重新验证并更新hash。

物体摆放：按照该片段的参考初始物体位置/姿态，结合校准后的参考世界系落地定位。质量、尺寸、摩擦和把持方式需与实验模型核对。当前模型largebox质量2kg、几何缩放0.83属于仿真假设，不能当成真实箱子的测量值。手和夹具不同也会改变接触；当前policy动作只有G1的29关节，不控制独立手指抓握。

## 9. 最终验收：测量任务结果，不能只看“没倒”

每次保留：checkpoint/config/hash、真实箱子参数、VIO标定与原始数据、runner console/npz、`contact_trace_*.jsonl`、同步视频、故障和人工干预记录。

根/关节误差可直接评估：

```bash
python scripts/deploy/contact_real_report.py \
  --trace deploy_logs/某次实验/contact_trace_实际时间.jsonl \
  --reference "$ART/reference.npz" \
  --output /tmp/contact_real_metrics.json
```

没有物体实测数据时，报告明确标记物体未验证、`tracking_criteria_pass=null`。

若有独立相机/动捕的物体测量，增加 `--object-track object_measurements.jsonl`。每行提供与trace同一PC单调时钟的 `t`、参考世界系的 `pose_wxyz=[x,y,z,w,x,y,z]`，最大对时差20ms。若要自动计算接触成功，还需实测二值 `actual_contacts[10]`，顺序为左腕/右腕/左踝/右踝/pelvis × terrain/object；不能填期望标签或世界模型预测来冒充实测接触。

完整策略帧序列、平均root误差<0.2m、物体位置误差<0.15m、姿态误差<30°、期望物体接触覆盖率≥0.9，才满足当前任务指标；同时必须检查runner无意外ABORT、无人工扶持搬运、无明显滑落/碰撞、结束站立稳定。自动指标不能替代这部分日志/视频审查。至少重复多次并记录每次结果，不只挑一段成功视频。

当前没有物体位姿测量和接触传感方案，所以最终HOI成功率尚不能从VIO单独得到。

## 文件与复用边界

- `src/supermimic/deploy/contact.py`：共用ContactCore、ONNX接口、VIO接收/验证。
- `scripts/deploy/contact_g1_runner.py`：将contact适配接入既有runner。
- `scripts/deploy/g1_runner.py`：增加可选runtime钩子；默认单人路径保留。
- `scripts/deploy/contact_dds_sim.py` / `test_contact_dds.sh`：loopback DDS仿真与故障注入。
- `scripts/deploy/contact_vio_check.py`：只读VIO检查。
- `scripts/deploy/contact_real_report.py`：实测轨迹评分，缺失证据不判成功。

VIO具体驱动适配需要你提供产品/算法和数据接口后才能完成。当前这套协议和runner已经就绪，但不声称实际设备适配、真机控制或全部部署门禁已经通过。
