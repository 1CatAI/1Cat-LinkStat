# 1CatLinkStat

`1CatLinkStat` 是一个面向 Linux NVIDIA 平台的实时 GPU 终端监控工具。它的启动方式尽量接近 `nvtop`，可以直接从终端一键启动，同时补上 `nvtop` 在 V100 这类平台上通常没有提供的 NVLink 可视化能力。

<img width="1978" height="1549" alt="image" src="https://github.com/user-attachments/assets/fe562895-4571-47fd-a745-ff94694ef613" />


## 版本信息

- 程序版本：`0.1.0`
- Debian 包版本：`0.1.0-1`
- 最后更新时间：`2026-03-25`

## 功能概览

`1CatLinkStat` 当前支持：

- 实时显示 GPU 利用率、显存占用、温度、功耗、时钟
- 实时显示 PCIe 代际、链路宽度、RX、TX
- 实时显示 NVLink 活跃通道数
- 实时显示每条 NVLink 的配置速率
- 实时显示总 NVLink 配置带宽
- 在平台支持时，通过 DCGM 实时显示每张 GPU 的 NVLink RX/TX 总带宽
- 基于 `nvidia-smi topo -m` 的 NVLink Matrix 拓扑模式
- 展开式逐链路 NVLink 明细模式
- NVLink 状态、速率、数据源变化事件跟踪
- GPU 进程列表，包括计算/图形类型、SM 占用、GPU 显存、CPU、主机内存

运行时不依赖第三方 Python 包，直接通过 `ctypes` 调用 `libnvidia-ml.so.1`。

## 启动命令

安装完成后，可以直接使用以下任一命令启动：

```bash
1catlinkstat
1CatLinkStat
```

其中主入口命令是 `1catlinkstat`。

## TUI 模式

当前支持三种 NVLink 展示模式：

- `summary`：紧凑总览模式
- `expanded`：逐链路明细模式
- `matrix`：NVLink 拓扑矩阵模式，并附带每张 GPU 的实时 RX/TX 条形图

```bash
1catlinkstat --links matrix
```

实时监控界面当前包括：

- 全屏 TUI 主界面
- 更长的 `GPU`、`MEM`、`TEMP`、`POW`、`NVRX`、`NVTX` 进度条
- `Activity` 历史趋势区
- `Processes` 进程区
- 状态栏
- 可选的最近 NVLink 事件提示行

## 使用示例

```bash
1catlinkstat
1catlinkstat --interval 0.5
1catlinkstat --links expanded
1catlinkstat --links matrix
1catlinkstat --events 10
1catlinkstat --once
1catlinkstat --once --json
1catlinkstat --no-color
1catlinkstat --no-bars
1catlinkstat --no-clear
```

## 命令行参数

```text
--interval <seconds>   刷新间隔，默认 1.0 秒
--once                 只输出一次快照后退出
--json                 以 JSON 输出，必须和 --once 一起使用
--links <mode>         NVLink 展示模式，可选 summary | expanded | matrix
--events <count>       在实时模式中显示最近的 NVLink 变化事件数量
--no-color             禁用 ANSI 颜色
--no-bars              禁用条形进度条
--no-clear             刷新时不清屏
```

## 使用 Python 安装

```bash
python3 -m pip install .
1catlinkstat
```

## 使用 APT 安装

仓库已经包含 Debian 打包配置，可以直接构建 `.deb` 并用 `apt` 安装。

1. 构建安装包：

```bash
bash packaging/build-apt-package.sh
```

2. 安装生成的 `.deb`：

```bash
sudo apt install ./dist/1catlinkstat_0.1.0-1_all.deb
```

3. 启动程序：

```bash
1catlinkstat
```

这样就可以像 `nvtop` 一样一条命令启动，但额外获得 NVLink 的监控和矩阵视图。

## V100 上的真实 NVLink RX/TX

在 Tesla V100 上，NVML 以及 `nvidia-smi nvlink -gt` 通常可以提供链路状态和配置速率，但往往不能直接提供实时 NVLink 吞吐。因此 `1CatLinkStat` 会优先使用 DCGM 获取真实的 NVLink RX/TX 总量：

```bash
sudo apt-get install -y datacenter-gpu-manager
sudo systemctl start nvidia-dcgm
1catlinkstat
```

当 DCGM 可用时，界面中会显示每张 GPU 的真实 NVLink RX/TX 总带宽。

目前在测试机上已经确认的 V100 能力边界如下：

- 每张 GPU 的真实 NVLink RX/TX 总量：支持，来源为 DCGM
- 每条 NVLink 的真实实时 RX/TX：当前这套 V100 环境不支持
- 每条 NVLink 的配置速率：支持
- NVLink 活跃通道数：支持
- NVLink Matrix 拓扑：支持

## 回退行为

如果当前平台拿不到真实实时吞吐，`1CatLinkStat` 仍会保持实时 NVLink 监控，不会直接失效。此时程序会退回到轮询和变化检测模式，继续提供：

- 活跃链路数量
- Link up/down 状态
- 每条链路的配置速率
- 总配置带宽
- 最近的 link 数、速率、数据源变化事件

## 已验证环境

当前已验证环境包括：

- Ubuntu `24.04.2`
- NVIDIA 驱动 `580.126.09`
- `Tesla V100-SXM2-16GB`
- `nvtop`
- `datacenter-gpu-manager`

## 开发与测试

运行测试：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
