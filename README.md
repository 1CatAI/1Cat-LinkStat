# 1CatLinkStat

`1CatLinkStat` 是一个面向 Linux NVIDIA 平台的实时 GPU 终端监控工具。它的交互方式尽量贴近 `nvtop`，可以直接从终端一键启动，同时补上 `nvtop` 在很多 V100 / NVLink 场景下没有直接提供的链路信息、矩阵视图和实时 NVLink 带宽显示。

## 版本信息

- 程序版本：`0.1.1`
- Debian 包版本：`0.1.1-1`
- 最后更新时间：`2026-03-26`

## 0.1.1 更新内容

`0.1.1` 这一版相对 `0.1.0` 主要补了三类内容：

- 新增 `1catlinkstat-bench` 微型带宽压测工具
- 压测工具以 CUDA 源码形式随仓库和安装包一起提供，首次运行会自动编译
- README 补全了新系统安装步骤、压测工具用法、验证方式和已知限制说明

另外，`0.1.0-2` 修掉的 DCGM 首帧采样问题也已经包含在 `0.1.1` 中，因此 V100 上的总 `NVLink RX/TX` 读数会比早期包更可靠。

## 功能概览

`1CatLinkStat` 当前支持：

<img width="1654" height="1351" alt="image" src="https://github.com/user-attachments/assets/aa9cb633-818a-46f5-8cab-5663f9cb5b7d" />

<img width="1236" height="665" alt="image" src="https://github.com/user-attachments/assets/14a19fed-31a4-4093-bb22-5cf6bd22a983" />

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
- 自带 `1catlinkstat-bench` 微型 NVLink 带宽压测工具

运行时不依赖第三方 Python 包，主监控程序直接通过 `ctypes` 调用 `libnvidia-ml.so.1`。

## 启动命令

安装完成后，可以直接使用以下任一命令启动：

```bash
1catlinkstat
1CatLinkStat
```

其中主入口命令是 `1catlinkstat`。

常用启动方式和示例：

```bash
# 默认实时监控
1catlinkstat

# 更快刷新
1catlinkstat --interval 0.5

# 查看 NVLink 明细
1catlinkstat --links expanded

# 查看 NVLink matrix 模式
1catlinkstat --links matrix

# 显示最近的 NVLink 变化事件
1catlinkstat --links matrix --events 10

# 只输出一次后退出
1catlinkstat --once

# 输出一次 JSON 快照
1catlinkstat --once --json

# 关闭颜色
1catlinkstat --no-color
```

如果你只是第一次验证安装是否成功，推荐先运行：

```bash
1catlinkstat --once
1catlinkstat --once --links matrix
```

## TUI 模式

当前支持三种 NVLink 展示模式：

- `summary`：紧凑总览模式
- `expanded`：逐链路明细模式
- `matrix`：NVLink 拓扑矩阵模式，并附带每张 GPU 的实时 RX/TX 条形图

实时监控界面当前包括：

- 全屏 TUI 主界面
- 更长的 `GPU`、`MEM`、`TEMP`、`POW`、`NVRX`、`NVTX` 进度条
- `Activity` 历史趋势区
- `Processes` 进程区
- 状态栏
- 可选的最近 NVLink 事件提示行

## 监控工具使用示例

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

## 监控工具命令行参数

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

## 在新系统上安装 1CatLinkStat

下面是一套推荐的完整安装顺序，适合新的 Ubuntu / Debian 类 Linux 系统。

### 1. 准备 NVIDIA 驱动

首先确认系统已经正确安装 NVIDIA 驱动，并且 `nvidia-smi` 正常：

```bash
nvidia-smi
```

如果这一步不通，`1CatLinkStat` 和压测工具都无法正常工作。

### 2. 安装可选依赖

如果你希望 `1CatLinkStat` 显示真实的总 `NVLink RX/TX`，建议安装并启动 DCGM：

```bash
sudo apt-get update
sudo apt-get install -y datacenter-gpu-manager
sudo systemctl enable --now nvidia-dcgm
```

如果你希望使用新加入的微型压测工具 `1catlinkstat-bench`，还需要保证系统中有 `nvcc`。最简单的方式之一是：

```bash
sudo apt-get install -y nvidia-cuda-toolkit build-essential
```

如果你的环境已经安装了 NVIDIA CUDA Toolkit，只要 `nvcc` 在 `PATH` 中即可。

### 3. 安装 Debian 包

如果你已经拿到了构建好的 Debian 包，例如：

```text
1catlinkstat_0.1.1-1_all.deb
```

那么直接安装：

```bash
sudo apt install ./1catlinkstat_0.1.1-1_all.deb
```

安装完成后，会得到以下命令：

- `1catlinkstat`
- `1CatLinkStat`
- `1catlinkstat-bench`
- `1CatLinkStat-bench`

### 4. 安装后验证

安装完成后，建议立刻跑一遍基础验证：

```bash
1catlinkstat --once
1catlinkstat --once --links matrix
1catlinkstat --once --json
```

如果你已经安装并启动了 DCGM，还可以确认数据源是否是 `dcgm-total`：

```bash
1catlinkstat --once --json
```

检查输出中的：

```text
"nvlink_metrics_source": "dcgm-total"
```

### 5. 从源码安装

如果你不走 Debian 包，也可以直接本地安装：

```bash
python3 -m pip install .
1catlinkstat
```

## 从仓库构建 Debian 包

仓库自带 Debian 打包配置，可以直接在 Linux 环境构建：

```bash
bash packaging/build-apt-package.sh
```

生成的文件会出现在：

```text
dist/1catlinkstat_0.1.1-1_all.deb
```

安装方式：

```bash
sudo apt install ./dist/1catlinkstat_0.1.1-1_all.deb
```

## 微型带宽压测工具：1catlinkstat-bench

`1catlinkstat-bench` 是 `0.1.1` 新加入的微型 NVLink / P2P 带宽压测工具。它适合做几件事：

- 快速验证两张 GPU 之间是否真的存在 P2P / NVLink 可达路径
- 快速测一组 GPU 的单向复制带宽
- 和 `1catlinkstat --links matrix` 联动，边压测边观察实时 `RX/TX`
- 在新机器验收时做简单健康检查

### 工具的工作方式

这个工具本身是仓库内自带的一份 CUDA 源码。安装 `1CatLinkStat` 后，运行：

```bash
1catlinkstat-bench
```

它会先检查本机是否存在 `nvcc`。如果存在，就会把仓库内打包进去的 CUDA 源码自动编译成一个本地缓存二进制，然后再执行压测。

默认缓存位置：

```text
~/.cache/1catlinkstat/1catlinkstat-nvlink-bench-0.1.1
```

如果源码版本没变，后续运行不会每次都重新编译。

### 压测工具的前置要求

要运行 `1catlinkstat-bench`，至少需要：

- NVIDIA 驱动正常
- 至少两张 NVIDIA GPU
- 这两张 GPU 之间存在 CUDA P2P 可达路径
- 系统里有 `nvcc`

### 先列出可压测的 GPU 对

推荐第一步先看有哪些 GPU 对支持双向 peer access：

```bash
1catlinkstat-bench --list
```

这个命令会打印：

- 检测到的 GPU 列表
- 支持双向 peer access 的 GPU 对

### 默认用法

如果不传 `--src` 和 `--dst`，工具会自动选择第一组支持双向 peer access 的 GPU 对，并顺序测试：

- `GPUA -> GPUB`
- `GPUB -> GPUA`

命令：

```bash
1catlinkstat-bench
```

默认参数：

- 每次传输大小：`256 MiB`
- 预热次数：`20`
- 计时次数：`200`

### 指定单向测试

如果你只想测一个方向，例如 `GPU0 -> GPU1`：

```bash
1catlinkstat-bench --src 0 --dst 1
```

### 调整传输块大小和迭代次数

例如：

```bash
1catlinkstat-bench --src 0 --dst 1 --size-mib 256 --warmup 20 --iters 400
```

参数说明：

- `--size-mib`：单次传输块大小，单位 MiB
- `--bytes`：如果你希望按字节数精确指定块大小，也可以直接传字节
- `--warmup`：预热迭代次数
- `--iters`：真正参与计时的迭代次数

### 压测所有可用 GPU 对

如果机器里不止两张卡，想把所有双向 peer-accessible 的 GPU 对都跑一遍：

```bash
1catlinkstat-bench --all-pairs
```

### JSON 输出

如果你想把压测结果喂给脚本或记录系统：

```bash
1catlinkstat-bench --json
```

或者：

```bash
1catlinkstat-bench --src 0 --dst 1 --json
```

### 只编译，不立即运行

如果你只想先把工具编译出来，确认路径：

```bash
1catlinkstat-bench --build-only
```

如果你想强制重新编译：

```bash
1catlinkstat-bench --rebuild
```

### 推荐联动方式

最直观的做法是开两个终端：

第一个终端：

```bash
1catlinkstat --links matrix
```

第二个终端：

```bash
1catlinkstat-bench --src 0 --dst 1
```

这样你能在 `matrix` 视图里直接看到 `NVRX/NVTX` 和矩阵区带宽条一起跳起来。

## V100 上的真实 NVLink RX/TX

在 Tesla V100 上，NVML 以及 `nvidia-smi nvlink -gt` 通常可以提供链路状态和配置速率，但往往不能直接提供实时 NVLink 吞吐。因此 `1CatLinkStat` 会优先使用 DCGM 获取真实的 NVLink RX/TX 总量：

```bash
sudo apt-get install -y datacenter-gpu-manager
sudo systemctl enable --now nvidia-dcgm
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
- `nvcc`

## 开发与测试

运行 Python 单元测试：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

如果要在测试机上联调压测工具，可以同时运行：

```bash
1catlinkstat --links matrix
1catlinkstat-bench --src 0 --dst 1
```
