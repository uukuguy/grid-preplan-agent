# 电网设备故障预案l辅助决策智能体

An AI agent framework for power grid dispatch decision-making based on standard operation preplans.

基于标准预案的电网调度辅助决策智能体框架

> 基于AutoGen + LangGraph + Smolagents的电网设备故障预案辅助决策智能体，实现预案自动解析、执行和报告生成。

## 项目背景

电网调度中心面对设备故障、运行异常等场景，需要快速依据 标准预案库 做出调度决策。
传统人工流程存在以下痛点：

- 查阅预案耗时长；
- 人工计算代入易出错；
- 缺乏透明可追溯性。

grid-preplan-agent 旨在通过 AutoGen + LangGraph/Smolagents 框架，
实现 预案文本 → JSON Schema → 智能体执行 → 报告输出 的自动化闭环。

## 项目目标

- 支持**标准预案文本**的自然语言编写（面向调度员，无需代码）。
- 自动解析为**Plan JSON DSL**，统一描述步骤（rag/tool/compute）。
- 提供 **LangGraph Executor** 与 **Smolagents Executor** 两种执行器。
- 由 **AutoGen Controller** 负责预案调度、路由执行。
- 生成 **透明可审计的决策报告**（背景、数据、公式、结论）。

## 系统特色

- **预案自然语言化**：调度人员只需用规范化文本编写预案，无需编程
- **自动解析执行**：系统自动将预案文本转换为Plan JSON并执行
- **双执行引擎**：LangGraph处理线性预案，Smolagents处理复杂场景  
- **智能路由**：AutoGen控制器根据复杂度自动选择合适的执行器
- **透明可追溯**：生成包含完整推导过程的专业决策报告

## 🏗️ 系统架构

```java
用户输入 (告警/场景)
        ↓
AutoGen Controller (中控智能体)
   ├─ 预案解析器 (文本 → Plan JSON)
   ├─ 复杂度判断
   │    • 线性简单 → LangGraph Executor
   │    • 线性简单 → Smolagents Executor  
   │    • 复杂分支 → AutoGen Sub-agents
   ↓
执行器 (LangGraph / Smolagents)
        ↓
执行结果 (变量值 + 公式推导 + 工具来源)
        ↓
Decision Agent (生成Markdown/JSON报告)
        ↓
调度员查看/确认
```

## 📦 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境

```bash
cp .env.example .env
# 编辑.env文件，配置OpenAI API密钥
```

### 3. 运行演示

```bash
python main.py --mode demo
```

### 4. 交互模式  

```bash
python main.py --mode interactive
```

## 📑 预案示例

```ruby
# 电网调度标准预案写作规范
# 1. 步骤必须编号，并包含【输入】和【输出】。
# 2. 变量必须在“变量定义”中定义，保持与公式一致。
# 3. 每个变量需注明单位。
# 4. 计算公式必须用 LaTeX 或清晰表达式。

设备故障直流限额计算预案

步骤：
1. 查询停运的<设备>影响<直流线路>，并判定其位于送端还是受端。
   输入：<设备>, <直流线路>
   输出：side_info
2. 计算电网运行方式：送端限额与受端限额的最小值。
   输入：P_max_send, P_max_receive
   输出：P_max_net
3. 计算设备传输能力：换流器运行容量 × 电流限额 × 换流器个数。
   输入：P_max_convert, F_current, N_convert
   输出：P_dcsystem
4. 计算设备故障直流限额：min(P_max_net, P_dcsystem)。
   输入：P_max_net, P_dcsystem
   输出：P_max_device

变量定义：
- 送端电网限额：$P_{max\_send}$ （MW）
- 受端电网限额：$P_{max\_receive}$ （MW）
- 电网运行方式传输限额：$P_{max\_net} = \min(P_{max\_send}, P_{max\_receive})$
- 换流器运行容量：$P_{max\_convert}$ （MW）
- 电流限值：$F_{current}$ （kA）
- 换流器个数：$N_{convert}$ （count）
- 设备传输能力：$P_{dcsystem} = P_{max\_convert} \times F_{current} \times N_{convert}$
- 设备故障直流限额：$P_{max\_device} = \min(P_{max\_net}, P_{dcsystem})$
```

## 🔧 Plan JSON 输出（解析器生成）

```json
{
  "plan_id": "dc_limit_fault",
  "description": "设备故障直流限额计算预案",
  "steps": [
    {"id":"step1","type":"rag","query":"判定停运设备影响直流线路送/受端","inputs":{"device":"<设备>","line":"<直流线路>"},"output":["side_info"]},
    {"id":"tool_send","type":"tool","tool_name":"query_send_limit","inputs":{"line":"{hvdc_line}"},"output":["P_max_send"]},
    {"id":"tool_recv","type":"tool","tool_name":"query_recv_limit","inputs":{"line":"{hvdc_line}"},"output":["P_max_receive"]},
    {"id":"compute_net","type":"compute","formula":"min(P_max_send,P_max_receive)","inputs":{"P_max_send":"{P_max_send}","P_max_receive":"{P_max_receive}"},"output":["P_max_net"]},
    {"id":"compute_final","type":"compute","formula":"min(P_max_net,P_dcsystem)","inputs":{"P_max_net":"{P_max_net}","P_dcsystem":"{P_dcsystem}"},"output":["P_max_device"]}
  ]
}
```

## 📊 执行效果示例（直流限额）

### 输入场景

> 天哈一线停运，影响天中直流

### 执行结果

> $P_{max_send} = 3000$ MW
>
> $P_{max_receive} = 2800$ MW
>
> $P_{max_net} = min(3000,2800) = 2800$ MW
>
> $P_{dcsystem} = 3200$ MW
>
> $P_{max_device} = min(2800, 3200) = 2800$ MW

### 报告输出

> 直流限额计算结果
>
> 背景：天哈一线停运，需评估直流输电能力
>
> 数据来源：调度监控系统 & 稳规文档
>
> 推导过程：如上公式代入计算
>
> 结论：直流限额 = 2800 MW，约束主因 = 受端电网限额

## 🚀 路线图

- MVP：接入直流限额预案，打通端到端流程
- 扩展：加入断面校核、备用容量预案
- 生产化：全量预案接入，日志与回归测试
- 智能化升级：增强解析器与 **Controller** 智能化能力

## 📜 License

MIT
