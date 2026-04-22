# BiteMe — 双 Agent 项目理解助手

## 概述

BiteMe 是一个双 Agent 协作系统，通过"提问-解答"对话帮助用户深入理解项目内容。一个 Agent 负责提问，另一个负责解答，用户默认旁观但可随时介入引导方向。

## 核心角色

### Question Planner
- **职责**：在对话初期基于项目内容规划提问方向和问题列表；随对话推进动态更新计划
- **输入**：项目内容 + 用户可选的关键字/关注点
- **输出**：有序的问题计划（question plan），包含主题、优先级、依赖关系
- **何时更新**：每轮对话结束后评估是否有新方向需要探索，或已有问题需要调整优先级

### Questioner Agent（提问者）
- **职责**：根据 question plan 和对话历史，生成有深度、有逻辑的问题
- **输入**：question plan + 共享 memory（项目内容、对话历史）
- **私有 memory**：已问过的问题、追问链、当前探索方向
- **行为**：按 plan 顺序提问，但可根据解答者的回答灵活追问

### Answerer Agent（解答者）
- **职责**：基于项目内容给出准确、结构化的解答
- **输入**：共享 memory（项目内容、对话历史）
- **私有 memory**：已回答的问题、引用过的代码/文档片段、解答置信度
- **行为**：解答时引用具体代码/文档位置，对不确定的内容明确标注
- **模拟面试模式**：用户回答后，Answerer 仍生成自己的参考答案供对比，但共享 memory 中记录的是用户的原始回答（非 agent 答案），确保总结和评估基于用户真实表现

### Summarizer Agent（总结者）
- **职责**：对话结束后生成结构化总结
- **输入**：共享 memory（项目内容、完整对话历史、question plan 及完成状态）
- **行为**：梳理对话脉络，提炼关键知识点，标注未覆盖方向，在模拟面试模式下额外给出用户理解度评估

### Orchestrator（编排层）
- **职责**：最小编排 — 启动/终止对话、处理用户介入、管理对话轮次
- **不干预**：不控制对话内容、不决定谁说什么
- **用户介入**：用户可随时插入问题或调整方向，orchestrator 将用户输入注入对话流

## Memory 架构

### 共享 Memory
- 项目原始内容（代码、文档、文件结构）
- 完整对话历史
- Question plan（由 planner 写入，两个 agent 读取）

### 私有 Memory
- Questioner：已问问题列表、追问链、当前探索方向
- Answerer：已回答问题、引用片段、置信度标注

## 输入处理

支持以下输入方式，统一转换为结构化项目内容存入共享 memory：

| 输入类型 | 处理方式 |
|---------|---------|
| 纯文本/文档 | 直接存储 |
| 本地文件夹路径 | 递归读取文件，识别项目介绍文件（README 等） |
| 代码仓库路径 | 克隆后递归读取 |
| 文件上传 | 解析文件内容（PDF/Markdown/文本） |
| URL | 抓取页面内容 |

## 交互流程

```
1. 用户输入项目内容 + 可选关键字
2. 内容处理器 → 结构化存入共享 memory
3. Question Planner → 生成初始 question plan
4. 循环：
   a. Questioner 根据 plan + 对话历史 → 提问
   b. Answerer 基于项目内容 → 解答
   c. Question Planner 评估 → 更新 plan
   d. 用户可随时介入（插入问题/调整方向）
   e. 检查终止条件（plan 完成 / 用户主动结束 / 达到轮次上限）
5. Summarizer Agent → 生成对话总结
```

## 用户交互模式

**混合模式**：默认旁观两个 agent 对话，可随时介入：
- 插入自己的问题
- 指定新的探索方向
- 跳过当前话题
- 终止对话

**模拟面试模式**：用户不仅可提问，也可作为回答方参与：
- 用户可选择接管 Answerer 的角色，自己回答 Questioner 的问题
- Agent 根据项目内容评估用户回答的准确性和完整性，给出反馈
- 用户回答后可选择继续下一个问题，或查看标准答案对比
- 适用于自测对项目的理解程度

**关键字引导**：用户在初期可选择输入关键字，question planner 围绕这些关键字规划问题；不输入则由 planner 自主探索。

## 技术选型

- **语言**：Python
- **核心**：纯 LLM API + 自定义编排（无框架依赖）
- **模型**：可配置，默认 OpenAI 兼容 API
- **界面**：CLI 优先（Rich 库），后期扩展 Web UI
- **插件化**：后期考虑接入 Claude Code / Cursor 插件

## 项目结构

```
biteme/
├── src/
│   ├── agents/
│   │   ├── base.py              # Agent 基类
│   │   ├── questioner.py        # 提问者 Agent
│   │   ├── answerer.py          # 解答者 Agent
│   │   ├── planner.py           # Question Planner
│   │   └── summarizer.py        # 总结者 Agent
│   ├── memory/
│   │   ├── shared.py            # 共享 Memory
│   │   └── private.py           # 私有 Memory
│   ├── input/
│   │   ├── processor.py         # 输入统一处理
│   │   ├── file_reader.py       # 文件/文件夹读取
│   │   ├── repo_reader.py       # 代码仓库读取
│   │   └── url_fetcher.py       # URL 抓取
│   ├── orchestrator.py          # 编排层
│   ├── cli.py                   # CLI 入口
│   └── config.py                # 配置管理
├── tests/
├── pyproject.toml
└── README.md
```

## 终止条件

对话在以下任一条件满足时终止：
1. Question plan 中所有问题已回答
2. 用户主动终止
3. 达到配置的最大轮次上限

## 输出

对话结束后由 Summarizer Agent 生成总结，包含：
- 探索过的主题列表
- 关键发现/知识点
- 用户可能还想深入的方向（基于 plan 中未覆盖的部分）
- 模拟面试模式下：用户理解度评估、薄弱环节提示
