# BiteMe Roadmap

## P0 — 核心功能（MVP）

最小可运行版本，实现"输入项目 → 两个 agent 对话 → 用户旁观"的基本闭环。

- [ ] Agent 基类（`base.py`）：统一的 LLM 调用、prompt 模板、消息格式
- [ ] 共享 Memory（`shared.py`）：存储项目内容、对话历史
- [ ] 输入处理 — 本地文件夹路径读取（`file_reader.py`）：递归读取文件，识别 README 等介绍文件
- [ ] 输入处理 — 纯文本/文档（`processor.py`）：直接存储文本内容
- [ ] Questioner Agent（`questioner.py`）：基于项目内容生成问题
- [ ] Answerer Agent（`answerer.py`）：基于项目内容回答问题
- [ ] Orchestrator（`orchestrator.py`）：启动/终止对话、管理轮次循环
- [ ] CLI 入口（`cli.py`）：基本命令行交互，输入项目路径、启动对话
- [ ] 配置管理（`config.py`）：模型 API 配置、最大轮次等

## P1 — 重要功能

让对话更有条理、用户可介入，形成完整体验。

- [ ] Question Planner（`planner.py`）：初期规划问题列表，每轮更新 plan
- [ ] 私有 Memory（`private.py`）：各 agent 维护自己的状态（已问问题、引用片段等）
- [ ] 用户介入：旁观模式下随时插入问题、调整方向、跳过话题
- [ ] 关键字引导：用户输入关键字，planner 围绕其规划问题
- [ ] Summarizer Agent（`summarizer.py`）：对话结束后生成结构化总结
- [ ] 终止条件完善：plan 完成 / 用户终止 / 轮次上限

## P2 — 增强功能

模拟面试模式，让用户从旁观者变为参与者。

- [ ] 模拟面试模式：用户接管 Answerer 角色自己回答
- [ ] Answerer 参考答案：用户回答后展示 agent 的参考答案对比
- [ ] 回答评估：agent 评估用户回答的准确性和完整性
- [ ] Memory 记录用户回答：共享 memory 记录用户原始回答而非 agent 答案
- [ ] 面试模式总结：理解度评估、薄弱环节提示

## P3 — 扩展功能

更多输入方式和输出形式。

- [ ] 输入处理 — 代码仓库路径（`repo_reader.py`）：克隆后递归读取
- [ ] 输入处理 — 文件上传：解析 PDF/Markdown 等文件
- [ ] 输入处理 — URL 抓取（`url_fetcher.py`）：抓取网页内容
- [ ] Web UI：浏览器界面，更丰富的展示和交互
- [ ] 对话导出：将对话和总结导出为 Markdown/JSON

## P4 — 生态集成

- [ ] Claude Code 插件接入
- [ ] Cursor 插件接入
- [ ] 可配置多模型支持（切换不同 LLM 后端）
