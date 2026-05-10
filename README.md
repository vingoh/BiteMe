# BiteMe

双 Agent 问答系统。给定任意内容（代码仓库、文档、简历），
一个 Agent 负责提问，一个负责回答，人类可旁观或接管任意一侧。

## 安装

```bash
pip install -e ".[dev]"
cp .env.example .env
# 填入 OPENAI_API_KEY
```

## 快速开始

```bash
# 小文件（自动 direct 策略）
biteme run ./my-resume.md --mode interview --hitl answerer

# 大型代码仓（先建索引）
biteme index ./my-repo
biteme run ./my-repo --mode learn --turns 15

# 查看历史会话
biteme list

# 恢复中断的会话
biteme resume <session-id>
```

## HITL 选项

| `--hitl` | 效果 |
|----------|------|
| `none`（默认）| 纯观察，两侧均为 AI |
| `questioner` | 人类控制提问侧 |
| `answerer` | 人类控制回答侧 |
| `both` | 两侧均由人类输入 |
