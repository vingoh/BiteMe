import pytest
from unittest.mock import MagicMock, patch
from biteme.graph.state import SessionState, Turn, KeywordScore
from biteme.graph.nodes import questioner_node, answerer_node, reviewer_node
from biteme.graph.prompts import get_prompts

def test_session_state_has_outline_field():
    state = make_state(outline=["Q1", "Q2"])
    assert state["outline"] == ["Q1", "Q2"]

def test_session_state_outline_defaults_to_empty():
    state = make_state()
    assert state["outline"] == []

def make_state(**kwargs) -> SessionState:
    defaults = dict(
        mode="learn",
        messages=[],
        current_speaker="questioner",
        hitl_flags=[],
        turn_count=0,
        max_turns=10,
        context_strategy="direct",
        source_path="/tmp/fake",
        outline=[],
        llm_reference_answer="",
        review_history=[],
    )
    defaults.update(kwargs)
    return defaults  # type: ignore

def test_questioner_node_appends_turn(tmp_path):
    state = make_state(source_path=str(tmp_path))
    (tmp_path / "a.py").write_text("def foo(): pass")

    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "What does foo do?"

    with patch("biteme.graph.nodes.ChatOpenAI", return_value=mock_llm), \
         patch("biteme.graph.nodes.create_provider") as mock_factory:
        mock_provider = MagicMock()
        mock_provider.get_overview.return_value = ["def foo(): pass"]
        mock_factory.return_value = mock_provider
        result = questioner_node(state)

    mock_provider.get_overview.assert_called_once()  # 第一轮走 get_overview，不走 retrieve
    mock_provider.retrieve.assert_not_called()
    assert result["current_speaker"] == "answerer"
    assert result["turn_count"] == 1
    assert len(result["messages"]) == 1
    assert result["messages"][0]["speaker"] == "questioner"

def test_answerer_node_always_retrieves(tmp_path):
    """answerer_node must call provider.retrieve() regardless of ReAct."""
    state = make_state(
        source_path=str(tmp_path),
        current_speaker="answerer",
        messages=[{"speaker": "questioner", "content": "What does foo do?", "retrieved_chunks": []}],
    )
    (tmp_path / "a.py").write_text("def foo(): pass")

    mock_react_agent = MagicMock()
    mock_react_agent.invoke.return_value = {
        "messages": [MagicMock(content="foo returns None.")]
    }

    with patch("biteme.graph.nodes.create_agent", return_value=mock_react_agent), \
         patch("biteme.graph.nodes.ChatOpenAI"), \
         patch("biteme.graph.nodes.create_provider") as mock_factory:
        mock_provider = MagicMock()
        mock_provider.retrieve.return_value = ["def foo(): pass"]
        mock_factory.return_value = mock_provider
        result = answerer_node(state)

    mock_provider.retrieve.assert_called_once()
    assert result["current_speaker"] == "questioner"
    assert result["messages"][-1]["speaker"] == "answerer"
    assert len(result["messages"][-1]["retrieved_chunks"]) > 0


def test_answerer_node_calls_react_agent(tmp_path):
    """AI branch should use create_agent instead of direct llm.invoke()."""
    state = make_state(
        source_path=str(tmp_path),
        current_speaker="answerer",
        messages=[{"speaker": "questioner", "content": "Explain the architecture.", "retrieved_chunks": []}],
    )
    (tmp_path / "a.py").write_text("class App: pass")

    mock_react_agent = MagicMock()
    mock_react_agent.invoke.return_value = {
        "messages": [MagicMock(content="The architecture is based on...")]
    }

    with patch("biteme.graph.nodes.create_agent", return_value=mock_react_agent) as mock_create, \
         patch("biteme.graph.nodes.ChatOpenAI") as mock_chat, \
         patch("biteme.graph.nodes.create_provider") as mock_factory:
        mock_provider = MagicMock()
        mock_provider.retrieve.return_value = ["class App: pass"]
        mock_factory.return_value = mock_provider
        result = answerer_node(state)

    mock_create.assert_called_once()
    mock_react_agent.invoke.assert_called_once()
    call_kwargs = mock_react_agent.invoke.call_args
    assert call_kwargs[1]["recursion_limit"] == 12
    assert result["messages"][-1]["content"] == "The architecture is based on..."


def test_get_prompts_learn_has_planner():
    prompts = get_prompts("learn")
    assert "planner" in prompts
    assert isinstance(prompts["planner"], str)
    assert len(prompts["planner"]) > 0

def test_get_prompts_interview_has_planner():
    prompts = get_prompts("interview")
    assert "planner" in prompts
    assert isinstance(prompts["planner"], str)
    assert len(prompts["planner"]) > 0


from biteme.graph.nodes import _parse_outline

def test_parse_outline_strips_numbering():
    text = "1. 什么是架构设计？\n2. 如何处理异常？\n3. 性能优化有哪些方法？"
    result = _parse_outline(text)
    assert result == ["什么是架构设计？", "如何处理异常？", "性能优化有哪些方法？"]

def test_parse_outline_ignores_empty_lines():
    text = "1. 问题一\n\n2. 问题二\n\n"
    result = _parse_outline(text)
    assert result == ["问题一", "问题二"]

def test_parse_outline_handles_parenthesis_numbering():
    text = "1) 问题一\n2) 问题二"
    result = _parse_outline(text)
    assert result == ["问题一", "问题二"]


from biteme.graph.nodes import planner_node

def test_planner_node_populates_outline(tmp_path):
    state = make_state(source_path=str(tmp_path), max_turns=3)

    outline_response = (
        "1. 什么是BiteMe？\n2. 它有哪些核心功能？\n"
        "3. 如何启动会话？\n4. HITL如何工作？\n5. 如何恢复会话？"
    )

    with patch("biteme.graph.nodes.create_agent") as mock_create, \
         patch("biteme.graph.nodes.stream_agent", return_value=outline_response) as mock_stream, \
         patch("biteme.graph.nodes.ChatOpenAI"), \
         patch("biteme.graph.nodes.Console"):
        result = planner_node(state)

    mock_create.assert_called_once()
    mock_stream.assert_called_once()
    assert len(result["outline"]) == 5  # max_turns(3) + 2
    assert result["outline"][0] == "什么是BiteMe？"

def test_planner_node_interview_mode(tmp_path):
    state = make_state(source_path=str(tmp_path), max_turns=2, mode="interview")

    outline_response = "1. 接口设计原则是什么？\n2. 如何处理版本兼容？\n3. 错误码规范？\n4. 认证机制？"

    with patch("biteme.graph.nodes.create_agent") as mock_create, \
         patch("biteme.graph.nodes.stream_agent", return_value=outline_response) as mock_stream, \
         patch("biteme.graph.nodes.ChatOpenAI"), \
         patch("biteme.graph.nodes.Console"):
        result = planner_node(state)

    mock_create.assert_called_once()
    mock_stream.assert_called_once()
    assert len(result["outline"]) == 4
    assert "outline" in result


def test_graph_entry_is_planner():
    from biteme.graph.graph import build_graph
    graph = build_graph(checkpointer=None)
    node_names = set(graph.nodes.keys())
    assert "planner" in node_names
    assert "questioner" in node_names
    assert "answerer" in node_names


def test_questioner_includes_outline_in_prompt(tmp_path):
    outline = ["问题一：架构设计", "问题二：数据流"]
    state = make_state(source_path=str(tmp_path), outline=outline)
    (tmp_path / "a.py").write_text("def foo(): pass")

    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "What does foo do?"

    with patch("biteme.graph.nodes.ChatOpenAI", return_value=mock_llm), \
         patch("biteme.graph.nodes.create_provider") as mock_factory:
        mock_provider = MagicMock()
        mock_provider.get_overview.return_value = ["def foo(): pass"]
        mock_factory.return_value = mock_provider
        questioner_node(state)

    call_args = mock_llm.invoke.call_args[0][0]
    human_message_content = call_args[1].content
    assert "本轮话题方向" in human_message_content
    assert "问题一：架构设计" in human_message_content

def test_questioner_no_outline_section_when_empty(tmp_path):
    state = make_state(source_path=str(tmp_path), outline=[])
    (tmp_path / "a.py").write_text("def foo(): pass")

    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "What does foo do?"

    with patch("biteme.graph.nodes.ChatOpenAI", return_value=mock_llm), \
         patch("biteme.graph.nodes.create_provider") as mock_factory:
        mock_provider = MagicMock()
        mock_provider.get_overview.return_value = ["def foo(): pass"]
        mock_factory.return_value = mock_provider
        questioner_node(state)

    call_args = mock_llm.invoke.call_args[0][0]
    human_message_content = call_args[1].content
    assert "本轮话题方向" not in human_message_content

def test_questioner_hitl_uses_outline_as_suggestion(tmp_path):
    outline = ["第一个建议问题", "第二个建议问题"]
    state = make_state(
        source_path=str(tmp_path),
        hitl_flags=["questioner"],
        outline=outline,
        turn_count=0,
    )

    interrupted_value = None

    def fake_interrupt(msg):
        nonlocal interrupted_value
        interrupted_value = msg
        raise Exception("interrupt_called")

    with patch("biteme.graph.nodes.interrupt", side_effect=fake_interrupt):
        try:
            questioner_node(state)
        except Exception:
            pass

    assert interrupted_value is not None
    assert "第一个建议问题" in interrupted_value

def test_questioner_hitl_fallback_when_outline_exhausted(tmp_path):
    state = make_state(
        source_path=str(tmp_path),
        hitl_flags=["questioner"],
        outline=["唯一问题"],
        turn_count=5,  # beyond outline range
    )

    interrupted_value = None

    def fake_interrupt(msg):
        nonlocal interrupted_value
        interrupted_value = msg
        raise Exception("interrupt_called")

    with patch("biteme.graph.nodes.interrupt", side_effect=fake_interrupt):
        try:
            questioner_node(state)
        except Exception:
            pass

    assert interrupted_value is not None
    assert "提问者" in interrupted_value


def test_answerer_prompt_no_context_placeholder():
    """Answerer prompts should not contain {context} — context is now passed via HumanMessage."""
    for mode in ("learn", "interview"):
        prompts = get_prompts(mode)
        assert "{context}" not in prompts["answerer"], (
            f"{mode} answerer prompt still contains '{{context}}' placeholder"
        )


def test_keyword_score_typeddict():
    ks: KeywordScore = {"keyword": "向量检索", "score": 8}
    assert ks["keyword"] == "向量检索"
    assert ks["score"] == 8

def test_session_state_has_review_history_field():
    state = make_state()
    assert state["review_history"] == []

def test_session_state_review_history_accumulates():
    turn_keywords = [{"keyword": "embedding", "score": 9}]
    state = make_state(review_history=[turn_keywords])
    assert state["review_history"] == [turn_keywords]


def test_get_prompts_learn_has_reviewer():
    prompts = get_prompts("learn")
    assert "reviewer" in prompts
    assert isinstance(prompts["reviewer"], str)
    assert len(prompts["reviewer"]) > 0

def test_get_prompts_interview_has_reviewer():
    prompts = get_prompts("interview")
    assert "reviewer" in prompts
    assert isinstance(prompts["reviewer"], str)
    assert len(prompts["reviewer"]) > 0

def test_reviewer_prompt_requests_json_output():
    prompts = get_prompts("learn")
    assert "JSON" in prompts["reviewer"]
    assert "keywords" in prompts["reviewer"]

def test_reviewer_prompt_is_shared_constant():
    from biteme.graph.prompts import REVIEWER
    assert get_prompts("learn")["reviewer"] is REVIEWER
    assert get_prompts("interview")["reviewer"] is REVIEWER


def test_reviewer_node_appends_to_review_history():
    """reviewer_node must append one list[KeywordScore] per call."""
    state = make_state(
        hitl_flags=["answerer"],
        messages=[
            {"speaker": "questioner", "content": "什么是 RAG？", "retrieved_chunks": []},
            {"speaker": "human", "content": "RAG 是检索增强生成。", "retrieved_chunks": []},
        ],
        llm_reference_answer="RAG（Retrieval-Augmented Generation）结合检索与生成，先从知识库检索相关片段，再输入 LLM 生成答案。核心组件：向量检索、embedding 模型、LLM。",
        review_history=[],
    )
    llm_response = '{"keywords": [{"keyword": "向量检索", "score": 7}, {"keyword": "embedding", "score": 5}]}'

    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = llm_response

    with patch("biteme.graph.nodes.ChatOpenAI", return_value=mock_llm), \
         patch("biteme.graph.nodes.Console"):
        result = reviewer_node(state)

    assert "review_history" in result
    assert len(result["review_history"]) == 1
    turn_keywords = result["review_history"][0]
    assert len(turn_keywords) == 2
    assert turn_keywords[0]["keyword"] == "向量检索"
    assert turn_keywords[0]["score"] == 7


def test_reviewer_node_handles_malformed_json():
    """If LLM returns invalid JSON, reviewer_node appends an empty list (no crash)."""
    state = make_state(
        hitl_flags=["answerer"],
        messages=[
            {"speaker": "questioner", "content": "问题", "retrieved_chunks": []},
            {"speaker": "human", "content": "回答", "retrieved_chunks": []},
        ],
        llm_reference_answer="参考答案",
        review_history=[],
    )

    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "这不是JSON"

    with patch("biteme.graph.nodes.ChatOpenAI", return_value=mock_llm), \
         patch("biteme.graph.nodes.Console"):
        result = reviewer_node(state)

    assert result["review_history"] == [[]]


def test_reviewer_node_accumulates_across_turns():
    """review_history grows by one entry each time reviewer_node runs."""
    existing = [[{"keyword": "旧关键词", "score": 6}]]
    state = make_state(
        hitl_flags=["answerer"],
        messages=[
            {"speaker": "questioner", "content": "新问题", "retrieved_chunks": []},
            {"speaker": "human", "content": "新回答", "retrieved_chunks": []},
        ],
        llm_reference_answer="新参考",
        review_history=existing,
    )

    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = '{"keywords": [{"keyword": "新关键词", "score": 9}]}'

    with patch("biteme.graph.nodes.ChatOpenAI", return_value=mock_llm), \
         patch("biteme.graph.nodes.Console"):
        result = reviewer_node(state)

    assert len(result["review_history"]) == 2
    assert result["review_history"][0][0]["keyword"] == "旧关键词"
    assert result["review_history"][1][0]["keyword"] == "新关键词"


from biteme.graph.graph import _after_answerer

def test_after_answerer_routes_to_reviewer_when_hitl():
    state = make_state(hitl_flags=["answerer"], turn_count=2, max_turns=5)
    assert _after_answerer(state) == "reviewer"

def test_after_answerer_routes_to_end_when_max_turns_non_hitl():
    state = make_state(hitl_flags=[], turn_count=5, max_turns=5)
    from langgraph.graph import END
    assert _after_answerer(state) == END

def test_after_answerer_routes_to_questioner_otherwise():
    state = make_state(hitl_flags=[], turn_count=2, max_turns=5)
    assert _after_answerer(state) == "questioner"

def test_graph_has_reviewer_node():
    from biteme.graph.graph import build_graph
    graph = build_graph(checkpointer=None)
    assert "reviewer" in set(graph.nodes.keys())
