"""Unit tests for the tool-call parser."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agent.parser import parse_tool_calls, format_tool_result, strip_tool_blocks


def test_xml_format():
    text = '<tool_call name="bash">{"cmd": "ls -la"}</tool_call>'
    calls, rest = parse_tool_calls(text, valid_tools=["bash", "read_file"])
    assert len(calls) == 1
    assert calls[0].name == "bash"
    assert calls[0].arguments == {"cmd": "ls -la"}
    assert rest == ""


def test_fenced_json_format():
    text = 'Let me run a command:\n```json\n{"name": "bash", "arguments": {"cmd": "pwd"}}\n```\nDone.'
    calls, rest = parse_tool_calls(text, valid_tools=["bash"])
    assert len(calls) == 1
    assert calls[0].name == "bash"
    assert calls[0].arguments == {"cmd": "pwd"}
    assert "Let me run a command" in rest
    assert "```" not in rest


def test_react_format():
    text = """Thought: I need to list files.
Action: list_dir
Action Input: {"path": "."}
"""
    calls, _ = parse_tool_calls(text, valid_tools=["list_dir"])
    assert len(calls) == 1
    assert calls[0].name == "list_dir"
    assert calls[0].arguments == {"path": "."}


def test_unknown_tool_filtered():
    text = '<tool_call name="dangerous_tool">{}</tool_call>'
    calls, _ = parse_tool_calls(text, valid_tools=["bash"])
    assert calls == []


def test_multiple_calls_in_text():
    text = (
        '<tool_call name="read_file">{"path": "a.py"}</tool_call>\n'
        '<tool_call name="read_file">{"path": "b.py"}</tool_call>'
    )
    calls, rest = parse_tool_calls(text, valid_tools=["read_file"])
    assert len(calls) == 2
    assert calls[0].arguments["path"] == "a.py"
    assert calls[1].arguments["path"] == "b.py"


def test_strip_tool_blocks():
    text = 'Hello <tool_call name="x">{}</tool_call> world'
    out = strip_tool_blocks(text)
    assert "tool_call" not in out
    assert "Hello" in out
    assert "world" in out


def test_format_tool_result():
    out = format_tool_result("bash", "hello\nworld", error=False)
    assert 'name="bash"' in out
    assert "status=\"ok\"" in out
    assert "hello" in out


def test_real_world_qwen_output():
    """Simulate the kind of output Qwen3-Coder produces."""
    text = """I'll explore the project first.

<tool_call name="bash">{"cmd": "ls -la"}</tool_call>"""
    calls, rest = parse_tool_calls(text, valid_tools=["bash", "read_file", "write_file", "edit_file", "list_dir", "search_files"])
    assert len(calls) == 1
    assert calls[0].name == "bash"
    assert "I'll explore the project first." in rest
    print("✓ test_real_world_qwen_output")


def test_no_tool_calls():
    text = "Just a regular response with no tool calls."
    calls, rest = parse_tool_calls(text, valid_tools=["bash"])
    assert calls == []
    assert rest == text


if __name__ == "__main__":
    test_xml_format()
    print("✓ test_xml_format")
    test_fenced_json_format()
    print("✓ test_fenced_json_format")
    test_react_format()
    print("✓ test_react_format")
    test_unknown_tool_filtered()
    print("✓ test_unknown_tool_filtered")
    test_multiple_calls_in_text()
    print("✓ test_multiple_calls_in_text")
    test_strip_tool_blocks()
    print("✓ test_strip_tool_blocks")
    test_format_tool_result()
    print("✓ test_format_tool_result")
    test_real_world_qwen_output()
    test_no_tool_calls()
    print("✓ test_no_tool_calls")
    print("\nAll parser tests passed.")
