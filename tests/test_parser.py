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
    print("✓ test_xml_format")


def test_fenced_json_format():
    text = 'Let me run a command:\n```json\n{"name": "bash", "arguments": {"cmd": "pwd"}}\n```\nDone.'
    calls, rest = parse_tool_calls(text, valid_tools=["bash"])
    assert len(calls) == 1
    assert calls[0].name == "bash"
    assert calls[0].arguments == {"cmd": "pwd"}
    assert "Let me run a command" in rest
    assert "```" not in rest
    print("✓ test_fenced_json_format")


def test_react_format():
    text = """Thought: I need to list files.
Action: list_dir
Action Input: {"path": "."}
"""
    calls, _ = parse_tool_calls(text, valid_tools=["list_dir"])
    assert len(calls) == 1
    assert calls[0].name == "list_dir"
    assert calls[0].arguments == {"path": "."}
    print("✓ test_react_format")


def test_unknown_tool_filtered():
    text = '<tool_call name="dangerous_tool">{}</tool_call>'
    calls, _ = parse_tool_calls(text, valid_tools=["bash"])
    assert calls == []
    print("✓ test_unknown_tool_filtered")


def test_multiple_calls_in_text():
    text = (
        '<tool_call name="read_file">{"path": "a.py"}</tool_call>\n'
        '<tool_call name="read_file">{"path": "b.py"}</tool_call>'
    )
    calls, rest = parse_tool_calls(text, valid_tools=["read_file"])
    assert len(calls) == 2
    assert calls[0].arguments["path"] == "a.py"
    assert calls[1].arguments["path"] == "b.py"
    print("✓ test_multiple_calls_in_text")


def test_strip_tool_blocks():
    text = 'Hello <tool_call name="x">{}</tool_call> world'
    out = strip_tool_blocks(text)
    assert "tool_call" not in out
    assert "Hello" in out
    assert "world" in out
    print("✓ test_strip_tool_blocks")


def test_format_tool_result():
    out = format_tool_result("bash", "hello\nworld", error=False)
    assert 'name="bash"' in out
    assert "status=\"ok\"" in out
    assert "hello" in out
    print("✓ test_format_tool_result")


def test_real_world_qwen_output():
    """Simulate the kind of output Qwen3-Coder produces."""
    text = """I'll explore the project first.

<tool_call name="bash">{"cmd": "ls -la"}</tool_call>"""
    calls, rest = parse_tool_calls(
        text,
        valid_tools=["bash", "read_file", "write_file", "edit_file", "list_dir", "search_files"],
    )
    assert len(calls) == 1
    assert calls[0].name == "bash"
    assert "I'll explore the project first." in rest
    print("✓ test_real_world_qwen_output")


def test_no_tool_calls():
    text = "Just a regular response with no tool calls."
    calls, rest = parse_tool_calls(text, valid_tools=["bash"])
    assert calls == []
    assert rest == text
    print("✓ test_no_tool_calls")


def test_qwen3_unclosed_tool_call():
    """Qwen3 often emits <tool_call> without closing tag, followed by </think>."""
    text = '''ated

<tool_call>
{"name": "read_file", "arguments": {"path": "test.py"}}
</think>

محتوای فایل `test.py`:

```python
def add(a, b):
    return a + b
```'''
    calls, rest = parse_tool_calls(text, valid_tools=["read_file", "bash"])
    assert len(calls) == 1, f"Expected 1 call, got {len(calls)}"
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {"path": "test.py"}
    # The markdown description should remain in the display text
    assert "محتوای فایل" in rest
    print("✓ test_qwen3_unclosed_tool_call")


def test_json_anywhere_fallback():
    """When other patterns fail, find a {name, arguments} JSON anywhere in text."""
    text = 'I will run the tool now: {"name": "bash", "arguments": {"cmd": "ls"}}, and continue.'
    calls, _ = parse_tool_calls(text, valid_tools=["bash"])
    assert len(calls) == 1
    assert calls[0].name == "bash"
    assert calls[0].arguments == {"cmd": "ls"}
    print("✓ test_json_anywhere_fallback")


def test_pipe_style_mistral():
    """Mistral / Llama-3 pipe-style: <|tool_call name="X">...</tool_call|>."""
    text = '<|tool_call name="bash">{"cmd": "pwd"}</tool_call|>'
    calls, rest = parse_tool_calls(text, valid_tools=["bash", "read_file"])
    assert len(calls) == 1, f"Expected 1 call, got {len(calls)}"
    assert calls[0].name == "bash"
    assert calls[0].arguments == {"cmd": "pwd"}
    assert rest == ""
    print("✓ test_pipe_style_mistral")


def test_pipe_style_no_closing():
    """Pipe-style with no closing tag — until next <|tool_call|> or <think>."""
    text = '<|tool_call name="read_file">\n{"path": "a.py"}\n<|tool_call|>'
    calls, _ = parse_tool_calls(text, valid_tools=["read_file"])
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {"path": "a.py"}
    print("✓ test_pipe_style_no_closing")


def test_pipe_style_mistral_real_world():
    """The actual format observed from Mistral-NeMo-12B-abliterated on the live URL."""
    text = 'Let me check the directory.\n<|tool_call name="bash">{"cmd": "pwd"}</tool_call|>\nDone.'
    calls, rest = parse_tool_calls(text, valid_tools=["bash"])
    assert len(calls) == 1
    assert calls[0].name == "bash"
    assert calls[0].arguments == {"cmd": "pwd"}
    assert "Let me check" in rest
    assert "Done" in rest
    print("✓ test_pipe_style_mistral_real_world")


def test_strip_pipe_blocks():
    """strip_tool_blocks should also strip <|tool_call|> blocks."""
    text = 'Hello <|tool_call name="x">{}</tool_call|> world'
    out = strip_tool_blocks(text)
    assert "tool_call" not in out
    assert "Hello" in out
    assert "world" in out
    print("✓ test_strip_pipe_blocks")


def test_pipe_style_noname_with_json():
    """<|tool_call|>{json with name}</tool_call|> — pipe-style without name attribute."""
    text = '<|tool_call|>\n{"name": "list_dir", "arguments": {"path": "."}}\n</tool_call|>'
    calls, _ = parse_tool_calls(text, valid_tools=["list_dir", "bash"])
    assert len(calls) == 1
    assert calls[0].name == "list_dir"
    assert calls[0].arguments == {"path": "."}
    print("✓ test_pipe_style_noname_with_json")


def test_strip_incomplete_pipe():
    """strip_tool_blocks should drop incomplete '<|im_start|>tool_call' fragments."""
    text = 'Working... <|im_start|>tool_call'
    out = strip_tool_blocks(text)
    assert "Working..." in out
    assert "im_start" not in out
    assert "tool_call" not in out
    print("✓ test_strip_incomplete_pipe")


if __name__ == "__main__":
    test_xml_format()
    test_fenced_json_format()
    test_react_format()
    test_unknown_tool_filtered()
    test_multiple_calls_in_text()
    test_strip_tool_blocks()
    test_format_tool_result()
    test_real_world_qwen_output()
    test_no_tool_calls()
    test_qwen3_unclosed_tool_call()
    test_json_anywhere_fallback()
    test_pipe_style_mistral()
    test_pipe_style_no_closing()
    test_pipe_style_mistral_real_world()
    test_strip_pipe_blocks()
    test_pipe_style_noname_with_json()
    test_strip_incomplete_pipe()
    print("\nAll parser tests passed.")
