"""Unit tests for the tools."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.tools import make_context, get_tool
from app.tools.base import ToolContext


async def run_tests():
    workdir = Path("/tmp/argo_test")
    workdir.mkdir(parents=True, exist_ok=True)
    ctx = make_context(workdir=workdir, auto_approve=True)

    # bash
    t = get_tool("bash", ctx)
    r = await t.run(cmd="echo hello")
    assert "hello" in r.output
    assert r.error is False
    print("✓ bash: echo")

    # bash with error
    r = await t.run(cmd="false")
    assert r.error is True
    print("✓ bash: error capture")

    # bash timeout (we use a small timeout via context)
    ctx2 = make_context(workdir=workdir, auto_approve=True, bash_timeout=1)
    t2 = get_tool("bash", ctx2)
    r = await t2.run(cmd="sleep 5")
    assert "timeout" in r.output.lower()
    print("✓ bash: timeout")

    # bash denied
    r = await t.run(cmd="rm -rf /")
    assert r.error is True
    assert "denied" in r.output.lower()
    print("✓ bash: deny list")

    # write_file
    t = get_tool("write_file", ctx)
    r = await t.run(path="hello.txt", content="Hello, world!")
    assert "[written]" in r.output
    assert (workdir / "hello.txt").read_text() == "Hello, world!"
    print("✓ write_file")

    # read_file
    t = get_tool("read_file", ctx)
    r = await t.run(path="hello.txt")
    assert "Hello, world!" in r.output
    print("✓ read_file")

    # edit_file
    t = get_tool("edit_file", ctx)
    r = await t.run(path="hello.txt", old_text="world", new_text="Argo")
    assert r.error is False
    assert (workdir / "hello.txt").read_text() == "Hello, Argo!"
    print("✓ edit_file")

    # edit_file not found
    r = await t.run(path="hello.txt", old_text="missing", new_text="x")
    assert r.error is True
    print("✓ edit_file: not found")

    # list_dir
    t = get_tool("list_dir", ctx)
    r = await t.run(path=".")
    assert "hello.txt" in r.output
    print("✓ list_dir")

    # search_files
    t = get_tool("search_files", ctx)
    r = await t.run(pattern="Argo", glob="*")
    assert "hello.txt" in r.output
    print("✓ search_files")

    # path escape
    t = get_tool("read_file", ctx)
    r = await t.run(path="../../../etc/passwd")
    assert r.error is True
    assert "outside" in r.output.lower() or "escapes" in r.output.lower()
    print("✓ read_file: sandbox escape blocked")

    # cleanup
    import shutil
    shutil.rmtree(workdir, ignore_errors=True)
    print("\nAll tool tests passed.")


if __name__ == "__main__":
    asyncio.run(run_tests())
