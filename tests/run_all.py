"""Run all tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if __name__ == "__main__":
    print("=== Parser tests ===")
    from tests.test_parser import (
        test_xml_format, test_fenced_json_format, test_react_format,
        test_unknown_tool_filtered, test_multiple_calls_in_text,
        test_strip_tool_blocks, test_format_tool_result,
        test_real_world_qwen_output, test_no_tool_calls,
    )
    test_xml_format(); print("✓ xml_format")
    test_fenced_json_format(); print("✓ fenced_json_format")
    test_react_format(); print("✓ react_format")
    test_unknown_tool_filtered(); print("✓ unknown_tool_filtered")
    test_multiple_calls_in_text(); print("✓ multiple_calls")
    test_strip_tool_blocks(); print("✓ strip_tool_blocks")
    test_format_tool_result(); print("✓ format_tool_result")
    test_real_world_qwen_output(); print("✓ real_world_qwen_output")
    test_no_tool_calls(); print("✓ no_tool_calls")

    print("\n=== Tool tests ===")
    from tests.test_tools import run_tests
    import asyncio
    asyncio.run(run_tests())

    print("\n=== All tests passed ===")
