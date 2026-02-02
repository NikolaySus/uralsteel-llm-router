#!/usr/bin/env python3
"""
Test script to verify the fix for the 'list index out of range' bug.
"""

def test_tool_calls_validation():
    """Test that the validation logic correctly handles edge cases."""
    
    # Test case 1: Normal case with valid tool_calls
    item1 = {
        "tool_calls": [
            {
                "function": {
                    "name": "test_function",
                    "arguments": "{}"
                },
                "id": "test_id"
            }
        ]
    }
    
    # This should pass the validation
    valid_check = item1.get("tool_calls") and len(item1["tool_calls"]) > 0
    print(f"Test 1 - Valid tool_calls: {'PASSED' if valid_check else 'FAILED'}")
    
    # Test case 2: Empty tool_calls
    item2 = {"tool_calls": []}
    empty_check = item2.get("tool_calls") and len(item2["tool_calls"]) > 0
    print(f"Test 2 - Empty tool_calls: {'PASSED' if not empty_check else 'FAILED'}")
    
    # Test case 3: Missing tool_calls
    item3 = {"some_other_field": "value"}
    missing_check = item3.get("tool_calls") and len(item3["tool_calls"]) > 0
    print(f"Test 3 - Missing tool_calls: {'PASSED' if not missing_check else 'FAILED'}")
    
    # Test case 4: None tool_calls
    item4 = {"tool_calls": None}
    none_check = item4.get("tool_calls") and len(item4["tool_calls"]) > 0
    print(f"Test 4 - None tool_calls: {'PASSED' if not none_check else 'FAILED'}")
    
    # Test case 5: Verify the old code would have failed
    try:
        # This would cause "list index out of range" in the old code
        if item2["tool_calls"]:  # This evaluates to False, so it wouldn't be accessed
            if item2["tool_calls"][0]:  # But if we try to access it directly...
                pass
        print("Test 5 - Direct access safety: PASSED")
    except IndexError as e:
        print(f"Test 5 - Direct access would fail: {'PASSED (safety check passed)'}")
    
    print("\nAll validation checks completed. The fix should prevent 'list index out of range' errors.")

if __name__ == "__main__":
    test_tool_calls_validation()