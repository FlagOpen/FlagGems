"""
Test hello functionality.
"""

import pytest


def test_hello_import():
    """Test that hello functions can be imported."""
    from flag_gems.utils.hello import hello, hello_info, hello_world
    
    # Basic function availability
    assert callable(hello)
    assert callable(hello_info)
    assert callable(hello_world)


def test_hello_function():
    """Test the hello function returns correct message."""
    from flag_gems.utils.hello import hello
    
    result = hello()
    assert isinstance(result, str)
    assert "Hello from FlagGems!" in result


def test_hello_info():
    """Test hello_info returns a dictionary."""
    from flag_gems.utils.hello import hello_info
    
    info = hello_info()
    assert isinstance(info, dict)
    assert "message" in info
    assert "Hello from FlagGems!" in info["message"]


def test_hello_world():
    """Test hello_world function runs without error."""
    from flag_gems.utils.hello import hello_world
    
    # Should not raise any exceptions
    hello_world()


def test_hello_standalone():
    """Test hello functions work without full flag_gems import."""
    # This tests the standalone functionality
    import sys
    import subprocess
    
    # Test direct execution of the hello module
    result = subprocess.run(
        [sys.executable, "-c", "from flag_gems.utils.hello import hello; print(hello())"],
        capture_output=True,
        text=True,
        cwd="/home/runner/work/FlagGems/FlagGems",
        env={"PYTHONPATH": "/home/runner/work/FlagGems/FlagGems/src"}
    )
    
    assert result.returncode == 0
    assert "Hello from FlagGems!" in result.stdout