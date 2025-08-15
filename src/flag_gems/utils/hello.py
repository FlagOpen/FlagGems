"""
Hello utilities for FlagGems.

Simple utility functions for demonstration and getting started.
"""


def hello():
    """
    A simple hello function that returns a greeting message.
    
    Returns:
        str: A greeting message from FlagGems
    """
    return "Hello from FlagGems!"


def hello_info():
    """
    Return information about FlagGems.
    
    Returns:
        dict: Information about the FlagGems library
    """
    try:
        from flag_gems import __version__, device, vendor_name
        return {
            "message": "Hello from FlagGems!",
            "version": __version__,
            "device": device,
            "vendor": vendor_name
        }
    except ImportError:
        return {
            "message": "Hello from FlagGems!",
            "status": "Basic import only"
        }


def hello_world():
    """
    Print a hello world message with FlagGems information.
    """
    info = hello_info()
    print("=" * 40)
    print(info.get("message", "Hello from FlagGems!"))
    print("=" * 40)
    
    for key, value in info.items():
        if key != "message":
            print(f"{key.capitalize()}: {value}")
    
    print("=" * 40)


if __name__ == "__main__":
    # Test the hello functions when run directly
    print(hello())
    hello_world()