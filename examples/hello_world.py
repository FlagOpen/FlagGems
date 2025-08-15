#!/usr/bin/env python3
"""
Hello World example for FlagGems.

This simple example demonstrates basic usage of the FlagGems library
and serves as a starting point for new users.
"""

# Try to import dependencies, but handle gracefully if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available")

try:
    import flag_gems
    FLAGGEMS_AVAILABLE = True
    print(f"FlagGems version: {flag_gems.__version__}")
    if hasattr(flag_gems, 'device'):
        print(f"Device: {flag_gems.device}")
    if hasattr(flag_gems, 'vendor_name'):
        print(f"Vendor: {flag_gems.vendor_name}")
except ImportError:
    FLAGGEMS_AVAILABLE = False
    print("FlagGems not available")


def hello_flaggems():
    """
    A simple hello world function that demonstrates basic FlagGems usage.
    """
    print("Hello, FlagGems!")
    
    if not TORCH_AVAILABLE:
        print("PyTorch is required for tensor operations")
        return "Hello from FlagGems!"
    
    # Create a simple tensor
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    print(f"Original tensor: {x}")
    
    if FLAGGEMS_AVAILABLE:
        # Enable FlagGems optimizations
        try:
            with flag_gems.use_gems():
                # Perform a simple operation that could benefit from FlagGems
                y = torch.relu(x)
                z = torch.sum(y)
                print(f"After ReLU (FlagGems): {y}")
                print(f"Sum (FlagGems): {z}")
        except Exception as e:
            print(f"FlagGems error: {e}, falling back to PyTorch")
            y = torch.relu(x)
            z = torch.sum(y)
            print(f"After ReLU (PyTorch): {y}")
            print(f"Sum (PyTorch): {z}")
    else:
        # Fallback to standard PyTorch
        y = torch.relu(x)
        z = torch.sum(y)
        print(f"After ReLU (PyTorch): {y}")
        print(f"Sum (PyTorch): {z}")
    
    return z


def main():
    """Main function to run the hello world example."""
    print("=" * 50)
    print("FlagGems Hello World Example")
    print("=" * 50)
    
    result = hello_flaggems()
    
    print("=" * 50)
    print(f"Final result: {result}")
    print("Hello World example completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()