"""
Sample Python code for testing AIMULGENT system
"""

import os
import sys
from typing import List, Dict, Optional
import json


class Calculator:
    """A simple calculator with security issues for testing."""
    
    def __init__(self):
        self.history = []
        self.secret_key = "hardcoded_password_123"  # Security issue
    
    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def divide(self, a: int, b: int) -> float:
        """Divide two numbers with potential error."""
        if b == 0:
            raise ValueError("Division by zero")
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        return result
    
    def execute_command(self, cmd: str) -> str:
        """Execute system command - SECURITY RISK."""
        return os.system(cmd)  # Command injection vulnerability
    
    def complex_nested_function(self, x, y, z, a, b, c, d, e):
        """A deliberately complex function to test complexity analysis."""
        if x > 0:
            if y > 0:
                if z > 0:
                    if a > 0:
                        if b > 0:
                            if c > 0:
                                if d > 0:
                                    if e > 0:
                                        return x + y + z + a + b + c + d + e
                                    else:
                                        return -1
                                else:
                                    return -2
                            else:
                                return -3
                        else:
                            return -4
                    else:
                        return -5
                else:
                    return -6
            else:
                return -7
        else:
            return -8


def process_user_data(data: Dict) -> Dict:
    """Process user data with minimal validation."""
    # Missing input validation - another issue
    name = data["name"]
    age = data["age"]
    email = data["email"]
    
    return {
        "processed_name": name.upper(),
        "processed_age": age * 2,
        "processed_email": email.lower()
    }


def main():
    """Main function demonstrating the calculator."""
    calc = Calculator()
    
    print("Testing Calculator:")
    print(f"5 + 3 = {calc.add(5, 3)}")
    print(f"10 / 2 = {calc.divide(10, 2)}")
    
    # Test complex function
    result = calc.complex_nested_function(1, 1, 1, 1, 1, 1, 1, 1)
    print(f"Complex calculation result: {result}")
    
    print("Calculator History:")
    for entry in calc.history:
        print(f"  {entry}")


if __name__ == "__main__":
    main()