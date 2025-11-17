#!/usr/bin/env python3
import sys
print("Content-Type: text/plain\n")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path[:5]}")
