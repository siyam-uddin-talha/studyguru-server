#!/usr/bin/env python3
"""
Debug script to check configuration
"""

import os
from app.core.config import settings

print("Environment variables:")
for key, value in os.environ.items():
    if "DATABASE" in key or "DB" in key:
        print(f"  {key}: {value}")

print(f"\nSettings DATABASE_URL: {settings.DATABASE_URL}")
print(f"Settings ENVIRONMENT: {settings.ENVIRONMENT}")
