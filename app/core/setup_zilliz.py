#!/usr/bin/env python3
"""
Setup script for Zilliz Cloud configuration
"""
import os
from pathlib import Path


def setup_zilliz_credentials():
    """Interactive setup for Zilliz Cloud credentials"""
    print("🔧 Zilliz Cloud Setup")
    print("=" * 40)

    print("\n📋 You'll need the following from https://cloud.zilliz.com/:")
    print("   1. URI (e.g., https://your-cluster.zillizcloud.com)")
    print("   2. Token (API Key)")
    print("   3. Collection name (optional, defaults to 'document_embeddings')")

    # Get credentials from user
    uri = input("\n🔗 Enter your Zilliz URI: ").strip()
    token = input("🔑 Enter your Zilliz Token: ").strip()
    collection = input(
        "📁 Enter collection name (or press Enter for default): "
    ).strip()

    if not collection:
        collection = "document_embeddings"

    # Create .env file
    env_file = Path(".env")
    env_content = []

    # Read existing .env if it exists
    if env_file.exists():
        with open(env_file, "r") as f:
            env_content = f.readlines()

    # Update or add Zilliz settings
    zilliz_settings = {
        "ZILLIZ_URI": uri,
        "ZILLIZ_TOKEN": token,
        "ZILLIZ_COLLECTION": collection,
        "ZILLIZ_DIMENSION": "1536",
        "ZILLIZ_INDEX_METRIC": "IP",
        "ZILLIZ_CONSISTENCY_LEVEL": "Bounded",
    }

    # Remove existing Zilliz settings
    env_content = [
        line for line in env_content if not line.startswith(("ZILLIZ_", "# Zilliz"))
    ]

    # Add new settings
    env_content.append("\n# Zilliz Vector Database Configuration\n")
    for key, value in zilliz_settings.items():
        env_content.append(f"{key}={value}\n")

    # Write updated .env file
    with open(env_file, "w") as f:
        f.writelines(env_content)

    print(f"\n✅ Configuration saved to {env_file}")
    print("\n📋 Your Zilliz settings:")
    for key, value in zilliz_settings.items():
        if "TOKEN" in key:
            print(f"   {key}: {'*' * 10}...{value[-4:]}")
        else:
            print(f"   {key}: {value}")

    print("\n🚀 You can now run the vector database test:")
    print("   python3 test_vector_db.py")


if __name__ == "__main__":
    setup_zilliz_credentials()
