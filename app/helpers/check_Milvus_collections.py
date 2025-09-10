#!/usr/bin/env python3
"""
Check and clean up Milvus collections
"""
import sys
import os

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

from pymilvus import connections, utility, Collection
from app.core.config import settings


def check_and_clean_collections():
    """Check existing collections and clean them up"""
    print("🔍 Checking Milvus Collections...")
    print("=" * 50)

    try:
        # Connect to Milvus
        connections.connect(
            alias="default",
            uri=settings.ZILLIZ_URI,
            token=settings.ZILLIZ_TOKEN,
            secure=True,
        )

        # List all collections
        all_collections = utility.list_collections()
        print(f"📋 All collections: {all_collections}")

        # Drop all collections
        for coll_name in all_collections:
            print(f"🗑️  Dropping collection: {coll_name}")
            collection = Collection(coll_name)
            collection.drop()

        print("\n✅ All collections cleaned up!")
        print("🚀 You can now run the vector database test:")
        print("   python3 test/test_vector_db.py")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    check_and_clean_collections()
