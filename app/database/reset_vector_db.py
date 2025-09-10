#!/usr/bin/env python3
"""
Reset vector database collection with correct schema
"""
import sys
import os

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

from pymilvus import connections, utility
from app.core.config import settings


def reset_collection():
    """Reset the Milvus collection with correct schema"""
    print("🔄 Resetting Vector Database Collection...")
    print("=" * 50)

    try:
        # Connect to Milvus
        connections.connect(
            alias="default",
            uri=settings.ZILLIZ_URI,
            token=settings.ZILLIZ_TOKEN,
            secure=True,
        )

        collection_name = settings.ZILLIZ_COLLECTION

        # Check if collection exists
        if utility.has_collection(collection_name):
            print(f"📋 Found existing collection: {collection_name}")

            # Drop the collection
            from pymilvus import Collection

            collection = Collection(collection_name)
            collection.drop()
            print(f"🗑️  Dropped collection: {collection_name}")
        else:
            print(f"📋 No existing collection found: {collection_name}")

        print("\n✅ Collection reset complete!")
        print("🚀 You can now run the vector database test:")
        print("   python3 test/test_vector_db.py")

    except Exception as e:
        print(f"❌ Error resetting collection: {e}")


if __name__ == "__main__":
    reset_collection()
