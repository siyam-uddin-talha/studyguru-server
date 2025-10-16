#!/usr/bin/env python3
"""
Vector Collection Management Script for StudyGuru Pro
Handles collection creation, migration, and cleanup for both GPT and Gemini models
"""

import os
import asyncio
from dotenv import load_dotenv
from pymilvus import connections, Collection, utility
from pymilvus import FieldSchema, CollectionSchema, DataType

# Load environment variables
load_dotenv()

from app.core.config import settings
from app.config.langchain_config import StudyGuruConfig


class VectorCollectionManager:
    """Manages vector collections for different embedding models"""

    def __init__(self):
        self.base_collection = settings.ZILLIZ_COLLECTION
        self.gpt_collection = f"{self.base_collection}_gpt"
        self.gemini_collection = f"{self.base_collection}_gemini"

    def connect(self):
        """Connect to Milvus"""
        try:
            connections.connect(
                alias="default",
                uri=settings.ZILLIZ_URI,
                token=settings.ZILLIZ_TOKEN,
                secure=True,
            )
            print("✅ Connected to Milvus")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Milvus: {e}")
            return False

    def list_collections(self):
        """List all collections"""
        try:
            collections = utility.list_collections()
            print(f"\n📋 Available collections:")
            for collection in collections:
                print(f"   - {collection}")
            return collections
        except Exception as e:
            print(f"❌ Failed to list collections: {e}")
            return []

    def create_collection(self, collection_name: str, dimension: int):
        """Create a new collection with specified dimension"""
        try:
            if utility.has_collection(collection_name):
                print(f"⚠️ Collection {collection_name} already exists")
                return True

            # Define schema
            fields = [
                FieldSchema(
                    name="id",
                    dtype=DataType.VARCHAR,
                    is_primary=True,
                    auto_id=True,
                    max_length=64,
                ),
                FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(
                    name="interaction_id",
                    dtype=DataType.VARCHAR,
                    max_length=64,
                    nullable=True,
                ),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
                FieldSchema(
                    name="metadata",
                    dtype=DataType.VARCHAR,
                    max_length=4096,
                    nullable=True,
                ),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
            ]

            schema = CollectionSchema(
                fields=fields,
                description=f"StudyGuru embeddings collection ({dimension}D)",
                enable_dynamic_field=False,
            )

            collection = Collection(
                name=collection_name, schema=schema, consistency_level="Bounded"
            )

            # Create index
            collection.create_index(
                field_name="vector",
                index_params={
                    "index_type": "IVF_FLAT",
                    "metric_type": settings.ZILLIZ_INDEX_METRIC,
                    "params": {"nlist": 1024},
                },
            )

            print(
                f"✅ Created collection {collection_name} with {dimension} dimensions"
            )
            return True

        except Exception as e:
            print(f"❌ Failed to create collection {collection_name}: {e}")
            return False

    def delete_collection(self, collection_name: str):
        """Delete a collection"""
        try:
            if not utility.has_collection(collection_name):
                print(f"⚠️ Collection {collection_name} does not exist")
                return True

            utility.drop_collection(collection_name)
            print(f"✅ Deleted collection {collection_name}")
            return True

        except Exception as e:
            print(f"❌ Failed to delete collection {collection_name}: {e}")
            return False

    def get_collection_info(self, collection_name: str):
        """Get information about a collection"""
        try:
            if not utility.has_collection(collection_name):
                print(f"⚠️ Collection {collection_name} does not exist")
                return None

            collection = Collection(collection_name)
            collection.load()

            # Get collection stats
            num_entities = collection.num_entities

            print(f"\n📊 Collection: {collection_name}")
            print(f"   Entities: {num_entities}")
            print(f"   Schema: {collection.schema}")

            return {
                "name": collection_name,
                "entities": num_entities,
                "schema": collection.schema,
            }

        except Exception as e:
            print(f"❌ Failed to get collection info for {collection_name}: {e}")
            return None

    def setup_model_collections(self):
        """Setup collections for both GPT and Gemini models"""
        print("🔧 Setting up model-specific collections...")

        # Create GPT collection (1536 dimensions)
        gpt_success = self.create_collection(self.gpt_collection, 1536)

        # Create Gemini collection (768 dimensions)
        gemini_success = self.create_collection(self.gemini_collection, 768)

        if gpt_success and gemini_success:
            print("✅ Both model collections created successfully")
            return True
        else:
            print("❌ Failed to create some collections")
            return False

    def cleanup_old_collection(self):
        """Clean up the old single collection if it exists"""
        if utility.has_collection(self.base_collection):
            print(f"\n⚠️ Found old collection: {self.base_collection}")
            response = input("Do you want to delete it? (y/N): ").strip().lower()
            if response == "y":
                return self.delete_collection(self.base_collection)
            else:
                print("Keeping old collection")
        return True

    def show_status(self):
        """Show current collection status"""
        print("\n📊 Vector Collection Status")
        print("=" * 50)

        collections = self.list_collections()

        # Check each collection
        for collection_name in [
            self.base_collection,
            self.gpt_collection,
            self.gemini_collection,
        ]:
            if collection_name in collections:
                self.get_collection_info(collection_name)
            else:
                print(f"\n❌ Collection: {collection_name} (not found)")


async def main():
    """Main function"""
    print("🚀 StudyGuru Pro - Vector Collection Manager")
    print("=" * 60)

    manager = VectorCollectionManager()

    if not manager.connect():
        return

    while True:
        print("\n📋 Available Operations:")
        print("1. Show collection status")
        print("2. Setup model-specific collections")
        print("3. Create GPT collection (1536D)")
        print("4. Create Gemini collection (768D)")
        print("5. Delete collection")
        print("6. Cleanup old collection")
        print("7. Exit")

        choice = input("\nSelect operation (1-7): ").strip()

        if choice == "1":
            manager.show_status()

        elif choice == "2":
            manager.setup_model_collections()

        elif choice == "3":
            manager.create_collection(manager.gpt_collection, 1536)

        elif choice == "4":
            manager.create_collection(manager.gemini_collection, 768)

        elif choice == "5":
            collections = manager.list_collections()
            if collections:
                collection_name = input("Enter collection name to delete: ").strip()
                manager.delete_collection(collection_name)
            else:
                print("No collections found")

        elif choice == "6":
            manager.cleanup_old_collection()

        elif choice == "7":
            print("👋 Goodbye!")
            break

        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    asyncio.run(main())
