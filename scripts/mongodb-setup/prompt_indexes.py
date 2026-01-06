#!/usr/bin/env python3
"""
MongoDB Indexes for Prompts Collection
Creates indexes for efficient prompt searching and deduplication.
"""

import logging
import os
import sys

from pymongo import MongoClient, TEXT
from pymongo.errors import ConnectionFailure

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def create_prompt_indexes(dry_run: bool = False) -> bool:
    """
    Create indexes for the prompts collection.

    Args:
        dry_run: If True, only print what would be created

    Returns:
        True if successful, False otherwise
    """
    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        logger.error("❌ MONGODB_URI environment variable not set")
        return False

    try:
        client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
        client.server_info()  # Test connection

        # Get database name from URI or use default
        db_name = "bmc_chat"
        if "/" in mongodb_uri:
            parts = mongodb_uri.split("/")
            if len(parts) > 3:
                potential_db = parts[-1].split("?")[0]
                if potential_db and ":" not in potential_db:
                    db_name = potential_db

        db = client[db_name]
        prompts = db["prompts"]

        logger.info(f"📦 Connected to database: {db_name}")
        logger.info("📑 Collection: prompts")

        # Define indexes
        indexes = [
            {
                "name": "hash_unique",
                "keys": [("hash", 1)],
                "unique": True,
                "description": "Unique constraint on content hash for deduplication",
            },
            {
                "name": "domain_type_idx",
                "keys": [("domain", 1), ("prompt_type", 1)],
                "unique": False,
                "description": "Compound index for filtered queries by domain and type",
            },
            {
                "name": "keywords_text_idx",
                "keys": [("keywords", TEXT)],
                "unique": False,
                "description": "Text index on keywords for search",
            },
            {
                "name": "content_text_idx",
                "keys": [("title", TEXT), ("content", TEXT)],
                "unique": False,
                "description": "Full-text search on title and content",
            },
            {
                "name": "indexed_at_idx",
                "keys": [("indexed_at", -1)],
                "unique": False,
                "description": "Sort by indexing date",
            },
            {
                "name": "source_path_idx",
                "keys": [("source_path", 1)],
                "unique": False,
                "description": "Lookup by source file path",
            },
        ]

        if dry_run:
            logger.info("\n🔍 DRY RUN - Would create the following indexes:\n")
            for idx in indexes:
                logger.info(f"  • {idx['name']}: {idx['description']}")
            return True

        logger.info("\n📝 Creating indexes...\n")

        for idx in indexes:
            try:
                # Handle text indexes differently
                if any(k[1] == TEXT for k in idx["keys"]):
                    prompts.create_index(
                        idx["keys"], name=idx["name"], default_language="spanish"
                    )
                else:
                    prompts.create_index(
                        idx["keys"], name=idx["name"], unique=idx.get("unique", False)
                    )
                logger.info(f"  ✅ Created: {idx['name']}")
            except Exception as e:
                if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                    logger.info(f"  ⏭️  Exists: {idx['name']}")
                else:
                    logger.warning(f"  ⚠️  Failed: {idx['name']} - {e}")

        # Show final index list
        logger.info("\n📊 Current indexes on 'prompts' collection:")
        for idx in prompts.list_indexes():
            logger.info(f"  • {idx['name']}")

        logger.info("\n✅ Prompt indexes setup complete!")
        return True

    except ConnectionFailure as e:
        logger.error(f"❌ Connection failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False


def verify_indexes() -> bool:
    """Verify that all required indexes exist."""
    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        logger.error("❌ MONGODB_URI not set")
        return False

    try:
        client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
        db = client["bmc_chat"]
        prompts = db["prompts"]

        existing = {idx["name"] for idx in prompts.list_indexes()}
        required = {
            "hash_unique",
            "domain_type_idx",
            "indexed_at_idx",
            "source_path_idx",
        }

        missing = required - existing
        if missing:
            logger.warning(f"⚠️  Missing indexes: {missing}")
            return False

        logger.info("✅ All required indexes present")
        return True

    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MongoDB Prompt Indexes Setup")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be created"
    )
    parser.add_argument("--verify", action="store_true", help="Verify indexes exist")
    args = parser.parse_args()

    if args.verify:
        success = verify_indexes()
    else:
        success = create_prompt_indexes(dry_run=args.dry_run)

    sys.exit(0 if success else 1)
