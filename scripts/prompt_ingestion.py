#!/usr/bin/env python3
"""
Prompt Ingestion Module
Scans directories, extracts from zips, and ingests prompts into MongoDB.
"""

import hashlib
import logging
import os
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Generator, List, Optional, Set

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class PromptFile:
    """Represents a discovered prompt file."""

    path: str
    filename: str
    content: str
    size_bytes: int
    source_type: str  # "file", "zip", "directory"
    modified_at: Optional[datetime] = None
    zip_source: Optional[str] = None  # If extracted from zip


@dataclass
class IngestionStats:
    """Statistics from an ingestion run."""

    files_scanned: int = 0
    prompts_found: int = 0
    prompts_ingested: int = 0
    duplicates_skipped: int = 0
    errors: int = 0
    domains: dict = field(default_factory=dict)


class PromptScanner:
    """Scans filesystem for prompt files."""

    # File patterns to match
    TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}

    # Directories to always exclude
    EXCLUDE_DIRS = {
        "node_modules",
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".turbo",
        "dist",
        "build",
        ".next",
        ".nuxt",
        "coverage",
        ".pytest_cache",
    }

    def __init__(
        self,
        include_md: bool = True,
        max_file_size_mb: float = 10.0,
        exclude_patterns: Optional[List[str]] = None,
    ):
        self.include_md = include_md
        self.max_file_size = int(max_file_size_mb * 1024 * 1024)
        self.exclude_patterns = exclude_patterns or []
        self._seen_hashes: Set[str] = set()

    def scan_directory(
        self, root_path: str, max_depth: int = 10
    ) -> Generator[PromptFile, None, None]:
        """
        Recursively scan a directory for prompt files.

        Args:
            root_path: Directory to scan
            max_depth: Maximum recursion depth

        Yields:
            PromptFile objects for each discovered prompt
        """
        root = Path(root_path).expanduser().resolve()
        if not root.exists():
            logger.error(f"Path not found: {root}")
            return

        logger.info(f"📂 Scanning: {root}")

        def should_exclude(path: Path) -> bool:
            """Check if path should be excluded."""
            if path.name in self.EXCLUDE_DIRS:
                return True
            for pattern in self.exclude_patterns:
                if re.search(pattern, str(path), re.IGNORECASE):
                    return True
            return False

        def scan_recursive(current: Path, depth: int):
            if depth > max_depth:
                return

            try:
                for item in current.iterdir():
                    if item.is_dir():
                        if not should_exclude(item):
                            yield from scan_recursive(item, depth + 1)
                    elif item.is_file():
                        # Check for prompt txt/md files
                        if self._is_prompt_file(item):
                            prompt = self._read_file(item)
                            if prompt:
                                yield prompt

                        # Check for prompt zip files
                        elif self._is_prompt_zip(item):
                            yield from self._extract_zip(item)
            except PermissionError:
                pass  # Skip directories we can't access
            except Exception as e:
                logger.debug(f"Error scanning {current}: {e}")

        yield from scan_recursive(root, 0)

    def _is_prompt_file(self, path: Path) -> bool:
        """Check if file is a prompt file based on name and extension."""
        name_lower = path.name.lower()

        # Must have "prompt" in name
        if "prompt" not in name_lower:
            return False

        # Check extension
        ext = path.suffix.lower()
        if ext == ".txt":
            return True
        if ext in {".md", ".markdown"} and self.include_md:
            return True

        return False

    def _is_prompt_zip(self, path: Path) -> bool:
        """Check if file is a zip containing prompts."""
        name_lower = path.name.lower()
        return path.suffix.lower() == ".zip" and "prompt" in name_lower

    def _read_file(self, path: Path) -> Optional[PromptFile]:
        """Read a prompt file and return PromptFile object."""
        try:
            # Check size
            size = path.stat().st_size
            if size > self.max_file_size:
                logger.debug(f"Skipping large file: {path} ({size} bytes)")
                return None

            # Read content
            content = path.read_text(encoding="utf-8", errors="ignore")
            if not content.strip():
                return None

            # Get modification time
            mtime = datetime.fromtimestamp(path.stat().st_mtime)

            return PromptFile(
                path=str(path),
                filename=path.name,
                content=content,
                size_bytes=size,
                source_type="file",
                modified_at=mtime,
            )
        except Exception as e:
            logger.debug(f"Error reading {path}: {e}")
            return None

    def _extract_zip(self, zip_path: Path) -> Generator[PromptFile, None, None]:
        """Extract prompt files from a zip archive."""
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                for name in zf.namelist():
                    # Skip directories and hidden files
                    if name.endswith("/") or name.startswith("__MACOSX"):
                        continue

                    # Check if it's a prompt file
                    name_lower = name.lower()
                    ext = Path(name).suffix.lower()

                    is_prompt = "prompt" in name_lower and ext in self.TEXT_EXTENSIONS

                    # Also include any txt/md from prompt zips
                    is_text_in_prompt_zip = ext in self.TEXT_EXTENSIONS

                    if is_prompt or is_text_in_prompt_zip:
                        try:
                            content = zf.read(name).decode("utf-8", errors="ignore")
                            if content.strip():
                                info = zf.getinfo(name)
                                yield PromptFile(
                                    path=f"{zip_path}!{name}",
                                    filename=Path(name).name,
                                    content=content,
                                    size_bytes=info.file_size,
                                    source_type="zip",
                                    zip_source=str(zip_path),
                                )
                        except Exception as e:
                            logger.debug(
                                f"Error extracting {name} from {zip_path}: {e}"
                            )
        except zipfile.BadZipFile:
            logger.warning(f"⚠️  Bad zip file: {zip_path}")
        except Exception as e:
            logger.debug(f"Error processing zip {zip_path}: {e}")

    def compute_hash(self, content: str) -> str:
        """Compute SHA256 hash of content for deduplication."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def is_duplicate(self, content: str) -> bool:
        """Check if content has already been seen."""
        content_hash = self.compute_hash(content)
        if content_hash in self._seen_hashes:
            return True
        self._seen_hashes.add(content_hash)
        return False


class PromptIngester:
    """Ingests prompts into MongoDB."""

    def __init__(self, mongodb_uri: Optional[str] = None):
        self.mongodb_uri = mongodb_uri or os.getenv("MONGODB_URI")
        self._client = None
        self._db = None
        self._collection = None

    def connect(self) -> bool:
        """Establish MongoDB connection."""
        if not self.mongodb_uri:
            logger.error("❌ MONGODB_URI not set")
            return False

        try:
            from pymongo import MongoClient

            self._client = MongoClient(self.mongodb_uri, serverSelectionTimeoutMS=5000)
            self._client.server_info()

            # Get database
            db_name = "bmc_chat"
            if "/" in self.mongodb_uri:
                parts = self.mongodb_uri.split("/")
                if len(parts) > 3:
                    potential_db = parts[-1].split("?")[0]
                    if potential_db and ":" not in potential_db:
                        db_name = potential_db

            self._db = self._client[db_name]
            self._collection = self._db["prompts"]
            logger.info(f"✅ Connected to MongoDB: {db_name}")
            return True
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            return False

    def ingest(
        self, prompt: PromptFile, classification: dict, dry_run: bool = False
    ) -> bool:
        """
        Ingest a single prompt into MongoDB.

        Args:
            prompt: PromptFile to ingest
            classification: Classification results from PromptClassifier
            dry_run: If True, don't actually insert

        Returns:
            True if ingested, False if skipped/failed
        """

        scanner = PromptScanner()
        content_hash = scanner.compute_hash(prompt.content)

        doc = {
            "hash": content_hash,
            "title": self._extract_title(prompt.content, prompt.filename),
            "content": prompt.content,
            "source_path": prompt.path,
            "source_type": prompt.source_type,
            "domain": classification.get("domain", "general"),
            "prompt_type": classification.get("prompt_type", "user_prompt"),
            "keywords": classification.get("keywords", []),
            "version": classification.get("version"),
            "language": classification.get("language", "mixed"),
            "quality_score": classification.get("quality_score", 0.5),
            "size_bytes": prompt.size_bytes,
            "created_at": prompt.modified_at or datetime.utcnow(),
            "indexed_at": datetime.utcnow(),
            "metadata": {"filename": prompt.filename, "zip_source": prompt.zip_source},
        }

        if dry_run:
            logger.info(f"  [DRY] Would ingest: {prompt.filename}")
            return True

        try:
            # Upsert by hash
            result = self._collection.update_one(
                {"hash": content_hash}, {"$set": doc}, upsert=True
            )
            return result.upserted_id is not None or result.modified_count > 0
        except Exception as e:
            logger.error(f"  ❌ Failed to ingest {prompt.filename}: {e}")
            return False

    def _extract_title(self, content: str, filename: str) -> str:
        """Extract title from content or use filename."""
        # Try to find markdown header
        match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if match:
            return match.group(1).strip()

        # Try to find title: pattern
        match = re.search(r"^title:\s*(.+)$", content, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()

        # Use filename without extension
        return Path(filename).stem

    def get_stats(self) -> dict:
        """Get collection statistics."""
        if not self._collection:
            return {}

        try:
            count = self._collection.count_documents({})
            domains = self._collection.aggregate(
                [{"$group": {"_id": "$domain", "count": {"$sum": 1}}}]
            )
            types = self._collection.aggregate(
                [{"$group": {"_id": "$prompt_type", "count": {"$sum": 1}}}]
            )

            return {
                "total_prompts": count,
                "by_domain": {d["_id"]: d["count"] for d in domains},
                "by_type": {t["_id"]: t["count"] for t in types},
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}


def run_ingestion(
    source_paths: List[str], dry_run: bool = False, include_md: bool = True
) -> IngestionStats:
    """
    Run a full ingestion from specified paths.

    Args:
        source_paths: List of directories to scan
        dry_run: If True, don't insert into database
        include_md: Include .md files alongside .txt

    Returns:
        IngestionStats with results
    """
    from prompt_classifier import PromptClassifier

    stats = IngestionStats()
    scanner = PromptScanner(include_md=include_md)
    classifier = PromptClassifier()
    ingester = PromptIngester()

    if not dry_run:
        if not ingester.connect():
            logger.error("Failed to connect to MongoDB")
            return stats

    logger.info("\n🚀 Starting prompt ingestion...\n")

    for source in source_paths:
        logger.info(f"\n📁 Processing: {source}")

        for prompt in scanner.scan_directory(source):
            stats.files_scanned += 1

            # Check for duplicates
            if scanner.is_duplicate(prompt.content):
                stats.duplicates_skipped += 1
                continue

            stats.prompts_found += 1

            # Classify
            classification = classifier.classify(prompt.content, prompt.filename)
            class_dict = {
                "domain": classification.domain,
                "prompt_type": classification.prompt_type,
                "keywords": classification.keywords,
                "version": classification.version,
                "language": classification.language,
                "quality_score": classification.quality_score,
            }

            # Track domains
            stats.domains[classification.domain] = (
                stats.domains.get(classification.domain, 0) + 1
            )

            # Ingest
            if ingester.ingest(prompt, class_dict, dry_run=dry_run):
                stats.prompts_ingested += 1
                logger.info(f"  ✅ {prompt.filename} [{classification.domain}]")
            else:
                stats.errors += 1

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 INGESTION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"  Files scanned:     {stats.files_scanned}")
    logger.info(f"  Prompts found:     {stats.prompts_found}")
    logger.info(f"  Prompts ingested:  {stats.prompts_ingested}")
    logger.info(f"  Duplicates:        {stats.duplicates_skipped}")
    logger.info(f"  Errors:            {stats.errors}")
    logger.info("\n  By domain:")
    for domain, count in sorted(stats.domains.items(), key=lambda x: -x[1]):
        logger.info(f"    {domain}: {count}")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prompt Ingestion Tool")
    parser.add_argument("paths", nargs="+", help="Directories to scan")
    parser.add_argument(
        "--dry-run", action="store_true", help="Don't insert into database"
    )
    parser.add_argument("--no-md", action="store_true", help="Exclude .md files")
    args = parser.parse_args()

    run_ingestion(
        source_paths=args.paths, dry_run=args.dry_run, include_md=not args.no_md
    )
