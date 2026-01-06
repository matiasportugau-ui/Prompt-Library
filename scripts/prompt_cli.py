#!/usr/bin/env python3
"""
Prompt Library CLI
Command-line interface for managing the searchable prompt library.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def get_db_name(mongodb_uri: str) -> str:
    """Extract database name from MongoDB URI."""
    db_name = "bmc_chat"  # Default
    if "/" in mongodb_uri:
        parts = mongodb_uri.split("/")
        if len(parts) > 3:
            potential_db = parts[-1].split("?")[0]
            if potential_db and ":" not in potential_db:
                db_name = potential_db
    return db_name


def cmd_scan(args):
    """Scan directories for prompts (dry-run by default)."""
    from prompt_ingestion import PromptScanner
    from prompt_classifier import PromptClassifier

    scanner = PromptScanner(include_md=not args.no_md)
    classifier = PromptClassifier()

    total = 0
    domains = {}

    for source in args.paths:
        logger.info(f"\n📂 Scanning: {source}")

        for prompt in scanner.scan_directory(source):
            if scanner.is_duplicate(prompt.content):
                continue

            total += 1
            classification = classifier.classify(prompt.content, prompt.filename)

            domains[classification.domain] = domains.get(classification.domain, 0) + 1

            if args.verbose:
                logger.info(
                    f"  📄 {prompt.filename} "
                    f"[{classification.domain}] "
                    f"({prompt.size_bytes} bytes)"
                )

    logger.info(f"\n{'=' * 50}")
    logger.info("📊 SCAN RESULTS (dry-run)")
    logger.info(f"{'=' * 50}")
    logger.info(f"  Total unique prompts: {total}")
    logger.info("\n  By domain:")
    for domain, count in sorted(domains.items(), key=lambda x: -x[1]):
        logger.info(f"    {domain}: {count}")

    if not args.verbose:
        logger.info("\n  💡 Use --verbose to see individual files")


def cmd_ingest(args):
    """Ingest prompts into MongoDB."""
    from prompt_ingestion import run_ingestion

    if args.dry_run:
        logger.info("🔍 DRY RUN MODE - No database changes will be made\n")

    stats = run_ingestion(
        source_paths=args.paths, dry_run=args.dry_run, include_md=not args.no_md
    )

    return 0 if stats.errors == 0 else 1


def cmd_search(args):
    """Search prompts in the database."""
    from pymongo import MongoClient

    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        logger.error("❌ MONGODB_URI not set")
        return 1

    try:
        client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
        db = client[get_db_name(mongodb_uri)]
        prompts = db["prompts"]

        # Build query
        query = {}

        if args.query:
            query["$text"] = {"$search": args.query}

        if args.domain:
            query["domain"] = args.domain

        if args.type:
            query["prompt_type"] = args.type

        # Execute search
        cursor = prompts.find(
            query,
            {"content": 0},  # Exclude full content for listing
        ).limit(args.limit)

        results = list(cursor)

        if not results:
            logger.info("No prompts found matching your query.")
            return 0

        logger.info(f"\n📋 Found {len(results)} prompts:\n")

        for i, p in enumerate(results, 1):
            logger.info(f"{i}. {p.get('title', 'Untitled')}")
            logger.info(f"   Domain: {p.get('domain')} | Type: {p.get('prompt_type')}")
            logger.info(f"   Keywords: {', '.join(p.get('keywords', [])[:5])}")
            logger.info(f"   Source: {p.get('source_path', 'unknown')}")
            logger.info("")

        if args.json:
            print(json.dumps(results, default=str, indent=2))

        return 0

    except Exception as e:
        logger.error(f"❌ Search failed: {e}")
        return 1


def cmd_stats(args):
    """Show collection statistics."""
    from pymongo import MongoClient

    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        logger.error("❌ MONGODB_URI not set")
        return 1

    try:
        client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
        db = client[get_db_name(mongodb_uri)]
        prompts = db["prompts"]

        total = prompts.count_documents({})

        if total == 0:
            logger.info("📭 No prompts in database. Run 'ingest' first.")
            return 0

        # Aggregate stats
        domains = list(
            prompts.aggregate([{"$group": {"_id": "$domain", "count": {"$sum": 1}}}])
        )

        types = list(
            prompts.aggregate(
                [{"$group": {"_id": "$prompt_type", "count": {"$sum": 1}}}]
            )
        )

        languages = list(
            prompts.aggregate([{"$group": {"_id": "$language", "count": {"$sum": 1}}}])
        )

        avg_quality = list(
            prompts.aggregate(
                [{"$group": {"_id": None, "avg": {"$avg": "$quality_score"}}}]
            )
        )

        # Print stats
        logger.info(f"\n{'=' * 50}")
        logger.info("📊 PROMPT LIBRARY STATISTICS")
        logger.info(f"{'=' * 50}")
        logger.info(f"\n  Total prompts: {total}")

        if avg_quality:
            logger.info(f"  Avg quality score: {avg_quality[0]['avg']:.2f}")

        logger.info("\n  By Domain:")
        for d in sorted(domains, key=lambda x: -x["count"]):
            logger.info(f"    {d['_id']}: {d['count']}")

        logger.info("\n  By Type:")
        for t in sorted(types, key=lambda x: -x["count"]):
            logger.info(f"    {t['_id']}: {t['count']}")

        logger.info("\n  By Language:")
        for lang in sorted(languages, key=lambda x: -x["count"]):
            logger.info(f"    {lang['_id']}: {lang['count']}")

        if args.json:
            stats = {
                "total": total,
                "by_domain": {d["_id"]: d["count"] for d in domains},
                "by_type": {t["_id"]: t["count"] for t in types},
                "by_language": {lang["_id"]: lang["count"] for lang in languages},
            }
            print(json.dumps(stats, indent=2))

        return 0

    except Exception as e:
        logger.error(f"❌ Stats failed: {e}")
        return 1


def cmd_show(args):
    """Show a specific prompt by ID or title."""
    from pymongo import MongoClient
    from bson import ObjectId

    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        logger.error("❌ MONGODB_URI not set")
        return 1

    try:
        client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
        db = client[get_db_name(mongodb_uri)]
        prompts = db["prompts"]

        # Try to find by ID or title
        query = {}
        try:
            query["_id"] = ObjectId(args.identifier)
        except Exception:
            query["title"] = {"$regex": args.identifier, "$options": "i"}

        prompt = prompts.find_one(query)

        if not prompt:
            logger.error(f"❌ Prompt not found: {args.identifier}")
            return 1

        logger.info(f"\n{'=' * 50}")
        logger.info(f"📄 {prompt.get('title', 'Untitled')}")
        logger.info(f"{'=' * 50}")
        logger.info(f"Domain: {prompt.get('domain')}")
        logger.info(f"Type: {prompt.get('prompt_type')}")
        logger.info(f"Language: {prompt.get('language')}")
        logger.info(f"Version: {prompt.get('version', 'N/A')}")
        logger.info(f"Quality: {prompt.get('quality_score', 0):.2f}")
        logger.info(f"Keywords: {', '.join(prompt.get('keywords', []))}")
        logger.info(f"Source: {prompt.get('source_path')}")
        logger.info(f"\n{'─' * 50}")
        logger.info("CONTENT:")
        logger.info(f"{'─' * 50}\n")

        content = prompt.get("content", "")
        if args.truncate and len(content) > args.truncate:
            logger.info(content[: args.truncate] + "\n... [truncated]")
        else:
            logger.info(content)

        return 0

    except Exception as e:
        logger.error(f"❌ Show failed: {e}")
        return 1


def cmd_export(args):
    """Export prompts to JSON or markdown files."""
    from pymongo import MongoClient

    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        logger.error("❌ MONGODB_URI not set")
        return 1

    try:
        client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
        db = client[get_db_name(mongodb_uri)]
        prompts = db["prompts"]

        query = {}
        if args.domain:
            query["domain"] = args.domain

        cursor = prompts.find(query)
        results = list(cursor)

        if not results:
            logger.info("No prompts to export.")
            return 0

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.format == "json":
            # Export as single JSON file
            output_file = output_dir / "prompts_export.json"

            # Convert ObjectId to string
            for r in results:
                r["_id"] = str(r["_id"])
                if r.get("created_at"):
                    r["created_at"] = r["created_at"].isoformat()
                if r.get("indexed_at"):
                    r["indexed_at"] = r["indexed_at"].isoformat()

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            logger.info(f"✅ Exported {len(results)} prompts to {output_file}")

        else:
            # Export as individual markdown files
            for p in results:
                safe_title = "".join(
                    c for c in p.get("title", "untitled") if c.isalnum() or c in " -_"
                )[:50]
                filename = f"{safe_title}.md"

                with open(output_dir / filename, "w", encoding="utf-8") as f:
                    f.write(f"# {p.get('title', 'Untitled')}\n\n")
                    f.write(f"**Domain:** {p.get('domain')}\n")
                    f.write(f"**Type:** {p.get('prompt_type')}\n")
                    f.write(f"**Keywords:** {', '.join(p.get('keywords', []))}\n\n")
                    f.write("---\n\n")
                    f.write(p.get("content", ""))

            logger.info(f"✅ Exported {len(results)} prompts to {output_dir}/")

        return 0

    except Exception as e:
        logger.error(f"❌ Export failed: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="🗂️  Prompt Library CLI - Manage your searchable prompt collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prompt_cli.py scan ~/Documents/prompts
  python prompt_cli.py ingest ~/Documents/prompts --dry-run
  python prompt_cli.py search "shopify chatbot"
  python prompt_cli.py stats
  python prompt_cli.py show "BMC Uruguay"
  python prompt_cli.py export --output ./exported_prompts
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # scan command
    scan_parser = subparsers.add_parser(
        "scan", help="Scan directories for prompts (dry-run)"
    )
    scan_parser.add_argument("paths", nargs="+", help="Directories to scan")
    scan_parser.add_argument("--no-md", action="store_true", help="Exclude .md files")
    scan_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show all files"
    )
    scan_parser.set_defaults(func=cmd_scan)

    # ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest prompts into MongoDB")
    ingest_parser.add_argument("paths", nargs="+", help="Directories to ingest")
    ingest_parser.add_argument(
        "--dry-run", action="store_true", help="Don't insert into database"
    )
    ingest_parser.add_argument("--no-md", action="store_true", help="Exclude .md files")
    ingest_parser.set_defaults(func=cmd_ingest)

    # search command
    search_parser = subparsers.add_parser("search", help="Search prompts")
    search_parser.add_argument("query", nargs="?", help="Search query")
    search_parser.add_argument("--domain", help="Filter by domain")
    search_parser.add_argument("--type", help="Filter by prompt type")
    search_parser.add_argument("--limit", type=int, default=20, help="Max results")
    search_parser.add_argument("--json", action="store_true", help="Output as JSON")
    search_parser.set_defaults(func=cmd_search)

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show collection statistics")
    stats_parser.add_argument("--json", action="store_true", help="Output as JSON")
    stats_parser.set_defaults(func=cmd_stats)

    # show command
    show_parser = subparsers.add_parser("show", help="Show a specific prompt")
    show_parser.add_argument("identifier", help="Prompt ID or title")
    show_parser.add_argument(
        "--truncate", type=int, default=2000, help="Max content chars"
    )
    show_parser.set_defaults(func=cmd_show)

    # export command
    export_parser = subparsers.add_parser("export", help="Export prompts to files")
    export_parser.add_argument(
        "--output", "-o", default="./prompt_export", help="Output directory"
    )
    export_parser.add_argument("--domain", help="Filter by domain")
    export_parser.add_argument(
        "--format", choices=["json", "md"], default="json", help="Export format"
    )
    export_parser.set_defaults(func=cmd_export)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
