# 🗂️ Prompt Library

A searchable prompt library with MongoDB full-text search, automatic classification, and a CLI interface.

## Features

- **📥 Ingestion**: Scan directories and ZIP archives for prompt files (`.txt`, `.md`)
- **🏷️ Auto-Classification**: Automatically detect domain, type, language, and extract keywords
- **🔍 Full-Text Search**: MongoDB-powered search with filtering by domain/type
- **🔗 Deduplication**: SHA256 hash-based deduplication prevents duplicate prompts
- **📊 Statistics**: View collection stats by domain, type, and language
- **📤 Export**: Export prompts to JSON or individual Markdown files

## Installation

```bash
# Clone the repository
git clone https://github.com/matiasportugau-ui/Prompt-Library.git
cd Prompt-Library

# Install dependencies
pip install -r requirements.txt

# Set MongoDB connection
export MONGODB_URI="mongodb+srv://..."
```

## Quick Start

```bash
# 1. Create MongoDB indexes (run once)
python scripts/mongodb-setup/prompt_indexes.py

# 2. Scan for prompts (dry-run)
python scripts/prompt_cli.py scan ~/Documents/prompts

# 3. Ingest prompts into database
python scripts/prompt_cli.py ingest ~/Documents/prompts

# 4. Search prompts
python scripts/prompt_cli.py search "shopify chatbot"

# 5. View statistics
python scripts/prompt_cli.py stats
```

## CLI Commands

| Command          | Description                            |
| ---------------- | -------------------------------------- |
| `scan <paths>`   | Scan directories for prompts (dry-run) |
| `ingest <paths>` | Ingest prompts into MongoDB            |
| `search <query>` | Search prompts with optional filters   |
| `stats`          | Show collection statistics             |
| `show <id>`      | Display a specific prompt              |
| `export`         | Export prompts to files                |

### Examples

```bash
# Scan with verbose output
python scripts/prompt_cli.py scan ~/Documents/prompts --verbose

# Ingest with dry-run (preview only)
python scripts/prompt_cli.py ingest ~/Documents/prompts --dry-run

# Search by domain
python scripts/prompt_cli.py search --domain shopify

# Export to JSON
python scripts/prompt_cli.py export --output ./exported --format json

# Export to Markdown files
python scripts/prompt_cli.py export --output ./exported --format md
```

## API Integration

The library includes FastAPI endpoints for HTTP access:

```bash
# Search via API
curl "http://localhost:8000/api/prompts/search?q=shopify"

# Get statistics
curl "http://localhost:8000/api/prompts/stats"
```

## Classification System

Prompts are automatically classified by:

| Category     | Values                                                                                      |
| ------------ | ------------------------------------------------------------------------------------------- |
| **Domain**   | chatbot, shopify, audio, automation, coding, ai_agent, mercadolibre, documentation, general |
| **Type**     | system_prompt, template, protocol, master_prompt, user_prompt                               |
| **Language** | en, es, mixed                                                                               |

## MongoDB Schema

```javascript
{
  hash: String,          // SHA256 for deduplication
  title: String,         // Extracted or filename-based
  content: String,       // Full prompt text
  domain: String,        // Auto-classified domain
  prompt_type: String,   // Auto-classified type
  keywords: [String],    // Extracted keywords
  language: String,      // en/es/mixed
  quality_score: Number, // 0.0 - 1.0
  source_path: String,   // Original file path
  indexed_at: Date
}
```

## Project Structure

```
Prompt-Library/
├── scripts/
│   ├── prompt_cli.py        # Main CLI interface
│   ├── prompt_ingestion.py  # Scanner and ingester
│   ├── prompt_classifier.py # Auto-classification logic
│   └── mongodb-setup/
│       └── prompt_indexes.py # MongoDB index setup
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- MongoDB 4.4+ (with text search enabled)
- pymongo

## License

MIT
