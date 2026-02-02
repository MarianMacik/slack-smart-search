# Slack Smart Search MCP Server

A Model Context Protocol (MCP) server that provides semantic search over historical Slack messages using vector embeddings. This server enables AI assistants to search through Slack conversation history by meaning rather than exact keyword matching, making it ideal for discovering past discussions, troubleshooting threads, and finding relevant context.

## Overview

The Slack Smart Search MCP server consists of two main components:

1. **Data Ingestion Pipeline** (`src/slack_dump.py`) - Dumps Slack messages and converts them to vector embeddings stored in Milvus databases
2. **MCP Server** (`src/smart_search_mcp.py`) - Exposes semantic search capabilities via the Model Context Protocol

The server uses sentence transformers to create vector embeddings of messages and stores them in separate databases for public and private channels, enabling efficient semantic nearest-neighbor search.

## Features

- **Semantic Search**: Find messages by meaning, not just keywords - describe what you're looking for in natural language
- **Public/Private Separation**: Separate databases for public channels (shareable) and private channels (user-specific)
- **HTTP Transport**: Designed to primarily run remotely with HTTP transport, stdio is supported as well
- **Containerized**: Ready to deploy in containers using Podman or Docker
- **Incremental Updates**: Idempotent message ingestion with interrupt/resume support
- **User Mention Resolution**: Automatically resolves user IDs to display names in search results
- **Interactive Search**: CLI tool for testing searches locally

## Architecture

```
src/
├── smart_search_mcp.py    # MCP server exposing search tools
├── slack_dump.py          # Data ingestion pipeline
├── search.py              # Interactive CLI search tool
├── config.py              # Configuration management
├── milvus_store.py        # Vector database operations
├── slack_client.py        # Slack API client
└── helpers.py             # Utility functions

config/
└── dump_config.template.json  # Configuration template

db/
├── slack_public.db        # Public channels vector database
└── slack_private.db       # Private channels vector database
```

## Quick Start

### Running the MCP Server

The server is designed to run in a container:

```bash
# Build the container
make build

# Run locally
make run-mcp-local

# Or run from registry
make run-mcp
```

The server will be available at `http://localhost:8000/mcp`

### Connecting to the MCP Server

Add to your MCP client configuration (e.g., Claude Desktop):

```json
{
  "mcpServers": {
    "smart-search": {
      "type": "http",
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

Or use Claude CLI:

```bash
make claude-mcp-import
```

## Claude Code Plugin

For enhanced functionality with Claude Code, a ready-to-use plugin is available in the `plugin/` directory. The plugin combines this smart search MCP server with a Slack MCP server to provide:

- **Slash Commands**: `/smart-search`, `/answer-questions`, `/get-latest-build`, `/rhoai-build`
- **Automated Q&A**: Automatically find and answer questions in Slack channels
- **Build Tracking**: Get latest RHOAI and ODH build information
- **Skills Integration**: Pre-configured workflows for common tasks

The plugin is particularly useful for Red Hat OpenShift AI teams but can be adapted for other use cases.

**Quick Setup:**
```bash
# Use the plugin with Claude Code
claude --plugin-dir plugin

# Or copy to your plugins directory
cp -r plugin ~/.config/claude-code/plugins/smart-search
```

See [`plugin/README.md`](plugin/README.md) for detailed installation instructions, available skills, and configuration options.

## Data Ingestion

Before using the search server, you need to dump Slack messages into the vector databases.

### Configuration

1. Copy the template configuration:
   ```bash
   cp config/dump_config.template.json config/dump_config.json
   ```

2. Edit `config/dump_config.json` with your settings:
   ```json
   {
     "public_channels": ["C12345678", "C87654321"],
     "private_channels": ["D12345678", "G12345678"],
     "start_timestamp": "2024-01-01",
     "public_db": "./db/slack_public.db",
     "private_db": "./db/slack_private.db",
     "collection_name": "slack_messages",
     "embedding_model": "all-MiniLM-L6-v2",
     "workspace_url": "https://your-workspace.slack.com",
     "request_delay": 1.0
   }
   ```

3. Set Slack tokens in environment variables:
   ```bash
   export SLACK_XOXC_TOKEN="xoxc-..."
   export SLACK_XOXD_TOKEN="xoxd-..."
   ```

### Running the Dump

```bash
make dump
```

This will:
- Fetch messages from configured channels
- Generate vector embeddings using sentence-transformers
- Store them in separate Milvus databases (public/private)
- Save progress to allow interrupt/resume
- Resolve user mentions to display names

The process is idempotent - you can run it multiple times to index new messages without creating duplicates.

## MCP Tools

The server exposes two MCP tools:

### `smart_search`

Deep semantic nearest-neighbor search over historical Slack messages.

**Parameters:**
- `query` (required): Natural language description of what you're looking for
- `top_k` (optional): Number of results to return (default: 10)
- `search_scope` (optional): "public", "private", or "all" (default: "public")

**Returns:**
- `message`: Summary of search results
- `results`: Array of matching messages with text, user, timestamp, and URL

**Example usage:**
```
Find discussions about database performance issues in RHOAI
```

### `search_stats`

Get statistics about the search databases.

**Returns:**
- Database availability and paths
- Message counts for public/private databases
- Embedding model information
- Workspace URL

## Interactive Search

For local testing, use the interactive search tool:

```bash
make search
```

Commands:
- `<query>` - Search for messages
- `/public` - Search only public channels (default)
- `/private` - Search only private channels
- `/all` - Search both databases
- `/quit` - Exit

## Environment Variables

The MCP server supports the following environment variables:

- `MCP_TRANSPORT`: Transport type (default: "http")
- `DB_PATH`: Path to database directory (default: "/data/db")
- `COLLECTION_NAME`: Milvus collection name (default: "slack_messages")
- `EMBEDDING_MODEL`: Sentence transformer model (default: "all-MiniLM-L6-v2")
- `WORKSPACE_URL`: Slack workspace URL (default: "https://redhat-internal.slack.com")
- `TOP_K`: Default number of results (default: 10)

## Development

### Prerequisites

- Python 3.11+
- Podman or Docker (for containerized deployment)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Project Structure

The project uses a modular structure:
- **Config** (`config.py`): Centralized configuration management
- **Slack Client** (`slack_client.py`): Async Slack API wrapper with user caching
- **Milvus Store** (`milvus_store.py`): Vector database operations
- **Helpers** (`helpers.py`): Progress tracking and utilities

### Building Containers

```bash
# Build multi-platform manifest (amd64 + arm64)
make build

# Rebuild without cache
make rebuild

# Push to registry
make push
```

### Managing Databases

```bash
# Delete all databases (requires confirmation)
make nuke-dbs
```

## Use Cases

- **Knowledge Discovery**: Find past discussions about specific topics without knowing exact keywords
- **Troubleshooting**: Locate similar issues and their solutions from conversation history
- **Context Retrieval**: Get background information on projects, decisions, or technical discussions
- **Team Onboarding**: Help new team members discover relevant historical context
- **Documentation Mining**: Extract insights from informal Slack conversations

## Technical Details

### Embedding Model

Default model: `all-MiniLM-L6-v2` (384 dimensions)
- Fast inference
- Good balance of quality and performance
- CPU-optimized for containerized deployment

Other supported models:
- `all-mpnet-base-v2` (768 dimensions, higher quality)
- `paraphrase-MiniLM-L6-v2` (384 dimensions)

### Vector Database

Uses Milvus Lite for embedded vector storage:
- SQLite-based storage (no external dependencies)
- Efficient nearest-neighbor search
- Separate databases for public/private data
- In-memory caching for fast queries

### Message Processing

- User mentions (`<@USER_ID>`) resolved to display names
- Original raw JSON preserved for reference
- Timestamps converted to readable format
- Direct message URLs constructed for easy access

## Makefile Targets

- `make dump` - Run data ingestion pipeline
- `make search` - Run interactive search CLI
- `make build` - Build container image
- `make rebuild` - Build without cache
- `make run-mcp-local` - Run MCP server locally
- `make run-mcp` - Run MCP server from registry
- `make stop-mcp` - Stop running MCP server
- `make claude-mcp-import` - Import to Claude CLI
- `make nuke-dbs` - Delete all databases
- `make clean` - Clean build artifacts
- `make push` - Push to container registry

## License

See LICENSE file for details.


## Contributors
* Andrej Podhradsky ([@apodhrad](https://github.com/apodhrad))
* Marian Macik ([@MarianMacik](https://github.com/MarianMacik))
* Jakub Stetina ([@jstetina](https://github.com/jstetina))
* Filip Roman ([@RomanFilip](https://github.com/RomanFilip))
* Karel Suta ([@sutaakar](https://github.com/sutaakar))
* Jiri Danek ([@jiridanek](https://github.com/jiridanek))
