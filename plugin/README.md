# Smart Search Plugin for Claude Code

A Claude Code plugin that enables semantic search and intelligent Slack interaction for the Red Hat OpenShift AI team. This plugin combines two powerful MCP servers to provide AI-powered search capabilities and automated question answering within Slack workspaces.

## Overview

This plugin integrates two MCP servers:
- **Slack MCP**: Direct interaction with Slack (reading messages, posting, reactions, channel management)
- **Smart Search MCP**: Semantic search over Slack message history using vector embeddings and Milvus

Together, they enable Claude to understand and navigate your Slack workspace more intelligently than simple keyword search.

## Features

### Semantic Search
- Search Slack conversations by meaning, not just keywords
- Find relevant discussions even when you don't know exact terminology
- Ranked results based on semantic similarity
- Search across public and private channels

### Build Information Tracking
- Find latest RHOAI (Red Hat OpenShift AI) and ODH (Open Data Hub) builds
- Track build status and versions
- Quick access to build artifacts and links

### Automated Question Answering
- Monitor channels for unanswered questions
- Automatically search for relevant answers in Slack history
- Post comprehensive answers with links to related discussions and Jira issues
- Mark questions as resolved

## Installation

### Prerequisites

1. **Slack Tokens**: You need two Slack tokens (XOXC and XOXD) from your workspace
   - See [Slack token extractor](https://github.com/maorfr/slack-token-extractor) for token extraction instructions

2. **Smart Search Server**: The smart-search MCP server must be running and accessible
   - Can run locally or remotely
   - Requires a populated Milvus vector database with Slack message embeddings

### Setup

1. Copy the plugin directory to your Claude Code plugins folder:
   ```bash
   cp -r plugin ~/.config/claude-code/plugins/smart-search
   ```
   or run Claude Code using
   ```bash
   claude --plugin-dir plugin
   ```

2. Set required environment variables:
   ```bash
   export SLACK_XOXC_TOKEN="your-xoxc-token"
   export SLACK_XOXD_TOKEN="your-xoxd-token"
   export SMART_SEARCH_URL="http://localhost:8000/mcp"  # or your remote URL
   ```

3. Optionally configure logging channel:
   ```bash
   export LOGS_CHANNEL_ID="C0123456789"  # Channel ID for Slack operation logs
   ```

## Configuration

The plugin uses the MCP configuration file at `plugin/.mcp.json`:

```json
{
  "mcpServers": {
    "slack": {
      "command": "podman",
      "args": ["run", "-i", "--rm", "-e", "SLACK_XOXC_TOKEN", "-e", "SLACK_XOXD_TOKEN", "-e", "MCP_TRANSPORT", "-e", "LOGS_CHANNEL_ID", "quay.io/redhat-ai-tools/slack-mcp"],
      "env": {
        "SLACK_XOXC_TOKEN": "${SLACK_XOXC_TOKEN}",
        "SLACK_XOXD_TOKEN": "${SLACK_XOXD_TOKEN}",
        "MCP_TRANSPORT": "stdio",
        "LOGS_CHANNEL_ID": ""
      }
    },
    "smart-search": {
      "type": "http",
      "url": "${SMART_SEARCH_URL:-http://localhost:8000/mcp}"
    }
  }
}
```

**Note**: You can use Docker instead of Podman by changing the `command` to `"docker"`.

## Available Skills

### `/smart-search` - Semantic Search

Search for relevant Slack conversations using natural language queries.

**Usage:**
```bash
/smart-search <query>
```

**Examples:**
```bash
/smart-search my model deployment is stuck and not starting
/smart-search dashboard not loading in RHOAI 3.3
/smart-search how to configure service mesh
/smart-search DSCI not ready state
```

**Output:**
- Ranked list of relevant Slack conversations
- Message previews with context
- Links to Slack threads
- Related Jira issues (RHOAIENG-*, RHAIENG-*)
- Timestamps and channel information

### `/answer-questions` - Automated Q&A

Finds unanswered questions in a Slack channel and posts intelligent answers.

**Usage:**
```bash
/answer-questions [CHANNEL_NAME_OR_ID]
```

**Examples:**
```bash
/answer-questions                                    # Use default channel
/answer-questions wg-3_3-openshift-ai-release       # Specific channel by name
/answer-questions C0A6R461R46                        # Specific channel by ID
```

**What it does:**
1. Searches for messages starting with "Question:" (including in threads)
2. Skips questions already marked with ✓ reaction
3. Uses semantic search to find relevant answers
4. Posts comprehensive responses with:
   - Related Slack discussions
   - Jira issue links
   - Common causes and solutions
   - Troubleshooting steps
5. Marks questions as done with ✓ reaction

### `/get-latest-build` - Latest Build Information

Finds the most recent OpenShift AI build information.

**Usage:**
```bash
/get-latest-build [TYPE] [BUILD_TYPE] [VERSION]
```

**Arguments:**
- `TYPE`: `RHOAI` or `ODH` (default: RHOAI)
- `BUILD_TYPE`: `NIGHTLY` or `CI` (default: NIGHTLY, RHOAI only)
- `VERSION`: Specific version like "2.15" or "3.3" (optional)

**Examples:**
```bash
/get-latest-build                    # Latest RHOAI nightly build
/get-latest-build RHOAI CI           # Latest RHOAI CI build
/get-latest-build RHOAI NIGHTLY 3.3  # Latest RHOAI 3.3 nightly
/get-latest-build ODH                # Latest ODH build
```

**Output:**
- Build status (SUCCESS/FAILED)
- Build timestamp
- Container image URL
- Commit URL (RHOAI)
- Build system URL
- Slack message link

### `/rhoai-build` - RHOAI Build History

Fetches successful RHOAI builds from build notifications channel.

**Usage:**
```bash
/rhoai-build [VERSION] [LIMIT] [DAYS]
```

**Arguments:**
- `VERSION`: Filter by version (e.g., "3.4", "3.3")
- `LIMIT`: Number of builds to show (default: 3 for specific version, 1 per version otherwise)
- `DAYS`: Days to look back (default: 7)

**Examples:**
```bash
/rhoai-build           # Latest build for each version from last 7 days
/rhoai-build 3.4       # Last 3 builds for version 3.4
/rhoai-build 3.4 5     # Last 5 builds for version 3.4
/rhoai-build "" 5      # Last 5 builds for each version
```

**Output:**
- Build timestamp
- Container image URL with SHA256 digest
- Slack thread link
- Grouped by version

## MCP Tools Reference

### Slack MCP Tools

The Slack MCP server provides these tools:
- `get_channel_history` - Read message history from a channel
- `post_message` - Post messages to channels or threads
- `post_command` - Execute Slack commands
- `add_reaction` - Add emoji reactions to messages
- `join_channel` - Join a Slack channel
- `send_dm` - Send direct messages
- `search_messages` - Search for messages using Slack search
- `whoami` - Check authentication status

### Smart Search MCP Tools

The smart-search MCP server provides:
- `smart_search` - Semantic nearest-neighbor search over Slack messages
  - Parameters:
    - `query`: Natural language search query
    - `top_k`: Number of results (default: 10)
    - `search_scope`: "public", "private", or "all" (default: "public")
    - `user`: Filter by user ID or username
    - `start_date`: Filter by date range (YYYY-MM-DD)
    - `end_date`: Filter by date range (YYYY-MM-DD)
- `search_stats` - Get database statistics (message counts, etc.)

## Running the Smart Search Server Locally

### Quick Start with Make

```bash
# Run the smart-search MCP server locally
make run-mcp-local
```

This starts the server on `http://localhost:8000/mcp`.

### Manual Container Run

```bash
# With Podman
podman run -d -p 8000:8000 \
  -e SLACK_XOXC_TOKEN="$SLACK_XOXC_TOKEN" \
  -e SLACK_XOXD_TOKEN="$SLACK_XOXD_TOKEN" \
  -e MILVUS_URI="your-milvus-uri" \
  -e MILVUS_TOKEN="your-milvus-token" \
  quay.io/redhat-ai-tools/slack-smart-search

# With Docker
docker run -d -p 8000:8000 \
  -e SLACK_XOXC_TOKEN="$SLACK_XOXC_TOKEN" \
  -e SLACK_XOXD_TOKEN="$SLACK_XOXD_TOKEN" \
  -e MILVUS_URI="your-milvus-uri" \
  -e MILVUS_TOKEN="your-milvus-token" \
  quay.io/redhat-ai-tools/slack-smart-search
```

## Troubleshooting

### Slack Authentication Issues

If you get authentication errors:
1. Verify your XOXC and XOXD tokens are correct
2. Check token hasn't expired
3. Ensure your Slack workspace allows the tokens

### Smart Search Not Finding Results

If semantic search returns no results:
1. Verify the smart-search server is running and accessible
2. Check that the Milvus database has been populated with message embeddings
3. Try broader search terms
4. Check the `search_scope` parameter (use "all" for public + private)

### Plugin Not Loading

If the plugin doesn't appear in Claude Code:
1. Verify the plugin directory is in the correct location
2. Check that `.mcp.json` is valid JSON
3. Ensure environment variables are set
4. Restart Claude Code

### Channel Access Issues

If you can't access certain channels:
1. Use `/join_channel` skill to join the channel first
2. Verify your Slack tokens have access to the workspace
3. Check channel permissions

## Development

### Project Structure

```
plugin/
├── .claude-plugin/
│   └── plugin.json          # Plugin metadata
├── .mcp.json                # MCP server configuration
├── README.md                # This file
└── skills/
    ├── answer-questions/
    │   └── SKILL.md         # Answer questions skill definition
    ├── get-latest-build/
    │   └── SKILL.md         # Get latest build skill definition
    ├── rhoai-build/
    │   └── SKILL.md         # RHOAI build history skill definition
    └── smart-search/
        └── SKILL.md         # Smart search skill definition
```

### Adding New Skills

To add a new skill:
1. Create a new directory under `skills/`
2. Add a `SKILL.md` file with the skill definition
3. Follow the existing skill format with frontmatter and instructions

### Modifying MCP Servers

To use different MCP server versions or configurations:
1. Edit `.mcp.json`
2. Update server URLs, environment variables, or Docker images
3. Restart Claude Code to apply changes

## Use Cases

### For Engineers
- Quickly find solutions to technical problems from past discussions
- Track build statuses and access build artifacts
- Get context on ongoing issues without searching manually

### For Team Leads
- Monitor unanswered questions and ensure team support
- Track recurring issues across conversations
- Build knowledge base from Slack history

### For Support Teams
- Automate first-level support responses
- Find relevant past solutions for customer issues
- Maintain response consistency with historical answers

## Privacy & Security

- Slack tokens are stored in environment variables, not in code
- Smart search respects Slack channel permissions
- Private channel content only accessible with proper scope
- All MCP communication uses secure protocols
- Optional logging channel for audit trails

## Contributing

This plugin is part of the Red Hat OpenShift AI tooling ecosystem. For questions or contributions:
- Create issues in the project repository
- Follow the contribution guidelines
- Test changes thoroughly before submitting

## License

See the main project LICENSE file for details.

## Support

For issues or questions:
- Check this documentation first
- Review skill definitions in `skills/*/SKILL.md`
- Open an issue in the project repository
- Contact the Red Hat OpenShift AI team

## Related Projects

- [Slack MCP Server](https://github.com/redhat-community-ai-tools/slack-mcp)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Claude Code](https://claude.ai/code)
- [Milvus Vector Database](https://milvus.io/)
