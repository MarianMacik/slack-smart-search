---
name: get-latest-build

description: Finds the latest OpenShift AI build info on rhoai-build-notifications (for RHOAI) and wg-odh-nightly (for ODH) channels

argument-hint: [TYPE] [BUILD_TYPE] [VERSION]

disable-model-invocation:

user-invocable: true

allowed-tools:

model:

context:

agent:

hooks:
  PreToolUse:
    - matcher: ""
      command: ""

  PostToolUse:
    - matcher: ""
      command: ""

  UserPromptSubmit:
    - command: ""

  ModelResponse:
    - command: ""

---

# ============================================================================
# SKILL CONTENT - Instructions for Claude
# ============================================================================

## Arguments

This skill accepts up to 3 optional arguments:

1. **TYPE** - Type of the build: `RHOAI` or `ODH`
   - Default: `RHOAI` if not provided

2. **BUILD_TYPE** - Build type: `NIGHTLY` or `CI`
   - Only relevant for RHOAI builds
   - Ignored for ODH builds
   - Default: `NIGHTLY` if not provided

3. **VERSION** - Specific version to search for (e.g., "2.15", "2.16")
   - If not provided, find the latest build regardless of version

## Channel Selection

- **RHOAI**: Use channel `rhoai-build-notifications`
- **ODH**: Use channel `wg-odh-nightly`

## Search Strategy

1. **Always use Slack MCP server tools** - The Slack MCP server is connected to the live Slack instance with real-time data
   - Primary tool: `search_messages` tool of the Slack MCP server
   - DO NOT use smart search semantic search for this task

2. **Search query construction**:
   - For RHOAI: Include BUILD_TYPE (CI/NIGHTLY) in the search
   - If VERSION is specified, include it in the search query
   - Search recent messages first (use `sort: "timestamp"` parameter)

3. **Result filtering**:
   - Look for messages containing build notifications
   - Identify success/failure status from message content
   - Extract URLs for images, commits, and build logs

4. **Fallback**: If search doesn't return results, try:
   - Broader search terms
   - Check channel history directly using `get_channel_history`
   - Adjust VERSION specificity if too restrictive

## Information Extraction

Parse the message content to extract:
- Build status (SUCCESS/FAILURE)
- Build timestamp or message timestamp
- Image URL (container registry URL)
- Commit URL (GitHub commit link for RHOAI)
- Build URL (link to CI/build system)
- Slack message permalink

## Output Format

Present the information clearly with the following structure:

**Build Information**
- **Type**: RHOAI/ODH
- **Build Type**: CI/NIGHTLY (for RHOAI only)
- **Version**: [version number if applicable]
- **Status**: [🟢 SUCCESS | 🔴 FAILED]
- **Date**: [Full timestamp with date and time]
- **Image URL**: [container image URL]
- **Commit URL**: [GitHub commit link] (RHOAI only)
- **Build URL**: [CI/build system link]
- **Slack Message**: [Direct link to message]

## Error Handling

- If no builds found: Clearly state "No builds found matching the criteria"
- If channel access fails: Suggest checking channel permissions
- If message format is unexpected: Show what information was found and what's missing
