---
name: rhoai-build

description: Fetches successful RHOAI builds from rhoai-build-notifications channel with optional version filtering

argument-hint: [VERSION] [LIMIT] [DAYS]

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

This skill fetches successful RHOAI builds from the `rhoai-build-notifications` Slack channel.

## Arguments

1. **VERSION** (optional): Specific RHOAI version to filter by (e.g., `3.4`, `3.3`)
2. **LIMIT** (optional): Maximum number of builds to show per version (default: 3 for specific versions, 1 per version for all builds)
3. **DAYS** (optional): Number of days to look back (default: 7)

## Search Query Format

**For all successful builds (no version specified):**
```
in:rhoai-build-notifications :solid-success: after:7days
```

**For specific version:**
```
in:rhoai-build-notifications :solid-success: 3.4 after:7days
```

**Note:** Adjust `after:Xdays` based on DAYS argument if provided.

## Output Format

Display only the image and link to the Slack thread:

```
### RHOAI v3.4.0

1. **2026-01-28 06:43:26**
   - Image: `quay.io/rhoai/rhoai-fbc-fragment:rhoai-3.4@sha256:cfca3fc6e9289b6484faee3acd39d77653cf2947bd4f42d4a89c8b1cc096a3d3`
   - [Slack Thread](https://redhat-internal.slack.com/archives/C07ANR2U56C/p1769582707254509)
```

## What to Include

- Build timestamp (from message)
- Image URL (full quay.io path with SHA256 digest)
- Link to Slack thread (permalink)

## What to Exclude

- Commit hashes
- Build URLs (Konflux links)
- Failed builds (only show `:solid-success:` builds)

## Implementation Notes

1. Use the Slack MCP `search_messages` tool to find messages
2. Parse messages to extract:
   - Timestamp
   - Version (from message content, displayed as full version like v3.4.0)
   - Image URL with SHA256 digest
3. Generate permalink for each message
4. Group by version if multiple versions found
5. Sort by timestamp (most recent first)
6. Limit results based on LIMIT argument:
   - When VERSION is specified: Show LIMIT builds for that version (default: 3)
   - When VERSION is not specified: Show LIMIT builds per version, grouped by version (default: 1 per version)

## Example Usage

```
/rhoai-build           # Show latest build for each version from last 7 days
/rhoai-build 3.4       # Show last 3 builds for version 3.4
/rhoai-build 3.4 5     # Show last 5 builds for version 3.4
/rhoai-build "" 5      # Show last 5 builds for each version
```
