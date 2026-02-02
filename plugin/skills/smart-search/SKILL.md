# Smart Search Skill

This skill searches for relevant Slack conversations using semantic search and displays the results with formatted links.

## Arguments

1. **QUERY** (required): The search query or question to find relevant Slack conversations about
   - Example: `my model deployment is stuck and not starting`
   - Example: `how to configure GPU nodes in RHOAI`
   - Example: `dashboard not loading in 3.3`

## Workflow

1. **Execute Smart Search**
   - Use the `smart_search` MCP tool with the provided query
   - Set `top_k: 10` to get up to 10 relevant results
   - Set `search_scope: "all"` to search both public and private channels

2. **Format and Display Results**
   - Show the most relevant conversations found
   - Include plain Slack URLs (no display text formatting needed for terminal output)
   - Include timestamps, channels, and message previews
   - Extract any Jira issue references (RHOAIENG-*, RHAIENG-*) and format as links

3. **Result Structure**
   Each result should include:
   - Brief message preview or summary
   - Plain Slack thread URL
   - Timestamp (human-readable format)
   - Channel name or ID if available
   - User who posted (if available)

## Output Format

```
Found X relevant Slack conversations about [QUERY]:

1. [Brief description or message preview]
   Slack-URL - [Channel] - [Timestamp]
   [Additional context if relevant]

2. [Brief description or message preview]
   Slack-URL - [Channel] - [Timestamp]
   [Additional context if relevant]

...

**Related Jira Issues Found:**
- https://issues.redhat.com/browse/RHOAIENG-12345
- https://issues.redhat.com/browse/RHOAIENG-67890
```

## Error Handling

- If no results found: Display "No relevant Slack conversations found for: [QUERY]"
- If search fails: Display error message and suggest alternative search terms
- If query is empty: Prompt user to provide a search query

## Example Usage

```
/smart-search my model deployment is stuck and not starting
/smart-search dashboard not loading in RHOAI 3.3
/smart-search how to configure service mesh
/smart-search DSCI not ready state
```

## Implementation Notes

1. **Always use the MCP smart-search tool** - Do not use regular Slack search
2. **Use plain URLs** - Print plain Slack URLs without display text formatting (terminal output, not Slack messages)
3. **Extract Jira references** - Look for patterns like `RHOAIENG-\d+` or `RHAIENG-\d+` in the results
4. **Sort by relevance** - Results are already sorted by relevance from smart-search
5. **Limit context** - Show brief previews, not full message text unless particularly relevant
6. **Include metadata** - Timestamp, channel, user info when available

## Example Output

```
Found 5 relevant Slack conversations about "model deployment stuck":

1. Deployment stuck, pods scaled to 0 after creation
   https://redhat-internal.slack.com/archives/C03UGJY6Z1A/p1738273428739749 - #forum-openshift-ai - Jan 30, 2026
   Issue: vLLM deployment creates pod, then scales to 0 after a few minutes

2. Granite model deployment status never completes
   https://redhat-internal.slack.com/archives/C03UGJY6Z1A/p1745923705566819 - #forum-openshift-ai - Apr 29, 2025
   RHOAI 2.13.1 - Predictor pods running but deployment stuck

3. KServe Raw Deployment - no resources created
   https://redhat-internal.slack.com/archives/C07KPDHBR4J/p1739362667725819 - #forum-rhoai-platform - Feb 12, 2025
   Error: "Knative Serving is not available"

**Related Jira Issues Found:**
- https://issues.redhat.com/browse/RHOAIENG-44454
```
