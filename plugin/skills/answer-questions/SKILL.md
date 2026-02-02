---
name: answer-questions

description: Searches for question messages in a Slack channel, finds answers using smart-search, replies to threads, and marks questions as done

argument-hint: [CHANNEL_NAME_OR_ID]

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

This skill automates the process of finding unanswered questions in Slack channels, searching for answers using smart-search, and responding to them.

## Arguments

1. **CHANNEL_NAME_OR_ID** (optional): Slack channel name (e.g., `wg-openshift-ai-hack-ai-thon-2026q1-brno-group11`) or channel ID (e.g., `C0ABACW6CG1`)
   - If not provided, use the default channel: `wg-openshift-ai-hack-ai-thon-2026q1-brno-group11`

## Workflow Steps

### Step 1: Search for Question Messages

Use the Slack MCP `search_messages` tool to find messages that start with "Question":

```
Query: Question in:#CHANNEL_NAME
Sort: timestamp
```

**Important:**
- Only process questions that do NOT have a `white_check_mark` (done) reaction already.
- **Questions can appear in threads!** The `search_messages` API may not reliably return all questions posted as thread replies. After searching, also use `get_channel_history` to check for questions that were posted in threads.
- When checking channel history, look at both top-level messages AND thread replies to find all questions.

### Step 2: Process Each Question

For each unanswered question found:

1. **Extract the question text** (remove the "Question:" prefix)
2. **Search for answers** using the `smart_search` MCP tool:
   - Query: The extracted question text
   - top_k: 10
   - search_scope: "all" (searches both public and private channels)

3. **Format the response** with:
   - Most relevant Slack conversations
   - Links to Jira issues (if found)
   - Common causes or solutions
   - Troubleshooting steps (if applicable)

4. **Post the response** as a reply to the question thread using `post_message`:
   - channel_id: The channel where the question was posted
   - thread_ts: The timestamp of the question message
   - message: The formatted smart-search response

5. **Mark the question as done** using `add_reaction`:
   - channel_id: The channel where the question was posted
   - message_ts: The timestamp of the question message
   - reaction: "white_check_mark"

### Step 3: Summary

After processing all questions, provide a summary:
```
Processed X question(s):
- Question 1: [Brief summary] - Answered and marked as done
- Question 2: [Brief summary] - Answered and marked as done
```

## Response Format Template

When posting answers to question threads, use this format:

```
Here are the most relevant Slack conversations about [TOPIC]:

**Most Recent Issue:**
• [Brief description] - [Slack link]
  Issue: [Jira link if available]

**Known Blocker/Issue:**
• [Issue description]
  Issue: [Jira link]
  Reported by: [Name]

**Related Issues:**
• [Related issue 1]
• [Related issue 2]

**Common Causes:**
- Cause 1
- Cause 2
- Cause 3

**Troubleshooting Steps:**
[Relevant commands or steps to diagnose/fix the issue]
```

## What to Include in Responses

- Most relevant and recent Slack messages
- Full Slack thread links (permalinks)
- Jira issue links from search results
- Common causes or patterns
- Actionable troubleshooting steps
- Component or service names mentioned

## What to Exclude

- Irrelevant search results (low relevance score)
- Duplicate information
- Overly verbose explanations
- Personal opinions or speculation

## Error Handling

- If no questions are found: Report "No unanswered questions found in #CHANNEL_NAME"
- If smart-search returns no results: Post a reply saying "No relevant information found in Slack history for this question."
- If posting fails: Report the error and skip marking as done

## Implementation Notes

1. **Always check for existing reactions** before processing to avoid duplicate work
2. **Check both top-level messages AND thread replies** - Questions can be posted as replies in threads, and `search_messages` may not always return them. Use `get_channel_history` after searching to find questions in threads.
3. **Process questions sequentially** to maintain order and avoid rate limits
4. **Use full permalinks** for Slack threads in responses
5. **Extract Jira issue keys** (RHOAIENG-*, RHAIENG-*) from search results and format as links
6. **Sort search results** by relevance and recency
7. **Group related issues** by topic or component when formatting responses

## Example Usage

```
/answer-questions                                    # Process questions in default channel
/answer-questions wg-3_3-openshift-ai-release       # Process questions in specific channel
/answer-questions C0A6R461R46                        # Process questions using channel ID
```

## Example Question Processing

### Example 1: Top-Level Question

**Input Question:**
```
Question: why dsc is not in ready state in 3.3?
```

**Smart-Search Query:**
```
why dsc is not in ready state in 3.3?
```

**Response Format:**
```
Here are the most relevant Slack conversations about DSC not being in ready state in 3.3:

**Most Recent Issue:**
• Trainer is not coming up with the DSC on 3.3 - https://redhat-internal.slack.com/archives/...
  Issue: https://issues.redhat.com/browse/RHOAIENG-44454

**Known Blocker (RHOAI 3.3):**
• DSC shows Not Ready when kueue is Unmanaged in latest 3.3 build
  Issue: https://issues.redhat.com/browse/RHOAIENG-43686
  Reported by: Anthony Coughlin

[... additional context ...]

Check the component statuses with: `oc get dsc default-dsc -o yaml`
```

**Then mark with ✓ reaction**

### Example 2: Question Posted in Thread

**Scenario:** A question is posted as a reply in an existing thread (e.g., as a follow-up to "Question: how many components are supported for rhoai?")

**Input Question (in thread):**
```
Question: which components are not supported in rhoai 3.3?
```

**How to Find:**
1. `search_messages` may not return this question since it's in a thread
2. Use `get_channel_history` to check channel messages including thread replies
3. Look for messages with `thread_ts` and check their reactions

**Processing:**
- Extract question text: "which components are not supported in rhoai 3.3"
- Search using smart-search
- Post reply to the thread using the `thread_ts` from the question message
- Add white_check_mark reaction to the question message timestamp
