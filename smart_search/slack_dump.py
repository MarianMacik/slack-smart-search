#!/usr/bin/env python3
"""
Slack Channel Dump Script

This script dumps Slack channel messages to Milvus vector databases.
It uses two separate collections:
  - Public channels: Can be shared with others
  - Private channels: User-specific, not shared

It is idempotent (won't add duplicates) and supports interrupt/resume.

Usage:
    python slack_dump.py

Configuration via environment variables or config file (dump_config.json).
"""

import os
import sys
import json
import signal
import asyncio
import logging
import time
import warnings
from datetime import datetime
from typing import Any
from pathlib import Path

# Suppress verbose HuggingFace/transformers logging and warnings before imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import httpx
from pymilvus import (
    MilvusClient,
    DataType,
)

# Suppress HF warning during sentence_transformers import
import io
_old_stderr = sys.stderr
sys.stderr = io.StringIO()
try:
    from sentence_transformers import SentenceTransformer
finally:
    sys.stderr = _old_stderr

# Model dimensions (must match the model used)
EMBEDDING_DIMS = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "paraphrase-MiniLM-L6-v2": 384,
}
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Configuration defaults
DEFAULT_CONFIG_PATH = Path(__file__).parent / "dump_config.json"
DEFAULT_PROGRESS_PATH = Path(__file__).parent / "dump_progress.json"
DEFAULT_RAW_RESPONSES_DIR = Path(__file__).parent / "raw_responses"
SLACK_API_BASE = "https://slack.com/api"

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global shutdown_requested
    print("\n[!] Shutdown requested. Finishing current operation...")
    shutdown_requested = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class RawResponseLogger:
    """Logs raw API responses to JSONL files for future reference."""

    def __init__(self, output_dir: Path = DEFAULT_RAW_RESPONSES_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log_messages(self, channel_id: str, messages: list[dict], source: str = "history"):
        """Append raw messages to channel's JSONL file."""
        if not messages:
            return

        filepath = self.output_dir / f"{channel_id}.jsonl"
        with open(filepath, "a", encoding="utf-8") as f:
            entry = {
                "source": source,
                "timestamp": datetime.now().isoformat(),
                "count": len(messages),
                "messages": messages,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def parse_timestamp(value: str) -> str:
    """
    Parse a timestamp value to Unix timestamp string.

    Supports:
      - Unix timestamp (e.g., "1704067200")
      - YYYY-MM-DD format (e.g., "2024-01-01")
      - YYYY-MM-DD HH:MM:SS format (e.g., "2024-01-01 12:00:00")

    Returns Unix timestamp as string.
    """
    if not value or value == "0":
        return "0"

    # Check if it's already a Unix timestamp (all digits, optionally with decimal)
    if value.replace(".", "").isdigit():
        return value

    # Try parsing as date/datetime
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
        try:
            dt = datetime.strptime(value, fmt)
            return str(dt.timestamp())
        except ValueError:
            continue

    print(f"[!] Warning: Could not parse timestamp '{value}', using 0")
    return "0"


class Config:
    """Configuration holder."""

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH):
        self.config_path = config_path
        self.load()

    def load(self):
        """Load configuration from file or environment."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                data = json.load(f)
        else:
            data = {}

        # Slack tokens from env or config
        self.xoxc_token = os.environ.get("SLACK_XOXC_TOKEN", data.get("xoxc_token", ""))
        self.xoxd_token = os.environ.get("SLACK_XOXD_TOKEN", data.get("xoxd_token", ""))

        # Milvus database files (separate files for public/private so public can be shared)
        # Use ".db" extension for Milvus Lite (file-based, no server needed)
        self.public_db = data.get("public_db", "./slack_public.db")
        self.private_db = data.get("private_db", "./slack_private.db")
        self.milvus_token = os.environ.get("MILVUS_TOKEN", data.get("milvus_token", ""))

        # Collection name (same for both databases since they're separate files)
        self.collection_name = data.get("collection_name", "slack_messages")

        # Embedding model
        self.embedding_model = data.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
        self.embedding_dim = EMBEDDING_DIMS.get(self.embedding_model, 384)

        # Channels to dump - now split by visibility
        # Public channels: can be shared with others
        self.public_channels = data.get("public_channels", [])
        # Private channels: user-specific, DMs, private groups
        self.private_channels = data.get("private_channels", [])

        # Start timestamp - only index messages after this time
        # Format: Unix timestamp or YYYY-MM-DD (e.g., "1704067200" or "2024-01-01")
        self.start_timestamp = parse_timestamp(data.get("start_timestamp", "0"))

        # Rate limiting
        self.request_delay = data.get("request_delay", 1.0)  # seconds between requests


class ProgressTracker:
    """Tracks progress for interrupt/resume support."""

    def __init__(self, progress_path: Path = DEFAULT_PROGRESS_PATH):
        self.progress_path = progress_path
        self.progress: dict[str, dict] = {}
        self.load()

    def load(self):
        """Load progress from file."""
        if self.progress_path.exists():
            with open(self.progress_path) as f:
                self.progress = json.load(f)
        else:
            self.progress = {}

    def save(self):
        """Save progress to file."""
        with open(self.progress_path, "w") as f:
            json.dump(self.progress, f, indent=2)

    def get_channel_progress(self, channel_id: str) -> dict:
        """Get progress for a specific channel."""
        return self.progress.get(channel_id, {})

    def update_channel_progress(
        self,
        channel_id: str,
        oldest_ts: str | None = None,
        newest_ts: str | None = None,
        messages_indexed: int | None = None,
        completed: bool = False,
    ):
        """Update progress for a channel."""
        if channel_id not in self.progress:
            self.progress[channel_id] = {
                "oldest_ts": None,
                "newest_ts": None,
                "messages_indexed": 0,
                "completed": False,
                "last_updated": None,
            }

        if oldest_ts is not None:
            self.progress[channel_id]["oldest_ts"] = oldest_ts
        if newest_ts is not None:
            self.progress[channel_id]["newest_ts"] = newest_ts
        if messages_indexed is not None:
            self.progress[channel_id]["messages_indexed"] = messages_indexed
        if completed:
            self.progress[channel_id]["completed"] = True

        self.progress[channel_id]["last_updated"] = datetime.now().isoformat()
        self.save()

    def is_channel_completed(self, channel_id: str, start_timestamp: str) -> bool:
        """Check if a channel has been fully indexed since start_timestamp."""
        progress = self.get_channel_progress(channel_id)
        if not progress.get("completed"):
            return False
        # Check if we've indexed back to the start timestamp
        oldest = progress.get("oldest_ts")
        if oldest and float(oldest) <= float(start_timestamp):
            return True
        return False


class SlackClient:
    """Slack API client."""

    def __init__(self, config: Config):
        self.config = config
        self._client: httpx.AsyncClient | None = None
        self._user_cache: dict[str, str] = {}  # user_id -> display_name

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()

    async def _request(
        self,
        endpoint: str,
        method: str = "POST",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Make a request to the Slack API."""
        url = f"{SLACK_API_BASE}/{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.config.xoxc_token}",
            "Content-Type": "application/json",
            "User-Agent": "SlackDump/1.0",
        }
        cookies = {"d": self.config.xoxd_token}

        try:
            if method.upper() == "GET":
                response = await self._client.request(
                    method, url, headers=headers, cookies=cookies, params=payload
                )
            else:
                response = await self._client.request(
                    method, url, headers=headers, cookies=cookies, json=payload
                )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"[!] API request failed: {e}")
            return None

    async def get_channel_history(
        self,
        channel_id: str,
        cursor: str | None = None,
        oldest: str | None = None,
        latest: str | None = None,
        limit: int = 100,
    ) -> tuple[list[dict], str | None, bool]:
        """
        Get channel history with pagination.

        Returns: (messages, next_cursor, has_more)
        """
        payload = {"channel": channel_id, "limit": limit}
        if cursor:
            payload["cursor"] = cursor
        if oldest:
            payload["oldest"] = oldest
        if latest:
            payload["latest"] = latest

        data = await self._request("conversations.history", payload=payload)
        if data and data.get("ok"):
            messages = data.get("messages", [])
            response_metadata = data.get("response_metadata", {})
            next_cursor = response_metadata.get("next_cursor")
            has_more = data.get("has_more", False)
            return messages, next_cursor, has_more

        error = data.get("error", "unknown") if data else "request failed"
        print(f"[!] Failed to get channel history: {error}")
        return [], None, False

    async def get_thread_replies(
        self, channel_id: str, thread_ts: str
    ) -> list[dict]:
        """Get all replies in a thread."""
        payload = {"channel": channel_id, "ts": thread_ts}
        data = await self._request("conversations.replies", payload=payload)
        if data and data.get("ok"):
            # First message is the parent, rest are replies
            messages = data.get("messages", [])
            return messages[1:] if len(messages) > 1 else []

        return []

    async def get_channel_info(self, channel_id: str) -> dict | None:
        """Get channel information."""
        payload = {"channel": channel_id}
        data = await self._request("conversations.info", payload=payload)
        if data and data.get("ok"):
            return data.get("channel")
        return None

    async def get_user_info(self, user_id: str) -> dict | None:
        """Get user information."""
        payload = {"user": user_id}
        data = await self._request("users.info", payload=payload)
        if data and data.get("ok"):
            return data.get("user")
        return None

    async def get_user_display_name(self, user_id: str) -> str:
        """Get user display name, with caching."""
        if user_id in self._user_cache:
            return self._user_cache[user_id]

        user_info = await self.get_user_info(user_id)
        if user_info:
            # Prefer display_name (non-empty), then real_name, then name
            profile = user_info.get("profile", {})
            display_name = (
                (profile.get("display_name") or "").strip()
                or (profile.get("real_name") or "").strip()
                or (user_info.get("real_name") or "").strip()
                or (user_info.get("name") or "").strip()
                or user_id
            )
            self._user_cache[user_id] = display_name
            return display_name

        # Cache failures too to avoid repeated lookups
        self._user_cache[user_id] = user_id
        return user_id

    async def resolve_user_mentions(self, text: str) -> str:
        """Resolve <@USER_ID> or <@USER_ID|name> mentions in text to @display_name."""
        import re

        # Pattern matches <@U12345678> or <@U12345678|display_name>
        # Group 1: user ID, Group 2: optional display name after pipe
        pattern = r"<@([A-Z0-9]+)(?:\|([^>]+))?>"

        # Find all matches first
        matches = list(re.finditer(pattern, text))
        if not matches:
            return text

        # Resolve all user IDs
        result = text
        for match in reversed(matches):  # Reverse to preserve positions
            user_id = match.group(1)
            # Use the display name from Slack if present, otherwise look it up
            slack_provided_name = match.group(2)
            if slack_provided_name:
                display_name = slack_provided_name
            else:
                display_name = await self.get_user_display_name(user_id)
            result = result[:match.start()] + f"@{display_name}" + result[match.end():]

        return result

    async def resolve_channel(self, channel: str) -> str | None:
        """
        Resolve a channel name or ID to a channel ID.

        Accepts:
          - Channel ID (C12345678, D12345678, G12345678) - returned as-is
          - Channel name with or without # (e.g., "general" or "#general")

        Returns the channel ID or None if not found.
        """
        # Strip # prefix if present
        channel = channel.lstrip("#")

        # Check if it looks like a channel ID already
        # Channel IDs start with C (public), D (DM), or G (private group)
        # followed by alphanumeric characters
        if len(channel) >= 9 and channel[0] in "CDG" and channel[1:].isalnum():
            return channel

        # Otherwise, treat it as a name and look it up
        print(f"[*] Resolving channel name: {channel}")

        # Search through all conversation types
        cursor = None
        while True:
            payload = {
                "types": "public_channel,private_channel,mpim,im",
                "limit": 200,
            }
            if cursor:
                payload["cursor"] = cursor

            data = await self._request("conversations.list", payload=payload)
            if not data or not data.get("ok"):
                error = data.get("error", "unknown") if data else "request failed"
                print(f"[!] Failed to list conversations: {error}")
                return None

            channels = data.get("channels", [])
            for ch in channels:
                if ch.get("name") == channel or ch.get("name_normalized") == channel:
                    channel_id = ch.get("id")
                    print(f"[*] Resolved '{channel}' to {channel_id}")
                    return channel_id

            # Check for pagination
            response_metadata = data.get("response_metadata", {})
            cursor = response_metadata.get("next_cursor")
            if not cursor:
                break

        print(f"[!] Could not find channel: {channel}")
        return None

    async def resolve_channels(self, channels: list[str]) -> list[str]:
        """Resolve a list of channel names/IDs to channel IDs."""
        resolved = []
        for channel in channels:
            channel_id = await self.resolve_channel(channel)
            if channel_id:
                resolved.append(channel_id)
            else:
                print(f"[!] Skipping unresolved channel: {channel}")
        return resolved


class EmbeddingModel:
    """Singleton embedding model to avoid loading multiple times."""

    _model = None
    _model_name = None

    @classmethod
    def get_model(cls, model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
        if cls._model is None or cls._model_name != model_name:
            print(f"[*] Loading embedding model: {model_name}")
            cls._model = SentenceTransformer(model_name)
            cls._model_name = model_name
        return cls._model

    @classmethod
    def encode(cls, text: str, model_name: str = DEFAULT_EMBEDDING_MODEL) -> list[float]:
        """Generate embedding for a single text."""
        model = cls.get_model(model_name)
        embedding = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return embedding.tolist()

    @classmethod
    def encode_batch(cls, texts: list[str], model_name: str = DEFAULT_EMBEDDING_MODEL) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        model = cls.get_model(model_name)
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()


class MilvusStore:
    """Milvus vector database storage for a specific collection."""

    def __init__(self, milvus_uri: str, milvus_token: str, collection_name: str, embedding_dim: int = 384, embedding_model: str = DEFAULT_EMBEDDING_MODEL):
        self.milvus_uri = milvus_uri
        self.milvus_token = milvus_token
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.embedding_model = embedding_model
        self.client: MilvusClient | None = None

    def connect(self):
        """Connect to Milvus and ensure collection exists."""
        print(f"[*] Connecting to Milvus at {self.milvus_uri}")
        # Ensure parent directory exists for local file-based Milvus
        if self.milvus_uri.startswith("./") or self.milvus_uri.startswith("/"):
            db_path = Path(self.milvus_uri)
            db_path.parent.mkdir(parents=True, exist_ok=True)
        self.client = MilvusClient(
            uri=self.milvus_uri,
            token=self.milvus_token if self.milvus_token else None,
        )
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        if self.client.has_collection(self.collection_name):
            print(f"[*] Collection '{self.collection_name}' already exists")
            return

        print(f"[*] Creating collection '{self.collection_name}'")

        # Define schema for Slack messages
        # We store the full message JSON for reconstruction
        # Primary key is channel_id + ts (message timestamp) as a composite key string
        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=True,  # Allow storing additional fields dynamically
        )

        # Primary key: unique message identifier (channel_id:ts)
        schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            max_length=100,
            is_primary=True,
        )

        # Channel ID for filtering
        schema.add_field(
            field_name="channel_id",
            datatype=DataType.VARCHAR,
            max_length=50,
        )

        # Message timestamp for ordering/filtering
        schema.add_field(
            field_name="ts",
            datatype=DataType.VARCHAR,
            max_length=30,
        )

        # Thread timestamp (if this is a reply)
        schema.add_field(
            field_name="thread_ts",
            datatype=DataType.VARCHAR,
            max_length=30,
        )

        # User ID
        schema.add_field(
            field_name="user",
            datatype=DataType.VARCHAR,
            max_length=50,
        )

        # User display name (resolved from ID)
        schema.add_field(
            field_name="user_name",
            datatype=DataType.VARCHAR,
            max_length=200,
        )

        # Message text (for search)
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=65535,  # Max VARCHAR length
        )

        # Message type
        schema.add_field(
            field_name="msg_type",
            datatype=DataType.VARCHAR,
            max_length=50,
        )

        # Full message JSON for reconstruction
        schema.add_field(
            field_name="raw_json",
            datatype=DataType.VARCHAR,
            max_length=65535,
        )

        # Vector field for semantic search embeddings
        schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.embedding_dim,
        )

        # Create index for vector field (required by Milvus)
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="FLAT",
            metric_type="L2",
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        print(f"[*] Collection '{self.collection_name}' created")

    def message_exists(self, channel_id: str, ts: str) -> bool:
        """Check if a message already exists in the database."""
        msg_id = f"{channel_id}:{ts}"
        results = self.client.query(
            collection_name=self.collection_name,
            filter=f'id == "{msg_id}"',
            output_fields=["id"],
            limit=1,
        )
        return len(results) > 0

    def get_newest_message_ts(self, channel_id: str) -> str | None:
        """Get the newest message timestamp for a channel."""
        results = self.client.query(
            collection_name=self.collection_name,
            filter=f'channel_id == "{channel_id}"',
            output_fields=["ts"],
            limit=1000,  # Get all and find max
        )
        if not results:
            return None

        # Find the newest timestamp
        timestamps = [r["ts"] for r in results if r.get("ts")]
        if timestamps:
            return max(timestamps, key=float)
        return None

    def get_oldest_message_ts(self, channel_id: str) -> str | None:
        """Get the oldest message timestamp for a channel."""
        results = self.client.query(
            collection_name=self.collection_name,
            filter=f'channel_id == "{channel_id}"',
            output_fields=["ts"],
            limit=1000,
        )
        if not results:
            return None

        timestamps = [r["ts"] for r in results if r.get("ts")]
        if timestamps:
            return min(timestamps, key=float)
        return None

    def get_newest_message_ts(self, channel_id: str) -> str | None:
        """Get the newest message timestamp for a channel."""
        results = self.client.query(
            collection_name=self.collection_name,
            filter=f'channel_id == "{channel_id}"',
            output_fields=["ts"],
            limit=1000,
        )
        if not results:
            return None

        timestamps = [r["ts"] for r in results if r.get("ts")]
        if timestamps:
            return max(timestamps, key=float)
        return None

    def insert_message(self, channel_id: str, message: dict) -> bool:
        """Insert a message into the database."""
        ts = message.get("ts", "")
        msg_id = f"{channel_id}:{ts}"

        # Prepare message data
        text = message.get("text", "")
        # Truncate text if too long
        if len(text) > 65000:
            text = text[:65000] + "..."

        raw_json = json.dumps(message, ensure_ascii=False)
        if len(raw_json) > 65000:
            # If too long, store essential fields only
            essential = {
                k: message[k]
                for k in ["ts", "user", "text", "type", "thread_ts", "reply_count", "reactions", "files", "attachments"]
                if k in message
            }
            raw_json = json.dumps(essential, ensure_ascii=False)
            if len(raw_json) > 65000:
                raw_json = raw_json[:65000]

        # Generate embedding for the text
        embedding_text = text if text else "(empty message)"
        vector = EmbeddingModel.encode(embedding_text, self.embedding_model)

        data = {
            "id": msg_id,
            "channel_id": channel_id,
            "ts": ts,
            "thread_ts": message.get("thread_ts", ""),
            "user": message.get("user", message.get("bot_id", "")),
            "text": text,
            "msg_type": message.get("type", "message"),
            "raw_json": raw_json,
            "vector": vector,
        }

        try:
            self.client.insert(
                collection_name=self.collection_name,
                data=[data],
            )
            return True
        except Exception as e:
            print(f"[!] Failed to insert message {msg_id}: {e}")
            return False

    def insert_messages_batch(self, channel_id: str, messages: list[dict]) -> int:
        """Insert multiple messages in a batch. Returns count of inserted messages."""
        if not messages:
            return 0

        # Prepare texts for batch embedding
        texts = []
        prepared_data = []

        for message in messages:
            ts = message.get("ts", "")
            msg_id = f"{channel_id}:{ts}"

            text = message.get("text", "")
            if len(text) > 65000:
                text = text[:65000] + "..."

            # Use original raw JSON if available (before mention resolution)
            raw_json = message.get("_raw_json_original") or json.dumps(message, ensure_ascii=False)
            if len(raw_json) > 65000:
                # Fall back to essential fields if too large
                essential = {
                    k: message[k]
                    for k in ["ts", "user", "text", "type", "thread_ts", "reply_count", "reactions", "files", "attachments"]
                    if k in message
                }
                raw_json = json.dumps(essential, ensure_ascii=False)
                if len(raw_json) > 65000:
                    raw_json = raw_json[:65000]

            texts.append(text if text else "(empty message)")
            user_id = message.get("user", message.get("bot_id", ""))
            user_name = message.get("user_name", user_id)  # Fall back to ID if not resolved
            prepared_data.append({
                "id": msg_id,
                "channel_id": channel_id,
                "ts": ts,
                "thread_ts": message.get("thread_ts", ""),
                "user": user_id,
                "user_name": user_name,
                "text": text,
                "msg_type": message.get("type", "message"),
                "raw_json": raw_json,
            })

        # Generate embeddings in batch (more efficient)
        vectors = EmbeddingModel.encode_batch(texts, self.embedding_model)

        # Add vectors to data
        data_list = []
        for i, data in enumerate(prepared_data):
            data["vector"] = vectors[i]
            data_list.append(data)

        try:
            self.client.insert(
                collection_name=self.collection_name,
                data=data_list,
            )
            return len(data_list)
        except Exception as e:
            print(f"[!] Batch insert failed: {e}")
            # Fall back to individual inserts
            count = 0
            for msg in messages:
                if self.insert_message(channel_id, msg):
                    count += 1
            return count


async def resolve_mentions_in_messages(
    slack: SlackClient,
    messages: list[dict],
) -> list[dict]:
    """Resolve user mentions and author IDs in messages.

    - Stores original message as '_raw_json_original' before any modifications
    - Transforms <@USER_ID> to @display_name in the text field
    - Adds 'user_name' field with resolved author name

    Modifies messages in place and returns them.
    """
    for msg in messages:
        # Resolve mentions in text
        text = msg.get("text", "")
        if text and "<@" in text:
            resolved_text = await slack.resolve_user_mentions(text)
            msg["text"] = resolved_text

        # Resolve author user ID to display name
        user_id = msg.get("user") or msg.get("bot_id", "")
        if user_id:
            user_name = await slack.get_user_display_name(user_id)
            msg["user_name"] = user_name

    return messages


async def dump_channel(
    slack: SlackClient,
    store: MilvusStore,
    progress: ProgressTracker,
    raw_logger: RawResponseLogger,
    channel_id: str,
    start_timestamp: str,
    request_delay: float,
    visibility: str,
) -> int:
    """
    Dump a single channel's messages.

    Returns the number of messages indexed.
    """
    global shutdown_requested

    print(f"\n[*] Processing {visibility} channel: {channel_id}")

    # Get channel info
    channel_info = await slack.get_channel_info(channel_id)
    if channel_info:
        channel_name = channel_info.get("name", channel_id)
        print(f"[*] Channel name: #{channel_name}")

    channel_progress = progress.get_channel_progress(channel_id)
    messages_indexed = channel_progress.get("messages_indexed", 0)
    cursor = None
    total_new = 0
    newest_ts_seen: str | None = None
    
    # Determine fetch mode: incremental (new messages) or historical (backfill)
    is_completed = progress.is_channel_completed(channel_id, start_timestamp)
    
    # Capture "now" at the start to avoid missing messages that arrive during processing
    fetch_start_ts = str(time.time())
    
    if is_completed:
        # Incremental mode: fetch messages newer than what we have
        saved_newest = channel_progress.get("newest_ts")
        if not saved_newest:
            # Fallback: look up from database
            saved_newest = store.get_newest_message_ts(channel_id)
            if saved_newest:
                print(f"[*] Recovered newest_ts from database: {saved_newest}")
                # Save it to progress for next time
                progress.update_channel_progress(channel_id, newest_ts=saved_newest)
        
        if saved_newest:
            print(f"[*] Fetching new messages since {saved_newest}")
            # Don't use oldest parameter - we'll stop manually when we hit old messages
            # This is more efficient because Slack returns newest-first
            oldest_for_fetch = None
            latest_for_fetch = fetch_start_ts  # Up to when we started
            stop_at_ts = float(saved_newest)  # We'll stop when we hit this timestamp
        else:
            print(f"[*] Channel completed but no messages found, skipping")
            return 0
    else:
        # Historical mode: continue backfilling from where we left off
        stop_at_ts = None  # No early stop in historical mode
        latest_for_fetch = channel_progress.get("oldest_ts")
        oldest_for_fetch = start_timestamp
        
        if latest_for_fetch:
            print(f"[*] Resuming historical fetch from timestamp {latest_for_fetch}")
        else:
            # Check if we have any messages in Milvus for this channel
            existing_oldest = store.get_oldest_message_ts(channel_id)
            if existing_oldest:
                print(f"[*] Found existing messages, oldest: {existing_oldest}")
                latest_for_fetch = existing_oldest

    while not shutdown_requested:
        # Fetch messages
        messages, next_cursor, has_more = await slack.get_channel_history(
            channel_id,
            cursor=cursor,
            oldest=oldest_for_fetch,
            latest=latest_for_fetch,
            limit=100,
        )

        if not messages:
            print(f"[*] No more messages to fetch")
            break

        # Log raw messages before any processing
        raw_logger.log_messages(channel_id, messages, source="history")

        # Filter out duplicates
        new_messages = []
        for msg in messages:
            ts = msg.get("ts", "")
            if not store.message_exists(channel_id, ts):
                new_messages.append(msg)

        # Resolve user mentions in texts
        if new_messages:
            await resolve_mentions_in_messages(slack, new_messages)

        # Insert new messages
        if new_messages:
            inserted = store.insert_messages_batch(channel_id, new_messages)
            total_new += inserted
            messages_indexed += inserted
            print(f"[*] Indexed {inserted} messages (total: {messages_indexed})")

        # Track oldest and newest timestamps from this batch
        if messages:
            oldest_in_batch = min(messages, key=lambda m: float(m.get("ts", "0")))
            newest_in_batch = max(messages, key=lambda m: float(m.get("ts", "0")))
            oldest_ts = oldest_in_batch.get("ts")
            batch_newest_ts = newest_in_batch.get("ts")
            
            # Track the overall newest message we've seen
            if newest_ts_seen is None or float(batch_newest_ts) > float(newest_ts_seen):
                newest_ts_seen = batch_newest_ts
            
            # Update progress
            progress.update_channel_progress(
                channel_id,
                oldest_ts=oldest_ts if not is_completed else None,
                newest_ts=newest_ts_seen,
                messages_indexed=messages_indexed,
            )
            
            # Early stop for incremental mode: if we hit messages older than our checkpoint, we're done
            if stop_at_ts and float(oldest_ts) <= stop_at_ts:
                print(f"[*] Reached previously indexed messages (oldest in batch: {oldest_ts})")
                # Still process thread replies for new messages only
                new_message_ts_set = {m.get("ts") for m in new_messages}
                for msg in messages:
                    if shutdown_requested:
                        break
                    msg_ts = msg.get("ts")
                    if msg_ts in new_message_ts_set and msg.get("reply_count", 0) > 0:
                        print(f"[*] Fetching thread replies for {msg_ts}")
                        replies = await slack.get_thread_replies(channel_id, msg_ts)
                        if replies:
                            raw_logger.log_messages(channel_id, replies, source=f"thread:{msg_ts}")
                            new_replies = [r for r in replies if not store.message_exists(channel_id, r.get("ts", ""))]
                            if new_replies:
                                await resolve_mentions_in_messages(slack, new_replies)
                                inserted = store.insert_messages_batch(channel_id, new_replies)
                                total_new += inserted
                                messages_indexed += inserted
                                print(f"[*] Indexed {inserted} thread replies")
                        await asyncio.sleep(request_delay)
                # Update with fetch_start_ts and finish
                progress.update_channel_progress(
                    channel_id,
                    newest_ts=fetch_start_ts,
                    messages_indexed=messages_indexed,
                )
                print(f"[*] Channel {channel_id} incremental update complete")
                break

        # Fetch thread replies for messages with replies
        for msg in messages:
            if shutdown_requested:
                break
            if msg.get("reply_count", 0) > 0:
                thread_ts = msg.get("ts")
                print(f"[*] Fetching thread replies for {thread_ts}")
                replies = await slack.get_thread_replies(channel_id, thread_ts)
                if replies:
                    # Log raw replies before processing
                    raw_logger.log_messages(channel_id, replies, source=f"thread:{thread_ts}")
                    new_replies = [r for r in replies if not store.message_exists(channel_id, r.get("ts", ""))]
                    if new_replies:
                        await resolve_mentions_in_messages(slack, new_replies)
                        inserted = store.insert_messages_batch(channel_id, new_replies)
                        total_new += inserted
                        messages_indexed += inserted
                        print(f"[*] Indexed {inserted} thread replies")
                await asyncio.sleep(request_delay)

        if not has_more:
            # We've reached the end - set newest_ts to fetch_start_ts so next run starts from there
            progress.update_channel_progress(
                channel_id,
                completed=True if not is_completed else None,  # Don't re-complete
                newest_ts=fetch_start_ts,  # Always use when we started, so next run catches everything
                messages_indexed=messages_indexed,
            )
            if is_completed:
                print(f"[*] Channel {channel_id} incremental update complete")
            else:
                print(f"[*] Channel {channel_id} fully indexed")
            break

        cursor = next_cursor
        await asyncio.sleep(request_delay)

    return total_new


async def main():
    """Main entry point."""
    global shutdown_requested

    print("=" * 60)
    print("Slack Channel Dump Script")
    print("=" * 60)

    # Load configuration
    config = Config()

    if not config.xoxc_token or not config.xoxd_token:
        print("[!] Slack tokens not configured.")
        print("    Set SLACK_XOXC_TOKEN and SLACK_XOXD_TOKEN environment variables")
        print("    or create dump_config.json with xoxc_token and xoxd_token fields")
        sys.exit(1)

    if not config.public_channels and not config.private_channels:
        print("[!] No channels configured.")
        print("    Add 'public_channels' and/or 'private_channels' lists to dump_config.json")
        sys.exit(1)

    print(f"[*] Public channels configured: {len(config.public_channels)}")
    print(f"[*] Private channels configured: {len(config.private_channels)}")
    print(f"[*] Start timestamp: {config.start_timestamp}")
    if config.start_timestamp != "0":
        dt = datetime.fromtimestamp(float(config.start_timestamp))
        print(f"    ({dt.isoformat()})")

    # Initialize progress tracker
    progress = ProgressTracker()

    # Process channels
    total_indexed = 0
    async with SlackClient(config) as slack:
        # Resolve channel names to IDs
        print("\n[*] Resolving channel names...")
        public_channel_ids = await slack.resolve_channels(config.public_channels)
        private_channel_ids = await slack.resolve_channels(config.private_channels)

        print(f"[*] Resolved {len(public_channel_ids)} public channels")
        print(f"[*] Resolved {len(private_channel_ids)} private channels")

        if not public_channel_ids and not private_channel_ids:
            print("[!] No valid channels found after resolution.")
            sys.exit(1)

        # Initialize Milvus stores (separate database files for public/private)
        public_store = None
        private_store = None

        if public_channel_ids:
            print(f"\n[*] Setting up public database: {config.public_db}")
            public_store = MilvusStore(
                config.public_db,
                config.milvus_token,
                config.collection_name,
                config.embedding_dim,
                config.embedding_model,
            )
            public_store.connect()

        if private_channel_ids:
            print(f"\n[*] Setting up private database: {config.private_db}")
            private_store = MilvusStore(
                config.private_db,
                config.milvus_token,
                config.collection_name,
                config.embedding_dim,
                config.embedding_model,
            )
            private_store.connect()

        # Initialize raw response logger
        raw_logger = RawResponseLogger()
        print(f"[*] Raw API responses will be saved to: {raw_logger.output_dir}")

        # Process public channels
        if public_store and public_channel_ids:
            print("\n" + "-" * 40)
            print("[*] Processing PUBLIC channels")
            print("-" * 40)
            for channel_id in public_channel_ids:
                if shutdown_requested:
                    print("\n[!] Shutdown requested, saving progress...")
                    break

                try:
                    indexed = await dump_channel(
                        slack,
                        public_store,
                        progress,
                        raw_logger,
                        channel_id,
                        config.start_timestamp,
                        config.request_delay,
                        "public",
                    )
                    total_indexed += indexed
                except Exception as e:
                    print(f"[!] Error processing channel {channel_id}: {e}")
                    continue

        # Process private channels
        if not shutdown_requested and private_store and private_channel_ids:
            print("\n" + "-" * 40)
            print("[*] Processing PRIVATE channels")
            print("-" * 40)
            for channel_id in private_channel_ids:
                if shutdown_requested:
                    print("\n[!] Shutdown requested, saving progress...")
                    break

                try:
                    indexed = await dump_channel(
                        slack,
                        private_store,
                        progress,
                        raw_logger,
                        channel_id,
                        config.start_timestamp,
                        config.request_delay,
                        "private",
                    )
                    total_indexed += indexed
                except Exception as e:
                    print(f"[!] Error processing channel {channel_id}: {e}")
                    continue

    print("\n" + "=" * 60)
    print(f"[*] Done! Total messages indexed: {total_indexed}")
    if public_channel_ids:
        print(f"    Public database: {config.public_db}")
    if private_channel_ids:
        print(f"    Private database: {config.private_db}")
    if shutdown_requested:
        print("[*] Script interrupted. Run again to continue from where you left off.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
