"""
Data models for the Community Research MCP.

Provides Pydantic models for request validation and
configuration enums for controlling analysis behavior.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "ResponseFormat",
    "ThinkingMode",
    "SearchInput",
    "AnalyzeInput",
    # Legacy aliases
    "CommunitySearchInput",
    "DeepAnalyzeInput",
]

# ══════════════════════════════════════════════════════════════════════════════
# Configuration Enums
# ══════════════════════════════════════════════════════════════════════════════


class ResponseFormat(str, Enum):
    """Output format for tool responses."""

    MARKDOWN = "markdown"
    JSON = "json"
    RAW = "raw"  # Full unprocessed results for LLM synthesis


class ThinkingMode(str, Enum):
    """Analysis depth modes."""

    QUICK = "quick"  # Fast, basic analysis
    BALANCED = "balanced"  # Default, good balance
    DEEP = "deep"  # Thorough, slower


# ══════════════════════════════════════════════════════════════════════════════
# Input Models
# ══════════════════════════════════════════════════════════════════════════════


class SearchInput(BaseModel):
    """Input model for community search with contextual grounding support."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    language: str = Field(
        ...,
        description="Programming language (e.g., 'Python', 'JavaScript')",
        min_length=2,
        max_length=50,
    )

    topic: str = Field(
        ...,
        description=(
            "Specific topic to search for. Be detailed! "
            "IMPORTANT: Use the user's EXACT words and terminology, do NOT rephrase or 'improve' the query. "
            "If user asks about 'audio transcription in my app', use that exact phrase. "
            "Example: 'FastAPI background tasks with Redis queue'"
        ),
        min_length=10,
        max_length=500,
    )

    user_original_query: str = Field(
        ...,
        description=(
            "REQUIRED: The user's EXACT original question/request, word-for-word, unmodified. "
            "Copy-paste the user's message here. The search WILL FAIL without this field. "
            "This prevents AI 'improvements' that cause irrelevant results."
        ),
        min_length=5,
        max_length=1000,
    )

    goal: Optional[str] = Field(
        default=None,
        description="What you want to achieve",
        max_length=500,
    )

    current_setup: Optional[str] = Field(
        default=None,
        description="Your current tech stack (highly recommended)",
        max_length=1000,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # CONTEXTUAL GROUNDING FIELDS
    # These fields allow the LLM to provide rich context gathered from examining
    # the user's codebase, chat history, and specific error situations.
    # The MCP uses this context to make searches more targeted and relevant.
    # ══════════════════════════════════════════════════════════════════════════

    project_context: Optional[str] = Field(
        default=None,
        description=(
            "IMPORTANT: Summary of what the project/app does based on examining the codebase. "
            "Include: app purpose, architecture patterns, key dependencies, relevant file structure. "
            "Example: 'Desktop app using Tauri + React for UI, with a Rust backend that handles "
            "voice transcription via WinRT Speech API. Uses async channels for IPC.'"
        ),
        max_length=2000,
    )

    error_context: Optional[str] = Field(
        default=None,
        description=(
            "The EXACT error message and relevant surrounding code/config. "
            "Include: full error text, the file/function where it occurs, relevant config. "
            "Example: 'Error 0x80072EFD in speech_handler.rs:142 when calling "
            "SpeechRecognizer::CreateAsync(). Config: uses system default mic, no proxy.'"
        ),
        max_length=2000,
    )

    implementation_details: Optional[str] = Field(
        default=None,
        description=(
            "How the specific feature/component is currently implemented. "
            "Include: which files handle it, what APIs/services it uses, data flow. "
            "Example: 'Voice recording in src/audio/recorder.ts uses Web Audio API, "
            "sends PCM to backend via IPC, backend calls Azure Speech SDK for transcription.'"
        ),
        max_length=2000,
    )

    files_examined: Optional[str] = Field(
        default=None,
        description=(
            "List of files already examined when gathering context. "
            "Helps avoid redundant suggestions. "
            "Example: 'package.json, src/config.ts, src/services/speech.ts, Cargo.toml'"
        ),
        max_length=1000,
    )

    chat_history_summary: Optional[str] = Field(
        default=None,
        description=(
            "Summary of relevant prior conversation context. "
            "Include: what was already tried, user preferences, constraints mentioned. "
            "Example: 'User tried reinstalling Speech SDK, issue persists. "
            "They prefer not to use cloud APIs due to privacy concerns.'"
        ),
        max_length=1500,
    )

    response_format: ResponseFormat = Field(
        default=ResponseFormat.RAW,
        description="Output format: 'raw' (default - full data for LLM synthesis), 'markdown', or 'json'",
    )

    thinking_mode: ThinkingMode = Field(
        default=ThinkingMode.BALANCED,
        description="Analysis depth: 'quick', 'balanced', or 'deep'",
    )

    expanded_mode: bool = Field(
        default=True,
        description="Enable expanded result limits for deeper, wider searches (default: True)",
    )

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, v: str) -> str:
        """Ensure topic is specific enough."""
        v = v.strip()

        vague_terms = {
            "settings",
            "config",
            "setup",
            "performance",
            "best practices",
            "tutorial",
            "basics",
            "help",
            "issue",
            "problem",
            "error",
            "install",
        }

        words = v.lower().split()
        if len(words) <= 2 and any(t in v.lower() for t in vague_terms):
            raise ValueError(
                f"Topic '{v}' is too vague. Be more specific! "
                "Include technologies, libraries, or patterns."
            )

        return v


class AnalyzeInput(BaseModel):
    """Input model for workspace analysis."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    query: str = Field(
        ...,
        description="What you want to understand about your codebase",
        min_length=10,
        max_length=1000,
    )

    workspace_path: Optional[str] = Field(
        default=None,
        description="Path to analyze (defaults to current directory)",
    )

    language: Optional[str] = Field(
        default=None,
        description="Language to focus on",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Legacy Aliases (backward compatibility)
# ══════════════════════════════════════════════════════════════════════════════

CommunitySearchInput = SearchInput
DeepAnalyzeInput = AnalyzeInput
