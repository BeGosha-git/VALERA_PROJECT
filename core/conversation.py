"""Multi-turn conversation management with Qwen3-Omni format."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""

    role: str  # "user" or "assistant"
    text: Optional[str] = None
    audio_path: Optional[str] = None  # path to saved audio file


@dataclass
class Conversation:
    """Manages conversation history in Qwen3-Omni chat format."""

    system_prompt: str = (
        "You are Valera, a smart voice assistant created for personal use. "
        "You are a helpful, friendly, and concise voice assistant. "
        "You communicate with the user in Russian. "
        "Interact with users using short (no more than 50 words), brief, "
        "straightforward language, maintaining a natural conversational tone. "
        "Never use formal phrasing, mechanical expressions, bullet points, "
        "or overly structured language. "
        "Your output must consist only of the spoken content you want the "
        "user to hear. Do not include any descriptions of actions, emotions, "
        "sounds, or voice changes."
    )
    history: list[ConversationTurn] = field(default_factory=list)
    max_history: int = 20  # max number of turns to keep

    def add_user_message(
        self, text: Optional[str] = None, audio_path: Optional[str] = None
    ) -> None:
        """Add a user turn."""
        self.history.append(ConversationTurn(
            role="user", text=text, audio_path=audio_path
        ))
        self._trim_history()

    def add_assistant_message(
        self, text: str, audio_path: Optional[str] = None
    ) -> None:
        """Add an assistant turn."""
        self.history.append(ConversationTurn(
            role="assistant", text=text, audio_path=audio_path
        ))
        self._trim_history()

    def _trim_history(self) -> None:
        """Keep only the last max_history turns."""
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def to_model_format(self) -> list[dict]:
        """Convert conversation to Qwen3-Omni model format.

        Returns:
            List of messages in the format expected by the model.
        """
        messages = []

        # Add system prompt
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            })

        for turn in self.history:
            content = []

            # Audio goes first (before text, as recommended by Qwen)
            if turn.audio_path:
                content.append({"type": "audio", "audio": turn.audio_path})

            if turn.text:
                content.append({"type": "text", "text": turn.text})

            # If no content, skip this turn
            if not content:
                continue

            messages.append({
                "role": turn.role,
                "content": content,
            })

        return messages

    def clear(self) -> None:
        """Reset conversation history."""
        self.history.clear()

    def get_last_n_turns(self, n: int) -> list[ConversationTurn]:
        """Get the last n turns."""
        return self.history[-n:] if n > 0 else []

    def __len__(self) -> int:
        return len(self.history)


# Global conversation instance (one per session)
conversation = Conversation()


def create_new_conversation(system_prompt: Optional[str] = None) -> Conversation:
    """Create a fresh conversation with optional custom system prompt."""
    conv = Conversation()
    if system_prompt:
        conv.system_prompt = system_prompt
    return conv
