"""
Клиент основной озвучки (XTTS-v2) для voice_assistant.py.

Silero остаётся запасным голосом: если сервис XTTS не поднят или упал,
``XttsClient.synthesize`` возвращает None, а ассистент синтезирует Silero.
После сбоя клиент делает паузу (COOLDOWN), чтобы каждая фраза не ждала таймаут.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

import requests

XTTS_URL = os.getenv("XTTS_URL", "http://127.0.0.1:8151").rstrip("/")
CONNECT_TIMEOUT = 0.7
READ_TIMEOUT = float(os.getenv("XTTS_TIMEOUT", "25"))
COOLDOWN = float(os.getenv("XTTS_COOLDOWN", "30"))


class XttsClient:
    def __init__(self, url: str = XTTS_URL):
        self.url = url.rstrip("/")
        self._down_until = 0.0
        self._session = requests.Session()
        self._session.trust_env = False  # локальный сервис — мимо HTTP_PROXY

    def is_up(self) -> bool:
        try:
            r = self._session.get(f"{self.url}/health", timeout=CONNECT_TIMEOUT + 1)
            return r.ok and r.json().get("status") == "ok"
        except (requests.RequestException, ValueError):
            return False

    def speakers(self) -> list:
        """Имена встроенных голосов сервиса (пусто, если сервис не отвечает)."""
        try:
            r = self._session.get(f"{self.url}/speakers", timeout=CONNECT_TIMEOUT + 2)
            return list(r.json().get("speakers", [])) if r.ok else []
        except (requests.RequestException, ValueError):
            return []

    def synthesize(self, text: str, speaker: Optional[str], sample_rate: int,
                   speed: float = 1.0) -> Optional[bytes]:
        """PCM S16_LE моно или None (сервис недоступен — нужен запасной голос)."""
        if time.time() < self._down_until:
            return None
        try:
            r = self._session.post(
                f"{self.url}/tts",
                json={"text": text, "speaker": speaker,
                      "sample_rate": sample_rate, "speed": speed},
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            )
            if r.status_code == 200 and r.content:
                return r.content
            reason = f"HTTP {r.status_code}"
        except requests.RequestException as exc:
            reason = exc.__class__.__name__
        self._down_until = time.time() + COOLDOWN
        print(f"⚠️ XTTS недоступен ({reason}) — {COOLDOWN:.0f} с говорю голосом Silero.",
              file=sys.stderr)
        return None


_client: Optional[XttsClient] = None


def get_xtts_client() -> XttsClient:
    global _client
    if _client is None:
        _client = XttsClient()
    return _client
