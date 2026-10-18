"""
valera_runtime.py
=================
Общее рантайм-состояние проекта «Валера» (робот-экскурсовод, Unitree H1).

Модуль решает три задачи:

1. **Переключатели функций на лету.** Единый JSON-файл ``valera_config.json``
   хранит, что включено (шумодав, авто-чувствительность VAD, подтверждение
   вопроса, веб-поиск, поиск по базе МИРЭА, barge-in, озвучка). И ``voice_assistant.py``
   (CLI), и ``valera_gui.py`` читают/пишут одно и то же состояние, поэтому
   галочки в GUI применяются мгновенно, без перезапуска.

2. **Триггеры «слово → сетевая команда».»** Список правил вида
   ``{"phrase": "Привет", "command": "curl ..."}``. Как только распознанная
   речь содержит ключевое слово — выполняется заданная команда (например,
   запрос к API робота Unitree H1: помахать рукой, кивнуть, включить подсветку).
   У каждого правила свой режим сравнения, кулдаун и вкл/выкл.

3. **Буферы аудио для GUI.** Живой осциллограф и «склейка» последней фразы
   (что пришло с микрофона → что вырезал шумодав → что ушло Валере) требуют
   доступа к сигналам внутри пайплайна. Ассистент складывает их сюда, GUI читает.

Модуль потокобезопасен: все обращения к состоянию идут под ``threading.RLock``,
потому что аудиопоток, поток распознавания и поток GUI работают одновременно.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import re
import subprocess
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.getenv(
    "VALERA_CONFIG", os.path.join(PROJECT_DIR, "valera_config.json")
)
TRIGGER_LOG = os.path.join(PROJECT_DIR, "triggers.log")

log = logging.getLogger("valera.runtime")

# ---------------------------------------------------------------------------
# Конфигурация по умолчанию
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "_comment": (
        "Рантайм-настройки ассистента «Валера». Файл общий для voice_assistant.py "
        "и valera_gui.py: правки в GUI сохраняются сюда и применяются на лету."
    ),
    "features": {
        # Шумоподавление (спектральное вычитание по профилю шума) + энергетический гейт
        "noise_reduction": True,
        # Автоподстройка чувствительности VAD под уровень шума/микрофона
        "auto_vad": True,
        # Спрашивать подтверждение перед отправкой вопроса в ИИ
        "confirm_before_answer": False,
        # Искать ответ в интернете (SearXNG) для не-МИРЭА вопросов
        "web_search": True,
        # Искать в коллекции «МИРЭА» (база знаний вуза)
        "mirea_collection": True,
        # Прерывать озвучку, если пользователь заговорил
        "barge_in": False,
        # Озвучивать ответы (TTS). Выключите для «сухого» тестирования.
        "tts": True,
        # Выполнять триггеры «слово → curl»
        "triggers_enabled": True,
        # Пересказывать сырой ответ поиска через LLM перед озвучкой
        "rephrase_answers": True,
        # Автоматически начинать НОВЫЙ чат после длительной тишины
        # (порог — rag.auto_new_chat_after_silence_sec). Чистит контекст
        # между подходами разных гостей.
        "auto_new_chat": True,
        # Отвечать на поддакивания («угу», «ага», «ясно») короткой репликой.
        # По умолчанию ВЫКЛ: это не вопрос, и на него лучше не отвечать —
        # иначе робот говорил бессмыслицу или повторялся.
        "react_to_backchannel": False,
    },
    "vad": {
        # Ручной порог VAD (используется, когда auto_vad = false)
        "threshold": 0.3,
        # Во сколько раз речь должна превышать шумовой пол (для авто-режима)
        "margin": 2.5,
        # Насколько агрессивно авто-режим отклоняется от базового порога
        "auto_base": 0.3,
        # Пауза тишины перед завершением фразы, сек
        "silence_timeout": 0.75,
        # Минимальная длительность фразы, сек
        "min_speech": 0.5,
    },
    "audio": {
        # Устройство ЗАХВАТА (микрофон): имя ALSA/PipeWire из `arecord -L`.
        # 'default' — системный по умолчанию (PipeWire/Pulse, умеет ресемплинг).
        "input_device": "default",
        # Устройство ВОСПРОИЗВЕДЕНИЯ (колонки): имя из `aplay -L`.
        # 'default' — системное по умолчанию.
        "output_device": "default",
        # (старое имя input_device — оставлено для совместимости)
        "device": "default",
        # Целевой RMS для АРУ (автоматическая регулировка усиления тихого микрофона)
        "target_rms": 0.15,
        # Максимальное усиление АРУ
        "max_gain": 8.0,
    },
    "tts": {
        "speaker": "eugene",
        "enabled": True,
    },
    "network": {
        # Сетевой интерфейс (NIC), через который ассистент общается с роботом
        # Unitree H1. 'auto' — не привязываться к конкретному интерфейсу.
        "interface": "auto",
    },
    "rag": {
        "base_url": "http://localhost:5000",
        "username": "admin",
        "password": "change_me_in_production",
        # Имя коллекции с базой знаний МИРЭА (создаётся kb_ingest.py)
        "collection_name": "МИРЭА",
        # Автоматически начинать НОВЫЙ чат после стольких секунд тишины
        # (0 = выключено). Помогает очищать контекст между подходами гостей.
        "auto_new_chat_after_silence_sec": 0.0,
    },
    "llm": {
        "ollama_url": "http://localhost:11434",
        "model": "true_valera:latest",
    },
    "robot": {
        # Базовый адрес API робота Unitree H1 (подставляется в команды как {robot})
        "base_url": "http://192.168.123.161:8080",
        "enabled": True,
        # Как сегодня работает робот-экскурсовод: «сидя» или «стоя».
        # Подставляется в промпт, чтобы Валера не утверждал лишнего
        # (например, «я стою в холле», когда он сидит на стойке).
        "posture": "сидя",
    },
    "triggers": [
        {
            "id": "greet-wave",
            "phrase": "Привет",
            "match": "contains",
            "command": (
                "curl -s -X POST {robot}/api/action "
                "-H 'Content-Type: application/json' "
                "-d '{\"action\":\"wave_hand\"}'"
            ),
            "enabled": True,
            "cooldown_sec": 5.0,
            "comment": "Валера машет рукой, когда слышит приветствие",
        }
    ],
}

# Линии визуализации: сколько секунд «живого» сигнала держим в буфере
WAVE_WINDOW_SEC = 2.0
LEVEL_HISTORY_LEN = 600


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------
def _deep_merge(base: dict, override: dict) -> dict:
    """Рекурсивно накладывает ``override`` на ``base`` (не мутируя входные)."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _norm_phrase(text: str) -> str:
    """Нормализует текст для сравнения с триггером: нижний регистр, ё→е, без пунктуации."""
    text = (text or "").lower().replace("ё", "е")
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Перечисление аудиоустройств и сетевых интерфейсов
# ---------------------------------------------------------------------------
def list_audio_inputs() -> List[Dict[str, str]]:
    """
    Возвращает список микрофонов (устройств ЗАХВАТА) из ``arecord -L``.

    Формат arecord -L: строка-имя (без отступа), затем с отступом описание.
    Возвращаем список словарей ``{name, description}`` — GUI/веб показывают
    описание, а в конфиг пишется ``name``.
    """
    return _parse_alsa_devices("arecord")


def list_audio_outputs() -> List[Dict[str, str]]:
    """Возвращает список устройств ВОСПРОИЗВЕДЕНИЯ (колонок) из ``aplay -L``."""
    return _parse_alsa_devices("aplay")


def _parse_alsa_devices(tool: str) -> List[Dict[str, str]]:
    """
    Разбирает вывод ``arecord -L`` / ``aplay -L`` в список устройств.

    Пример вывода::

        default
            Default ALSA Output
        hw:CARD=Generic_1,DEV=0
            HD-Audio Generic, ALC897 Analog

    Устройства ``null`` отбрасываются (заглушка ALSA). Если утилита недоступна
    или ничего не найдено — возвращаем только ``default``.
    """
    devices: List[Dict[str, str]] = []
    try:
        proc = subprocess.run(
            [tool, "-L"], capture_output=True, text=True, timeout=10
        )
        lines = (proc.stdout or "").splitlines()
    except Exception:  # noqa: BLE001
        lines = []

    current: Optional[Dict[str, str]] = None
    for raw in lines:
        if not raw.strip():
            continue
        if not raw[0].isspace():
            # Новая строка-имя устройства
            name = raw.strip()
            if name == "null":
                current = None
                continue
            current = {"name": name, "description": ""}
            devices.append(current)
        elif current is not None:
            # Строка описания (с отступом) — берём первую непустую
            if not current["description"]:
                current["description"] = raw.strip()

    if not devices:
        devices = [{"name": "default", "description": "Системное устройство по умолчанию"}]
    return devices


def list_network_interfaces() -> List[Dict[str, str]]:
    """
    Возвращает сетевые интерфейсы (NIC) с адресами.

    Используется, чтобы выбрать интерфейс, через который ассистент общается
    с роботом Unitree H1 (обычно это Ethernet-порт робота или Wi-Fi).
    Виртуальные интерфейсы (docker, veth, br-, virbr) отфильтровываются:
    к роботу они отношения не имеют.
    Возвращает список ``{name, address, state, description}``.
    """
    VIRTUAL_PREFIXES = (
        "docker", "veth", "br-", "virbr", "vmnet", "tun", "tap",
        "wg", "zt", "lo", "ham", "dummy",
    )

    def _is_virtual(name: str) -> bool:
        low = name.lower()
        return low == "lo" or any(low.startswith(p) for p in VIRTUAL_PREFIXES)

    result: List[Dict[str, str]] = []
    try:
        proc = subprocess.run(
            ["ip", "-j", "addr"], capture_output=True, text=True, timeout=10
        )
        import json as _json

        for iface in _json.loads(proc.stdout or "[]"):
            name = iface.get("ifname", "")
            if not name or _is_virtual(name):
                continue
            ip = ""
            for info in iface.get("addr_info", []):
                if info.get("family") == "inet":
                    ip = info.get("local", "")
                    break
            state = iface.get("operstate", "?")
            result.append({
                "name": name,
                "address": ip,
                "state": state,
                "description": f"{name} — {ip or 'без адреса'} ({state})",
            })
    except Exception:  # noqa: BLE001
        pass

    # Всегда даём выбор «авто» — не привязываться к интерфейсу
    result.insert(0, {
        "name": "auto", "address": "", "state": "auto",
        "description": "Автоматически (интерфейс не задан)",
    })
    return result


def device_exists(tool: str, name: str) -> bool:
    """Проверяет, что устройство ``name`` реально доступно для ``arecord``/``aplay``."""
    devices = _parse_alsa_devices(tool)
    return any(d["name"] == name for d in devices)


# ---------------------------------------------------------------------------
# Автоподстройка чувствительности VAD
# ---------------------------------------------------------------------------
class SensitivityController:
    """
    Автоматически подбирает порог VAD и «энергетический гейт» под условия.

    Идея простая и объяснимая:

    * Шумовой пол (``noise_rms``) — экспоненциальное среднее RMS только тех
      чанков, которые VAD счёл тишиной. Это «сколько шумит микрофон».
    * Уровень речи (``speech_rms``) — то же, но по чанкам, где VAD услышал речь.
    * Если шум громкий, порог вероятности поднимаем (меньше ложных срабатываний).
      Если шум тихий и микрофон тихий — порог опускаем (выше чувствительность).
    * Гейт обрезает всё, что тише ``noise_rms * margin`` — ровно это и рисуется
      в GUI как «что шумодав выключает».
    * АРУ (``gain``) подтягивает тихий микрофон к целевому RMS, чтобы STT
      получал стабильный уровень.
    """

    def __init__(
        self,
        base_threshold: float = 0.3,
        margin: float = 2.5,
        target_rms: float = 0.15,
        max_gain: float = 8.0,
        alpha: float = 0.05,
        reference_rms: float = 0.001,
    ):
        self.base_threshold = float(base_threshold)
        self.margin = float(margin)
        self.target_rms = float(target_rms)
        self.max_gain = float(max_gain)
        self.alpha = float(alpha)
        self.reference_rms = float(reference_rms)

        self.noise_rms: Optional[float] = None
        self.speech_rms: Optional[float] = None
        self.gain: float = 1.0
        self._lock = threading.Lock()

    # -- обновление оценок -------------------------------------------------
    def update(self, rms: float, is_speech: bool = False) -> None:
        rms = float(max(rms, 0.0))
        with self._lock:
            if is_speech:
                if self.speech_rms is None:
                    self.speech_rms = rms
                else:
                    self.speech_rms += self.alpha * (rms - self.speech_rms)
            else:
                # Пока речи мало, считаем шумовой пол (медленно, чтобы не «дёргался»)
                if self.noise_rms is None:
                    self.noise_rms = rms
                else:
                    self.noise_rms += self.alpha * (rms - self.noise_rms)
            self._recompute_gain()

    def _recompute_gain(self) -> None:
        # АРУ усиливает ТОЛЬКО когда речь тише целевого уровня. Опираться на
        # шумовой пол нельзя: в тишине это раздуло бы шум до максимума.
        ref = self.speech_rms
        if not ref or ref <= 1e-6:
            self.gain = 1.0
            return
        gain = self.target_rms / ref
        self.gain = float(min(max(gain, 1.0), self.max_gain))

    # -- вычисляемые пороги -------------------------------------------------
    def auto_vad_threshold(self) -> float:
        """
        Порог вероятности Silero VAD, подстроенный под условия.

        Логика:
        * Если мы уже слышали речь — считаем отношение сигнал/шум (SNR).
          Хороший SNR (речь явно громче шума) → порог ниже, слушаем чутко.
          Плохой SNR (шумно) → порог выше, чтобы не срабатывать на шум.
        * Если речи ещё не было — смотрим только на абсолютный шумовой пол
          (громкий шум → выше порог).
        """
        with self._lock:
            noise = self.noise_rms
            speech = self.speech_rms
        if not noise or noise <= 1e-6:
            return self.base_threshold

        if speech and speech > noise:
            snr_db = 20.0 * math.log10(speech / noise)
            # 15 дБ — «нормально»: ниже SNR → порог выше, выше → порог ниже
            adjustment = (15.0 - snr_db) * 0.012
        else:
            noise_db = 20.0 * math.log10(max(noise, 1e-7) / self.reference_rms)
            adjustment = 0.03 * noise_db
        adjustment = max(-0.12, min(0.25, adjustment))
        return float(max(0.10, min(0.70, self.base_threshold + adjustment)))

    def gate_threshold(self) -> float:
        """Порог энергии: чанки тише него считаются вырезанным шумом."""
        with self._lock:
            noise = self.noise_rms
        if not noise:
            return 0.0
        return float(noise * self.margin)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "noise_rms": self.noise_rms,
                "speech_rms": self.speech_rms,
                "gain": self.gain,
                "auto_threshold": self.auto_vad_threshold(),
                "gate_threshold": self.gate_threshold(),
            }


# ---------------------------------------------------------------------------
# Рантайм-состояние
# ---------------------------------------------------------------------------
class RuntimeState:
    """
    Потокобезопасный снимок состояния приложения + буферы для GUI.

    Экземпляр создаётся один раз (``get_runtime()``) и передаётся и в
    ``VoiceAssistant``, и в GUI — тогда галочки и тумблеры применяются мгновенно.
    """

    def __init__(self, path: str = CONFIG_PATH, sample_rate: int = 16000):
        self.path = path
        self.sample_rate = sample_rate
        self._lock = threading.RLock()
        self.config: Dict[str, Any] = copy.deepcopy(DEFAULT_CONFIG)
        self._listeners: List[Callable[[str, Any], None]] = []

        # --- буферы визуализации ---
        self.levels: deque = deque(maxlen=LEVEL_HISTORY_LEN)   # (ts, mic, gated, out)
        self.raw_wave: deque = deque(maxlen=int(sample_rate * WAVE_WINDOW_SEC))
        self.gated_wave: deque = deque(maxlen=int(sample_rate * WAVE_WINDOW_SEC))
        self.phrase: Optional[Dict[str, Any]] = None           # последняя фраза
        self.status: str = "Ожидание"
        self.events: deque = deque(maxlen=200)                  # (ts, level, text)
        self.last_transcript: str = ""
        self.last_answer: str = ""
        self.metrics: Dict[str, Any] = {}                       # живая телеметрия VAD/шумодава
        self._trigger_last_run: Dict[str, float] = {}
        self._last_activity: Optional[float] = None             # время последнего вопроса
        self.current_chat_id: Optional[str] = None              # активный чат WebRAgent
        self.current_chat_title: str = ""

        self.load()

    # ------------------------------------------------------------------
    # Загрузка/сохранение
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Читает конфиг с диска (отсутствующие поля берутся из DEFAULT_CONFIG)."""
        data: Dict[str, Any] = {}
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f) or {}
            except Exception as exc:  # noqa: BLE001
                log.warning("Не удалось прочитать %s: %s", self.path, exc)
        merged = _deep_merge(DEFAULT_CONFIG, data)
        with self._lock:
            self.config = merged
        # Если файла не было — сразу создаём с дефолтами, чтобы GUI было что править
        if not os.path.exists(self.path):
            self.save()

    def save(self) -> None:
        """Пишет конфиг на диск атомарно (через временный файл)."""
        with self._lock:
            payload = copy.deepcopy(self.config)
        try:
            tmp = f"{self.path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except Exception as exc:  # noqa: BLE001
            log.warning("Не удалось сохранить %s: %s", self.path, exc)

    # ------------------------------------------------------------------
    # Доступ к настройкам
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self.config)

    def get(self, *keys: str, default: Any = None) -> Any:
        """``rt.get("features", "web_search", default=True)``."""
        node: Any = self.config
        with self._lock:
            for key in keys:
                if not isinstance(node, dict) or key not in node:
                    return default
                node = node[key]
            return copy.deepcopy(node)

    def set(self, *keys: str, value: Any) -> None:
        """Пишет значение по пути ключей и уведомляет слушателей."""
        if not keys:
            return
        with self._lock:
            node = self.config
            for key in keys[:-1]:
                if not isinstance(node.get(key), dict):
                    node[key] = {}
                node = node[key]
            node[keys[-1]] = value
        self._notify(".".join(keys), value)
        self.save()

    def feature(self, name: str, default: bool = True) -> bool:
        """Быстрая проверка флага из секции ``features``."""
        value = self.get("features", name, default=default)
        return bool(default if value is None else value)

    def set_feature(self, name: str, value: bool) -> None:
        self.set("features", name, value=bool(value))

    # ------------------------------------------------------------------
    # Слушатели (GUI подписывается, чтобы обновлять виджеты)
    # ------------------------------------------------------------------
    def add_listener(self, callback: Callable[[str, Any], None]) -> None:
        with self._lock:
            self._listeners.append(callback)

    def _notify(self, key: str, value: Any) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for cb in listeners:
            try:
                cb(key, value)
            except Exception as exc:  # noqa: BLE001
                log.debug("Слушатель упал: %s", exc)

    # ------------------------------------------------------------------
    # Триггеры «слово → сетевая команда»
    # ------------------------------------------------------------------
    def triggers(self) -> List[Dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self.config.get("triggers", []))

    def set_triggers(self, triggers: List[Dict[str, Any]]) -> None:
        self.set("triggers", value=list(triggers))

    def _new_trigger_id(self) -> str:
        return f"trg-{int(time.time() * 1000) % 10_000_000}"

    def add_trigger(
        self,
        phrase: str,
        command: str,
        match: str = "contains",
        enabled: bool = True,
        cooldown_sec: float = 5.0,
        comment: str = "",
    ) -> Dict[str, Any]:
        trigger = {
            "id": self._new_trigger_id(),
            "phrase": phrase,
            "match": match,
            "command": command,
            "enabled": bool(enabled),
            "cooldown_sec": float(cooldown_sec),
            "comment": comment,
        }
        with self._lock:
            self.config.setdefault("triggers", []).append(trigger)
        self._notify("triggers", self.triggers())
        self.save()
        return trigger

    def update_trigger(self, trigger_id: str, **fields: Any) -> bool:
        with self._lock:
            for trigger in self.config.get("triggers", []):
                if trigger.get("id") == trigger_id:
                    trigger.update(fields)
                    updated = True
                    break
            else:
                updated = False
        if updated:
            self._notify("triggers", self.triggers())
            self.save()
        return updated

    def remove_trigger(self, trigger_id: str) -> bool:
        with self._lock:
            before = self.config.get("triggers", [])
            after = [t for t in before if t.get("id") != trigger_id]
            self.config["triggers"] = after
        if len(after) != len(before):
            self._notify("triggers", self.triggers())
            self.save()
            return True
        return False

    def match_triggers(self, text: str) -> List[Dict[str, Any]]:
        """Возвращает все включённые триггеры, чьё слово встретилось в ``text``."""
        if not text or not self.feature("triggers_enabled"):
            return []
        norm = _norm_phrase(text)
        matched: List[Dict[str, Any]] = []
        for trigger in self.triggers():
            if not trigger.get("enabled", True):
                continue
            phrase = _norm_phrase(trigger.get("phrase", ""))
            if not phrase:
                continue
            mode = trigger.get("match", "contains")
            hit = False
            if mode == "equals":
                hit = norm == phrase
            elif mode == "word":
                hit = re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", norm) is not None
            elif mode == "regex":
                try:
                    hit = re.search(trigger.get("phrase", ""), text, re.IGNORECASE) is not None
                except re.error:
                    hit = False
            else:  # contains
                hit = phrase in norm
            if hit:
                matched.append(trigger)
        return matched

    def _render_command(self, trigger: Dict[str, Any], text: str, raw_text: str) -> str:
        """Подставляет плейсхолдеры в команду."""
        robot = self.get("robot", "base_url", default="http://192.168.123.161:8080")
        command = trigger.get("command", "")
        return (
            command.replace("{robot}", str(robot))
            .replace("{phrase}", trigger.get("phrase", ""))
            .replace("{text}", text or "")
            .replace("{raw}", raw_text or text or "")
        )

    def run_trigger(
        self,
        trigger: Dict[str, Any],
        text: str = "",
        raw_text: str = "",
        timeout: float = 10.0,
        respect_cooldown: bool = True,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Выполняет сетевую команду триггера.

        Возвращает словарь с результатом (``ok``, ``command``, ``stdout``, ``stderr``).
        Команда запускается через shell — так задумано: пользователь сам прописывает
        строку `curl` в GUI.
        """
        command = self._render_command(trigger, text, raw_text)
        trigger_id = trigger.get("id", command)

        if respect_cooldown:
            cooldown = float(trigger.get("cooldown_sec", 0) or 0)
            now = time.time()
            last = self._trigger_last_run.get(trigger_id, 0.0)
            if cooldown > 0 and (now - last) < cooldown:
                return {
                    "ok": False,
                    "skipped": True,
                    "reason": f"кулдаун {cooldown:.0f}с",
                    "command": command,
                }
            self._trigger_last_run[trigger_id] = now

        event = {
            "ok": True,
            "command": command,
            "phrase": trigger.get("phrase", ""),
            "stdout": "",
            "stderr": "",
        }
        if dry_run:
            self.log_event("info", f"[ТЕСТ] {trigger.get('phrase')} → {command}")
            return event

        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            event["stdout"] = (proc.stdout or "")[:500]
            event["stderr"] = (proc.stderr or "")[:500]
            event["returncode"] = proc.returncode
            if proc.returncode != 0:
                event["ok"] = False
                self.log_event(
                    "error",
                    f"Триггер «{trigger.get('phrase')}» вернул код {proc.returncode}: "
                    f"{(proc.stderr or '').strip()[:200]}",
                )
            else:
                self.log_event(
                    "ok", f"Триггер «{trigger.get('phrase')}» выполнен: {command[:120]}"
                )
        except subprocess.TimeoutExpired:
            event["ok"] = False
            event["stderr"] = "таймаут"
            self.log_event("error", f"Триггер «{trigger.get('phrase')}» не ответил за {timeout:.0f}с")
        except Exception as exc:  # noqa: BLE001
            event["ok"] = False
            event["stderr"] = str(exc)
            self.log_event("error", f"Триггер «{trigger.get('phrase')}» упал: {exc}")

        self._append_trigger_log(event)
        return event

    def fire_triggers(self, text: str, raw_text: str = "") -> List[Dict[str, Any]]:
        """Находит и выполняет все подходящие триггеры для распознанного текста."""
        results = []
        for trigger in self.match_triggers(text):
            results.append(self.run_trigger(trigger, text=text, raw_text=raw_text))
        return results

    def _append_trigger_log(self, event: Dict[str, Any]) -> None:
        try:
            with open(TRIGGER_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": time.time(), **event}, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Буферы для GUI
    # ------------------------------------------------------------------
    def push_level(self, mic_rms: float, gated_rms: float, out_rms: float = 0.0) -> None:
        with self._lock:
            self.levels.append((time.time(), float(mic_rms), float(gated_rms), float(out_rms)))

    def push_wave(self, raw=None, gated=None) -> None:
        """Добавляет чанки живой волны (то, что рисует осциллограф)."""
        with self._lock:
            if raw is not None:
                self.raw_wave.extend(raw.tolist() if hasattr(raw, "tolist") else raw)
            if gated is not None:
                self.gated_wave.extend(gated.tolist() if hasattr(gated, "tolist") else gated)

    def push_phrase(self, raw, cleaned) -> None:
        """
        Сохраняет «склейку» последней фразы для виджета сравнения.

        Считает вырезанный шумодавом остаток: ``removed = raw - cleaned``.
        Массивы прореживаются, чтобы matplotlib не рисовал 300 тысяч точек.
        """
        import numpy as np

        raw = np.asarray(raw, dtype=np.float32)
        cleaned = np.asarray(cleaned, dtype=np.float32)
        if cleaned.shape != raw.shape:
            n = min(len(raw), len(cleaned))
            raw, cleaned = raw[:n], cleaned[:n]
        removed = raw - cleaned

        # ВАЖНО: прореживаем до ~600 точек на кривую. Веб-пульт отдаёт эту
        # «склейку» по HTTP, а в этом окружении ответы больше ~16 КБ
        # обрезаются (VPN на loopback). 2000 точек × 3 кривые давали ~130 КБ
        # JSON и ~57 КБ после gzip — сайт «зависал». 600 точек хватает,
        # чтобы форма сигнала читалась, и влезает в лимит.
        target_points = 600
        step = max(1, len(raw) // target_points)
        payload = {
            "ts": time.time(),
            "duration": len(raw) / float(self.sample_rate),
            "raw": [round(float(v), 3) for v in raw[::step]],
            "removed": [round(float(v), 3) for v in removed[::step]],
            "clean": [round(float(v), 3) for v in cleaned[::step]],
            "raw_rms": float(np.sqrt(np.mean(raw**2))) if raw.size else 0.0,
            "clean_rms": float(np.sqrt(np.mean(cleaned**2))) if cleaned.size else 0.0,
        }
        with self._lock:
            self.phrase = payload

    def get_phrase(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self.phrase)

    def get_waves(self):
        """Возвращает копии буферов живой волны (для matplotlib)."""
        with self._lock:
            return list(self.raw_wave), list(self.gated_wave)

    def get_levels(self):
        with self._lock:
            return list(self.levels)

    def set_metrics(self, **values: Any) -> None:
        """Публикует живую телеметрию (RMS, пороги, усиление) для панели в GUI."""
        with self._lock:
            self.metrics.update(values)

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.metrics)

    # ------------------------------------------------------------------
    # Статус и события (для строки состояния GUI)
    # ------------------------------------------------------------------
    def set_status(self, text: str) -> None:
        with self._lock:
            self.status = text
        self._notify("status", text)

    def get_status(self) -> str:
        with self._lock:
            return self.status

    def set_transcript(self, text: str, answer: str = "") -> None:
        with self._lock:
            self.last_transcript = text or ""
            if answer:
                self.last_answer = answer
        self._notify("transcript", text)

    def log_event(self, level: str, text: str) -> None:
        with self._lock:
            self.events.append((time.time(), level, text))
        log.log(logging.WARNING if level == "error" else logging.INFO, "%s", text)
        self._notify("event", (level, text))

    def get_events(self) -> List[Any]:
        with self._lock:
            return list(self.events)

    # ------------------------------------------------------------------
    # Контекст диалога: авто-новый чат по тишине
    # ------------------------------------------------------------------
    def touch_activity(self) -> None:
        """
        Отмечает «последнюю активность» (был вопрос/ответ).

        По этой отметке ассистент решает, не пора ли начать новый чат после
        длительной тишины — см. ``should_auto_new_chat()``.
        """
        with self._lock:
            self._last_activity = time.time()

    def seconds_since_activity(self) -> Optional[float]:
        with self._lock:
            last = getattr(self, "_last_activity", None)
        if last is None:
            return None
        return time.time() - last

    def should_auto_new_chat(self) -> bool:
        """
        True, если тишина длится дольше настроенного порога.

        Порог берётся из ``rag.auto_new_chat_after_silence_sec`` (0 = выключено).
        После срабатывания отметку активности сбрасывают через
        ``mark_new_chat_started()``, чтобы не пересоздавать чат на каждой фразе.
        """
        limit = float(self.get("rag", "auto_new_chat_after_silence_sec", default=0.0) or 0.0)
        if limit <= 0:
            return False
        idle = self.seconds_since_activity()
        return idle is not None and idle >= limit

    def mark_new_chat_started(self) -> None:
        """Сбрасывает таймер тишины (вызывается после создания нового чата)."""
        self.touch_activity()

    def set_current_chat(self, chat_id: Optional[str], title: str = "") -> None:
        """Публикует текущий чат (для отображения в GUI/веб-интерфейсе)."""
        with self._lock:
            self.current_chat_id = chat_id
            self.current_chat_title = title

    def get_current_chat(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "chat_id": getattr(self, "current_chat_id", None),
                "title": getattr(self, "current_chat_title", ""),
            }


# ---------------------------------------------------------------------------
# Синглтон
# ---------------------------------------------------------------------------
_runtime: Optional[RuntimeState] = None
_runtime_lock = threading.Lock()


def get_runtime(path: str = None) -> RuntimeState:
    """Возвращает общий экземпляр ``RuntimeState`` (создаёт при первом вызове)."""
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = RuntimeState(path=path or CONFIG_PATH)
        return _runtime


def set_runtime(runtime: RuntimeState) -> None:
    """Подменяет синглтон (используется GUI, чтобы делиться состоянием)."""
    global _runtime
    with _runtime_lock:
        _runtime = runtime


if __name__ == "__main__":  # pragma: no cover - ручная проверка
    rt = get_runtime()
    print(json.dumps(rt.snapshot(), ensure_ascii=False, indent=2))
