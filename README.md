# 🎙️ Голосовой ассистент «Валера» (Микрофон → RAG → TTS)

Полноценный голосовой ассистент в стиле экскурсовода РТУ МИРЭА.
Распознаёт речь с микрофона, ведёт диалог с контекстом через RAG/LLM
и озвучивает ответ.

```
Микрофон → [STT] → текст вопроса → [WebRAgent / Ollama RAG] → ответ → [TTS] → динамики
```

---

## 📋 Быстрый старт

Всё запускается одной командой:

```bash
./run_all.sh voice-fast --stt nemotron
```

| Режим | Что делает |
|---|---|
| `talk` | Запускает `TALK.py` (простой диалог, по умолчанию) |
| `voice` | Голосовой ассистент **с подтверждением** вопроса перед запросом |
| `voice-fast` | Голосовой ассистент **без подтверждения** (сразу отвечает) |

### Опции (передаются после режима)

| Опция | Описание |
|---|---|
| `--stt kairos` | Лёгкая модель распознавания речи (по умолчанию) |
| `--stt nemotron` | NVIDIA **Nemotron 3.5 ASR** — точнее, но тяжелее (GPU) |
| `--stt auto` | Автовыбор: короткие фразы → kairos, длинные → nemotron |
| `--keep-ollama` | **Не** перезапускать Ollama (пригодится, если вы запустили её сами) |
| `--new-chat` | Начать **новый** разговор (сбросить контекст) |
| `--no-clean` | **Не** чистить текст вопроса (имена и мат сохраняются в запросе) |
| `--preload` | Прогреть STT-модель при старте (первая фраза быстрее) |
| `--barge-in` | Прерывать озвучку ответа, если пользователь начал говорить |
| `--speaker ИМЯ` | Голос TTS (`eugene`, `xenia`, `aidar`, `baya`, `kseniya`) |
| `--vad-threshold N` | Порог VAD (0..1, ниже = чувствительнее) |
| `--silence-timeout N` | Пауза тишины перед концом фразы (сек) |

### Примеры

```bash
# Голосовой ассистент на Nemotron, без подтверждения, новый чат каждый раз
./run_all.sh voice-fast --stt nemotron --new-chat

# Голосовой ассистент на Kairos с подтверждением
./run_all.sh voice --stt kairos

# Использовать уже запущенную Ollama (не перезапускать)
./run_all.sh voice-fast --stt nemotron --keep-ollama

# Отправлять в ИИ текст вопроса БЕЗ очистки (имя/мат сохраняются)
./run_all.sh voice-fast --stt nemotron --no-clean
```

---

## 🚀 Что запускает `run_all.sh`

Скрипт автоматически поднимает весь стек:

1. **Conda-окружения** — создаёт `tts_env` и `mic_to_text_env`, если их нет.
2. **Ollama** — **останавливает все чужие процессы** `ollama serve` и запускает
   собственную копию с моделью `valera_live:latest` (персона «Валера»).
   Чтобы НЕ трогать чужую Ollama — передайте `--keep-ollama`.
3. **WebRAgent** (Docker Compose) — веб-интерфейс, RAG, MongoDB, Qdrant, SearXNG.
4. **Приложение** — `voice_assistant.py` или `TALK.py`.

При завершении (Ctrl+C) скрипт корректно останавливает Docker-сервисы
и **собственную** Ollama (чужую — не трогает).

---

## 🧠 Про контекст разговора

Контекст вопросов/ответов хранится **на сервере WebRAgent** (в MongoDB),
а не в самом скрипте.

- При первом запуске создаётся чат, его `chat_id` сохраняется в файл
  `chat_id.txt` в корне проекта.
- При следующих запусках ассистент **продолжает тот же чат** — модель помнит
  предыдущие вопросы и ответы.
- Чтобы **начать новый разговор** (сбросить контекст) — используйте флаг
  `--new-chat`:

```bash
./run_all.sh voice-fast --stt nemotron --new-chat
```

Все заданные вопросы и ответы пишутся в лог `questions.log` (формат JSON —
одна строка на запись).

### Очистка текста вопроса

По умолчанию ассистент **очищает** распознанный текст перед отправкой в ИИ
(через `clean_question`): удаляет обращение «Валера» и нецензурную лексику.

Чтобы отправлять текст **как есть** (без очистки) — используйте `--no-clean`:

```bash
./run_all.sh voice-fast --stt nemotron --no-clean
```

---

## 📁 Структура проекта

```
VALERA_PROJECT/
├── run_all.sh                 # Главный запуск (Docker + Ollama + приложение)
├── voice_assistant.py         # Голосовой ассистент (микрофон → RAG → TTS)
├── Modelfile                  # Персона «Валера» для Ollama (valera_live)
├── chat_id.txt                # Текущий chat_id WebRAgent (генерируется)
├── questions.log              # Лог заданных вопросов (генерируется)
│
├── mic_to_text/               # Окружение и зависимости для распознавания
│   └── environment.yml        #   (kairos, nemotron, silero-vad, etc.)
│
├── nemotron-project/          # NVIDIA Nemotron 3.5 ASR (модель STT)
│   └── model.safetensors
│
├── nemotron_asr/              # Отдельное окружение для стримингового Nemotron
│   ├── nemotron_streaming_stt.py
│   ├── run.sh
│   └── setup_conda.sh
│
├── WebRAgent/                 # RAG-сервис (Docker compose up -d)
├── russian_text_to_speech/    # TTS (NeuralSpeaker, model.pt)
└── Speech-main/               # (опционально) NeMo Toolkit
```

---

## 🛠️ Требования

- **Linux** с `alsa-utils` (`arecord`), Docker, Conda
- **NVIDIA GPU** (рекомендуется) — для Nemotron STT и ускорения LLM
- **микрофон** (лучше USB, например DJI MIC MINI) и **динамики**

### Микрофон

Захват идёт через `arecord -D default` (PipeWire/Pulse), что надёжно работает
с USB-микрофонами. По умолчанию используется устройство `default`.
Устройство можно сменить в `voice_assistant.py`:

```python
AUDIO_DEVICE = "default"          # например: "plughw:CARD=MINI,DEV=0"
```

Посмотреть доступные микрофоны:
```bash
arecord -L
```

---

## 🔑 Переменные окружения (в `voice_assistant.py`)

| Переменная | Значение | Назначение |
|---|---|---|
| `RAG_BASE_URL` | `http://localhost:5000` | Адрес WebRAgent |
| `RAG_USERNAME` / `RAG_PASSWORD` | `admin` / `change_me_in_production` | Учётка WebRAgent |
| `TTS_SPEAKER` | `eugene` | Голос TTS |
| `NEMOTRON_MODEL_PATH` | `nemotron-project/` | Путь к модели Nemotron |
| `NEMOTRON_LANGUAGE` | `ru-RU` | Язык распознавания |
| `CHAT_ID_FILE` | `chat_id.txt` | Файл с chat_id для контекста |

---

## 🧪 STT-модели распознавания речи

### Kairos (по умолчанию, лёгкая)
- Работает на CPU/GPU, быстрая, средняя точность.
- Не требует отдельных зависимостей кроме `mic_to_text/environment.yml`.

### Nemotron 3.5 ASR (точная)
- NVIDIA-модель, требует GPU (проверено на RTX 5070 Ti / 5060 Ti).
- **Важно**: для Blackwell (RTX 50xx) нужен PyTorch `cu128+`.
  В `mic_to_text_env` уже стоит `torch 2.13+cu130` (от kairos), поэтому
  Nemotron работает прямо в этом окружении.
- Задержка стриминга задаётся `NEMOTRON_LATENCY_MS`
  (поддерживается 80 / 320 / 560 / 1120 мс).

---

## 🐛 Диагностика

### Не слышно голоса / ничего не распознаётся

1. Проверьте, что микрофон виден: `arecord -L`
2. Если распознавание пустое — уровень громкости слишком низкий.
   Проверьте уровень:
   ```bash
   arecord -D default -d 3 -f S16_LE test.raw && python3 -c "
   import numpy as np
   a = np.frombuffer(open('test.raw','rb').read(), dtype=np.int16)/32768
   print('peak:', np.abs(a).max())"
   ```
   Если peak < 0.05 — поднесите микрофон ближе или увеличьте громкость.

### Ollama перезапускается каждый раз

Это ожидаемо: `run_all.sh` по умолчанию останавливает внешнюю Ollama
и запускает свою. Чтобы сохранить свою — используйте `--keep-ollama`.

### Контекст разговора не работает

- Убедитесь, что WebRAgent запущен: `curl http://localhost:5000/auth/login`
- Убедитесь, что `chat_id.txt` существует. Если он удалён — создастся новый чат.
- Проверьте содержимое `questions.log` — там видны все запросы/ответы.

---

## 📝 Примечания по перезапуску Ollama

`run_all.sh` по умолчанию:

1. Находит все процессы `ollama` командой `pgrep -f "ollama"`.
2. Останавливает их (кроме своих собственных процессов скрипта).
3. Запускает свежий `ollama serve` для проекта.
4. Убеждается, что модель `valera_live:latest` создана (из `Modelfile`).
5. При выходе останавливает только **свою** Ollama.

Если Ollama нужна вам и вне этого проекта — запустите её отдельно и передайте
`--keep-ollama`, чтобы скрипт её не трогал.
