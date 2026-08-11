# 🎙️ QWEN-VALERA — Голосовой Ассистент

Голосовой ассистент на базе **Qwen3-Omni-30B-A3B-Instruct** с локальной базой знаний из ваших документов и доступом в интернет.

## ✨ Возможности

- 🎤 **Голосовой вход** — микрофон → распознавание речи (встроенный ASR модели)
- 🔊 **Голосовой выход** — ИИ → речь (встроенный TTS модели, голос "Ethan")
- 💬 **Текстовый чат** — полноценный многоходовой диалог
- 📄 **База знаний из документов** — загружайте `.doc`, `.docx`, `.pdf`, `.txt` — ИИ разобьёт их на чанки, токенизирует (эмбеддинги) и всегда сможет отвечать по их содержимому (RAG)
- 🌐 **Интернет-поиск** — DuckDuckGo (бесплатно, без API-ключа)
- 🗄️ **Локальная база данных** — SQLite + ChromaDB (векторный поиск)
- 🔌 **REST API + WebSocket** — для интеграции с любыми приложениями
- 🇷🇺 **Русский язык** — поддерживается моделью нативно (и вход, и выход)
- 📦 **Работает локально** — только поиск в интернете требует сети

## 🖥️ Аппаратные требования

| Параметр | Значение |
|----------|----------|
| Платформа | Jetson AGX Orin (или x86 + NVIDIA GPU) |
| JetPack | **6.x** (CUDA 12) |
| VRAM | 64 GB (у нас хватает с запасом) |
| Python | 3.10 |

> ⚠️ **Важно:** Qwen3-Omni требует JetPack 6.x. Если у вас JetPack 5.x — см. `JETPACK_UPGRADE.md`.

## 🧠 Выбор модели (квантование)

| Модель | Размер | Качество | Работает на Jetson (Ampere)? |
|--------|--------|----------|------------------------------|
| `AWQ-8bit` **(по умолчанию)** | ~42 GB | Лучшее | ✅ Да |
| `AWQ-4bit` | ~27 GB | Хорошее | ✅ Да |
| `NVFP4` (25 GB) | ~26 GB | Лучше 4-bit | ❌ **Нет** — нужен Blackwell (FP4-железо) |

**Почему не NVFP4?** Ваш Jetson AGX Orin — архитектура **Ampere (sm_87)**, в которой **нет FP4-аппаратных ядер** (они появились только в Blackwell: RTX 5090, B200). Модели NVFP4 (25 GB) физически не запустятся. AWQ — целочисленное квантование, работает на любой архитектуре.

Смена модели — в `.env`:
```ini
VALERA_MODEL_NAME_OR_PATH=cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-8bit
# или для экономии памяти:
# VALERA_MODEL_NAME_OR_PATH=cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit
```

## 🚀 Быстрый старт

### Шаг 1: Обновите JetPack до 6.x
См. подробную инструкцию в **[JETPACK_UPGRADE.md](JETPACK_UPGRADE.md)**.

### Шаг 2: Установка

```bash
cd QWEN-VALERA
chmod +x setup_jetson6.sh
bash setup_jetson6.sh
```

Скрипт установит:
- NVIDIA Jetson PyTorch 2.5.0 (CUDA 12)
- Transformers с поддержкой Qwen3-Omni
- Все зависимости проекта
- Модель AWQ 4-bit (~10 GB)

### Шаг 3: Запуск сервера

```bash
conda activate qwen-valera
python main.py
```

Сервер запустится на `http://localhost:8765`. Модель загрузится автоматически (1-2 минуты).

### Шаг 4: Голосовой клиент

В другом терминале:

```bash
conda activate qwen-valera
python client.py --mode voice
```

## 📚 API

| Метод | Endpoint | Описание |
|-------|----------|----------|
| `POST` | `/api/v1/chat/text` | Текстовый чат |
| `POST` | `/api/v1/chat/voice` | Голосовой чат (audio → text + audio) |
| `POST` | `/api/v1/chat/voice/raw` | Голосовой чат (возвращает WAV напрямую) |
| `GET` | `/api/v1/audio/{filename}` | Скачать сгенерированный аудиофайл |
| `POST` | `/api/v1/documents/upload` | Загрузить документ (.doc/.docx/.pdf/.txt/.md) |
| `GET` | `/api/v1/documents` | Список документов |
| `GET` | `/api/v1/documents/{id}` | Статус обработки документа |
| `DELETE` | `/api/v1/documents/{id}` | Удалить документ + его чанки |
| `POST` | `/api/v1/documents/search` | Семантический поиск по документам |
| `GET` | `/api/v1/documents/stats` | Статистика базы документов |
| `POST` | `/api/v1/knowledge` | Добавить в базу знаний |
| `POST` | `/api/v1/knowledge/search` | Поиск по базе знаний |
| `GET` | `/api/v1/sessions` | Активные сессии |
| `GET` | `/api/v1/sessions/{id}/history` | История сессии |
| `GET` | `/api/v1/health` | Статус сервера и GPU |
| `GET` | `/api/v1/devices` | Список аудиоустройств |
| `WS` | `/ws/chat` | Реалтайм-голосовой чат |
| `GET` | `/docs` | Swagger документация |

### 📄 База знаний из документов (RAG)

Загрузите свои документы — ИИ **токенизирует их** (разобьёт на чанки и превратит в векторные эмбеддинги) и будет **автоматически подгружать нужные фрагменты** при каждом вашем вопросе:

```python
import requests

# 1. Загружаем документ
with open("инструкция.docx", "rb") as f:
    resp = requests.post(
        "http://localhost:8765/api/v1/documents/upload",
        files={"file": ("инструкция.docx", f)},
    )
    doc = resp.json()
    print("Document ID:", doc["id"], "| status:", doc["status"])

# 2. Проверяем, что обработался
resp = requests.get(f"http://localhost:8765/api/v1/documents/{doc['id']}")
print(resp.json())  # status: "ready", num_chunks: N

# 3. Теперь можно просто задавать вопросы — нужные чанки подтянутся сами
resp = requests.post("http://localhost:8765/api/v1/chat/text", json={
    "text": "Что написано в инструкции о настройке?",
})
print(resp.json()["text"])
```

**Или через CLI** (без запуска сервера):
```bash
python docs_tool.py index path/to/file.docx   # добавить документ
python docs_tool.py index path/to/folder/     # добавить все документы из папки
python docs_tool.py search "мой вопрос"       # поиск по документам
python docs_tool.py list                      # список документов
python docs_tool.py test                      # тест: создать и проиндексировать пример
```

**Поддерживаемые форматы:** `.doc`, `.docx`, `.pdf`, `.txt`, `.md`, `.rtf`, `.log`

> 💡 **Как работает RAG:** при вопросе модель находит 5 (настраивается `VALERA_RAG_TOP_K`) самых похожих чанков в ChromaDB и подставляет их в промпт. Модель отвечает на основе ваших документов. Эмбеддинги считаются локально на CPU (`intfloat/multilingual-e5-small`, поддержка русского).

### Пример: голосовой запрос (Python)

```python
import requests

# Отправляем аудио, получаем ответ (текст + аудио URL)
with open("question.wav", "rb") as f:
    resp = requests.post(
        "http://localhost:8765/api/v1/chat/voice",
        files={"audio": ("q.wav", f, "audio/wav")},
        data={"session_id": "my_session"},
    )
    result = resp.json()
    print("Ответ:", result["text"])
    print("Аудио:", result["audio_url"])  # /api/v1/audio/assistant_xxx.wav
```

### Пример: добавление в базу знаний

```python
requests.post("http://localhost:8765/api/v1/knowledge", json={
    "title": "Мой пароль от Wi-Fi",
    "content": "Wi-Fi пароль: 12345678",
    "tags": "пароль,wifi,личное"
})
```

## 📁 Структура проекта

```
QWEN-VALERA/
├── main.py              # FastAPI сервер (точка входа)
├── client.py            # Терминальный голосовой/текстовый клиент
├── docs_tool.py         # CLI для управления документами (без сервера)
├── config.py            # Конфигурация
├── test_model.py        # Тест модели
├── download_model.py    # Скачивание модели
├── setup_jetson6.sh     # Установка (после обновления JetPack 6)
├── run.sh               # Быстрый запуск
├── api/
│   ├── routes.py        # REST endpoints
│   ├── document_routes.py # Загрузка и поиск документов
│   ├── ws.py            # WebSocket (реалтайм)
│   └── schemas.py       # Pydantic-схемы
├── core/
│   ├── model.py         # Загрузка модели и инференс
│   ├── audio_io.py      # Микрофон и динамики
│   ├── conversation.py  # Многоходовые диалоги
│   └── search.py        # Интернет-поиск
├── db/
│   ├── database.py      # SQLite (история, знания, документы)
│   ├── documents.py     # Парсинг/чанкование/эмбеддинг документов
│   └── knowledge_base.py # ChromaDB (векторный поиск)
└── utils/helpers.py     # Утилиты
```

## 🧠 Как это работает

Qwen3-Omni — это **end-to-end мультимодальная модель** с архитектурой Thinker-Talker:

```
Микрофон → [Qwen3-Omni Thinker] → текст-ответ
              ↓                        ↓
       [Qwen3-Omni Talker]      интернет-поиск
              ↓                   и база знаний
        Голос (WAV)                    ↓
              ↓                     (локально)
          Динамики
```

- **Thinker** — понимает речь, думает, формирует ответ (как ASR + LLM в одном)
- **Talker** — превращает текст-ответ в естественную речь (как TTS в одном)
- Модель поддерживает **119 языков текста**, **19 языков речи на вход**, **10 языков речи на выход** (включая русский)

## ⚙️ Конфигурация

Все настройки в `.env`:

```ini
# Модель
# Высокое качество (~42 GB): cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-8bit
# Сбалансировано (~27 GB):   cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit
VALERA_MODEL_NAME_OR_PATH=cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-8bit
VALERA_SPEAKER_VOICE=Ethan          # голос модели
VALERA_ATTN_IMPLEMENTATION=sdpa     # для Jetson (flash-attn не доступен)

# Сервер
VALERA_API_HOST=0.0.0.0
VALERA_API_PORT=8765

# Генерация
VALERA_MAX_NEW_TOKENS=2048
VALERA_TEMPERATURE=0.6

# Поиск
VALERA_SEARCH_ENABLED=true
VALERA_SEARCH_REGION=ru-ru

# Документы (RAG база знаний)
VALERA_RAG_ENABLED=true             # авто-поиск по документам при каждом вопросе
VALERA_RAG_TOP_K=5                  # сколько чанков подставлять в промпт
VALERA_CHUNK_SIZE=800               # размер чанка (символов)
VALERA_CHUNK_OVERLAP=150            # перекрытие чанков
```

## 🐛 Отладка

- Логи: `data/valera.log`
- Swagger: `http://localhost:8765/docs`
- Статус: `curl http://localhost:8765/api/v1/health`
- Аудиоустройства: `python client.py --list-devices`
- Документы: `python docs_tool.py test`

## 📝 Лицензия

Модель: Apache-2.0 (Qwen). Проект: для личного использования.
