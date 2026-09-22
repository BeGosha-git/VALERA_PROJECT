2
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

## 🧠 Выбор модели

| Модель | Размер | Работает на Jetson 64 GB? |
|--------|--------|---------------------------|
| `Qwen/Qwen2.5-Omni-7B` **(по умолчанию)** | ~20.8 GB (bf16) | ✅ Да, «из коробки» |
| `Qwen/Qwen2.5-Omni-3B` | ~11.2 GB (bf16) | ✅ Да, быстрее, качество ниже |
| `Qwen/Qwen3-Omni-30B-A3B-Instruct` | ~60 GB (bf16) | ❌ Не влезает |
| `cyankiwi/Qwen3-Omni-…-AWQ-4bit` (compressed-tensors) | 25.7 GB на диске | ❌ См. ниже |
| любые `NVFP4` / `FP8` | — | ❌ Нужен Blackwell (FP4/FP8-ядра) |

**Почему не квантизованные 30B?** Ключевое ограничение — не диск, а память:

- `transformers` **всегда материализует веса квантованной модели в fp16**. Для
  30B это ~60 GB — при 61 GB общей памяти (Jetson делит её с GPU) не влезает.
  Модель `compressed-tensors` (`format: pack-quantized`) вдобавок требует
  декомпрессии, поэтому падает с `'Linear' object has no attribute 'weight'`.
- `NVFP4` и `FP8` требуют аппаратных ядер, которых в **Ampere (sm_87)** нет —
  они появились только в Blackwell (RTX 5090, B200).

Поэтому используется модель **без квантования**, которая запускается сразу.
Семейство определяется автоматически по `architectures` в `config.json`
(см. `core/model.py`), так что Qwen3-Omni тоже поддерживается — если памяти хватит.

Смена модели — в `.env`:
```ini
VALERA_MODEL_NAME_OR_PATH=Qwen/Qwen2.5-Omni-7B
# легче и быстрее:
# VALERA_MODEL_NAME_OR_PATH=Qwen/Qwen2.5-Omni-3B
```

## 🚀 Быстрый старт

### Шаг 1: Обновите JetPack до 6.x
См. подробную инструкцию в **[JETPACK_UPGRADE.md](JETPACK_UPGRADE.md)**.

### Шаг 2: Внешний диск (диск D) — обязательно

Внутренний eMMC всего **57 GB** и он уже заполнен → PyTorch (~10 GB) + модель (27 GB) туда не влезают. Поэтому окружение, кэши и модель переносятся на внешний диск:

```bash
cd QWEN-VALERA
sudo bash setup_external_storage.sh
```

Скрипт:
- подключает внешний диск и настраивает **автоподключение при загрузке** (systemd-служба `valera-storage.service`);
- создаёт хранилище `/mnt/valera` и переносит туда модель, conda-окружения и кэши (`PIP_CACHE_DIR`, `HF_HOME`, `TMPDIR`, `CONDA_ENVS_PATH`);
- делает `models/` символической ссылкой на `/mnt/valera/models`;
- освобождает ~30 GB на внутреннем диске.

> ⚠️ **Про exFAT.** Диск D отформатирован в **exFAT**, а exFAT не поддерживает symlink/hardlink и права доступа → conda-окружение там работать не может. Поэтому у скрипта два режима:

| Режим | Команда | Что делает | Минус |
|-------|---------|-----------|-------|
| `safe` (по умолчанию) | `sudo bash setup_external_storage.sh` | создаёт на диске D файл-образ ext4 и монтирует его в `/mnt/valera` | разово ~5–10 мин на создание образа |
| `ext4` | `sudo bash setup_external_storage.sh ext4` | переформатирует раздел диска D в ext4 (⚠️ данные удаляются) | диск D перестанет читаться на Windows |

Размер образа в режиме `safe` задаётся так: `sudo VALERA_IMG_SIZE=150G bash setup_external_storage.sh`.

Проверить состояние хранилища в любой момент:

```bash
mountpoint /mnt/valera && df -h /mnt/valera
sudo systemctl status valera-storage.service
```

### Шаг 3: Установка

```bash
cd QWEN-VALERA
chmod +x setup_jetson6.sh
bash setup_jetson6.sh
```

> Установку нужно запускать **при подключённом внешнем диске** — иначе скрипт остановится с подсказкой. Переменные окружения подхватываются автоматически из `/etc/profile.d/valera-storage.sh`.

Скрипт установит:
- NVIDIA Jetson PyTorch 2.5.0 (CUDA 12)
- Transformers с поддержкой Qwen-Omni
- Все зависимости проекта
- Патчи совместимости Jetson-сборки torch
- `torchvision` (из исходников, нужен процессору модели)
- Модель Qwen2.5-Omni-7B (~20.8 GB)

### Шаг 4: Запуск сервера

```bash
bash run_all.sh            # модель + WebRAgent
bash run_all.sh --client   # + голосовой клиент (слушай → отвечай)
bash run_all.sh --status   # что сейчас работает
bash run_all.sh --down     # остановить всё
```

Скрипт сам убивает процессы от прошлого запуска, ждёт загрузки модели
(~30 с–4 мин) и **по Ctrl+C гасит всё**, что поднял.

Запуск только API (без веб-интерфейса):

```bash
conda activate qwen-valera
python main.py
```

Сервер запустится на `http://localhost:8765`.

### Шаг 5: Голосовой клиент

В другом терминале:

```bash
conda activate qwen-valera
python client.py --mode voice
```

Клиент работает **непрерывным диалогом**: ждёт начало речи, останавливает запись
после секунды тишины, сразу отправляет аудио и сразу проигрывает озвученный ответ.
Нажимать ничего не нужно — выход по Ctrl+C.

```bash
python client.py --mode voice                 # авто-режим (по умолчанию)
python client.py --mode voice --tts model     # озвучить встроенным голосом Qwen
python client.py --mode voice --push-to-talk  # старый режим: Enter — запись
python client.py --mode text                  # текстовый чат
```

## � Установка на другом устройстве (воспроизводимо)

Все версии зафиксированы в файлах, поэтому на втором Jetson всё встанет один в один:

| Файл | Что содержит |
|------|--------------|
| `apt-deps.txt` | системные пакеты (ffmpeg, portaudio, antiword, catdoc, libopenblas…) |
| `requirements-jetson6.txt` | **точные** версии всех Python-пакетов (снято через `pip freeze`) |
| `patch_torch_jetson.py` | патчи совместимости Jetson-сборки PyTorch |
| `build_torchvision.sh` | сборка `torchvision` из исходников (15–40 минут, один раз) |
| `setup_external_storage.sh` | внешний диск, автоподключение, перенос модели и кэшей |
| `freeze_env.sh` | переснять lock-файл после установки новых пакетов |

Порядок установки:

```bash
cd VALERA_PROJECT
sudo bash setup_external_storage.sh ext4   # диск D → /mnt/valera
bash setup_jetson6.sh                      # apt-deps + requirements + патчи + модель
```

Вручную, без `setup_jetson6.sh`:

```bash
grep -vE '^\s*#|^\s*$' apt-deps.txt | xargs sudo apt-get install -y
conda create -n qwen-valera python=3.10 -y && conda activate qwen-valera
pip install -r requirements-jetson6.txt
python patch_torch_jetson.py               # ← обязательно
bash build_torchvision.sh                  # ← обязательно, 15-40 минут
python -c "import torch, transformers; print(torch.cuda.is_available(), transformers.is_torch_available())"
# ожидаемый вывод: True True
```

> ⚠️ В `requirements-jetson6.txt` `torch` указан ссылкой на репозиторий NVIDIA, а `transformers` зафиксирован на **4.57.6** (ветки 5.x требуют PyTorch ≥ 2.6, которого на Jetson нет). Не заменяйте их на версии с PyPI: для aarch64 там CPU-сборка torch без CUDA.
>
> ⚠️ `torchvision` **нельзя** ставить как `pip install torchvision` — колесо с PyPI собрано против другой сборки torch и падает с `undefined symbol: ...Node::nameEv`, а с зависимостями оно ещё и затирает Jetson-сборку torch. Только через `build_torchvision.sh`.

После установки новых пакетов обновите lock-файл:

```bash
bash freeze_env.sh
```

## �📚 API

| Метод | Endpoint | Описание |
|-------|----------|----------|
| `POST` | `/api/v1/chat/text` | Текстовый чат |
| `POST` | `/api/v1/llm/generate` | Чистая генерация текста (для внешних приложений, напр. WebRAgent) |
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

## 🌐 WebRAgent — веб-интерфейс RAG на локальной модели

В ветке **`MAIN_VALERA`** добавлен веб-проект **WebRAgent** (Flask + Qdrant) из
ветки `PC` — **с заменой Ollama на нашу Qwen2.5-Omni-7B**:

```
WebRAgent (Flask :5000) ──HTTP──▶ наш API (:8765) ──▶ Qwen2.5-Omni-7B на GPU
   RAG: Qdrant + эмбеддинги          /api/v1/llm/generate
```

Модель загружается **один раз** (второй копии в памяти нет), эмбеддинги
считаются на CPU, веб-поиск и авторизация работают без Docker/MongoDB.

Запуск:

```bash
/mnt/valera/conda-envs/qwen-valera/bin/pip install -r WebRAgent/requirements-jetson.txt
bash run_all.sh            # модель + WebRAgent + клиент по желанию
```

Открыть http://127.0.0.1:5000 (вход `admin` / `admin`).

Подробности — в **[WebRAgent/README_JETSON.md](WebRAgent/README_JETSON.md)**.

## 📁 Структура проекта

```
QWEN-VALERA/
├── main.py              # FastAPI сервер (точка входа)
├── client.py            # Терминальный голосовой/текстовый клиент
├── docs_tool.py         # CLI для управления документами (без сервера)
├── config.py            # Конфигурация
├── test_model.py        # Тест модели
├── download_model.py    # Скачивание модели
├── setup_external_storage.sh  # Перенос окружения на внешний диск (диск D)
├── setup_jetson6.sh     # Установка (после обновления JetPack 6)
├── apt-deps.txt         # Системные пакеты (apt) — список
├── requirements-jetson6.txt   # Точные версии Python-пакетов (lock-файл)
├── patch_torch_jetson.py      # Патчи совместимости Jetson-сборки PyTorch
├── build_torchvision.sh       # Сборка torchvision из исходников
├── freeze_env.sh        # Переснять lock-файл с текущего окружения
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

Qwen-Omni — это **end-to-end мультимодальная модель** (Thinker-Talker):

```
Микрофон → [Qwen-Omni Thinker] → текст-ответ
              ↓                        ↓
       синтез речи (на выбор)    интернет-поиск
              ↓                   и база знаний
        Голос (WAV)
              ↓
          Динамики
```

- **Thinker** — понимает речь, думает, формирует ответ (как ASR + LLM в одном)
- **Синтез речи** — два взаимозаменяемых бэкенда (`VALERA_TTS_BACKEND`):

| Бэкенд | Кто озвучивает | Где считается | Скорость на Jetson | Голоса |
|--------|---------------|---------------|--------------------|--------|
| `russian_tts` **(по умолчанию)** | Silero v3.1_ru | **CPU** (GPU свободен) | ~2 с на 1 с речи | xenia, eugene, aidar, baya, kseniya |
| `model` | встроенный Talker Qwen2.5-Omni | GPU | ~13 с на 1 с речи | Ethan, Chelsie |

Измерено на реальном запросе (ответ 68 символов):

```
russian_tts → 15.4 с всего, 3.7 с речи, 177 КБ WAV
model       → 93.1 с всего, 8.0 с речи, 383 КБ WAV
```

- Qwen2.5-Omni поддерживает десятки языков текста и речи (включая русский)

> 💡 **Персона ассистента.** Qwen2.5-Omni синтезирует голос только если первым
> сообщением идёт канонический системный промпт («You are Qwen, a virtual human…»).
> Поэтому `core/model.py` подставляет его первым, а персона «Валера» идёт вторым
> system-сообщением — так работают и голос, и характер.

## ⚙️ Конфигурация

Все настройки в `.env`:

```ini
# Модель
# Работает «из коробки»: Qwen/Qwen2.5-Omni-7B (20.8 GB) или Qwen/Qwen2.5-Omni-3B (11.2 GB)
VALERA_MODEL_NAME_OR_PATH=Qwen/Qwen2.5-Omni-7B
VALERA_SPEAKER_VOICE=Ethan          # голос модели (Chelsie — женский)
VALERA_ATTN_IMPLEMENTATION=sdpa     # для Jetson (flash-attn не доступен)

# Сервер
VALERA_API_HOST=0.0.0.0
VALERA_API_PORT=8765

# Генерация
VALERA_MAX_NEW_TOKENS=256
VALERA_TEMPERATURE=0.6

# Голосовой режим (каждая секунда озвучки ~13 с генерации на Jetson)
VALERA_VOICE_MAX_NEW_TOKENS=80      # длина ответа голосом (токенов)
VALERA_TALKER_MAX_NEW_TOKENS=400    # потолок озвучки (кадров, ~32 с)

# ---- Чем озвучивать ответ ----
# russian_tts (ПО УМОЛЧАНИЮ) — Silero v3.1_ru: русский голос, CPU, ~2 с на
#                             секунду речи, GPU остаётся свободным
# model                    — встроенный Talker Qwen2.5-Omni: GPU, ~13 с на
#                             секунду речи
VALERA_TTS_BACKEND=russian_tts
VALERA_SILERO_SPEAKER=eugene        # xenia (жен.) / eugene / aidar / baya
VALERA_SILERO_MODEL_PATH=           # пусто = russian_text_to_speech/model.pt

# Поиск
VALERA_SEARCH_ENABLED=true
VALERA_SEARCH_REGION=ru-ru

# Документы (RAG база знаний)
VALERA_RAG_ENABLED=true             # авто-поиск по документам при каждом вопросе
VALERA_RAG_TOP_K=5                  # сколько чанков подставлять в промпт
VALERA_CHUNK_SIZE=800               # размер чанка (символов)
VALERA_CHUNK_OVERLAP=150            # перекрытие чанков
```

## � Внешнее хранилище (диск D)

Всё «тяжёлое» лежит на внешнем диске в `/mnt/valera`:

| Что | Путь |
|-----|------|
| Conda-окружение `qwen-valera` | `/mnt/valera/conda-envs` |
| Кэш pip | `/mnt/valera/pip-cache` |
| Кэш HuggingFace | `/mnt/valera/hf-cache` |
| Модель | `/mnt/valera/models` (ссылка `./models`) |
| Временные файлы (`TMPDIR`) | `/mnt/valera/tmp` |

Автоподключение при загрузке обеспечивает служба `valera-storage.service`. Управление вручную:

```bash
sudo valera-storage.sh up      # подключить
sudo valera-storage.sh down    # отключить
sudo systemctl status valera-storage.service
```

Если диск не подключён, переменные окружения не выставляются (защита в `/etc/profile.d/valera-storage.sh`), а `run.sh` попробует подключить хранилище сам и подскажет, что делать.

Откат изменений:

```bash
sudo systemctl disable --now valera-storage.service
sudo rm /etc/systemd/system/valera-storage.service /usr/local/bin/valera-storage.sh /etc/profile.d/valera-storage.sh
sudo systemctl daemon-reload
```

## �🐛 Отладка

- Логи: `data/valera.log`
- Swagger: `http://localhost:8765/docs`
- Статус: `curl http://localhost:8765/api/v1/health`
- Аудиоустройства: `python client.py --list-devices`
- Документы: `python docs_tool.py test`
### Частые проблемы

> 💡 **Главное правило.** После установки или обновления любых пакетов запускайте:
> ```bash
> python patch_torch_jetson.py           # проверить и починить
> python patch_torch_jetson.py --check   # только проверить, ничего не менять
> python patch_torch_jetson.py --list    # показать текущее состояние
> ```

**`[transformers] Disabling PyTorch because PyTorch >= 2.5 is required but found 2.5.0a0+...`**

Версия Jetson-сборки `2.5.0a0+872d972e41.nv24.8` по правилам PEP 440 — это
**пре-релиз**, то есть «меньше» `2.5.0`. transformers это видит и **отключает
PyTorch** («Models won't be available»), после чего ничего не загружается.

Лечится `python patch_torch_jetson.py` — версия дистрибутива переписывается на
`2.5.0+nv24.08` (локальная версия PEP 440, сравнивается как больше `2.5.0`).

**`ModuleNotFoundError: Could not import module 'GenerationMixin'` (или `'Qwen3OmniMoeForConditionalGeneration'`)**

В Jetson-сборке torch вырезан публичный модуль `torch.nn.attention.flex_attention`
(остался только приватный `_flex_attention.py`), а transformers определяет его
доступность **только по версии**:

```python
def is_torch_flex_attn_available():
    return is_torch_available() and get_torch_version() >= "2.5.0"
```

Лечится `python patch_torch_jetson.py` — создаётся модуль-заглушка, повторяющая
ветку transformers «flex attention недоступна». Проект использует
`attn_implementation="sdpa"`, поэтому flex attention не вызывается.

**`Failed to load model: Qwen2VLVideoProcessor requires the Torchvision library but it was not found`**

Qwen3-Omni — omni-модель: её процессор (`Qwen3OmniMoeProcessor`) поднимает
`Qwen2VLImageProcessor` и `Qwen2VLVideoProcessor`, а они требуют `torchvision`.
Готовых сборок под Jetson нет (в репозитории NVIDIA лежит только torch), а колесо
с PyPI несовместимо с Jetson-сборкой torch:

```
ImportError: .../torchvision/_C.so: undefined symbol: _ZNK5torch8autograd4Node4nameEv
```

Решение — собрать из исходников (один раз, 15–40 минут):

```bash
bash build_torchvision.sh
```

> С зависимостями (`pip install torchvision`) ставить нельзя: он требует
> ровно `torch==2.5.1` и затирает Jetson-сборку — CUDA пропадёт.
> Скрипт ставит её через `--no-deps` + `--no-build-isolation` против
> установленного Jetson-torch.

**`ImportError: libcusparseLt.so.0: cannot open shared object file`**

Jetson-сборка torch слинкована с cuSPARSELt, которой нет ни в системе, ни в
репозиториях Jetson. Библиотека ставится из PyPI, а загрузчик ищет её рядом с
`libtorch_cuda.so` (RUNPATH = `$ORIGIN`), поэтому нужен симлинк в `torch/lib`:

```bash
pip install nvidia-cusparselt-cu12
ENV=$(python -c "import sys; print(sys.prefix)")
ln -sf "$ENV"/lib/python*/site-packages/nvidia/cusparselt/lib/libcusparseLt*.so* \
       "$ENV"/lib/python*/site-packages/torch/lib/
```
(в `setup_jetson6.sh` это делается автоматически)

**`torch.cuda.is_available()` возвращает `False`**

Почти всегда это CPU-сборка torch с PyPI, которая подменила Jetson-сборку.
Так делает `torchvision` из PyPI: он требует ровно `torch==2.5.1`. Проект
torchvision не использует.

```bash
pip uninstall -y torchvision torch
pip install "https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"
python patch_torch_jetson.py
python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
```

**`UserWarning: Failed to initialize NumPy: _ARRAY_API not found`**

Jetson-сборка torch собрана под numpy 1.x, с numpy 2.x не работает:

```bash
pip install "numpy<2"      # ставит 1.26.4
```

**`conda activate qwen-valera` → `EnvironmentNameNotFound: Could not find conda environment`**

Значит не смонтировано хранилище `/mnt/valera` — окружение лежит там.

```bash
mountpoint /mnt/valera
systemctl status valera-storage.service --no-pager
lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINT,UUID,LABEL
```

Две частые причины:

1. **udisks примонтировал раздел в другое место** (`/media/nv_3/…`) — при входе в
   систему он подхватывает диск раньше службы.
2. **Сменился UUID** — `mkfs.ext4` при форматировании создаёт новый UUID, а служба
   искала старый из настроек.

В обоих случаях лечится повторным запуском (скрипт идемпотентен, форматировать
ещё раз ничего не будет):

```bash
sudo bash setup_external_storage.sh ext4
```

Скрипт находит диск по UUID, затем по метке `VALERA`, затем автоопределением,
отмонтирует его из `/media/...` и поднимет в `/mnt/valera`.

**`'Linear' object has no attribute 'weight'` (при запросе к модели)**

Модель в формате `compressed-tensors` (`format: pack-quantized`) не была
декомпрессирована при загрузке. `transformers` **всегда** материализует веса
квантованной модели в fp16, а запуск «как есть» (packed) поддерживается только в
vLLM. Для 30B это ~60 GB, что не влезает в 61 GB общей памяти Jetson.

Стек подтверждает:

```
compressed_tensors/quantization/lifecycle/forward.py:272 in quantized_forward
    weight = self.weight  # onload only
AttributeError: 'Linear' object has no attribute 'weight'
```

**Решение — взять модель без квантования** (см. раздел «🧠 Выбор модели»):

```ini
VALERA_MODEL_NAME_OR_PATH=Qwen/Qwen2.5-Omni-7B
```

Классы модели/процессора при этом подбираются автоматически по `config.json`
(`core/model.py` → `resolve_model_classes()`).

**`ModuleNotFoundError: No module named 'chromadb'`** (или `pydub`, `onnxruntime`)

Пакет не попал в установку. Поставьте всё по lock-файлу:

```bash
pip install -r requirements-jetson6.txt
```

## 📝 Лицензия

Модель: Apache-2.0 (Qwen). Проект: для личного использования.
