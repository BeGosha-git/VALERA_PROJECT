# WebRAgent на Jetson — локальная Qwen2.5-Omni вместо Ollama

Эта папка — веб-приложение **WebRAgent** (Flask + Qdrant), перенесённое из ветки
`PC` и адаптированное под Jetson AGX Orin. Главное отличие от оригинала:

> **Ollama полностью заменена на локальную модель Qwen2.5-Omni-7B**
> (сервер `python main.py` из корня проекта).

Ни Ollama, ни Docker, ни MongoDB на Jetson не нужны.

## Как это связано с корневым проектом

```
┌──────────────────────────┐        HTTP         ┌───────────────────────────┐
│  WebRAgent (Flask :5000) │  ───────────────▶   │  Наш API (:8765)          │
│  RAG: Qdrant + эмбеддинги│                     │  Qwen2.5-Omni-7B на GPU   │
│  /api/v1/llm/generate    │  ◀───────────────   │  /api/v1/llm/generate     │
└──────────────────────────┘   чистый текст       └───────────────────────────┘
```

- WebRAgent делает RAG сам (ищет чанки в Qdrant, собирает контекст).
- Наш сервер отдаёт **чистую генерацию** — эндпоинт `/api/v1/llm/generate`
  (без персоны «Валера», без авто-RAG и без истории диалога, в отличие от
  `/api/v1/chat/text`).
- Модель загружена **один раз**: второй копии в памяти нет (на Jetson это
  критично — 61 ГБ общие с GPU).

## Что изменено относительно оригинала

| Файл | Что сделано |
|------|-------------|
| `app/services/valera_service.py` | **новый** провайдер: ходит на наш `/api/v1/llm/generate` |
| `app/services/llm_service.py` | `LLMFactory` умеет провайдер `valera` |
| `app/services/model_service.py` | провайдер `valera` в конфиге, списке и выборе по умолчанию; `openai`/`anthropic` импортируются лениво |
| `app/services/qdrant_service.py` | если сервер Qdrant недоступен — локальное хранилище на диске (без Docker); `SentenceTransformer` создаётся с явным `device` (в Jetson-сборке torch нет `torch.distributed`, иначе падает `get_device_name()`) и на CPU, чтобы не занимать GPU модели |
| `app/services/document_service.py` | `docling` стал опциональным (он тянет свой torch и может сломать Jetson-сборку); для PDF используется `PyPDF2` |
| `app/services/local_store.py` | **новый**: JSON-хранилище с API pymongo — замена MongoDB |
| `app/models/user.py`, `app/services/chat_service.py` | используют локальное хранилище, если MongoDB нет |
| `requirements-jetson.txt` | зависимости без torch/docling/openai/anthropic |
| `run_all.sh` (в корне) | запуск модели + WebRAgent одной командой |

## Установка

```bash
cd VALERA_PROJECT

# 1. Зависимости WebRAgent (в то же окружение, чтобы не появился второй torch)
/mnt/valera/conda-envs/qwen-valera/bin/pip install -r WebRAgent/requirements-jetson.txt
```

## Запуск

```bash
bash run_all.sh            # модель + WebRAgent
bash run_all.sh --client   # + голосовой клиент (слушай → отвечай)
bash run_all.sh --status   # что сейчас работает
bash run_all.sh --down     # остановить всё (или Ctrl+C в работающем скрипте)
bash run_all.sh --no-web   # только модель (API на :8765)
bash run_all.sh --no-model # только WebRAgent (если модель уже запущена)
```

`run_all.sh` сам гасит процессы от предыдущего запуска, чтобы не ловить
`Address already in use`, и по Ctrl+C останавливает всё, что поднял.

Открыть: **http://127.0.0.1:5000**, вход `admin` / `admin`
(меняется через `ADMIN_USERNAME` / `ADMIN_PASSWORD`).

> Первый запуск: модель грузится ~2–4 минуты (20.8 ГБ в память). `run_all.sh`
> дождётся готовности и только потом поднимет веб-интерфейс.

## Настройки (`.env`)

Создаётся автоматически из `.env.example`. Ключевые параметры:

```ini
VALERA_API_BASE=http://localhost:8765/api/v1   # адрес нашего сервера с моделью
VALERA_TIMEOUT=600                             # таймаут запроса к модели, сек
VALERA_MODEL_NAME_OR_PATH=Qwen/Qwen2.5-Omni-7B # имя модели для отображения
EMBEDDING_DEVICE=cpu                           # эмбеддинги на CPU (GPU — модели)
FORCE_LOCAL_STORE=1                            # не пытаться использовать MongoDB
```

## Выбор провайдера в веб-интерфейсе

`/admin/models` — список провайдеров. Провайдер **«Qwen2.5-Omni (локально)»**
идёт по умолчанию; OpenAI/Claude/Ollama остались, но требуют своих ключей и
серверов.

## Ollama больше не нужна

Код `app/services/ollama_service.py` оставлен только для совместимости — он не
используется при `llm_provider=valera`. Если Ollama выключена, всё работает:
WebRAgent ходит напрямую в наш API.

## Проверено на Jetson

```
✓ Вход в систему                      200
✓ Создание коллекции                  200
✓ Загрузка документа (эмбеддинги+Qdrant) 200
✓ Запрос через RAG                    200, ~32 c
  Контекст: «Компания VALERA разработала голосового ассистента на Jetson AGX Orin…»
  Ответ модели: «Кодовое слово проекта VALERA — 'ОРИОН-2026'.
                 Проект работает на модели Qwen2.5-Omni-7B.»
```

## Известные ограничения

- **Веб-поиск (SearXNG)** не работает — сервис не установлен (Docker недоступен).
  Поиск по документам (RAG) работает.
- **docling не установлен** → конвертация DOCX/PPTX/XLSX недоступна, для PDF
  используется `PyPDF2`. Если нужен docling, ставьте его вручную с осторожностью:
  он тянет собственный torch и может заменить Jetson-сборку на CPU-версию.
- **MongoDB не установлена** → пользователи и история чатов хранятся в
  `data/local_store/*.json`. Для продакшена можно поднять MongoDB и убрать
  `FORCE_LOCAL_STORE`.
- Оригинальные `Dockerfile`, `docker-compose*.yml` оставлены как есть — они
  рассчитаны на x86 и Ollama, на Jetson не используются.
