#!/usr/bin/env python3
"""
valera_web.py
=============
Веб-пульт робота-экскурсовода «Валера» (замена настольного GUI).

Что это
-------
Тот же функционал, что и ``valera_gui.py`` (тумблеры функций, визуализация
звука, триггеры «слово → curl», настройка чувствительности, управление чатами),
но в браузере. Работает на «невзрачном» порту (по умолчанию **8137**), поэтому
случайный человек в интернете его не найдёт, а вы открываете его с другого
устройства — например, с планшета или телефона рядом с роботом.

Запуск
------
    python valera_web.py                  # http://localhost:8137
    python valera_web.py --port 8137
    python valera_web.py --host 0.0.0.0   # доступ с других устройств сети
    python valera_web.py --stt kairos      # модель распознавания речи

Настольный GUI по-прежнему доступен: ``python valera_gui.py`` (или
``./run_all.sh gui``).

Архитектура
-----------
* Flask-приложение с JSON-API (``/api/...``) и одной HTML-страницей.
* Состояние (тумблеры, пороги, триггеры) читается/пишется через
  ``valera_runtime`` — тот же самый файл ``valera_config.json``, что и у GUI.
* Сам ассистент запускается в фоновом потоке через ``voice_assistant.build_assistant``
  — ровно тот же код, что и в CLI/GUI, никакой второй реализации.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from typing import Any, Dict, Optional

log = logging.getLogger("valera_web")

# Flask импортируется лениво (в main), чтобы модуль можно было тестировать
# функции без установленного Flask.
from valera_runtime import (
    get_runtime,
    list_audio_inputs,
    list_audio_outputs,
    list_network_interfaces,
)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PORT = 8137  # «невзрачный» порт по умолчанию

# Раньше при старом VPN ответы больше ~16 КБ на loopback обрезались, и пульт
# следил за размером. Новый VPN проверен (30.10.2026): 3 МБ по loopback и по LAN
# доходят целиком, предупреждение и лимит сняты. gzip оставлен — он просто ускоряет.


# ===========================================================================
# Фоновый запуск ассистента (тот же код, что в CLI и GUI)
# ===========================================================================
class AssistantRunner:
    """
    Держит ``VoiceAssistant`` в отдельном потоке, чтобы веб-интерфейс мог
    запускать/останавливать робота и показывать его состояние.

    Ничего не дублирует: вызывает ``voice_assistant.build_assistant`` — общий
    конструктор для CLI, GUI и веба.
    """

    def __init__(self, runtime):
        self.runtime = runtime
        self.assistant = None
        self.thread: Optional[threading.Thread] = None
        self.error: Optional[str] = None
        self._lock = threading.Lock()
        self.started_at: Optional[float] = None

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def status(self) -> Dict[str, Any]:
        return {
            "running": self.running,
            "error": self.error,
            "uptime_sec": (time.time() - self.started_at) if self.started_at and self.running else 0,
            "chat": self.runtime.get_current_chat(),
        }

    def start(self, stt: str = "kairos", new_chat: bool = False,
              no_clean: bool = False) -> bool:
        """Запускает ассистента. False, если уже запущен."""
        with self._lock:
            if self.running:
                return False
            self.error = None

        def _run():
            try:
                from voice_assistant import build_assistant

                self.runtime.set_status("Загрузка моделей…")
                self.assistant = build_assistant(
                    stt=stt, new_chat=new_chat, no_clean=no_clean,
                    runtime=self.runtime,
                )
                self.started_at = time.time()
                self.assistant.start()
                while self.assistant.is_recording:
                    time.sleep(0.25)
            except Exception as exc:  # noqa: BLE001
                self.error = str(exc)
                self.runtime.log_event("error", f"Ассистент упал: {exc}")
                self.runtime.set_status("Ошибка")
            finally:
                self.started_at = None
                if not self.error:
                    self.runtime.set_status("Остановлен")

        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()
        return True

    def stop(self) -> bool:
        """Останавливает ассистента. False, если он не запущен."""
        if not self.running:
            return False
        if self.assistant is not None:
            try:
                self.assistant.stop()
            except Exception as exc:  # noqa: BLE001
                self.runtime.log_event("error", f"Ошибка остановки: {exc}")
        self.thread = None
        return True


# ===========================================================================
# Приложение Flask
# ===========================================================================
def create_app(runtime=None, runner=None):
    """
    Собирает Flask-приложение. Вынесено отдельно, чтобы можно было поднять
    сервер и в тестах, и в бою.
    """
    from flask import Flask, jsonify, render_template_string, request

    rt = runtime if runtime is not None else get_runtime()
    asst_runner = runner if runner is not None else AssistantRunner(rt)

    app = Flask(__name__)
    app.config["JSON_AS_ASCII"] = False

    # ------------------------------------------------------------------
    # Сжатие всех ответов gzip
    # ------------------------------------------------------------------
    # Причина: в этом окружении несжатые ответы больше ~16 КБ «зависают» на
    # loopback (заголовки приходят, тело — нет). Это ломало и главную
    # страницу, и крупные JSON (список устройств, история уровней для
    # графиков). gzip уменьшает объём в 2-4 раза и снимает проблему.
    import gzip as _gzip

    @app.after_request
    def _compress(response):
        try:
            # Anti-cache: страница и API — «живые», браузер не должен
            # показывать устаревший интерфейс после наших правок.
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"

            accept = request.headers.get("Accept-Encoding", "").lower()
            if "gzip" not in accept:
                return response
            if response.direct_passthrough or response.status_code >= 300:
                return response
            if response.headers.get("Content-Encoding"):
                return response
            body = response.get_data()
            if len(body) < 1024:      # мелочь сжимать смысла нет
                return response
            compressed = _gzip.compress(body, 6)
            if len(compressed) >= len(body):
                return response
            response.set_data(compressed)
            response.headers["Content-Encoding"] = "gzip"
            response.headers["Content-Length"] = str(len(compressed))
            response.headers["Vary"] = "Accept-Encoding"
        except Exception:  # noqa: BLE001
            # Сжатие — оптимизация; если что-то пошло не так, отдаём как есть
            pass
        return response

    # ------------------------------------------------------------------
    # Страница (одностраничное приложение на чистом JS)
    # ------------------------------------------------------------------
    @app.get("/")
    def index():
        # Отдаём страницу как обычный HTML (не через Jinja): в CSS/JS есть
        # фигурные скобки, которые шаблонизатор принял бы за свои директивы.
        # Комментарии вырезаем — страница вплотную подходит к лимиту 16 КБ.
        return _html_response(_compact_html(_page_html()))

    # ------------------------------------------------------------------
    # ПРАВИЛА ДЛЯ ГОСТЕЙ — отдельная страница для второго экрана
    # ------------------------------------------------------------------
    @app.get("/rules")
    def rules():
        """
        Памятка для гостей + живой ответ робота.

        Зачем отдельная страница: её ставят на второй экран/планшет рядом с
        роботом, чтобы гость видел, КАК говорить с Валерой, и сразу читал
        ответ. Пульт (/) для этого не годится — там настройки и графики.
        """
        return _html_response(_compact_html(RULES_HTML))

    @app.get("/api/live")
    def api_live():
        """
        Минимум данных для страницы правил: статус, услышанное и ответ.

        Отдельный эндпоинт, а не ``/api/state``: страница правил опрашивается
        каждые полсекунды, а полный ``/api/state`` тащит ещё и буферы волн
        (~5 КБ gzip). Здесь ответ в разы меньше, и экран обновляется без
        задержек.
        """
        levels = rt.get_levels()
        mic = 0.0
        if levels:
            try:
                mic = float(levels[-1][1])
            except (TypeError, ValueError, IndexError):
                mic = 0.0
        asst = asst_runner.status()
        return jsonify({
            "status": rt.get_status(),
            "heard": rt.last_transcript or "",
            "answer": rt.last_answer or "",
            "running": bool(asst.get("running")),
            "error": asst.get("error"),
            "mic": round(mic, 4),
            "uptime": int(asst.get("uptime_sec") or 0),
        })

    # ------------------------------------------------------------------
    # Состояние: настройки, метрики, события
    # ------------------------------------------------------------------
    @app.get("/api/state")
    def api_state():
        """Один запрос — всё, что нужно странице для отрисовки."""
        metrics = rt.get_metrics()
        return jsonify({
            "config": rt.snapshot(),
            "features": rt.snapshot()["features"],
            "status": rt.get_status(),
            "metrics": metrics,
            "current_chat": rt.get_current_chat(),
            "assistant": asst_runner.status(),
            "transcript": rt.last_transcript,
            "answer": rt.last_answer,
            "phrase": rt.get_phrase(),
            "levels": _downsample_levels(rt.get_levels()),
            "waves": _downsample_waves(rt.get_waves()),
            "events": _recent_events(rt.get_events()),
        })

    @app.get("/api/models")
    def api_models():
        """
        Список установленных моделей Ollama + текущая выбранная.

        Отдаём именно то, что реально есть в Ollama (а не выдуманный список),
        чтобы на сайте не было «настроек-пустышек».
        """
        import requests as _rq

        current = rt.get("llm", "model", default="true_valera_fast:latest")
        models = []
        try:
            base = rt.get("llm", "ollama_url", default="http://localhost:11434")
            resp = _rq.get(f"{base}/api/tags", timeout=5)
            for m in resp.json().get("models", []):
                models.append({
                    "name": m.get("name"),
                    "size_gb": round((m.get("size") or 0) / 1e9, 1),
                })
        except Exception as exc:  # noqa: BLE001
            log.warning("Не удалось получить список моделей Ollama: %s", exc)
        return jsonify({"models": models, "current": current})

    @app.post("/api/models/warm")
    def api_models_warm():
        """Прогревает текущую модель в память, чтобы первый ответ был быстрым."""
        import requests as _rq

        model = rt.get("llm", "model", default="true_valera_fast:latest")
        base = rt.get("llm", "ollama_url", default="http://localhost:11434")
        try:
            _rq.post(f"{base}/api/generate", json={
                "model": model, "prompt": "привет", "stream": False,
                "keep_alive": -1, "options": {"num_predict": 3},
            }, timeout=180)
            return jsonify({"ok": True, "model": model})
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": str(exc)}), 500

    # ------------------------------------------------------------------
    # Состояние системы: сервисы, видеокарты, процессор и память
    # ------------------------------------------------------------------
    @app.get("/api/system")
    def api_system():
        import subprocess

        import requests as _rq

        def _get(url, timeout=1.5):
            try:
                s = _rq.Session()
                s.trust_env = False            # локальные сервисы — мимо HTTP_PROXY
                return s.get(url, timeout=timeout)
            except Exception:  # noqa: BLE001
                return None

        # --- видеокарты ---
        gpus = []
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=4).stdout
            roles_env = {"LLM": os.getenv("LLM_GPU"), "XTTS": os.getenv("XTTS_GPU"), "Распознавание+Silero": os.getenv("ASR_GPU")}
            for line in out.strip().splitlines():
                idx, name, used, total, util, temp = [x.strip() for x in line.split(",")]
                gpus.append({
                    "index": idx, "name": name, "used": int(float(used)), "total": int(float(total)),
                    "util": int(float(util)), "temp": int(float(temp)),
                    "roles": [r for r, g in roles_env.items() if g == idx],
                })
        except Exception:  # noqa: BLE001
            pass

        # --- сервисы (проверяем параллельно: каждая проверка до 1,5 с) ---
        from concurrent.futures import ThreadPoolExecutor

        base = rt.get("llm", "ollama_url", default="http://localhost:11434")
        rag_base = rt.get("rag", "base_url", default="http://localhost:5000")
        rb = rt.get("robot", "base_url", default="")

        def _xtts():
            try:
                from tts_backends import get_xtts_client
                xt = get_xtts_client()
                return xt.is_up(), (len(xt.speakers()) if xt.is_up() else 0)
            except Exception:  # noqa: BLE001
                return False, 0

        with ThreadPoolExecutor(max_workers=4) as pool:
            f_oll = pool.submit(_get, f"{base}/api/ps")
            f_rag = pool.submit(_get, rag_base + "/auth/login")
            f_rob = pool.submit(_get, rb, 1.0) if rb else None
            f_xtts = pool.submit(_xtts)
            r = f_oll.result()
            r_rag = f_rag.result()
            r_rob = f_rob.result() if f_rob else None
            xt_up, xt_n = f_xtts.result()

        services = []
        if r is not None and r.ok:
            models = (r.json() or {}).get("models") or []
            sub = ", ".join(f"{m.get('name')} · {round((m.get('size_vram') or 0)/1e9, 1)} ГБ видеопамяти" for m in models) or "модель не загружена"
            services.append({"name": "Ollama (языковая модель)", "ok": True, "sub": sub})
        else:
            services.append({"name": "Ollama (языковая модель)", "ok": False, "sub": "не отвечает"})
        services.append({"name": "XTTS-v2 (голос)", "ok": xt_up,
                         "sub": f"работает, голосов: {xt_n}" if xt_up else "не запущен — говорит Silero"})
        services.append({"name": "WebRAgent (база знаний)", "ok": bool(r_rag is not None and r_rag.status_code < 500),
                         "sub": rag_base if r_rag is not None else "не отвечает"})
        if rb:
            services.append({"name": "Робот Unitree H1", "ok": r_rob is not None,
                             "sub": rb if r_rob is not None else "нет связи (" + rb + ")"})

        # --- процессор и память ---
        mem = {}
        try:
            for line in open("/proc/meminfo", encoding="utf-8"):
                k, v = line.split(":")
                mem[k] = int(v.split()[0])      # КБ
        except Exception:  # noqa: BLE001
            pass
        cores = os.cpu_count() or 1
        load = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
        total_gb = mem.get("MemTotal", 0) / 1048576
        avail_gb = mem.get("MemAvailable", 0) / 1048576
        return jsonify({
            "gpus": gpus, "services": services,
            "cpu": {"cores": cores, "load": round(load, 2), "pct": min(100, int(load / cores * 100))},
            "mem": {"total_gb": round(total_gb, 1), "used_gb": round(total_gb - avail_gb, 1)},
        })

    # ------------------------------------------------------------------
    # Голос озвучки: XTTS-v2 (основной) ↔ Silero (запасной)
    # ------------------------------------------------------------------
    SILERO_VOICES = ["aidar", "baya", "kseniya", "xenia", "eugene"]

    @app.get("/api/tts")
    def api_tts():
        """Текущий движок озвучки, доступные голоса и состояние сервиса XTTS."""
        from tts_backends import get_xtts_client

        xtts = get_xtts_client()
        up = xtts.is_up()
        return jsonify({
            "backend": rt.get("tts", "backend", default="xtts"),
            "xtts_up": up,
            "xtts_speakers": xtts.speakers() if up else [],
            "xtts_speaker": rt.get("tts", "xtts_speaker", default="Damien Black"),
            "speed": rt.get("tts", "speed", default=1.0),
            "silero_speakers": SILERO_VOICES,
            "silero_speaker": rt.get("tts", "speaker", default="eugene"),
        })

    @app.post("/api/tts/preview")
    def api_tts_preview():
        """Озвучивает пробную фразу выбранным голосом XTTS и отдаёт WAV для браузера."""
        import io
        import wave

        from flask import Response
        from tts_backends import get_xtts_client

        payload = request.get_json(silent=True) or {}
        text = (payload.get("text") or "Привет! Я Валера, робот-экскурсовод.").strip()[:200]
        speaker = payload.get("speaker") or rt.get("tts", "xtts_speaker", default="Damien Black")
        rate = 24000
        try:
            speed = float(payload.get("speed") or rt.get("tts", "speed", default=1.0))
        except (TypeError, ValueError):
            speed = 1.0
        pcm = get_xtts_client().synthesize(text, speaker=speaker, sample_rate=rate, speed=speed)
        if not pcm:
            return jsonify({"ok": False, "error": "сервис XTTS не отвечает"}), 503
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(pcm)
        return Response(buf.getvalue(), mimetype="audio/wav")

    # ------------------------------------------------------------------
    # Выбор LLM-провайдера: локальная (Ollama) ↔ облачная (DeepSeek)
    # ------------------------------------------------------------------
    # Ответ на вопрос ищет и формулирует WebRAgent, поэтому провайдер
    # переключается ИМЕННО там (в его конфиге моделей), а не в
    # valera_config.json. Пульт выступает посредником: логинится в WebRAgent и
    # дёргает его админский API.
    def _rag_session():
        """
        Возвращает (base_url, requests.Session) с выполненным входом в WebRAgent.

        Возвращает ``(None, None)``, если вход не удался — вызывающий код
        отдаст понятную ошибку вместо падения.
        """
        import requests as _rq

        base = str(rt.get("rag", "base_url", default="http://localhost:5000")).rstrip("/")
        user = rt.get("rag", "username", default="admin")
        pwd = rt.get("rag", "password", default="change_me_in_production")
        session = _rq.Session()
        try:
            session.post(f"{base}/auth/login",
                         data={"username": user, "password": pwd},
                         timeout=10)
            return base, session
        except Exception as exc:  # noqa: BLE001
            log.warning("Не удалось войти в WebRAgent: %s", exc)
            return None, None

    @app.get("/api/llm-provider")
    def api_llm_provider():
        """
        Текущий провайдер LLM в WebRAgent + какие провайдеры доступны.

        Доступность считаем по ключу: DeepSeek без DEEPSEEK_API_KEY включить
        нельзя, и лучше показать это на сайте, чем получить ошибку при ответе.
        """
        base, session = _rag_session()
        if not base:
            return jsonify({
                "provider": None, "model": None,
                "available": {"ollama": False, "deepseek": False},
                "error": "WebRAgent недоступен",
            })

        provider, model, available = None, None, {"ollama": False, "deepseek": False}
        try:
            resp = session.get(f"{base}/admin/api/models/active", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                provider = (data.get("llm_provider") or "").lower() or None
                models = data.get("models") or {}
                if provider:
                    model = models.get(f"{provider}_llm") or models.get("embedding")
            resp = session.get(f"{base}/admin/api/models/providers", timeout=10)
            if resp.status_code == 200:
                for p in resp.json().get("providers", []):
                    pid = p.get("id")
                    if pid in available:
                        available[pid] = bool(p.get("available"))
        except Exception as exc:  # noqa: BLE001
            log.warning("Не удалось получить провайдера LLM: %s", exc)

        return jsonify({
            "provider": provider, "model": model, "available": available,
        })

    @app.post("/api/llm-provider")
    def api_llm_provider_set():
        """
        Переключает провайдера LLM в WebRAgent.

        Тело: ``{"provider": "ollama" | "deepseek"}``.
        Изменение применяется к следующему же ответу — WebRAgent создаёт
        LLM-сервис на каждый запрос и читает активную конфигурацию.
        """
        payload = request.get_json(silent=True) or {}
        provider = str(payload.get("provider") or "").strip().lower()
        if provider not in ("ollama", "deepseek"):
            return jsonify({"ok": False, "error": "Неизвестный провайдер"}), 400

        base, session = _rag_session()
        if not base:
            return jsonify({"ok": False, "error": "WebRAgent недоступен"}), 503

        try:
            resp = session.post(
                f"{base}/admin/api/models/set-provider",
                json={"provider": provider}, timeout=15,
            )
            data = resp.json() if resp.content else {}
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": str(exc)}), 500

        if not data.get("success"):
            return jsonify({
                "ok": False,
                "error": data.get("error") or "Не удалось переключить провайдера",
            }), 400

        # Сбрасываем кэш провайдера в ассистенте, чтобы он сразу увидел смену
        # (иначе до минуты продолжал бы считать модель локальной и пересказывать
        # ответ через 3B — а это портит факты облачной модели).
        try:
            assistant = getattr(asst_runner, "assistant", None)
            if assistant is not None:
                assistant._provider_cache = None
        except Exception:  # noqa: BLE001
            pass

        log.info("LLM-провайдер переключён на %s", provider)
        return jsonify({"ok": True, "provider": provider})

    @app.get("/api/devices")
    def api_devices():
        """Списки доступных микрофонов, колонок и сетевых интерфейсов."""
        return jsonify({
            "inputs": list_audio_inputs(),
            "outputs": list_audio_outputs(),
            "interfaces": list_network_interfaces(),
            "selected": {
                "input_device": rt.get("audio", "input_device", default="default"),
                "output_device": rt.get("audio", "output_device", default="default"),
                "interface": rt.get("network", "interface", default="auto"),
            },
        })

    @app.post("/api/settings")
    def api_settings():
        """Записывает настройки (тот же valera_config.json, что и у GUI)."""
        payload = request.get_json(silent=True) or {}
        for path, value in payload.items():
            keys = path.split(".")
            if not keys:
                continue
            rt.set(*keys, value=value)
        return jsonify({"ok": True, "config": rt.snapshot()})

    @app.post("/api/feature")
    def api_feature():
        """Быстрое переключение флага из секции features."""
        payload = request.get_json(silent=True) or {}
        name = payload.get("name")
        if not name:
            return jsonify({"ok": False, "error": "не указано имя флага"}), 400
        rt.set_feature(name, bool(payload.get("value")))
        return jsonify({"ok": True, "features": rt.snapshot()["features"]})

    # ------------------------------------------------------------------
    # Управление ассистентом
    # ------------------------------------------------------------------
    @app.post("/api/assistant/start")
    def api_assistant_start():
        payload = request.get_json(silent=True) or {}
        started = asst_runner.start(
            stt=payload.get("stt", "kairos"),
            new_chat=bool(payload.get("new_chat")),
            no_clean=bool(payload.get("no_clean")),
        )
        return jsonify({"ok": True, "started": started, "assistant": asst_runner.status()})

    @app.post("/api/assistant/stop")
    def api_assistant_stop():
        stopped = asst_runner.stop()
        return jsonify({"ok": True, "stopped": stopped, "assistant": asst_runner.status()})

    # ------------------------------------------------------------------
    # Триггеры «слово → сетевая команда»
    # ------------------------------------------------------------------
    @app.get("/api/triggers")
    def api_triggers_list():
        return jsonify({"triggers": rt.triggers()})

    @app.post("/api/triggers")
    def api_triggers_add():
        payload = request.get_json(silent=True) or {}
        phrase = (payload.get("phrase") or "").strip()
        command = (payload.get("command") or "").strip()
        if not phrase or not command:
            return jsonify({"ok": False, "error": "нужны «слово» и «команда»"}), 400
        trig = rt.add_trigger(
            phrase=phrase,
            command=command,
            match=payload.get("match", "contains"),
            enabled=bool(payload.get("enabled", True)),
            cooldown_sec=float(payload.get("cooldown_sec") or 5.0),
            comment=payload.get("comment", ""),
        )
        return jsonify({"ok": True, "trigger": trig, "triggers": rt.triggers()})

    @app.put("/api/triggers/<trigger_id>")
    def api_triggers_update(trigger_id: str):
        payload = request.get_json(silent=True) or {}
        fields: Dict[str, Any] = {}
        for key in ("phrase", "command", "match", "comment"):
            if key in payload:
                fields[key] = payload[key]
        if "enabled" in payload:
            fields["enabled"] = bool(payload["enabled"])
        if "cooldown_sec" in payload:
            try:
                fields["cooldown_sec"] = float(payload["cooldown_sec"] or 0)
            except (TypeError, ValueError):
                pass
        ok = rt.update_trigger(trigger_id, **fields)
        return jsonify({"ok": ok, "triggers": rt.triggers()})

    @app.delete("/api/triggers/<trigger_id>")
    def api_triggers_delete(trigger_id: str):
        ok = rt.remove_trigger(trigger_id)
        return jsonify({"ok": ok, "triggers": rt.triggers()})

    @app.post("/api/triggers/<trigger_id>/test")
    def api_triggers_test(trigger_id: str):
        """Прогоняет команду правила без ожидания микрофона."""
        trig = next((t for t in rt.triggers() if t["id"] == trigger_id), None)
        if not trig:
            return jsonify({"ok": False, "error": "правило не найдено"}), 404
        result = rt.run_trigger(trig, text=trig.get("phrase", ""),
                                respect_cooldown=False)
        return jsonify({"ok": True, "result": result})

    # ------------------------------------------------------------------
    # Чаты (контекст диалога)
    # ------------------------------------------------------------------
    @app.get("/api/chats")
    def api_chats_list():
        from valera_chats import ChatStore

        days = float(request.args.get("older_than_days", 1) or 1)
        store = ChatStore()
        try:
            return jsonify({
                "chats": store.list(limit=int(request.args.get("limit", 200) or 200)),
                "stats": store.stats(older_than_days=days),
                "current_chat_id": rt.get_current_chat().get("chat_id"),
            })
        finally:
            store.close()

    @app.post("/api/chats/delete")
    def api_chats_delete():
        """Удаляет выбранные чаты либо все «старые» (кнопка на странице)."""
        from valera_chats import ChatStore

        payload = request.get_json(silent=True) or {}
        store = ChatStore()
        try:
            keep = rt.get_current_chat().get("chat_id")
            if payload.get("older_than_days") is not None:
                deleted = store.delete_older_than(
                    float(payload["older_than_days"]), keep_chat_id=keep
                )
            elif payload.get("all"):
                deleted = store.delete_all(keep_chat_id=keep)
            else:
                deleted = store.delete(payload.get("chat_ids") or [])
            return jsonify({"ok": True, "deleted": deleted, "stats": store.stats()})
        finally:
            store.close()

    # ------------------------------------------------------------------
    # Новый чат вручную
    # ------------------------------------------------------------------
    @app.post("/api/chat/new")
    def api_chat_new():
        """Просит ассистента начать новый чат (сброс контекста)."""
        if asst_runner.assistant is not None:
            try:
                asst_runner.assistant._clear_chat_id_file()
                asst_runner.assistant._chat_id = asst_runner.assistant.rag.create_chat()
                new_id = asst_runner.assistant._chat_id
                if new_id:
                    asst_runner.assistant._save_chat_id(new_id)
                    rt.set_current_chat(new_id, "")
                rt.mark_new_chat_started()
                rt.log_event("info", "Новый чат создан вручную")
                return jsonify({"ok": True, "chat_id": new_id})
            except Exception as exc:  # noqa: BLE001
                return jsonify({"ok": False, "error": str(exc)}), 500
        return jsonify({"ok": False, "error": "ассистент не запущен"}), 409

    return app


# ---------------------------------------------------------------------------
# Вспомогательные: прореживание данных для графиков
# ---------------------------------------------------------------------------
def _downsample_levels(levels, max_points: int = 160):
    """Прореживает историю уровней, чтобы JSON не был огромным."""
    if not levels:
        return []
    step = max(1, len(levels) // max_points)
    out = []
    for ts, mic, gated, out_rms in levels[::step]:
        out.append([round(mic, 5), round(gated, 5), round(out_rms, 5)])
    return out


def _downsample_waves(waves, max_points: int = 200):
    """
    Прореживает буферы живой волны (осциллограф).

    Ограничение small (260 точек): ответ ``/api/state`` должен влезать в
    лимит ~16 КБ окружения (VPN на loopback обрезает крупные ответы).
    900 точек × 2 буфера × 6 знаков давали ~15 КБ одного только JSON —
    вместе со «склейкой» фразы ответ переваливал за лимит и сайт зависал.
    """
    raw, gated = waves
    def _thin(seq):
        if not seq:
            return []
        step = max(1, len(seq) // max_points)
        return [round(float(v), 3) for v in seq[::step]]
    return {"raw": _thin(raw), "gated": _thin(gated)}


def _recent_events(events, limit: int = 60, max_chars: int = 200):
    """
    Последние события триггеров для лога на странице.

    ``limit`` и ``max_chars`` держим маленькими специально: VPN на loopback
    обрезает ответы крупнее ~16 КБ, а ``/api/state`` опрашивается раз в
    секунду. Когда событий было 60 и текст не обрезался, ответ доходил до
    19 КБ — браузер висел в ожидании, и пульт «не грузился».
    """
    out = []
    for ts, level, text in events[-limit:]:
        text = (text or "").strip()
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "…"
        out.append({
            "time": time.strftime("%H:%M:%S", time.localtime(ts)),
            "level": level,
            "text": text,
        })
    return out


# ---------------------------------------------------------------------------
# Отдача HTML
# ---------------------------------------------------------------------------
def _html_response(html: str):
    """
    Возвращает HTML-ответ.

    Сжатие gzip делает общий ``after_request``-хук (см. ``create_app``), поэтому
    здесь достаточно отдать тело с правильным Content-Type. Отдаём именно
    ``Response`` с bytes, чтобы Flask не пытался прогнать страницу через Jinja2:
    в её CSS/JS есть фигурные скобки, которые шаблонизатор принял бы за свои
    директивы и сломал бы страницу.
    """
    from flask import Response

    return Response(html.encode("utf-8"), mimetype="text/html")


def _page_html() -> str:
    """Страница пульта: HTML + стили из web_ui/style.css (читаются при каждом запросе — правки видны сразу)."""
    try:
        with open(os.path.join(PROJECT_DIR, "web_ui", "style.css"), encoding="utf-8") as fh:
            css = fh.read()
    except OSError:
        css = ""
    return INDEX_HTML.replace("/*__STYLE__*/", css)


def _compact_html(html: str) -> str:
    """
    Убирает HTML-комментарии перед отдачей страницы.

    Зачем: в этом окружении VPN на loopback обрывает ответы крупнее ~16 КБ.
    Пульт вместе с встроенной дизайн-системой и JS занимает почти весь лимит,
    а комментарии — это несколько килобайт служебного текста, который нужен
    разработчику, но не браузеру. Сжатие снимает часть объёма, но запас в
    пару сотен байт слишком мал: любая следующая правка снова упёрлась бы в
    лимит и «пульт перестал грузиться».

    Комментарии в CSS/JS не трогаем — вырезаем только ``<!-- ... -->``,
    которых внутри стилей и скриптов нет. Дополнительно чистим
    ``/* ... */`` ВНУТРИ <style>: именно там их больше всего, и они самые
    длинные (пояснения к дизайн-системе).
    """
    import re as _re

    out = _re.sub(r"<!--.*?-->", "", html, flags=_re.S)

    def _strip_css(match):
        inner = _re.sub(r"/\*.*?\*/", "", match.group(1), flags=_re.S)
        # Схлопываем пустые строки, оставшиеся от комментариев.
        inner = _re.sub(r"\n{3,}", "\n\n", inner)
        return f"<style>{inner}</style>"

    return _re.sub(r"<style>(.*?)</style>", _strip_css, out, flags=_re.S)


# ===========================================================================
# Одностраничный интерфейс
# ===========================================================================
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Валера — пульт робота-экскурсовода</title>
<!-- Шрифты берём ЛОКАЛЬНЫЕ (Lato/Open Sans/Ubuntu есть в системе): робот на
     выставке может быть без интернета, а внешний Google Fonts ещё и тормозит
     первую загрузку. Стек с fallback-цепочкой на случай другой машины. -->
<style>/*__STYLE__*/</style>
</head>
<body>
<div class="app">
  <aside class="side">
    <div class="brand">
      <div class="logo">V</div>
      <div><h1>Валера</h1><p>ассистент лаборатории<br>РТУ МИРЭА · Unitree H1</p></div>
    </div>
    <nav class="tabs">
      <button class="tab active" data-pane="main"><span class="ico">🏠</span>Главная</button>
      <button class="tab" data-pane="sound"><span class="ico">🎙</span>Звук</button>
      <button class="tab" data-pane="ai"><span class="ico">🧠</span>ИИ и голос</button>
      <button class="tab" data-pane="devices"><span class="ico">🔌</span>Устройства</button>
      <button class="tab" data-pane="triggers"><span class="ico">⚡</span>Триггеры</button>
      <button class="tab" data-pane="chats"><span class="ico">💬</span>Чаты</button>
      <button class="tab" data-pane="log"><span class="ico">📜</span>Журнал</button>
      <button class="tab" data-pane="system"><span class="ico">🖥</span>Система</button>
    </nav>
    <div class="side-foot">
      <a href="/rules" target="_blank" title="Памятка для гостей на второй экран">📋 Правила для гостей</a>
      <button id="btnTheme" title="Сменить тему">🌓 Тема</button>
    </div>
  </aside>

  <main class="content">
    <header class="top">
      <div class="title"><h1 id="pageTitle">Главная</h1><small id="pageSub">состояние и последний диалог</small></div>
      <span class="pill"><span class="dot" id="dot"></span><span id="stateTxt">остановлен</span></span>
      <span class="pill" id="chatPill" title="Активный чат (контекст диалога)"><span id="chatTxt">чат: —</span></span>
      <button class="pri big" id="btnStart">▶ Запустить</button>
      <button class="dgr big" id="btnStop" disabled>■ Остановить</button>
    </header>

    <!-- ============ ГЛАВНАЯ ============ -->
    <div class="pane active" id="pane-main">
      <div class="hero">
      <div class="hero-ico" id="heroIco">⏹</div>
      <div class="hero-txt">
        <div class="k">Состояние робота</div>
        <div class="v" id="heroState">Ассистент остановлен</div>
        <div class="s" id="heroSub">Нажмите «Запустить», чтобы Валера начал слушать</div>
      </div>
      <div style="display:flex;gap:var(--sp-2);flex-wrap:wrap">
        <div class="metric" style="min-width:120px">
          <div class="k">Время работы</div><div class="v" id="heroUptime">—</div>
        </div>
        <div class="metric" style="min-width:150px">
          <div class="k">Уровень микрофона</div><div class="v" id="heroMic">—</div>
        </div>
      </div>
    </div>
      <div class="grid cols">
        <div class="stack">
          <div class="card">
          <h2>Диалог <span class="tag">последний обмен</span></h2><div class="bubbles">
          <div class="transcript user"><div class="lbl">Услышано</div><div class="txt muted" id="tUser">—</div></div>
          <div class="transcript bot"><div class="lbl">Ответ Валеры</div><div class="txt" id="tBot">—</div></div>
        </div></div>
          <div class="card">
          <h2>Телеметрия</h2>
          <div class="metrics" id="metrics"></div>
        </div>
        </div>
        <div class="stack">
          <div class="card">
          <h2>Функции <span class="tag">применяются на лету</span></h2>
          <div id="features"></div>
        </div>
        </div>
      </div>
    </div>

    <!-- ============ ЗВУК ============ -->
    <div class="pane" id="pane-sound">
      <div class="grid cols">
        <div class="stack"><div class="card">
          <h2>Чувствительность</h2>
          <label class="f">Порог VAD (ниже = чувствительнее)</label>
          <div class="row"><input type="range" id="sThr" min="0.05" max="0.9" step="0.01"><span class="val" id="vThr"></span></div>
          <label class="f">Пауза тишины, сек</label>
          <div class="row"><input type="range" id="sSil" min="0.3" max="3" step="0.05"><span class="val" id="vSil"></span></div>
          <label class="f">Мин. длина фразы, сек</label>
          <div class="row"><input type="range" id="sMin" min="0.2" max="3" step="0.05"><span class="val" id="vMin"></span></div>
        </div></div>
        <div class="stack"><div class="card">
          <h2>Что слышит Валера <span class="tag">микрофон → шумодав → итог</span></h2>
          <canvas id="cPhrase" height="155"></canvas>
          <div class="legend">
            <span><i style="background:var(--mic)"></i>с микрофона</span>
            <span><i style="background:var(--gate)"></i>вырезано шумодавом</span>
            <span><i style="background:var(--out)"></i>итог для Валеры</span>
          </div>
        </div><div class="card">
          <h2>Живой сигнал <span class="tag">2 секунды</span></h2>
          <canvas id="cLive" height="84"></canvas>
        </div><div class="card">
          <h2>Уровень и порог гейта</h2>
          <canvas id="cLevel" height="94"></canvas>
          <div class="legend">
            <span><i style="background:var(--mic)"></i>микрофон (RMS)</span>
            <span><i style="background:var(--out)"></i>после шумодава</span>
            <span><i style="background:var(--gate)"></i>порог гейта</span>
          </div>
        </div></div>
      </div>
    </div>

    <!-- ============ ИИ И ГОЛОС ============ -->
    <div class="pane" id="pane-ai">
      <div class="grid cols-3">
        <div class="card">
          <h2>Модель ИИ <span class="tag" id="modelTag">Ollama</span></h2>

          <!-- Выбор провайдера: локальная (Ollama) ↔ облачная (DeepSeek) -->
          <label class="f">Движок ответов</label>
          <div class="row" style="margin-bottom:var(--sp-2)">
            <select id="selProvider">
              <option value="ollama">Локальная — Ollama (быстро, без интернета)</option>
              <option value="deepseek">DeepSeek — облако (точнее, нужен интернет)</option>
            </select>
            <button class="pri" id="btnSaveProvider">Применить</button>
          </div>
          <small class="muted">Локальная модель бесплатна и работает без сети, но
            на фактах путается (например, имена). <b>DeepSeek</b> отвечает точнее
            по базе знаний, но требует интернет и стоит копейки за запрос.
            Смена применяется к следующему ответу.</small>

          <hr style="border:none;border-top:1px solid var(--border);margin:var(--sp-3) 0">

          <label class="f">Локальная модель Ollama</label>
          <div class="row">
            <select id="selModel"></select>
            <button class="pri" id="btnSaveModel">Применить</button>
          </div>
          <small class="muted">Быстрая <b>true_valera_fast</b> (3B) отвечает почти
            мгновенно; <b>true_valera</b> (8B) — умнее, но медленнее. Смена
            применится к следующему ответу, перезапуск не нужен.</small>
          <div class="row" style="margin-top:var(--sp-2)">
            <button class="sm" id="btnWarmModel">🔥 Прогреть модель</button>
            <span class="muted" id="modelStatus"></span>
          </div>
        </div>
        <div class="card">
          <h2>Голос озвучки <span class="tag" id="ttsTag">XTTS</span></h2>
          <label class="f">Движок</label>
          <div class="row" style="margin-bottom:var(--sp-2)">
            <select id="selTtsBackend">
              <option value="xtts">XTTS-v2 — живой голос (основной, на видеокарте)</option>
              <option value="silero">Silero — быстрый запасной</option>
            </select>
          </div>
          <label class="f">Голос</label>
          <div class="row">
            <select id="selTtsVoice"></select>
            <button class="pri" id="btnSaveTts">Применить</button>
            <button class="sm" id="btnPreviewTts">▶ Прослушать</button>
          </div>
          <label class="f" style="margin-top:var(--sp-2)">Скорость речи (XTTS):
            <b id="ttsSpeedVal">1.00</b>×</label>
          <input type="range" id="ttsSpeed" min="0.5" max="1.5" step="0.05" value="1.0" style="width:100%">
          <small class="muted" id="ttsInfo"></small>
          <span class="muted" id="ttsStatus"></span>
        </div>
      </div>
    </div>

    <div class="pane" id="pane-devices">
    <div class="grid cols-3">
      <div class="card">
        <h2>Микрофон (ввод)</h2>
        <label class="f">Устройство захвата</label>
        <select id="selIn"></select>
        <small class="muted">Список из <code>arecord -L</code>. Если микрофон не подключён —
          выберите устройство после подключения и нажмите «Обновить».</small>
      </div>
      <div class="card">
        <h2>Колонки (вывод)</h2>
        <label class="f">Устройство воспроизведения</label>
        <select id="selOut"></select>
        <small class="muted">Список из <code>aplay -L</code>. Голос Валеры пойдёт
          именно сюда (например, на DIN-выход к роботу).</small>
      </div>
      <div class="card">
        <h2>Сетевой интерфейс</h2>
        <label class="f">Интерфейс для связи с роботом</label>
        <select id="selNic"></select>
        <small class="muted">Через этот интерфейс уходят команды роботу
          (Unitree H1). Виртуальные интерфейсы скрыты.</small>
      </div>
    </div>
    <div class="card" style="margin-top:var(--sp-3)">
      <div class="row">
        <button class="pri" id="btnSaveDev">Сохранить устройства</button>
        <button id="btnRefreshDev">Обновить списки</button>
        <span class="muted" id="devStatus"></span>
      </div>
    </div>
  </div>
    <div class="pane" id="pane-triggers">
    <div class="card">
      <h2>Правила «слово → сетевая команда»</h2>
      <small class="muted" style="margin-bottom:var(--sp-3)">
        Когда в речи прозвучит ключевое слово — выполняется команда. Подстановки:
        <code>{robot}</code> — адрес API робота, <code>{text}</code> — распознанный текст,
        <code>{phrase}</code> — само слово.
      </small>
      <div class="scroll" style="margin-top:var(--sp-3)"><table>
        <thead><tr><th>Вкл</th><th>Слово</th><th>Сравнение</th><th>Кулдаун</th><th>Команда</th><th></th></tr></thead>
        <tbody id="trigBody"></tbody>
      </table></div>
    </div>

    <div class="card" style="margin-top:var(--sp-3)">
      <h2>Новое / редактирование правила</h2>
      <div class="grid cols-3">
        <div><label class="f">Ключевое слово</label><input type="text" id="tPhrase" placeholder="Привет"></div>
        <div><label class="f">Сравнение</label>
          <select id="tMatch">
            <option value="contains">содержит (по умолчанию)</option>
            <option value="word">целое слово</option>
            <option value="equals">точное совпадение</option>
            <option value="regex">регулярное выражение</option>
          </select></div>
        <div><label class="f">Кулдаун, сек</label><input type="number" id="tCool" value="5" min="0" step="0.5"></div>
      </div>
      <label class="f">Команда (curl)</label>
      <textarea id="tCmd" placeholder="curl -s -X POST {robot}/api/action -d '{&quot;action&quot;:&quot;wave_hand&quot;}'"></textarea>
      <div class="row" style="margin-top:var(--sp-3)">
        <button class="pri" id="btnAddTrg">＋ Добавить</button>
        <button id="btnUpdTrg">Обновить выбранное</button>
        <button class="dgr" id="btnDelTrg">Удалить</button>
        <button id="btnTestTrg">▶ Проверить команду</button>
        <input type="hidden" id="tId">
      </div>
      <p class="muted" style="margin-top:var(--sp-2)" id="trgMsg"></p>
    </div>
  </div>
    <div class="pane" id="pane-chats">
    <div class="grid cols-3" style="margin-bottom:var(--sp-3)">
      <div class="card"><h2>Всего чатов</h2><div class="metric"><div class="k">в базе</div><div class="v" id="stChats">—</div></div></div>
      <div class="card"><h2>Сообщений</h2><div class="metric"><div class="k">в истории</div><div class="v" id="stMsgs">—</div></div></div>
      <div class="card"><h2>Старых</h2><div class="metric"><div class="k" id="stOldK">старше суток</div><div class="v" id="stOld">—</div></div></div>
    </div>

    <div class="card" style="margin-bottom:var(--sp-3)">
      <h2>Авто-новый чат по тишине</h2>
      <label class="sw">
        <input type="checkbox" id="fAutoChat">
        <span class="track"></span>
        <span>Начинать новый чат после тишины<span class="hint">Сбрасывает контекст, когда гость давно ушёл</span></span>
      </label>
      <div class="grid cols-3" style="margin-top:var(--sp-3)">
        <div>
          <label class="f">Тишина до нового чата, сек</label>
          <input type="number" id="autoChatSec" min="0" step="10" value="0">
        </div>
        <div>
          <label class="f">Считать «старыми» старше, дней</label>
          <input type="number" id="oldDays" min="0.1" step="0.5" value="1">
        </div>
        <div style="display:flex;align-items:flex-end">
          <button class="pri" id="btnSaveChatCfg">Сохранить</button>
        </div>
      </div>
      <small class="muted">0 секунд = авто-новый чат выключен.
        Рекомендуется 60 секунд для экскурсий.</small>
    </div>

    <div class="card">
      <h2>Чаты WebRAgent <span class="tag" id="mongoState"></span></h2>
      <div class="row" style="margin-bottom:var(--sp-3)">
        <button class="pri" id="btnNewChat">＋ Новый чат сейчас</button>
        <button class="dgr" id="btnDelOld">Удалить старые</button>
        <button class="dgr" id="btnDelAll">Удалить все</button>
        <button id="btnDelSel">Удалить выбранные</button>
        <button id="btnRefreshChats">Обновить</button>
      </div>
      <div class="scroll"><table>
        <thead><tr><th style="width:44px"></th><th>Название</th><th>Сообщений</th><th>Возраст</th><th>Обновлён</th></tr></thead>
        <tbody id="chatBody"></tbody>
      </table></div>
      <p class="muted" style="margin-top:var(--sp-2)" id="chatMsg"></p>
    </div>
  </div>
    <div class="pane" id="pane-log">
    <div class="card">
      <h2>Журнал событий <span class="tag" id="logTag">последние 60</span></h2>
      <div class="log" id="logBox"></div>
      <div class="row" style="margin-top:var(--sp-3)">
        <button class="sm" id="btnClearLog">Очистить вид</button>
      </div>
    </div>
  </div>

    <!-- ============ СИСТЕМА ============ -->
    <div class="pane" id="pane-system">
      <div class="grid cols">
        <div class="stack">
          <div class="card"><h2>Сервисы <span class="tag" id="sysUpd">—</span></h2><div class="stack" id="svcList"></div></div>
          <div class="card"><h2>Процессор и память</h2><div class="metrics" id="sysMetrics"></div></div>
        </div>
        <div class="stack">
          <div class="card"><h2>Видеокарты <span class="tag" id="gpuTag"></span></h2><div class="stack" id="gpuList"></div></div>
        </div>
      </div>
    </div>
  </main>
</div>

<script>
const $ = id => document.getElementById(id);
const api = async (url, opts) => {
  // Таймаут на запрос: если сервер «затыкается», интерфейс не должен висеть
  // вечно — отдаём управление и пробуем на следующем тике.
  const ctrl = new AbortController();
  const timer = setTimeout(()=>ctrl.abort(), 4000);
  try{
    const r = await fetch(url, Object.assign(
      {headers:{'Content-Type':'application/json'}, signal: ctrl.signal}, opts));
    return await r.json();
  } finally { clearTimeout(timer); }
};

// ---------- тумблеры функций ----------
const FEATURES = [
  ['noise_reduction','Шумодав','Спектральное вычитание + энергетический гейт'],
  ['auto_vad','Авто-чувствительность','Порог VAD подстраивается под шум'],
  ['auto_new_chat','Авто-новый чат','Новый чат после длительной тишины'],
  ['react_to_backchannel','Реакция на «угу»','Отвечать на поддакивания (по умолчанию — молчать)'],
  ['web_search','Веб-поиск','Искать ответ в интернете'],
  ['mirea_collection','База МИРЭА и Студсоюза','Искать в коллекции вуза (МИРЭА, Студсоюз, профком…)'],
  ['confirm_before_answer','Подтверждение вопроса','Спрашивать перед отправкой в ИИ'],
  ['barge_in','Barge-in','Прерывать озвучку, если заговорили'],
  ['tts','Озвучка (TTS)','Произносить ответы голосом'],
  ['rephrase_answers','Пересказ через LLM','Живой текст вместо ссылок'],
  ['triggers_enabled','Триггеры','Выполнять «слово → команда»'],
];
let features = {}, charts = {};
// Флаги, которые пользователь только что переключил вручную. Пока сервер не
// подтвердит новое значение, мы не перетираем галочку из опроса /api/state —
// иначе чекбокс «мигает» и кажется, что настройка не применяется.
const pendingFeatures = {};

function buildFeatureToggles(){
  $('features').innerHTML = FEATURES.map(([k,t,h]) => `
    <label class="sw"><input type="checkbox" data-f="${k}">
      <span class="track"></span>
      <span>${t}<span class="hint">${h}</span></span></label>`).join('');
  document.querySelectorAll('[data-f]').forEach(cb => {
    cb.onchange = async () => {
      const want = cb.checked;
      pendingFeatures[cb.dataset.f] = want;
      try{
        await api('/api/feature', {method:'POST',
          body: JSON.stringify({name: cb.dataset.f, value: want})});
      }catch(e){
        cb.checked = !want;              // откат при ошибке сети
        delete pendingFeatures[cb.dataset.f];
      }
    };
  });
}
function syncFeatures(f){
  features = f;
  document.querySelectorAll('[data-f]').forEach(cb => {
    const k = cb.dataset.f;
    if(k in pendingFeatures){
      // Ждём, пока сервер вернёт ровно то, что просили — тогда снимаем блок.
      if(!!f[k] === pendingFeatures[k]) delete pendingFeatures[k];
      else return;
    }
    cb.checked = !!f[k];
  });
  $('fAutoChat').checked = !!f.auto_new_chat;
}

// ---------- графики на canvas ----------
function fitCanvas(cv){
  const r = cv.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  cv.width = Math.max(1, r.width * dpr);
  cv.height = Math.max(1, (cv.getAttribute('height')|0) * dpr);
  const c = cv.getContext('2d');
  c.setTransform(dpr,0,0,dpr,0,0);
  return {c, w: r.width, h: cv.getAttribute('height')|0};
}
function grid(c, w, h, color){
  c.strokeStyle = color || 'rgba(35,45,61,.75)'; c.lineWidth = 1;
  c.beginPath();
  for(let i=0;i<=4;i++){const y=~~(h*i/4)+.5; c.moveTo(0,y); c.lineTo(w,y);}
  c.stroke();
}
function drawSeries(c, w, h, data, color, maxAbs){
  if(!data || !data.length) return;
  const m = maxAbs || Math.max(0.001, ...data.map(Math.abs));
  c.strokeStyle = color; c.lineWidth = 1.2; c.beginPath();
  data.forEach((v,i)=>{
    const x = w * i / Math.max(1,data.length-1);
    const y = h/2 - (v/m) * (h/2 - 3);
    i ? c.lineTo(x,y) : c.moveTo(x,y);
  });
  c.stroke();
}
function drawPhrase(p){
  const {c,w,h} = fitCanvas($('cPhrase'));
  c.clearRect(0,0,w,h); grid(c,w,h);
  if(!p){ c.fillStyle='#5d6b7f'; c.font='13px Manrope'; c.textAlign='center';
    c.fillText('скажите что-нибудь — появится сравнение', w/2, h/2); return; }
  const m = Math.max(0.001, ...p.raw.map(Math.abs), ...p.removed.map(Math.abs));
  drawSeries(c,w,h,p.raw,'#38bdf8',m);
  drawSeries(c,w,h,p.removed,'#fb7185',m);
  drawSeries(c,w,h,p.clean,'#34d399',m);
  c.fillStyle='#5d6b7f'; c.font='11px JetBrains Mono'; c.textAlign='left';
  c.fillText(`${p.duration.toFixed(1)}с · микрофон ${p.raw_rms.toFixed(3)} → итог ${p.clean_rms.toFixed(3)}`, 7, 13);
}
function drawLive(wv){
  const {c,w,h} = fitCanvas($('cLive'));
  c.clearRect(0,0,w,h); grid(c,w,h);
  if(wv && wv.raw && wv.raw.length){ drawSeries(c,w,h,wv.raw,'#38bdf8',0.25); }
  else { c.fillStyle='#5d6b7f'; c.font='12px Manrope'; c.textAlign='center';
    c.fillText('нет сигнала', w/2, h/2); }
}
function drawLevel(lv, m){
  const {c,w,h} = fitCanvas($('cLevel'));
  c.clearRect(0,0,w,h); grid(c,w,h);
  if(!lv || !lv.length){ c.fillStyle='#5d6b7f'; c.font='12px Manrope'; c.textAlign='center';
    c.fillText('уровни появятся при работе микрофона', w/2, h/2); return; }
  const mic = lv.map(x=>x[0]), gat = lv.map(x=>x[1]);
  const maxV = Math.max(0.02, ...mic, ...gat, (m.gate_threshold||0)*1.3);
  const n = lv.length;
  c.strokeStyle='#38bdf8'; c.lineWidth=1.3; c.beginPath();
  mic.forEach((v,i)=>{const x=w*i/Math.max(1,n-1), y=h-(v/maxV)*(h-4)-2; i?c.lineTo(x,y):c.moveTo(x,y);}); c.stroke();
  c.strokeStyle='#34d399'; c.lineWidth=1.1; c.beginPath();
  gat.forEach((v,i)=>{const x=w*i/Math.max(1,n-1), y=h-(v/maxV)*(h-4)-2; i?c.lineTo(x,y):c.moveTo(x,y);}); c.stroke();
  if(m.gate_threshold > 0){
    const y = h-(m.gate_threshold/maxV)*(h-4)-2;
    c.strokeStyle='#fb7185'; c.setLineDash([5,4]); c.lineWidth=1.1;
    c.beginPath(); c.moveTo(0,y); c.lineTo(w,y); c.stroke(); c.setLineDash([]);
  }
}
function drawMetrics(m){
  const items = [
    ['Уровень микрофона', m.mic_rms], ['Вероятность речи', m.vad_prob],
    ['Порог VAD', m.vad_threshold], ['Шумовой пол', m.noise_rms],
    ['Уровень речи', m.speech_rms], ['Порог гейта', m.gate_threshold],
    ['Усиление (АРУ)', m.gain],
  ];
  $('metrics').innerHTML = items.map(([k,v]) => {
    if(k.startsWith('Усиление')) return `<div class="metric"><div class="k">${k}</div><div class="v">${v==null?'—':(+v).toFixed(2)+'×'}</div></div>`;
    return `<div class="metric"><div class="k">${k}</div><div class="v">${v==null?'—':(+v).toFixed(3)}</div></div>`;
  }).join('');
}

// ---------- поллинг состояния ----------
let chatsLoaded = false;
async function tick(){
  try{
    const s = await api('/api/state');
    syncFeatures(s.features);
    drawPhrase(s.phrase); drawLive(s.waves); drawLevel(s.levels, s.metrics);
    drawMetrics(s.metrics);
    $('tUser').textContent = s.transcript || '—';
    $('tBot').textContent  = s.answer || '—';

    // --- статус в шапке и в hero-блоке ---
    const running = s.assistant.running;
    const hasErr = !!s.assistant.error;
    $('stateTxt').textContent = running ? 'работает' : (hasErr ? 'ошибка' : 'остановлен');
    const d = $('dot');
    d.className = 'dot ' + (running ? 'on' : (hasErr ? 'err' : ''));

    $('heroIco').textContent = running ? '🎙' : (hasErr ? '⚠' : '⏹');
    $('heroState').textContent = running ? 'Валера слушает' : (hasErr ? 'Ошибка запуска' : 'Ассистент остановлен');
    $('heroSub').textContent = running
        ? (s.status || 'Говорите — я слушаю')
        : (hasErr ? s.assistant.error : 'Нажмите «Запустить», чтобы Валера начал слушать');

    // Время работы в читаемом виде
    const up = s.assistant.uptime_sec || 0;
    $('heroUptime').textContent = up > 0
        ? `${Math.floor(up/60)}:${String(Math.floor(up%60)).padStart(2,'0')}`
        : '—';
    const mic = s.metrics.mic_rms;
    $('heroMic').textContent = mic == null ? '—' : (+mic).toFixed(3);

    $('btnStart').disabled = running; $('btnStop').disabled = !running;
    $('chatTxt').textContent = 'чат: ' + (s.current_chat.chat_id ? s.current_chat.chat_id.slice(0,8) : '—');

    // слайдеры (только если не тянем прямо сейчас)
    const cfg = s.config;
    if(!dragging){
      $('sThr').value = cfg.vad.threshold; $('vThr').textContent = (+cfg.vad.threshold).toFixed(2);
      $('sSil').value = cfg.vad.silence_timeout; $('vSil').textContent = (+cfg.vad.silence_timeout).toFixed(2);
      $('sMin').value = cfg.vad.min_speech; $('vMin').textContent = (+cfg.vad.min_speech).toFixed(2);
      $('autoChatSec').value = cfg.rag.auto_new_chat_after_silence_sec;
    }
    // модель ИИ: показываем реальную из конфига; список моделей подгружаем один раз
    if(cfg.llm && cfg.llm.model){
      const cur = cfg.llm.model;
      // Тег в заголовке зависит от активного ДВИЖКА. Раньше здесь всегда
      // подставлялась локальная модель из valera_config.json, и при включённом
      // DeepSeek заголовок продолжал врать («Модель ИИ true_valera:latest»).
      if(activeProvider !== 'deepseek'){
        $('modelTag').textContent = cur;
      }
      const sel = $('selModel');
      if(sel && [...sel.options].some(o=>o.value===cur)) sel.value = cur;
    }
    if(!modelsLoaded) loadModels();
    // журнал
    const lb = $('logBox');
    lb.innerHTML = s.events.map(e=>`<div class="${e.level}">[${e.time}] ${escapeHtml(e.text)}</div>`).join('');
    lb.scrollTop = lb.scrollHeight;
  }catch(e){ /* сервер перезапускается — просто пропускаем кадр */ }
}
function escapeHtml(t){return (t||'').replace(/[&<>"]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}

// ---------- слайдеры ----------
let dragging = false;
['sThr','sSil','sMin'].forEach(id => {
  const el = $(id);
  const fmt = id==='sThr' ? 'vThr' : id==='sSil' ? 'vSil' : 'vMin';
  const key = id==='sThr' ? 'vad.threshold'
            : id==='sSil' ? 'vad.silence_timeout' : 'vad.min_speech';
  el.addEventListener('input', ()=>{ dragging = true; $(fmt).textContent = (+el.value).toFixed(2); });
  el.addEventListener('change', async ()=>{
    // Держим блокировку до подтверждения сервера, иначе следующий tick()
    // вернёт старое значение и ползунок «отскочит» назад.
    try{
      await api('/api/settings', {method:'POST', body: JSON.stringify({[key]: +el.value})});
    }catch(e){ /* сеть — оставим как есть, следующий tick поправит */ }
    dragging = false;
  });
});

// ---------- устройства ----------
async function loadDevices(){
  const d = await api('/api/devices');
  const fill = (sel, list, cur) => {
    sel.innerHTML = list.map(o =>
      `<option value="${o.name}" ${o.name===cur?'selected':''}>${escapeHtml(o.description||o.name)} — ${o.name}</option>`).join('');
    if(cur && !list.some(o=>o.name===cur)) sel.insertAdjacentHTML('afterbegin',
      `<option value="${cur}" selected>${cur} (текущее, не найдено в списке)</option>`);
  };
  fill($('selIn'), d.inputs, d.selected.input_device);
  fill($('selOut'), d.outputs, d.selected.output_device);
  $('selNic').innerHTML = d.interfaces.map(i =>
    `<option value="${i.name}" ${i.name===d.selected.interface?'selected':''}>${escapeHtml(i.description||i.name)}</option>`).join('');
}
$('btnSaveDev').onclick = async () => {
  await api('/api/settings', {method:'POST', body: JSON.stringify({
    'audio.input_device': $('selIn').value,
    'audio.output_device': $('selOut').value,
    'network.interface': $('selNic').value,
  })});
  $('devStatus').textContent = '✓ сохранено';
  setTimeout(()=>$('devStatus').textContent='', 2500);
};
$('btnRefreshDev').onclick = loadDevices;

// ---------- модель ИИ ----------
let modelsLoaded = false;
// Активный движок ответов ('ollama' | 'deepseek'). Хранится здесь, чтобы
// циклический опрос /api/state не перетирал заголовок локальной моделью.
let activeProvider = 'ollama';

// Выбор движка ответов: локальная Ollama ↔ облачный DeepSeek.
// Провайдер живёт в WebRAgent (он формирует ответ), поэтому пульт просто
// переключает его через прокси-эндпоинт /api/llm-provider.
async function loadProvider(){
  const d = await api('/api/llm-provider');
  const cur = (d.provider || 'ollama').toLowerCase();
  activeProvider = cur;
  $('selProvider').value = cur;
  $('modelTag').textContent = cur === 'deepseek'
    ? ('DeepSeek · ' + (d.model || 'deepseek-chat'))
    : ('Ollama · ' + (d.model || ''));

  // Если ключа DeepSeek нет — честно говорим об этом прямо в списке.
  const opt = $('selProvider').querySelector('option[value="deepseek"]');
  if(opt){
    const ok = d.available && d.available.deepseek;
    opt.disabled = !ok;
    opt.textContent = ok
      ? 'DeepSeek — облако (точнее, нужен интернет)'
      : 'DeepSeek — нет ключа DEEPSEEK_API_KEY';
  }
  if(d.error){
    $('modelStatus').textContent = '⚠ ' + d.error;
    setTimeout(()=>$('modelStatus').textContent='', 4000);
  }
}

$('btnSaveProvider').onclick = async () => {
  const provider = $('selProvider').value;
  $('modelStatus').textContent = '⏳ переключаю…';
  const r = await api('/api/llm-provider', {
    method:'POST', body: JSON.stringify({provider: provider})
  });
  if(r.ok){
    $('modelStatus').textContent = '✓ движок: ' + provider;
    await loadProvider();
  } else {
    $('modelStatus').textContent = '✗ ' + (r.error || 'не удалось');
  }
  setTimeout(()=>{ if($('modelStatus').textContent.startsWith('✓')) $('modelStatus').textContent=''; }, 3500);
};

async function loadModels(){
  const d = await api('/api/models');
  const list = d.models || [];
  $('selModel').innerHTML = list.map(m =>
    `<option value="${m.name}" ${m.name===d.current?'selected':''}>${escapeHtml(m.name)} — ${m.size_gb} ГБ</option>`).join('')
    || `<option value="${d.current}">${escapeHtml(d.current)} (список недоступен)</option>`;
  if(!list.some(m=>m.name===d.current)){
    $('selModel').insertAdjacentHTML('afterbegin',
      `<option value="${d.current}" selected>${escapeHtml(d.current)} (текущая)</option>`);
  }
  modelsLoaded = true;
  // Тег «Ollama» в заголовке теперь показывает реальный провайдер.
  loadProvider();
}
$('btnSaveModel').onclick = async () => {
  await api('/api/settings', {method:'POST', body: JSON.stringify({'llm.model': $('selModel').value})});
  $('modelStatus').textContent = '✓ модель: ' + $('selModel').value;
  $('modelTag').textContent = $('selModel').value;
  setTimeout(()=>$('modelStatus').textContent='', 3500);
};
$('btnWarmModel').onclick = async () => {
  $('modelStatus').textContent = '⏳ прогреваю…';
  const r = await api('/api/models/warm', {method:'POST', body:'{}'});
  $('modelStatus').textContent = r.ok ? ('✓ готова: ' + r.model) : ('✗ ' + (r.error||''));
};

// ---------- голос озвучки ----------
let ttsInfo = null;
function fillTtsVoices(){
  if(!ttsInfo) return;
  const be = $('selTtsBackend').value;
  const list = be==='xtts' ? ttsInfo.xtts_speakers : ttsInfo.silero_speakers;
  const cur  = be==='xtts' ? ttsInfo.xtts_speaker  : ttsInfo.silero_speaker;
  let opts = list.map(v => `<option value="${escapeHtml(v)}" ${v===cur?'selected':''}>${escapeHtml(v)}</option>`).join('');
  if(cur && !list.includes(cur)) opts = `<option value="${escapeHtml(cur)}" selected>${escapeHtml(cur)} (текущий)</option>` + opts;
  $('selTtsVoice').innerHTML = opts || '<option value="">нет голосов</option>';
  $('btnPreviewTts').disabled = (be!=='xtts' || !ttsInfo.xtts_up);
  $('ttsInfo').textContent = be==='xtts'
    ? (ttsInfo.xtts_up ? 'Сервис XTTS работает. Если он упадёт, Валера сам заговорит голосом Silero.'
                       : 'Сервис XTTS не запущен — пока говорит Silero. Запустите ./run_all.sh web.')
    : 'Говорит Silero (CPU). XTTS не используется, скорость на Silero не действует.';
}
async function loadTts(){
  ttsInfo = await api('/api/tts');
  $('selTtsBackend').value = ttsInfo.backend;
  $('ttsTag').textContent = ttsInfo.backend==='xtts' ? (ttsInfo.xtts_up ? 'XTTS' : 'XTTS · не запущен') : 'Silero';
  fillTtsVoices();
  if(!ttsDragging){
    $('ttsSpeed').value = ttsInfo.speed; $('ttsSpeedVal').textContent = (+ttsInfo.speed).toFixed(2);
  }
}
let ttsDragging = false;
$('ttsSpeed').addEventListener('input', () => { ttsDragging = true; $('ttsSpeedVal').textContent = (+$('ttsSpeed').value).toFixed(2); });
$('ttsSpeed').addEventListener('change', async () => {
  try{ await api('/api/settings', {method:'POST', body: JSON.stringify({'tts.speed': +$('ttsSpeed').value})}); }catch(e){}
  ttsDragging = false;
  $('ttsStatus').textContent = '✓ скорость ' + (+$('ttsSpeed').value).toFixed(2) + '×';
  setTimeout(()=>$('ttsStatus').textContent='', 2500);
});
$('selTtsBackend').onchange = fillTtsVoices;
$('btnSaveTts').onclick = async () => {
  const be = $('selTtsBackend').value, v = $('selTtsVoice').value;
  await api('/api/settings', {method:'POST', body: JSON.stringify(
    be==='xtts' ? {'tts.backend':'xtts','tts.xtts_speaker':v} : {'tts.backend':'silero','tts.speaker':v})});
  $('ttsStatus').textContent = '✓ применено: ' + v;
  setTimeout(()=>$('ttsStatus').textContent='', 3000);
  loadTts();
};
$('btnPreviewTts').onclick = async () => {
  $('ttsStatus').textContent = '⏳ синтез…';
  try{
    const r = await fetch('/api/tts/preview', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({speaker: $('selTtsVoice').value, speed: +$('ttsSpeed').value})});
    if(!r.ok) throw new Error((await r.json()).error || r.status);
    const url = URL.createObjectURL(await r.blob());
    const a = new Audio(url); a.onended = () => URL.revokeObjectURL(url); await a.play();
    $('ttsStatus').textContent = '';
  }catch(e){ $('ttsStatus').textContent = '✗ ' + e.message; }
};

// ---------- триггеры ----------
let triggers = [];
async function loadTriggers(){
  triggers = (await api('/api/triggers')).triggers;
  $('trigBody').innerHTML = triggers.map(t => `
    <tr><td>${t.enabled?'✅':'⛔'}</td><td><b>${escapeHtml(t.phrase)}</b></td>
    <td class="mono">${t.match}</td><td class="mono">${t.cooldown_sec}s</td>
    <td class="mono" style="max-width:420px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escapeHtml(t.command)}</td>
    <td><button class="sm" data-sel="${t.id}">выбрать</button>
        <button class="sm" data-tgl="${t.id}">${t.enabled?'⏸':'▶'}</button></td></tr>`).join('')
    || `<tr><td colspan="6" class="muted">Правил пока нет</td></tr>`;
  document.querySelectorAll('[data-sel]').forEach(b => b.onclick = () => selectTrigger(b.dataset.sel));
  document.querySelectorAll('[data-tgl]').forEach(b => b.onclick = async () => {
    const t = triggers.find(x=>x.id===b.dataset.tgl);
    await api('/api/triggers/'+t.id, {method:'PUT', body: JSON.stringify({enabled: !t.enabled})});
    loadTriggers();
  });
}
function selectTrigger(id){
  const t = triggers.find(x=>x.id===id); if(!t) return;
  $('tId').value=t.id; $('tPhrase').value=t.phrase; $('tMatch').value=t.match;
  $('tCool').value=t.cooldown_sec; $('tCmd').value=t.command;
}
$('btnAddTrg').onclick = async () => {
  const r = await api('/api/triggers', {method:'POST', body: JSON.stringify({
    phrase: $('tPhrase').value, command: $('tCmd').value, match: $('tMatch').value,
    cooldown_sec: +$('tCool').value })});
  $('trgMsg').textContent = r.ok ? '✓ правило добавлено' : ('✗ ' + r.error);
  loadTriggers();
};
$('btnUpdTrg').onclick = async () => {
  const id = $('tId').value; if(!id) return $('trgMsg').textContent='выберите правило';
  await api('/api/triggers/'+id, {method:'PUT', body: JSON.stringify({
    phrase: $('tPhrase').value, command: $('tCmd').value,
    match: $('tMatch').value, cooldown_sec: +$('tCool').value })});
  $('trgMsg').textContent = '✓ обновлено'; loadTriggers();
};
$('btnDelTrg').onclick = async () => {
  const id = $('tId').value; if(!id) return;
  if(!confirm('Удалить правило?')) return;
  await api('/api/triggers/'+id, {method:'DELETE'});
  $('tId').value=''; $('tPhrase').value=''; $('tCmd').value='';
  $('trgMsg').textContent = '✓ удалено'; loadTriggers();
};
$('btnTestTrg').onclick = async () => {
  const id = $('tId').value; if(!id) return $('trgMsg').textContent='выберите правило';
  $('trgMsg').textContent = '⏳ выполняю…';
  const r = await api('/api/triggers/'+id+'/test', {method:'POST'});
  const res = r.result || {};
  $('trgMsg').textContent = res.ok ? ('✓ выполнено: ' + (res.stdout||'').trim().slice(0,160))
                                   : ('✗ ' + (res.stderr || 'ошибка'));
};

// ---------- чаты ----------
async function loadChats(){
  const days = +($('oldDays').value || 1);
  const d = await api('/api/chats?older_than_days='+days);
  const st = d.stats;
  $('mongoState').textContent = st.available ? 'MongoDB подключена' : 'MongoDB недоступна';
  $('stChats').textContent = st.total_chats; $('stMsgs').textContent = st.total_messages;
  $('stOld').textContent = st.old_chats; $('stOldK').textContent = `старше ${st.older_than_days} дн.`;
  $('chatBody').innerHTML = d.chats.map(c => `
    <tr><td><input type="checkbox" class="ck" value="${c.chat_id}"
        ${c.chat_id===d.current_chat_id?'checked disabled':''}></td>
      <td>${escapeHtml(c.title)} ${c.chat_id===d.current_chat_id?'<span class="badge b-ok">текущий</span>':''}</td>
      <td class="mono">${c.message_count}</td>
      <td class="mono">${c.age_days==null?'—':c.age_days+' дн.'}</td>
      <td class="mono">${(c.updated_at||'').replace('T',' ').slice(0,16)}</td></tr>`).join('')
    || `<tr><td colspan="5" class="muted">Чатов нет</td></tr>`;
  chatsLoaded = true;
}
$('btnRefreshChats').onclick = loadChats;
$('btnDelOld').onclick = async () => {
  const days = +($('oldDays').value || 1);
  if(!confirm(`Удалить все чаты старше ${days} дн.? Текущий чат останется.`)) return;
  const r = await api('/api/chats/delete', {method:'POST', body: JSON.stringify({older_than_days: days})});
  $('chatMsg').textContent = `✓ удалено чатов: ${r.deleted}`; loadChats();
};
$('btnDelAll').onclick = async () => {
  if(!confirm('Удалить ВСЕ чаты, кроме текущего? Это очистит всю историю.')) return;
  const r = await api('/api/chats/delete', {method:'POST', body: JSON.stringify({all: true})});
  $('chatMsg').textContent = `✓ удалено чатов: ${r.deleted}`; loadChats();
};
$('btnDelSel').onclick = async () => {
  const ids = [...document.querySelectorAll('.ck:checked')].map(x=>x.value).filter(Boolean);
  if(!ids.length) return $('chatMsg').textContent='ничего не выбрано';
  if(!confirm(`Удалить выбранные чаты (${ids.length})?`)) return;
  const r = await api('/api/chats/delete', {method:'POST', body: JSON.stringify({chat_ids: ids})});
  $('chatMsg').textContent = `✓ удалено чатов: ${r.deleted}`; loadChats();
};
$('btnNewChat').onclick = async () => {
  const r = await api('/api/chat/new', {method:'POST'});
  $('chatMsg').textContent = r.ok ? ('✓ новый чат: ' + (r.chat_id||'').slice(0,8)) : ('✗ ' + (r.error||''));
  loadChats();
};
$('btnSaveChatCfg').onclick = async () => {
  await api('/api/settings', {method:'POST', body: JSON.stringify({
    'rag.auto_new_chat_after_silence_sec': +$('autoChatSec').value,
    'features.auto_new_chat': $('fAutoChat').checked,
  })});
  $('chatMsg').textContent = '✓ настройки сохранены'; loadChats();
};

// ---------- ассистент ----------
$('btnStart').onclick = async () => { await api('/api/assistant/start', {method:'POST', body:'{}'}); };
$('btnStop').onclick  = async () => { await api('/api/assistant/stop',  {method:'POST', body:'{}'}); };
$('btnClearLog').onclick = () => { $('logBox').innerHTML=''; };


// ---------- тема ----------
(function(){
  let t=''; try{ t = localStorage.getItem('valera_theme') || ''; }catch(e){}
  if(t) document.documentElement.dataset.theme = t;
  $('btnTheme').onclick = () => {
    const dark = document.documentElement.dataset.theme==='dark' ||
      (!document.documentElement.dataset.theme && matchMedia('(prefers-color-scheme: dark)').matches);
    const n = dark ? 'light' : 'dark';
    document.documentElement.dataset.theme = n;
    try{ localStorage.setItem('valera_theme', n); }catch(e){}
  };
})();

// ---------- вкладка «Система» ----------
let sysTimer = null;
function bar(pct, hot){ return `<div class="bar ${hot?'hot':''}"><i style="width:${Math.max(2,pct)}%"></i></div>`; }
async function loadSystem(){
  try{
    const ctrl = new AbortController(); const tm = setTimeout(()=>ctrl.abort(), 9000);
    const d = await (await fetch('/api/system', {signal: ctrl.signal})).json(); clearTimeout(tm);
    $('sysUpd').textContent = 'обновлено ' + new Date().toLocaleTimeString();
    $('svcList').innerHTML = d.services.map(s =>
      `<div class="svc"><span class="dot ${s.ok?'on':'err'}"></span><div><div class="nm">${escapeHtml(s.name)}</div><div class="sub">${escapeHtml(s.sub)}</div></div></div>`).join('');
    $('gpuTag').textContent = d.gpus.length + ' шт.';
    $('gpuList').innerHTML = d.gpus.map(g => {
      const pct = Math.round(g.used/g.total*100);
      const roles = g.roles.map(r=>`<span class="badge b-ok">${escapeHtml(r)}</span>`).join('');
      return `<div class="gpu"><div class="row"><div class="nm">GPU ${g.index} · ${escapeHtml(g.name)}</div><span class="muted" style="text-align:right">${g.temp} °C · загрузка ${g.util}%</span></div>
        <div class="muted">${(g.used/1024).toFixed(1)} / ${(g.total/1024).toFixed(1)} ГБ видеопамяти</div>${bar(pct, pct>90)}
        <div class="roles">${roles}</div></div>`;
    }).join('') || '<p class="muted">nvidia-smi не нашёл видеокарт</p>';
    $('sysMetrics').innerHTML =
      `<div class="metric"><div class="k">Процессор (${d.cpu.cores} потоков)</div><div class="v">${d.cpu.pct}%</div>${bar(d.cpu.pct, d.cpu.pct>85)}</div>` +
      `<div class="metric"><div class="k">Память</div><div class="v">${d.mem.used_gb} / ${d.mem.total_gb} ГБ</div>${bar(Math.round(d.mem.used_gb/Math.max(d.mem.total_gb,1)*100), d.mem.used_gb/Math.max(d.mem.total_gb,1)>0.9)}</div>`;
  }catch(e){ /* сеть — следующая попытка через 3 с */ }
}
const PAGE_TITLES = {main:['Главная','состояние и последний диалог'], sound:['Звук','чувствительность и графики микрофона'], ai:['ИИ и голос','модель, голос и скорость речи'], devices:['Устройства','микрофон, колонки, сеть'], triggers:['Триггеры','слово → команда роботу'], chats:['Чаты','контекст и история'], log:['Журнал','события в реальном времени'], system:['Система','сервисы, видеокарты, процессор']};

// переключение вкладок
function openPane(name){
  document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active', x.dataset.pane===name));
  document.querySelectorAll('.pane').forEach(x=>x.classList.toggle('active', x.id==='pane-'+name));
  const ti = PAGE_TITLES[name] || [name,''];
  $('pageTitle').textContent = ti[0]; $('pageSub').textContent = ti[1];
  try{ location.hash = name; }catch(e){}
  clearInterval(sysTimer);
  if(name==='system'){ loadSystem(); sysTimer = setInterval(loadSystem, 3000); }
  if(name==='chats' && !chatsLoaded) loadChats();
}
document.querySelectorAll('.tab').forEach(t => t.onclick = () => openPane(t.dataset.pane));
if(PAGE_TITLES[location.hash.slice(1)]) openPane(location.hash.slice(1));

// старт
buildFeatureToggles();
loadDevices(); loadTriggers(); loadTts();
const od = $('oldDays'); od.addEventListener('change', loadChats);
tick(); setInterval(tick, 1000);
window.addEventListener('resize', ()=>{ /* перерисуется на следующем кадре */ });
</script>
</body>
</html>
"""


RULES_HTML = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Валера — как со мной говорить</title>
<!--
  ПАМЯТКА ДЛЯ ГОСТЕЙ (страница /rules)
  Её ставят на второй экран рядом с роботом, поэтому:
   1) крупный шрифт — читается с 2-3 метров;
   2) минимум текста — три правила и всё;
   3) живой ответ робота + анимация генерации, чтобы было видно,
      что Валера услышал и что он сейчас думает.
  Шрифты локальные (как и на пульте): на выставке может не быть интернета.
-->
<style>
:root{
  --bg:#0a0f16; --surface-1:#111925; --surface-2:#16202e; --surface-3:#1d2937;
  --surface-4:#243244; --line:#26374a; --line-strong:#35495f;
  --fg:#f4f7fb; --fg-2:#a9bacd; --fg-3:#6f8098;
  --accent:#2dd4bf; --accent-strong:#14b8a6; --accent-soft:rgba(45,212,191,.14);
  --violet:#8b7cf6; --violet-soft:rgba(139,124,246,.14);
  --ok:#4ade80; --ok-soft:rgba(74,222,128,.14);
  --warn:#fbbf24; --warn-soft:rgba(251,191,36,.14);
  --err:#fb7185;
  --r-lg:22px; --r-md:15px; --r-sm:10px;
  --sp-1:6px; --sp-2:10px; --sp-3:14px; --sp-4:20px; --sp-5:28px; --sp-6:38px;
  --shadow:0 22px 50px -30px rgba(0,0,0,.95);
  --glow:0 0 0 1px rgba(45,212,191,.22), 0 16px 44px -20px rgba(45,212,191,.5);
  --sans:'Lato','Open Sans','Ubuntu','Noto Sans',system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
  --mono:'Ubuntu Mono','DejaVu Sans Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
}
*{box-sizing:border-box;margin:0;padding:0}
body{
  min-height:100vh;color:var(--fg);font-family:var(--sans);
  font-size:17px;line-height:1.5;letter-spacing:-.012em;
  background:
    radial-gradient(900px 520px at 4% -12%, rgba(45,212,191,.12), transparent 60%),
    radial-gradient(820px 480px at 104% -8%, rgba(139,124,246,.13), transparent 58%),
    radial-gradient(700px 420px at 50% 116%, rgba(45,212,191,.06), transparent 62%),
    var(--bg);
  -webkit-font-smoothing:antialiased;
}
::selection{background:rgba(45,212,191,.3);color:#04191b}
.wrap{max-width:1560px;margin:0 auto;padding:var(--sp-4) var(--sp-4) 48px}

/* ---------- шапка ---------- */
header{
  display:flex;align-items:center;gap:var(--sp-3);flex-wrap:wrap;
  padding-bottom:var(--sp-3);margin-bottom:var(--sp-4);
  border-bottom:1px solid var(--line);
}
.brand{display:flex;align-items:center;gap:var(--sp-3);margin-right:auto}
.logo{
  width:60px;height:60px;border-radius:var(--r-md);flex:0 0 auto;
  position:relative;display:grid;place-items:center;
  font-size:25px;font-weight:800;color:#04191b;
  background:linear-gradient(145deg,var(--accent),var(--violet));
  box-shadow:var(--glow);
}
h1{font-size:30px;font-weight:800;letter-spacing:-.9px;line-height:1.1}
.brand p{color:var(--fg-2);font-size:14px;font-weight:500;margin-top:2px}
.pill{
  display:inline-flex;align-items:center;gap:8px;
  padding:11px 18px;border-radius:100px;text-decoration:none;
  background:var(--surface-2);border:1px solid var(--line);
  font-size:14.5px;font-weight:700;color:var(--fg-2);
  transition:border-color .2s,background .2s;
}
.pill:hover{border-color:var(--line-strong);background:var(--surface-3)}
.dot{width:10px;height:10px;border-radius:50%;background:var(--fg-3);flex:0 0 auto;transition:all .25s}
.dot.on{background:var(--ok);box-shadow:0 0 0 4px var(--ok-soft);animation:breathe 2.2s ease-in-out infinite}
.dot.think{background:var(--warn);box-shadow:0 0 0 4px var(--warn-soft);animation:breathe 1s ease-in-out infinite}
.dot.err{background:var(--err);box-shadow:0 0 0 4px rgba(251,113,133,.18)}
@keyframes breathe{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.45;transform:scale(.8)}}

/* ---------- сетка ---------- */
.grid{display:grid;gap:var(--sp-3);grid-template-columns:minmax(0,1.05fr) minmax(0,1fr);align-items:start}
@media(max-width:1080px){.grid{grid-template-columns:1fr}}
.stack{display:grid;gap:var(--sp-3)}

/* ---------- карточки ---------- */
.card{
  background:linear-gradient(170deg,var(--surface-2),var(--surface-1));
  border:1px solid var(--line);border-radius:var(--r-lg);
  padding:var(--sp-5);box-shadow:var(--shadow);
}
.card h2{
  font-size:13px;font-weight:800;letter-spacing:1.4px;text-transform:uppercase;
  color:var(--fg-2);display:flex;align-items:center;gap:var(--sp-2);
  margin-bottom:var(--sp-4);
}
.card h2::before{
  content:'';width:5px;height:16px;border-radius:3px;flex:0 0 auto;
  background:linear-gradient(180deg,var(--accent),var(--violet));
}

/* ---------- правила ---------- */
.rule{
  display:flex;align-items:center;gap:var(--sp-4);
  padding:var(--sp-4);border-radius:var(--r-md);
  background:var(--surface-1);border:1px solid var(--line);
  transition:border-color .2s,transform .2s,background .2s;
}
.rule:hover{border-color:var(--line-strong);transform:translateX(3px);background:var(--surface-2)}
.rule + .rule{margin-top:var(--sp-2)}
.num{
  width:66px;height:66px;flex:0 0 auto;border-radius:var(--r-sm);
  display:grid;place-items:center;
  font-family:var(--mono);font-size:31px;font-weight:800;color:#04191b;
  background:linear-gradient(145deg,var(--accent),var(--violet));
  box-shadow:0 12px 28px -16px var(--accent);
}
.rule .r-txt{font-size:21px;font-weight:700;line-height:1.35;letter-spacing:-.3px}
.rule .r-sub{font-size:14px;font-weight:500;color:var(--fg-3);margin-top:3px}
.ico{font-size:26px;flex:0 0 auto;opacity:.85}

/* ---------- живой ответ ---------- */
.status-line{
  display:flex;align-items:center;gap:var(--sp-2);flex-wrap:wrap;
  margin-bottom:var(--sp-3);
}
.badge{
  display:inline-flex;align-items:center;gap:8px;
  font-size:12.5px;font-weight:800;letter-spacing:.4px;text-transform:uppercase;
  padding:7px 14px;border-radius:100px;border:1px solid transparent;
  background:var(--surface-3);color:var(--fg-2);
}
.badge.listen{background:var(--ok-soft);color:var(--ok);border-color:rgba(74,222,128,.32)}
.badge.think{background:var(--warn-soft);color:var(--warn);border-color:rgba(251,191,36,.32)}
.badge.speak{background:var(--accent-soft);color:var(--accent);border-color:rgba(45,212,191,.32)}
.badge.idle{background:var(--surface-3);color:var(--fg-3);border-color:var(--line)}
.badge.err{background:rgba(251,113,133,.14);color:var(--err);border-color:rgba(251,113,133,.32)}
.micbar{
  flex:1;min-width:120px;height:7px;border-radius:100px;
  background:var(--surface-4);overflow:hidden;
}
.micbar i{
  display:block;height:100%;width:0%;border-radius:100px;
  background:linear-gradient(90deg,var(--accent),var(--violet));
  transition:width .12s linear;
}

.answer-box{
  position:relative;min-height:220px;
  background:linear-gradient(180deg,#0b1119,#0e1520);
  border:1px solid var(--line);border-radius:var(--r-md);
  padding:var(--sp-4) var(--sp-4) var(--sp-5);
  overflow:hidden;
}
.answer-box.thinking{border-color:rgba(251,191,36,.38);box-shadow:0 0 0 1px rgba(251,191,36,.14)}
.answer-box.speaking{border-color:rgba(45,212,191,.38);box-shadow:0 0 0 1px rgba(45,212,191,.16)}
.answer-lbl{
  font-size:11.5px;font-weight:800;letter-spacing:1.2px;text-transform:uppercase;
  color:var(--fg-3);margin-bottom:var(--sp-2);
}
#answer{
  font-size:27px;font-weight:600;line-height:1.42;letter-spacing:-.5px;
  min-height:78px;word-break:break-word;
}
#answer.placeholder{color:var(--fg-3);font-weight:500;font-size:20px}
.caret{
  display:inline-block;width:3px;height:26px;margin-left:3px;
  vertical-align:-3px;border-radius:2px;background:var(--accent);
  animation:blink .9s steps(1) infinite;
}
@keyframes blink{0%,49%{opacity:1}50%,100%{opacity:0}}

/* ---------- анимация генерации ---------- */
.gen{
  display:none;align-items:center;gap:var(--sp-2);
  margin-top:var(--sp-3);font-size:14px;font-weight:700;color:var(--warn);
}
.gen.on{display:flex}
.dots{display:inline-flex;gap:5px}
.dots i{
  width:9px;height:9px;border-radius:50%;background:var(--warn);
  animation:jump 1.15s ease-in-out infinite;
}
.dots i:nth-child(2){animation-delay:.16s}
.dots i:nth-child(3){animation-delay:.32s}
@keyframes jump{0%,70%,100%{transform:translateY(0);opacity:.45}35%{transform:translateY(-7px);opacity:1}}

/* бегущая полоса «модель печатает ответ» */
.sweep{
  position:absolute;left:0;right:0;bottom:0;height:4px;
  background:rgba(251,191,36,.14);display:none;overflow:hidden;
}
.answer-box.thinking .sweep{display:block}
.sweep::after{
  content:'';position:absolute;top:0;left:-40%;width:40%;height:100%;
  background:linear-gradient(90deg,transparent,var(--warn),var(--accent),transparent);
  animation:run 1.25s linear infinite;
}
@keyframes run{0%{left:-40%}100%{left:100%}}

/* эквалайзер «робот говорит» */
.wave{display:none;align-items:flex-end;gap:3px;height:22px}
.answer-box.speaking .wave{display:inline-flex}
.wave i{
  width:4px;border-radius:3px;background:var(--accent);
  animation:eq .9s ease-in-out infinite;
}
.wave i:nth-child(1){height:8px;animation-delay:0s}
.wave i:nth-child(2){height:16px;animation-delay:.12s}
.wave i:nth-child(3){height:22px;animation-delay:.24s}
.wave i:nth-child(4){height:14px;animation-delay:.36s}
.wave i:nth-child(5){height:9px;animation-delay:.48s}
@keyframes eq{0%,100%{transform:scaleY(.45)}50%{transform:scaleY(1)}}

/* ---------- подсказка ---------- */
.hint{
  margin-top:var(--sp-3);padding:var(--sp-3) var(--sp-4);
  border-radius:var(--r-md);border:1px dashed var(--line-strong);
  background:rgba(139,124,246,.07);
  font-size:14.5px;font-weight:600;color:var(--fg-2);line-height:1.55;
}
.hint b{color:var(--violet)}

footer{
  margin-top:var(--sp-4);padding-top:var(--sp-3);
  border-top:1px solid var(--line);
  display:flex;gap:var(--sp-3);flex-wrap:wrap;align-items:center;
  font-size:13px;color:var(--fg-3);font-weight:600;
}
footer a{color:var(--accent);text-decoration:none;font-weight:700}
footer a:hover{text-decoration:underline}
</style>
</head>
<body>
<div class="wrap">

  <header>
    <div class="brand">
      <div class="logo">V</div>
      <div>
        <h1>Как со мной говорить</h1>
        <p>Валера · робот-экскурсовод РТУ МИРЭА</p>
      </div>
    </div>
    <span class="pill"><span class="dot" id="dot"></span><span id="stateTxt">проверяю…</span></span>
  </header>

  <div class="grid">

    <!-- ================= ПРАВИЛА ================= -->
    <div class="stack">
      <div class="card">
        <h2>Три простых правила</h2>

        <div class="rule">
          <div class="num">1</div>
          <div>
            <div class="r-txt">Говори на расстоянии 10–15 см от микрофона</div>
            <div class="r-sub">Ближе — звук искажается, дальше — Валера не разберёт слова</div>
          </div>
          <div class="ico">🎤</div>
        </div>

        <div class="rule">
          <div class="num">2</div>
          <div>
            <div class="r-txt">Дождись, пока робот ответит</div>
            <div class="r-sub">Он думает 2–4 секунды — не перебивай и не повторяй вопрос сразу</div>
          </div>
          <div class="ico">⏳</div>
        </div>

        <div class="rule">
          <div class="num">3</div>
          <div>
            <div class="r-txt">Задавай вопрос без больших пауз</div>
            <div class="r-sub">Длинная пауза — и Валера решит, что ты уже закончил говорить</div>
          </div>
          <div class="ico">💬</div>
        </div>
      </div>
    </div>

    <!-- ================= ЖИВОЙ ОТВЕТ ================= -->
    <div class="stack">
      <div class="card">
        <h2>Что сейчас происходит</h2>

        <div class="status-line">
          <span class="badge idle" id="badge">ожидание</span>
          <div class="micbar" title="Уровень микрофона"><i id="micFill"></i></div>
        </div>

        <div class="answer-box" id="answerBox">
          <div class="answer-lbl">Ответ Валеры</div>
          <div id="answer" class="placeholder">Поздоровайся — и я отвечу</div>

          <div class="gen" id="gen">
            <span class="dots"><i></i><i></i><i></i></span>
            <span id="genTxt">генерирую ответ…</span>
          </div>

          <div class="wave" id="wave"><i></i><i></i><i></i><i></i><i></i></div>
          <div class="sweep"></div>
        </div>
      </div>
    </div>

  </div>

  <footer>
    <span id="uptime">робот остановлен</span>
    <span>·</span>
    <a href="/" target="_blank">открыть пульт оператора</a>
  </footer>
</div>

<script>
const $ = id => document.getElementById(id);

// ---------------------------------------------------------------------------
// Состояние экрана
// ---------------------------------------------------------------------------
// answerStage: 0 — ответ не начат, 1 — печатается, 2 — показан полностью.
let shownAnswer = '';
let typedChars = 0;
let typeTimer = null;
let currentStatus = '';

// Статус робота → режим экрана.
// Статусы приходят из voice_assistant.py строками («Отвечаю…»,
// «Поиск в базе МИРЭА…»), поэтому определяем режим по подстрокам.
function modeOf(status, running) {
  const s = (status || '').toLowerCase();
  if (s.indexOf('ошибка') >= 0) return 'err';
  if (s.indexOf('голосом') >= 0) return 'speak';
  if (s.indexOf('отвеча') >= 0 || s.indexOf('поиск') >= 0 ||
      s.indexOf('распозна') >= 0 || s.indexOf('загрузка') >= 0 ||
      s.indexOf('агентн') >= 0) return 'think';
  if (!running) return 'idle';
  if (s.indexOf('слушаю') >= 0 || s.indexOf('тишина') >= 0 ||
      s.indexOf('услышал') >= 0) return 'listen';
  return 'idle';
}

const MODE_TEXT = {
  idle:  'ожидание',
  listen:'слушаю тебя',
  think: 'думаю',
  speak: 'отвечаю',
  err:   'ошибка'
};

// ---------------------------------------------------------------------------
// Печать ответа «по буквам» — эффект генерации
// ---------------------------------------------------------------------------
function startTyping(text) {
  if (typeTimer) { clearInterval(typeTimer); typeTimer = null; }
  shownAnswer = text;
  typedChars = 0;
  const el = $('answer');
  el.classList.remove('placeholder');
  el.textContent = '';
  // Курсор «печатающей машинки»: видно, что текст появляется прямо сейчас,
  // а не просто перерисовался экран.
  const caret = document.createElement('span');
  caret.className = 'caret';
  el.appendChild(caret);

  typeTimer = setInterval(() => {
    // 3 символа за кадр: длинный ответ появляется ~2 секунды, и это
    // выглядит как живая печать, а не как мгновенная подмена текста.
    typedChars = Math.min(shownAnswer.length, typedChars + 3);
    el.textContent = shownAnswer.slice(0, typedChars);
    if (typedChars >= shownAnswer.length) {
      clearInterval(typeTimer);
      typeTimer = null;          // курсор убираем — ответ готов
    } else {
      el.appendChild(caret);     // держим курсор в конце текста
    }
  }, 26);
}

// ---------------------------------------------------------------------------
// Отрисовка
// ---------------------------------------------------------------------------
function paint(d) {
  const mode = modeOf(d.status, d.running);
  if (d.status !== currentStatus) currentStatus = d.status || '';

  // Точка и подпись в шапке
  const dot = $('dot');
  dot.className = 'dot' + (mode === 'err' ? ' err'
      : mode === 'idle' ? ''
      : mode === 'think' ? ' think' : ' on');
  $('stateTxt').textContent = d.running ? (MODE_TEXT[mode] || 'работаю')
                                        : 'робот остановлен';
  if (d.error) $('stateTxt').textContent = 'ошибка';

  // Бейдж режима
  const badge = $('badge');
  badge.className = 'badge ' + mode;
  badge.textContent = d.running ? (MODE_TEXT[mode] || 'работаю') : 'выключен';

  // Уровень микрофона. mic — это RMS сигнала (0..1), у обычной речи ~0.15.
  // Корень делает тихие звуки заметными: 0.01 → 22%, 0.05 → 50%, 0.15 → 87%.
  const raw = Math.max(0, d.mic || 0);
  const lvl = Math.min(1, Math.sqrt(raw / 0.2));
  $('micFill').style.width = (lvl * 100).toFixed(0) + '%';

  // Рамка ответа + анимация
  const box = $('answerBox');
  box.classList.toggle('thinking', mode === 'think');
  box.classList.toggle('speaking', mode === 'speak');

  const gen = $('gen');
  if (mode === 'think') {
    gen.classList.add('on');
    const s = (d.status || '').toLowerCase();
    $('genTxt').textContent = s.indexOf('поиск') >= 0
      ? 'ищу в базе знаний…'
      : s.indexOf('распозна') >= 0 ? 'распознаю речь…'
      : 'генерирую ответ…';
  } else {
    gen.classList.remove('on');
  }

  // Ответ: печатаем заново только если он реально изменился.
  const ans = (d.answer || '').trim();
  if (ans && ans !== shownAnswer) {
    startTyping(ans);
  } else if (!ans && !shownAnswer) {
    $('answer').classList.add('placeholder');
  }

  // Время работы
  const up = d.uptime || 0;
  const mm = Math.floor(up / 60), ss = up % 60;
  $('uptime').textContent = d.running
    ? 'робот работает · ' + mm + ' мин ' + String(ss).padStart(2, '0') + ' с'
    : 'робот остановлен';
}

// ---------------------------------------------------------------------------
// Опрос (дважды в секунду — анимация должна быть отзывчивой)
// ---------------------------------------------------------------------------
async function tick() {
  try {
    const r = await fetch('/api/live', { cache: 'no-store' });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    paint(await r.json());
  } catch (e) {
    $('stateTxt').textContent = 'нет связи';
    $('dot').className = 'dot err';
  }
}

tick();
setInterval(tick, 500);
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Веб-пульт робота Валеры (замена настольного GUI)"
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="Адрес прослушивания (0.0.0.0 — доступ с других устройств)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"Порт (по умолчанию {DEFAULT_PORT})")
    parser.add_argument("--stt", default="kairos",
                        choices=["kairos"],
                        help="Модель распознавания речи")
    parser.add_argument("--no-autostart", action="store_true",
                        help="Не запускать ассистента автоматически")
    parser.add_argument("--proxy", default=None, help="HTTP-прокси для запросов")
    # Толерантность к общим флагам проекта (--barge-in и т.п.)
    args, _unknown = parser.parse_known_args()

    if args.proxy:
        os.environ["HTTP_PROXY"] = args.proxy
        os.environ["HTTPS_PROXY"] = args.proxy
        os.environ["ALL_PROXY"] = args.proxy
        os.environ.setdefault("NO_PROXY",
                              "localhost,127.0.0.1,::1,searxng,qdrant,mongodb,app")

    rt = get_runtime()
    runner = AssistantRunner(rt)
    app = create_app(rt, runner)
    app.config["PORT"] = args.port

    print("=" * 64)
    print("  💻  Веб-пульт «Валера»")
    print(f"  Откройте:  http://{'localhost' if args.host in ('127.0.0.1','0.0.0.0') else args.host}:{args.port}")
    if args.host == "0.0.0.0":
        print("  (доступен с других устройств сети — используйте IP этого компьютера)")
    print(f"  STT: {args.stt}")
    print("  Ctrl+C — остановить")
    print("=" * 64)

    if not args.no_autostart:
        # Небольшая задержка, чтобы сервер успел подняться до тяжёлой загрузки моделей
        threading.Timer(1.0, lambda: runner.start(stt=args.stt)).start()

    try:
        app.run(host=args.host, port=args.port, threaded=True, debug=False,
                use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        runner.stop()
        print("\n👋 Сервер остановлен.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
