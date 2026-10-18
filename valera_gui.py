#!/usr/bin/env python3
"""
valera_gui.py
=============
Пульт управления роботом-экскурсоводом «Валера» (Unitree H1).

Что даёт окно
-------------
1. **Тумблеры функций на лету** — шумодав, авто-чувствительность, подтверждение
   вопроса, веб-поиск, поиск по базе МИРЭА, barge-in, озвучка, пересказ ответов.
   Переключаются во время работы: ассистент читает флаги на каждом шаге.

2. **Настройка чувствительности** — либо «авто» (ассистент сам подстраивает
   порог VAD и усиление под шум), либо вручную ползунками (порог VAD, пауза
   тишины, минимальная длина фразы). Рядом — живая телеметрия: шумовой пол,
   уровень речи, текущее усиление, вычисленный порог.

3. **Визуализация звука на одном виджете** — три «дорожки» последней фразы:
   сверху то, что пришло с микрофона; посередине то, что вырезал шумодав
   (шум/тишина); снизу итоговый очищенный сигнал, который ушёл Валере.
   Плюс живой осциллограф и «бегущая» шкала уровня с отметкой порога.

4. **Триггеры «слово → сетевая команда»** — таблица правил: ключевое слово
   (например, «Привет») и `curl`-команда, которая выполняется, когда это слово
   распознано. Кнопка «Проверить» прогоняет команду без ожидания речи.

Запуск
------
    python valera_gui.py                       # с автозапуском ассистента
    python valera_gui.py --stt kairos          # модель распознавания речи
    python valera_gui.py --no-autostart        # только пульт, без микрофона

Требует: тот же conda-env, что и ``voice_assistant.py`` (tkinter, matplotlib).
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

# Обязываем matplotlib рисовать внутри Tk-окна
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from valera_runtime import (
    get_runtime,
    list_audio_inputs,
    list_audio_outputs,
    list_network_interfaces,
)  # noqa: E402

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Тема оформления
# ---------------------------------------------------------------------------
# Светлее и «цветнее», чем чистый чёрный: мягкий графитовый фон, бирюзово-
# фиолетовые акценты, цветные дорожки звука. Так окно не выглядит «мёртвым»,
# и одновременно видно, где что нажимать.
BG = "#141b26"            # основной фон окна (графит, не чёрный)
BG_PANEL = "#1c2534"      # фон карточек/панелей
BG_INPUT = "#0f1620"      # фон полей ввода и графиков
BG_HOVER = "#26334a"      # подсветка при наведении
LINE = "#33415a"          # разделители (заметны на фоне)
LINE_SOFT = "#2a3548"
FG = "#f2f6fc"            # основной текст
FG_DIM = "#a5b4c8"        # вторичный текст
FG_MUTE = "#7b8a9e"       # приглушённый текст
ACCENT = "#2dd4bf"        # акцент (бирюза)
ACCENT_2 = "#8b7cf6"      # второй акцент (фиолет) — для градиентов
ACCENT_TEXT = "#0b2a2a"   # тёмный текст на акцентной кнопке
OK = "#4ade80"
WARN = "#fbbf24"
ERR = "#fb7185"
MIC_COLOR = "#38bdf8"     # дорожка «с микрофона»
GATE_COLOR = "#fb7185"    # дорожка «вырезанный шум»
OUT_COLOR = "#4ade80"     # дорожка «итог для Валеры»

# Шрифт интерфейса. Lato мягче и «человечнее» системного DejaVu; если его нет,
# Tk подставит ближайший доступный, интерфейс не сломается.
FONT = "Lato"
FONT_MONO = "Ubuntu Mono"

# Символы, которые точно есть в шрифтах (емодзи Tk/Agg не рисует)
SYM_OK = "✓"
SYM_NO = "✗"
SYM_PLAY = "▶"
SYM_STOP = "■"
SYM_DOT = "●"
SYM_ARROW = "▾"


# ===========================================================================
# Вспомогательные утилиты и прокручиваемый контейнер
# ===========================================================================
def _format_trigger_result(res: dict) -> str:
    """Человекочитаемое описание результата триггера (для GUI и журнала)."""
    if res.get("skipped"):
        return f"Триггер пропущен ({res.get('reason')}): {res.get('command')}"
    if res.get("ok"):
        out = (res.get("stdout") or "").strip()[:300]
        return f"✓ Выполнено: {res.get('command')}\nВывод: {out or '(пусто)'}"
    return (f"✗ Ошибка: {res.get('stderr') or res.get('reason') or 'неизвестно'}\n"
            f"Команда: {res.get('command')}")


class ScrollableFrame(ttk.Frame):
    """
    Фрейм с вертикальной прокруткой.

    Нужен, чтобы интерфейс полностью помещался и на маленьком экране
    (например, 1280×720 или планшет): содержимое вкладки кладётся во
    ``inner``, а появившиеся лишние строки прокручиваются колесом мыши или
    полосой справа. Ширина содержимого подгоняется под ширину окна, поэтому
    горизонтальной прокрутки не возникает.
    """

    def __init__(self, parent, bg=BG, **kw):
        super().__init__(parent, style="Panel.TFrame", **kw)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview,
                                 style="Valera.Vertical.TScrollbar")
        self.canvas.configure(yscrollcommand=self.vsb.set)

        # Внутренний фрейм — сюда пишет весь код построения вкладки
        self.inner = ttk.Frame(self.canvas, style="Panel.TFrame")
        self._window = self.canvas.create_window(
            (0, 0), window=self.inner, anchor="nw"
        )

        # Обновляем область прокрутки при изменении содержимого
        self.inner.bind("<Configure>", self._on_inner_configure)
        # Растягиваем содержимое по ширине видимой области
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self.canvas.pack(side="left", fill="both", expand=True)
        # Полоса прокрутки всегда упакована: её видно и когда прокручивать
        # нечего — так понятно, что вкладку можно листать.
        self.vsb.pack(side="right", fill="y")

        # Колесо мыши работает, только пока курсор над этой вкладкой
        self.canvas.bind("<Enter>", self._bind_wheel)
        self.canvas.bind("<Leave>", self._unbind_wheel)
        self._wheel_bound = False

    def _on_inner_configure(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self._window, width=event.width)

    def _bind_wheel(self, _event=None):
        if not self._wheel_bound:
            # Linux: Button-4/5; Windows/macOS: MouseWheel
            self.canvas.bind_all("<Button-4>", self._on_wheel)
            self.canvas.bind_all("<Button-5>", self._on_wheel)
            self.canvas.bind_all("<MouseWheel>", self._on_wheel)
            self._wheel_bound = True

    def _unbind_wheel(self, _event=None):
        if self._wheel_bound:
            self.canvas.unbind_all("<Button-4>")
            self.canvas.unbind_all("<Button-5>")
            self.canvas.unbind_all("<MouseWheel>")
            self._wheel_bound = False

    def _on_wheel(self, event):
        if event.num == 4:
            delta = -1
        elif event.num == 5:
            delta = 1
        else:
            delta = -1 if event.delta > 0 else 1
        self.canvas.yview_scroll(delta, "units")


# ===========================================================================
# Поток ассистента
# ===========================================================================
class AssistantRunner:
    """
    Запускает ``voice_assistant.VoiceAssistant`` в отдельном потоке и держит
    ссылку на него, чтобы GUI мог стартовать/останавливать «робота».
    """

    def __init__(self, runtime):
        self.runtime = runtime
        self.assistant = None
        self.thread: Optional[threading.Thread] = None
        self.error: Optional[str] = None

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, stt: str = "kairos", new_chat: bool = False, no_clean: bool = False):
        if self.running:
            return
        self.error = None

        def _run():
            try:
                # Импорт внутри потока: он тянет torch/transformers и грузится долго
                from voice_assistant import build_assistant

                self.runtime.set_status("Загрузка моделей…")
                self.assistant = build_assistant(
                    stt=stt, new_chat=new_chat, no_clean=no_clean, runtime=self.runtime
                )
                self.assistant.start()
                while self.running or self.assistant.is_recording:
                    if not self.assistant.is_recording:
                        break
                    time.sleep(0.2)
            except Exception as exc:  # noqa: BLE001
                self.error = str(exc)
                self.runtime.log_event("error", f"Ассистент упал: {exc}")
                self.runtime.set_status("Ошибка")
            finally:
                self.runtime.set_status("Остановлен")

        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()

    def stop(self):
        if self.assistant is not None:
            try:
                self.assistant.stop()
            except Exception:  # noqa: BLE001
                pass
        self.thread = None


# ===========================================================================
# Главное окно
# ===========================================================================
class ValeraGUI:
    def __init__(self, root: tk.Tk, runtime, stt: str = "kairos", autostart: bool = True):
        self.root = root
        self.rt = runtime
        self.stt = stt
        self.runner = AssistantRunner(runtime)
        # Очередь «тяжёлых» обновлений из фонового потока (триггеры, лог)
        self.ui_queue: "queue.Queue[tuple]" = queue.Queue()
        # Адрес API робота — общий для диалогов правил (подставляется как {robot})
        self.var_robot = tk.StringVar()

        root.title("Валера — пульт робота-экскурсовода (РТУ МИРЭА / Unitree H1)")
        root.configure(bg=BG)
        root.geometry("1480x940")
        root.minsize(1180, 800)

        self._build_styles()
        self._build_layout()
        self._load_config_into_widgets()

        # Отрисовка графиков ~10 кадров/сек
        self._tick_graphs()
        self._tick_ui_queue()
        self._tick_status()

        root.protocol("WM_DELETE_WINDOW", self._on_close)
        if autostart:
            self.root.after(700, lambda: self._toggle_assistant(start=True))

    # ------------------------------------------------------------------
    # Стили и компоновка
    # ------------------------------------------------------------------
    def _build_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        f = lambda size, weight="normal": (FONT, size, weight)  # noqa: E731
        fm = lambda size: (FONT_MONO, size)  # noqa: E731

        # --- Полоса прокрутки: заметная, бирюзовый «бегунок» ---
        # В тёмных темах штатный скроллбар clam почти не виден — его легко
        # принять за отсутствие прокрутки. Делаем собственный, контрастный.
        for orient in ("Vertical", "Horizontal"):
            style.configure(
                f"Valera.{orient}.TScrollbar",
                background=BG_HOVER,        # цвет «бегунка»
                troughcolor=BG_INPUT,       # цвет жёлоба
                bordercolor=LINE,
                arrowcolor=ACCENT,
                darkcolor=BG_HOVER,
                lightcolor=BG_HOVER,
                relief="flat",
                width=13 if orient == "Vertical" else 0,
            )
            style.map(
                f"Valera.{orient}.TScrollbar",
                background=[("active", ACCENT), ("pressed", ACCENT)],
                arrowcolor=[("active", ACCENT_TEXT)],
            )

        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=BG_PANEL)
        style.configure("TLabel", background=BG, foreground=FG, font=f(10))
        style.configure("Panel.TLabel", background=BG_PANEL, foreground=FG, font=f(10))
        style.configure("H1.TLabel", background=BG, foreground=FG, font=f(20, "bold"))
        style.configure("Sub.TLabel", background=BG, foreground=FG_DIM, font=f(10))
        style.configure("Dim.TLabel", background=BG_PANEL, foreground=FG_MUTE, font=f(9))
        style.configure("Metric.TLabel", background=BG_INPUT, foreground=ACCENT, font=fm(13))
        style.configure("MetricKey.TLabel", background=BG_INPUT, foreground=FG_DIM, font=f(9))

        style.configure("TCheckbutton", background=BG_PANEL, foreground=FG, font=f(10))
        style.map("TCheckbutton", background=[("active", BG_PANEL)],
                  foreground=[("active", ACCENT)])

        style.configure("TButton", font=f(10, "bold"), padding=(13, 8),
                        background=BG_PANEL, foreground=FG, borderwidth=1,
                        focuscolor=BG_PANEL, relief="flat", bordercolor=LINE)
        style.map("TButton",
                  background=[("active", BG_HOVER), ("pressed", BG_HOVER),
                              ("disabled", BG_INPUT)],
                  foreground=[("active", ACCENT), ("disabled", FG_MUTE)],
                  bordercolor=[("active", ACCENT)])
        style.configure("Accent.TButton", background=ACCENT, foreground=ACCENT_TEXT,
                        font=f(10, "bold"), borderwidth=0, padding=(16, 9))
        style.map("Accent.TButton",
                  background=[("active", "#5eead4"), ("disabled", BG_INPUT)],
                  foreground=[("disabled", FG_MUTE)])
        style.configure("Stop.TButton", background=ERR, foreground="#2b0009",
                        font=f(10, "bold"), borderwidth=0, padding=(16, 9))
        style.map("Stop.TButton",
                  background=[("active", "#ff9fb0"), ("disabled", BG_INPUT)],
                  foreground=[("disabled", FG_MUTE)])

        style.configure("TScale", background=BG_PANEL, troughcolor=BG_INPUT,
                        bordercolor=LINE)
        style.configure("Treeview", background=BG_INPUT, fieldbackground=BG_INPUT,
                        foreground=FG, rowheight=28, font=f(9), borderwidth=0,
                        relief="flat")
        style.map("Treeview", background=[("selected", "#155e5a")],
                  foreground=[("selected", FG)])
        style.configure("Treeview.Heading", background=BG_PANEL, foreground=FG_DIM,
                        font=f(9, "bold"), borderwidth=0, relief="flat", padding=6)
        style.map("Treeview.Heading", background=[("active", BG_HOVER)],
                  foreground=[("active", ACCENT)])

        style.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(0, 0, 0, 0))
        style.configure("TNotebook.Tab", background=BG, foreground=FG_MUTE,
                        padding=(20, 10), font=f(10, "bold"), borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", BG_PANEL)],
                  foreground=[("selected", ACCENT)],
                  expand=[("selected", (0, 0, 0, 0))])

        style.configure("TLabelframe", background=BG_PANEL, foreground=ACCENT,
                        font=f(10, "bold"), borderwidth=1, relief="solid",
                        bordercolor=LINE)
        style.configure("TLabelframe.Label", background=BG_PANEL, foreground=ACCENT,
                        font=f(10, "bold"))

        style.configure("TEntry", fieldbackground=BG_INPUT, foreground=FG,
                        insertcolor=FG, borderwidth=1, relief="flat",
                        bordercolor=LINE, padding=7)
        style.map("TEntry", bordercolor=[("focus", ACCENT)])
        style.configure("TCombobox", fieldbackground=BG_INPUT, background=BG_INPUT,
                        foreground=FG, arrowcolor=ACCENT, borderwidth=1, padding=6,
                        bordercolor=LINE)
        style.map("TCombobox",
                  fieldbackground=[("readonly", BG_INPUT)],
                  foreground=[("readonly", FG)],
                  bordercolor=[("focus", ACCENT)])

    def _panel(self, parent, **kw):
        return ttk.Frame(parent, style="Panel.TFrame", **kw)

    def _build_layout(self):
        # --- верхняя панель управления ---
        top = self._panel(self.root)
        top.pack(fill="x", padx=14, pady=(14, 8))

        # Бренд: логотип-квадрат + название
        brand = self._panel(top)
        brand.pack(side="left", padx=(2, 22))
        tk.Label(brand, text="V", bg=ACCENT, fg="#04222b",
                 font=(FONT, 17, "bold"), width=2, height=1).pack(side="left")
        titles = self._panel(brand)
        titles.pack(side="left", padx=(10, 0))
        ttk.Label(titles, text="ВАЛЕРА", style="H1.TLabel").pack(anchor="w")
        ttk.Label(titles, text="робот-экскурсовод · РТУ МИРЭА · Unitree H1",
                  style="Sub.TLabel").pack(anchor="w")

        self.btn_start = ttk.Button(top, text=f"{SYM_PLAY}  Запустить",
                                    style="Accent.TButton", command=self._start_assistant)
        self.btn_start.pack(side="left", padx=4, pady=(6, 0))
        self.btn_stop = ttk.Button(top, text=f"{SYM_STOP}  Остановить", style="Stop.TButton",
                                   command=self._stop_assistant, state="disabled")
        self.btn_stop.pack(side="left", padx=4, pady=(6, 0))

        ttk.Label(top, text="STT:", style="Panel.TLabel").pack(side="left", padx=(20, 5), pady=(6, 0))
        self.var_stt = tk.StringVar(value=self.stt)
        ttk.Combobox(top, textvariable=self.var_stt, width=10, state="readonly",
                     values=["kairos"]).pack(side="left", pady=(6, 0))

        self.var_new_chat = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="новый чат", variable=self.var_new_chat).pack(
            side="left", padx=(14, 0), pady=(6, 0))
        self.var_no_clean = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="не чистить", variable=self.var_no_clean).pack(
            side="left", padx=(10, 0), pady=(6, 0))

        # Индикатор состояния + активный чат
        self.lbl_chat = ttk.Label(top, text="чат: —", style="Panel.TLabel",
                                  foreground=FG_DIM)
        self.lbl_chat.pack(side="right", padx=(8, 12), pady=(6, 0))
        self.lbl_state = ttk.Label(top, text=f"{SYM_STOP} остановлен", style="Panel.TLabel",
                                   font=(FONT, 10, "bold"), foreground=WARN)
        self.lbl_state.pack(side="right", padx=8, pady=(6, 0))

        # --- блокнот: аудио / устройства / триггеры / чаты / лог ---
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        self.nb = nb

        # Каждая вкладка — прокручиваемый контейнер: содержимое влезает
        # на любом экране, лишнее прокручивается колесом мыши.
        self.tab_audio = ScrollableFrame(nb)
        self.tab_devices = ScrollableFrame(nb)
        self.tab_trig = ScrollableFrame(nb)
        self.tab_chats = ScrollableFrame(nb)
        self.tab_log = ScrollableFrame(nb)

        # Дальше код строит содержимое внутри tab.inner (см. *_tab методы)
        self.tab_audio_inner = self.tab_audio.inner
        self.tab_devices_inner = self.tab_devices.inner
        self.tab_trig_inner = self.tab_trig.inner
        self.tab_chats_inner = self.tab_chats.inner
        self.tab_log_inner = self.tab_log.inner

        nb.add(self.tab_audio, text="  Звук и чувствительность  ")
        nb.add(self.tab_devices, text="  Устройства и сеть  ")
        nb.add(self.tab_trig, text="  Триггеры «слово → команда»  ")
        nb.add(self.tab_chats, text="  Чаты и контекст  ")
        nb.add(self.tab_log, text="  Журнал  ")
        # При переходе на вкладку чатов — обновляем список
        nb.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        self._build_audio_tab()
        self._build_devices_tab()
        self._build_triggers_tab()
        self._build_chats_tab()
        self._build_log_tab()

    def _on_tab_changed(self, _event=None):
        try:
            if self.nb.index(self.nb.select()) == 3:  # «Чаты и контекст»
                self._refresh_chats()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Вкладка «Устройства и сеть»
    # ------------------------------------------------------------------
    def _build_devices_tab(self):
        outer = self._panel(self.tab_devices_inner)
        outer.pack(fill="both", expand=True, padx=12, pady=12)

        cols = self._panel(outer)
        cols.pack(fill="x")

        def card(title, hint):
            box = ttk.Labelframe(cols, text=title)
            box.pack(side="left", fill="both", expand=True, padx=(0, 10))
            ttk.Label(box, text=hint, style="Dim.TLabel", wraplength=330,
                      justify="left").pack(anchor="w", padx=9, pady=(7, 4))
            return box

        in_box = card("Микрофон (ввод)",
                      "Устройство захвата из `arecord -L`. Смените и нажмите «Сохранить» — "
                      "применится без перезапуска.")
        self.var_dev_in = tk.StringVar()
        self.cmb_dev_in = ttk.Combobox(in_box, textvariable=self.var_dev_in,
                                       state="readonly", width=38)
        self.cmb_dev_in.pack(fill="x", padx=9, pady=(0, 10))

        out_box = card("Колонки (вывод)",
                       "Устройство воспроизведения из `aplay -L`. Сюда пойдёт голос Валеры "
                       "(например, на DIN-выход к роботу).")
        self.var_dev_out = tk.StringVar()
        self.cmb_dev_out = ttk.Combobox(out_box, textvariable=self.var_dev_out,
                                        state="readonly", width=38)
        self.cmb_dev_out.pack(fill="x", padx=9, pady=(0, 10))

        nic_box = card("Сетевой интерфейс",
                       "Через этот интерфейс уходят команды роботу (Unitree H1). "
                       "Виртуальные интерфейсы скрыты.")
        self.var_nic = tk.StringVar()
        self.cmb_nic = ttk.Combobox(nic_box, textvariable=self.var_nic,
                                    state="readonly", width=38)
        self.cmb_nic.pack(fill="x", padx=9, pady=(0, 10))

        actions = self._panel(outer)
        actions.pack(fill="x", pady=(14, 0))
        ttk.Button(actions, text="Сохранить устройства", style="Accent.TButton",
                   command=self._save_devices).pack(side="left")
        ttk.Button(actions, text="Обновить списки",
                   command=self._load_devices).pack(side="left", padx=8)
        self.lbl_dev_status = ttk.Label(actions, text="", style="Panel.TLabel",
                                        foreground=OK)
        self.lbl_dev_status.pack(side="left", padx=12)

        self._load_devices()

    def _load_devices(self):
        """Заполняет выпадающие списки устройств и интерфейсов."""
        try:
            inputs = list_audio_inputs()
            outputs = list_audio_outputs()
            nics = list_network_interfaces()
        except Exception as exc:  # noqa: BLE001
            self.lbl_dev_status.config(text=f"Ошибка чтения устройств: {exc}",
                                       foreground=ERR)
            return

        self._dev_inputs = {f"{d['description'] or d['name']} — {d['name']}": d["name"]
                            for d in inputs}
        self._dev_outputs = {f"{d['description'] or d['name']} — {d['name']}": d["name"]
                             for d in outputs}
        self.cmb_dev_in["values"] = list(self._dev_inputs)
        self.cmb_dev_out["values"] = list(self._dev_outputs)
        self.cmb_nic["values"] = [d["description"] for d in nics]
        self._nic_map = {d["description"]: d["name"] for d in nics}

        cur_in = self.rt.get("audio", "input_device", default="default")
        cur_out = self.rt.get("audio", "output_device", default="default")
        cur_nic = self.rt.get("network", "interface", default="auto")

        def _pick(combo, mapping, current, extra_label):
            for label, name in mapping.items():
                if name == current:
                    combo.set(label)
                    return
            # Текущее устройство не найдено (микрофон отключён) — предупреждаем
            combo.set(f"{extra_label} — {current}")

        _pick(self.cmb_dev_in, self._dev_inputs, cur_in, "не найдено в списке")
        _pick(self.cmb_dev_out, self._dev_outputs, cur_out, "не найдено в списке")
        for label, name in self._nic_map.items():
            if name == cur_nic:
                self.cmb_nic.set(label)
                break

        self.lbl_dev_status.config(
            text=f"найдено: {len(inputs)} микрофонов · {len(outputs)} выходов · "
                 f"{len(nics) - 1} интерфейсов",
            foreground=FG_DIM)

    def _save_devices(self):
        in_dev = self._dev_inputs.get(self.cmb_dev_in.get(), self.cmb_dev_in.get())
        out_dev = self._dev_outputs.get(self.cmb_dev_out.get(), self.cmb_dev_out.get())
        nic = self._nic_map.get(self.cmb_nic.get(), "auto")
        self.rt.set("audio", "input_device", value=in_dev)
        self.rt.set("audio", "output_device", value=out_dev)
        self.rt.set("network", "interface", value=nic)
        self.lbl_dev_status.config(text=f"✓ сохранено: {in_dev} → {out_dev}",
                                   foreground=OK)

    # ------------------------------------------------------------------
    # Вкладка «Звук и чувствительность»
    # ------------------------------------------------------------------
    def _build_audio_tab(self):
        wrap = self._panel(self.tab_audio_inner)
        wrap.pack(fill="both", expand=True, padx=8, pady=8)

        # --- левая колонка: тумблеры и ползунки ---
        left = self._panel(wrap)
        left.pack(side="left", fill="y", padx=(0, 10))

        feat = ttk.Labelframe(left, text="Функции (применяются на лету)")
        feat.pack(fill="x", pady=(0, 8))
        self.flags = {}
        flag_defs = [
            ("noise_reduction", "Шумодав (спектр. + энергетический гейт)"),
            ("auto_vad", "Авто-регулировка чувствительности"),
            ("confirm_before_answer", "Спрашивать подтверждение вопроса"),
            ("web_search", "Веб-поиск (интернет)"),
            ("mirea_collection", "Поиск по базе МИРЭА"),
            ("barge_in", "Barge-in (прерывать озвучку)"),
            ("tts", "Озвучивать ответы (TTS)"),
            ("rephrase_answers", "Пересказ ответов через LLM"),
            ("triggers_enabled", "Триггеры «слово → команда»"),
        ]
        for key, label in flag_defs:
            var = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(feat, text=label, variable=var,
                                 command=lambda k=key: self._on_flag_change(k))
            cb.pack(anchor="w", padx=8, pady=2)
            self.flags[key] = var

        # --- ручная чувствительность ---
        srv = ttk.Labelframe(left, text="Чувствительность (ручной режим)")
        srv.pack(fill="x")
        self.var_threshold = tk.DoubleVar(value=0.30)
        self.var_silence = tk.DoubleVar(value=0.75)
        self.var_min_speech = tk.DoubleVar(value=0.50)
        for label, var, lo, hi, step in [
            ("Порог VAD (ниже = чувствительнее)", self.var_threshold, 0.05, 0.90, 0.01),
            ("Пауза тишины, сек", self.var_silence, 0.3, 3.0, 0.05),
            ("Мин. длина фразы, сек", self.var_min_speech, 0.2, 3.0, 0.05),
        ]:
            ttk.Label(srv, text=label, style="Panel.TLabel").pack(anchor="w", padx=8, pady=(6, 0))
            row = self._panel(srv)
            row.pack(fill="x", padx=8)
            scale = ttk.Scale(row, from_=lo, to=hi, variable=var, orient="horizontal",
                              command=lambda _v, k=label: self._on_slider(k))
            scale.pack(side="left", fill="x", expand=True)
            ttk.Label(row, textvariable=var, style="Dim.TLabel", width=6).pack(side="left")

        # --- телеметрия ---
        tel = ttk.Labelframe(left, text="Телеметрия (живая)")
        tel.pack(fill="x", pady=(8, 0))
        self.tele_vars = {}
        for key, label in [
            ("mic", "Уровень микрофона"),
            ("vad", "Вероятность речи (VAD)"),
            ("thr", "Текущий порог VAD"),
            ("noise", "Шумовой пол"),
            ("speech", "Уровень речи"),
            ("gate", "Порог гейта (шумодав)"),
            ("gain", "Усиление (АРУ)"),
            ("status", "Статус"),
        ]:
            row = self._panel(tel)
            row.pack(fill="x", padx=8, pady=1)
            ttk.Label(row, text=label + ":", style="Dim.TLabel", width=20).pack(side="left")
            var = tk.StringVar(value="—")
            ttk.Label(row, textvariable=var, style="Panel.TLabel").pack(side="left")
            self.tele_vars[key] = var

        # --- правая колонка: графики ---
        right = self._panel(wrap)
        right.pack(side="left", fill="both", expand=True)

        self.fig = Figure(figsize=(10, 8), dpi=96, facecolor=BG)
        self.fig.subplots_adjust(left=0.07, right=0.98, top=0.94, bottom=0.06, hspace=0.55)

        # (1) три дорожки последней фразы — «на одном виджете»
        self.ax_phrase = self.fig.add_subplot(3, 1, 1)
        self.ax_phrase.set_facecolor("#0d1116")
        self.ax_phrase.set_title(
            "Последняя фраза:  МИКРОФОН  -->  вырезанный шумодавом шум  -->  ИТОГ для Валеры",
            color=FG, fontsize=10, loc="left")
        self.ax_phrase.tick_params(colors=FG_DIM, labelsize=7)
        for spine in self.ax_phrase.spines.values():
            spine.set_color("#2a3441")

        # (2) живой осциллограф
        self.ax_live = self.fig.add_subplot(3, 1, 2)
        self.ax_live.set_facecolor("#0d1116")
        self.ax_live.set_title("Живой сигнал с микрофона (2 сек)", color=FG,
                               fontsize=10, loc="left")
        self.ax_live.tick_params(colors=FG_DIM, labelsize=7)
        for spine in self.ax_live.spines.values():
            spine.set_color("#2a3441")

        # (3) бегущая шкала уровня + пороги
        self.ax_level = self.fig.add_subplot(3, 1, 3)
        self.ax_level.set_facecolor("#0d1116")
        self.ax_level.set_title("Уровень микрофона / после шумодава · порог гейта",
                                color=FG, fontsize=10, loc="left")
        self.ax_level.tick_params(colors=FG_DIM, labelsize=7)
        for spine in self.ax_level.spines.values():
            spine.set_color("#2a3441")

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    # Вкладка триггеров
    # ------------------------------------------------------------------
    def _build_triggers_tab(self):
        outer = self._panel(self.tab_trig_inner)
        outer.pack(fill="both", expand=True, padx=10, pady=10)

        head = self._panel(outer)
        head.pack(fill="x", pady=(0, 8))
        ttk.Label(
            head,
            text=("Правило срабатывает, когда распознанная речь содержит ключевое слово.\n"
                  "В команде можно использовать подстановки: {robot} — адрес API робота "
                  "(Unitree H1), {text} — распознанный текст, {phrase} — само ключевое слово."),
            style="Panel.TLabel", justify="left",
        ).pack(side="left", anchor="w")

        ttk.Button(head, text="＋ Новое правило", style="Accent.TButton",
                   command=lambda: self._edit_trigger_dialog(None)).pack(side="right")

        # --- таблица правил ---
        # Двойной клик по строке открывает диалог редактирования этого правила.
        cols = ("enabled", "phrase", "match", "cooldown", "command")
        self.tree = ttk.Treeview(outer, columns=cols, show="headings", height=9,
                                 selectmode="browse")
        for col, title, width, anchor in [
            ("enabled", "Вкл", 50, "center"),
            ("phrase", "Ключевое слово", 160, "w"),
            ("match", "Сравнение", 110, "w"),
            ("cooldown", "Кулдаун, с", 95, "center"),
            ("command", "Сетевая команда (curl)", 700, "w"),
        ]:
            self.tree.heading(col, text=title)
            self.tree.column(col, width=width, anchor=anchor)
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<Double-1>", self._on_trigger_double_click)
        self.tree.bind("<Return>", self._on_trigger_double_click)

        actions = self._panel(outer)
        actions.pack(fill="x", pady=(10, 0))
        ttk.Button(actions, text="✏️  Редактировать",
                   command=self._edit_selected_trigger).pack(side="left")
        ttk.Button(actions, text="⏯  Вкл/выкл",
                   command=self._toggle_selected_trigger).pack(side="left", padx=8)
        ttk.Button(actions, text="▶  Проверить команду",
                   command=self._test_selected_trigger).pack(side="left")
        ttk.Button(actions, text="🗑  Удалить", style="Stop.TButton",
                   command=self._remove_selected_trigger).pack(side="left", padx=8)

        ttk.Label(actions, text="Двойной клик по строке — редактировать",
                  style="Dim.TLabel").pack(side="right")

        self.lbl_trig_result = ttk.Label(outer, text="", style="Panel.TLabel",
                                         foreground=OK, wraplength=1300, justify="left")
        self.lbl_trig_result.pack(anchor="w", pady=(8, 0))

    # ------------------------------------------------------------------
    # Диалог редактирования правила
    # ------------------------------------------------------------------
    def _edit_trigger_dialog(self, trigger_id: Optional[str]):
        """
        Открывает модальное окно редактирования правила.

        ``trigger_id=None`` — создание нового правила. Диалог самодостаточен:
        содержит все поля (слово, сравнение, кулдаун, команда), а также
        адрес API робота для подстановки ``{robot}``.
        """
        trig = None
        if trigger_id:
            trig = next((t for t in self.rt.triggers() if t["id"] == trigger_id), None)
            if trig is None:
                messagebox.showwarning("Валера", "Правило не найдено — возможно, удалено.")
                return

        dlg = tk.Toplevel(self.root)
        dlg.title("Правило «слово → команда»" if trig else "Новое правило")
        dlg.configure(bg=BG)
        dlg.transient(self.root)
        dlg.geometry("760x560")
        dlg.minsize(640, 480)

        # Всё содержимое — в прокручиваемом фрейме (команда curl бывает длинной)
        scroll = ScrollableFrame(dlg)
        scroll.pack(fill="both", expand=True)
        body = ttk.Frame(scroll.inner, style="Panel.TFrame")
        body.pack(fill="both", expand=True, padx=16, pady=16)

        ttk.Label(body, text=("Правило «слово → команда»" if trig else "Новое правило"),
                  style="H1.TLabel").pack(anchor="w")
        ttk.Label(body, text="Когда в речи прозвучит ключевое слово — выполняется команда.",
                  style="Sub.TLabel").pack(anchor="w", pady=(2, 14))

        v_phrase = tk.StringVar(value=trig.get("phrase", "") if trig else "")
        v_match = tk.StringVar(value=trig.get("match", "contains") if trig else "contains")
        v_cool = tk.StringVar(value=str(trig.get("cooldown_sec", 5)) if trig else "5")
        v_cmd = tk.StringVar(value=trig.get("command", "") if trig else "")
        v_enabled = tk.BooleanVar(value=bool(trig.get("enabled", True)) if trig else True)

        ttk.Label(body, text="Ключевое слово", style="Panel.TLabel").pack(anchor="w")
        ttk.Entry(body, textvariable=v_phrase).pack(fill="x", pady=(3, 12))

        line2 = self._panel(body)
        line2.pack(fill="x", pady=(0, 12))
        box_m = self._panel(line2)
        box_m.pack(side="left", fill="x", expand=True, padx=(0, 10))
        ttk.Label(box_m, text="Как сравнивать", style="Panel.TLabel").pack(anchor="w")
        ttk.Combobox(box_m, textvariable=v_match, state="readonly",
                     values=["contains", "word", "equals", "regex"]).pack(fill="x", pady=(3, 0))
        box_c = self._panel(line2)
        box_c.pack(side="left")
        ttk.Label(box_c, text="Кулдаун, сек", style="Panel.TLabel").pack(anchor="w")
        ttk.Entry(box_c, textvariable=v_cool, width=12).pack(pady=(3, 0))

        ttk.Checkbutton(body, text="Правило включено", variable=v_enabled).pack(
            anchor="w", pady=(0, 12))

        ttk.Label(body, text="Команда (curl)", style="Panel.TLabel").pack(anchor="w")
        txt = tk.Text(body, bg=BG_INPUT, fg=FG, insertbackground=FG, height=6,
                      font=(FONT_MONO, 10), relief="flat", wrap="word",
                      highlightbackground=LINE, highlightthickness=1)
        txt.pack(fill="x", pady=(3, 6))
        txt.insert("1.0", v_cmd.get())
        ttk.Label(body, text="Подстановки: {robot} — адрес API робота, "
                             "{text} — распознанный текст, {phrase} — ключевое слово",
                  style="Dim.TLabel").pack(anchor="w", pady=(0, 12))

        self.var_robot = tk.StringVar()
        self.var_robot.set(str(self.rt.get("robot", "base_url", default="")))
        row_rb = self._panel(body)
        row_rb.pack(fill="x", pady=(0, 16))
        ttk.Label(row_rb, text="Адрес API робота ({robot}):", style="Panel.TLabel").pack(side="left")
        ttk.Entry(row_rb, textvariable=self.var_robot).pack(side="left", fill="x",
                                                            expand=True, padx=6)

        msg = ttk.Label(body, text="", style="Panel.TLabel", foreground=OK,
                        wraplength=680, justify="left")
        msg.pack(anchor="w", pady=(0, 8))

        buttons = self._panel(body)
        buttons.pack(fill="x")

        def _collect():
            return {
                "phrase": v_phrase.get().strip(),
                "match": v_match.get(),
                "cooldown_sec": float(v_cool.get() or 0),
                "command": txt.get("1.0", "end").strip(),
                "enabled": bool(v_enabled.get()),
            }

        def _save():
            data = _collect()
            if not data["phrase"] or not data["command"]:
                msg.config(text="Укажите ключевое слово и команду.", foreground=ERR)
                return
            # Сохраняем адрес робота тоже — он используется в командах
            self.rt.set("robot", "base_url", value=self.var_robot.get().strip())
            if trig:
                self.rt.update_trigger(trig["id"], **data)
                self.rt.log_event("ok", f"Правило «{data['phrase']}» обновлено")
            else:
                self.rt.add_trigger(**data)
                self.rt.log_event("ok", f"Правило «{data['phrase']}» добавлено")
            self._refresh_triggers()
            dlg.destroy()

        def _test():
            data = _collect()
            if not data["command"]:
                msg.config(text="Команда пуста.", foreground=ERR)
                return
            # Подставляем адрес робота из поля, чтобы тест был «как в бою»
            self.rt.set("robot", "base_url", value=self.var_robot.get().strip())
            msg.config(text="Выполняю…", foreground=WARN)
            dlg.update_idletasks()

            def _worker():
                res = self.rt.run_trigger(
                    {"id": "dialog-test", "phrase": data["phrase"],
                     "command": data["command"], "cooldown_sec": 0},
                    text=data["phrase"], respect_cooldown=False,
                )
                self.ui_queue.put(("dialog_result", res))

            threading.Thread(target=_worker, daemon=True).start()

        ttk.Button(buttons, text="💾 Сохранить", style="Accent.TButton",
                   command=_save).pack(side="left")
        ttk.Button(buttons, text="▶ Проверить команду",
                   command=_test).pack(side="left", padx=8)
        ttk.Button(buttons, text="Отмена", command=dlg.destroy).pack(side="left")

        dlg.bind("<Escape>", lambda _e: dlg.destroy())
        dlg.grab_set()
        dlg.focus_set()

    # ------------------------------------------------------------------
    # Действия над выбранным правилом
    # ------------------------------------------------------------------
    def _edit_selected_trigger(self):
        tid = self._selected_trigger()
        if not tid:
            messagebox.showinfo("Валера", "Выберите правило в таблице.")
            return
        self._edit_trigger_dialog(tid)

    def _on_trigger_double_click(self, _event=None):
        tid = self._selected_trigger()
        if tid:
            self._edit_trigger_dialog(tid)

    def _toggle_selected_trigger(self):
        tid = self._selected_trigger()
        if not tid:
            return
        trig = next((t for t in self.rt.triggers() if t["id"] == tid), None)
        if not trig:
            return
        self.rt.update_trigger(tid, enabled=not trig.get("enabled", True))
        self._refresh_triggers(tid)

    def _remove_selected_trigger(self):
        tid = self._selected_trigger()
        if not tid:
            return
        if messagebox.askyesno("Валера", "Удалить выбранное правило?"):
            self.rt.remove_trigger(tid)
            self._refresh_triggers()
            self.rt.log_event("info", "Правило удалено")

    def _test_selected_trigger(self):
        tid = self._selected_trigger()
        if not tid:
            messagebox.showinfo("Валера", "Выберите правило в таблице.")
            return
        trig = next((t for t in self.rt.triggers() if t["id"] == tid), None)
        if not trig:
            return
        self.lbl_trig_result.config(text="Выполняю…", foreground=WARN)
        self.root.update_idletasks()

        def _worker():
            res = self.rt.run_trigger(trig, text=trig.get("phrase", ""),
                                      respect_cooldown=False)
            self.ui_queue.put(("trigger_result", res))

        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Вкладка «Чаты и контекст»
    # ------------------------------------------------------------------
    def _build_chats_tab(self):
        outer = self._panel(self.tab_chats_inner)
        outer.pack(fill="both", expand=True, padx=12, pady=12)

        # --- сводка ---
        stats = self._panel(outer)
        stats.pack(fill="x", pady=(0, 12))
        self._stat_vars = {}
        for key, title in [("total_chats", "Всего чатов"),
                           ("total_messages", "Сообщений"),
                           ("old_chats", "Старых")]:
            box = tk.Frame(stats, bg=BG_INPUT, padx=14, pady=10,
                           highlightbackground=LINE, highlightthickness=1)
            box.pack(side="left", padx=(0, 10))
            ttk.Label(box, text=title, style="MetricKey.TLabel").pack(anchor="w")
            var = tk.StringVar(value="—")
            ttk.Label(box, textvariable=var, style="Metric.TLabel").pack(anchor="w")
            self._stat_vars[key] = var

        self.lbl_mongo = ttk.Label(stats, text="", style="Panel.TLabel",
                                   foreground=FG_DIM)
        self.lbl_mongo.pack(side="left", padx=10)

        # --- авто-новый чат по тишине ---
        auto = ttk.Labelframe(outer, text="Авто-новый чат по тишине")
        auto.pack(fill="x", pady=(0, 12))

        self.var_auto_chat = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            auto, text="Начинать новый чат, если гость давно молчит",
            variable=self.var_auto_chat, command=self._on_auto_chat_toggle,
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=10, pady=(8, 2))

        ttk.Label(auto, text="Тишина до нового чата, сек:", style="Panel.TLabel").grid(
            row=1, column=0, sticky="w", padx=10, pady=(4, 10))
        self.var_auto_sec = tk.StringVar(value="0")
        ttk.Entry(auto, textvariable=self.var_auto_sec, width=8).grid(
            row=1, column=1, sticky="w", pady=(4, 10))
        ttk.Button(auto, text="Сохранить", command=self._save_auto_chat).grid(
            row=1, column=2, sticky="w", padx=10, pady=(4, 10))
        ttk.Label(auto, text="0 = выключено · для экскурсий удобно 60 сек",
                  style="Dim.TLabel").grid(row=1, column=3, sticky="w", padx=(4, 10))

        # --- список чатов ---
        actions = self._panel(outer)
        actions.pack(fill="x", pady=(0, 8))
        ttk.Button(actions, text="＋ Новый чат сейчас", style="Accent.TButton",
                   command=self._new_chat_now).pack(side="left")
        ttk.Button(actions, text="🗑 Удалить старые",
                   style="Stop.TButton", command=self._delete_old_chats).pack(
            side="left", padx=8)
        ttk.Button(actions, text="🗑 Удалить все",
                   style="Stop.TButton", command=self._delete_all_chats).pack(side="left")
        ttk.Button(actions, text="🗑 Удалить выбранные",
                   command=self._delete_selected_chats).pack(side="left", padx=8)
        ttk.Button(actions, text="Обновить", command=self._refresh_chats).pack(side="left")

        ttk.Label(actions, text="считать старыми старше:", style="Dim.TLabel").pack(
            side="right", padx=(0, 5))
        self.var_old_days = tk.StringVar(value="1")
        ttk.Entry(actions, textvariable=self.var_old_days, width=6).pack(
            side="right", padx=(0, 6))
        ttk.Label(actions, text="дней", style="Dim.TLabel").pack(side="right", padx=(0, 12))

        cols = ("sel", "title", "count", "age", "updated")
        self.tree_chats = ttk.Treeview(outer, columns=cols, show="headings", height=12)
        for col, title, width, anchor in [
            ("sel", "✓", 40, "center"),
            ("title", "Название", 380, "w"),
            ("count", "Сообщений", 100, "center"),
            ("age", "Возраст", 90, "center"),
            ("updated", "Обновлён", 170, "w"),
        ]:
            self.tree_chats.heading(col, text=title)
            self.tree_chats.column(col, width=width, anchor=anchor)
        self.tree_chats.pack(fill="both", expand=True)
        # Отметка чатов «галочкой» по клику (для удаления выбранных)
        self.tree_chats.bind("<Button-1>", self._on_chat_click)

        self.lbl_chat_msg = ttk.Label(outer, text="", style="Panel.TLabel",
                                      foreground=OK, wraplength=1200, justify="left")
        self.lbl_chat_msg.pack(anchor="w", pady=(8, 0))
        self._chat_checked: set = set()
        self._chats: list = []

    def _refresh_chats(self):
        """Перечитывает список чатов из MongoDB и обновляет таблицу."""
        from valera_chats import ChatStore

        try:
            days = float(self.var_old_days.get() or 1)
        except ValueError:
            days = 1.0

        store = ChatStore()
        try:
            if not store.available:
                self.lbl_mongo.config(text="MongoDB недоступна — запустите стек (./run_all.sh)",
                                      foreground=WARN)
                self._chats = []
                self.tree_chats.delete(*self.tree_chats.get_children())
                return
            self.lbl_mongo.config(text="MongoDB подключена", foreground=OK)
            stats = store.stats(older_than_days=days)
            self._stat_vars["total_chats"].set(str(stats["total_chats"]))
            self._stat_vars["total_messages"].set(str(stats["total_messages"]))
            self._stat_vars["old_chats"].set(str(stats["old_chats"]))
            self._chats = store.list(limit=300)
        finally:
            store.close()

        current = self.rt.get_current_chat().get("chat_id")
        self.tree_chats.delete(*self.tree_chats.get_children())
        for chat in self._chats:
            is_current = chat["chat_id"] == current
            mark = "●" if is_current else ("✓" if chat["chat_id"] in self._chat_checked else "☐")
            title = chat["title"] + ("  (текущий)" if is_current else "")
            self.tree_chats.insert("", "end", iid=chat["chat_id"], values=(
                mark, title, chat["message_count"],
                "—" if chat["age_days"] is None else f"{chat['age_days']:.1f} дн.",
                (chat["updated_at"] or "").replace("T", " ")[:16],
            ))

    def _on_chat_click(self, event):
        """По клику на колонку «✓» отмечает чат для удаления."""
        row = self.tree_chats.identify_row(event.y)
        col = self.tree_chats.identify_column(event.x)
        if not row or col != "#1":
            return
        current = self.rt.get_current_chat().get("chat_id")
        if row == current:
            self.lbl_chat_msg.config(text="Текущий чат удалить нельзя.", foreground=WARN)
            return
        if row in self._chat_checked:
            self._chat_checked.discard(row)
        else:
            self._chat_checked.add(row)
        # Обновляем только отметку в строке
        vals = list(self.tree_chats.item(row, "values"))
        vals[0] = "✓" if row in self._chat_checked else "☐"
        self.tree_chats.item(row, values=vals)

    def _delete_selected_chats(self):
        if not self._chat_checked:
            self.lbl_chat_msg.config(text="Ни один чат не отмечен (кликните по ✓ слева).",
                                     foreground=WARN)
            return
        if not messagebox.askyesno("Валера", f"Удалить отмеченные чаты ({len(self._chat_checked)})?"):
            return
        from valera_chats import ChatStore
        store = ChatStore()
        try:
            deleted = store.delete(list(self._chat_checked))
        finally:
            store.close()
        self._chat_checked.clear()
        self.lbl_chat_msg.config(text=f"✓ удалено чатов: {deleted}", foreground=OK)
        self._refresh_chats()

    def _delete_old_chats(self):
        try:
            days = float(self.var_old_days.get() or 1)
        except ValueError:
            days = 1.0
        if not messagebox.askyesno(
            "Валера",
            f"Удалить все чаты, не обновлявшиеся дольше {days:g} дн.?\n"
            f"Текущий чат останется.",
        ):
            return
        from valera_chats import ChatStore
        store = ChatStore()
        try:
            keep = self.rt.get_current_chat().get("chat_id")
            deleted = store.delete_older_than(days, keep_chat_id=keep)
        finally:
            store.close()
        self.lbl_chat_msg.config(text=f"✓ удалено старых чатов: {deleted}", foreground=OK)
        self._refresh_chats()

    def _delete_all_chats(self):
        if not messagebox.askyesno("Валера",
                                   "Удалить ВСЕ чаты, кроме текущего?\nЭто очистит всю историю диалогов."):
            return
        from valera_chats import ChatStore
        store = ChatStore()
        try:
            keep = self.rt.get_current_chat().get("chat_id")
            deleted = store.delete_all(keep_chat_id=keep)
        finally:
            store.close()
        self.lbl_chat_msg.config(text=f"✓ удалено чатов: {deleted}", foreground=OK)
        self._refresh_chats()

    def _new_chat_now(self):
        """Просит ассистента начать новый чат (сброс контекста)."""
        asst = self.runner.assistant
        if asst is None:
            self.lbl_chat_msg.config(text="Ассистент не запущен — нечего переключать.",
                                     foreground=WARN)
            return
        try:
            asst._clear_chat_id_file()
            asst._chat_id = asst.rag.create_chat()
            if asst._chat_id:
                asst._save_chat_id(asst._chat_id)
                self.rt.set_current_chat(asst._chat_id, "")
            self.rt.mark_new_chat_started()
            self.lbl_chat_msg.config(text=f"✓ новый чат: {asst._chat_id}", foreground=OK)
        except Exception as exc:  # noqa: BLE001
            self.lbl_chat_msg.config(text=f"Ошибка: {exc}", foreground=ERR)
        self._refresh_chats()

    def _on_auto_chat_toggle(self):
        self.rt.set_feature("auto_new_chat", bool(self.var_auto_chat.get()))

    def _save_auto_chat(self):
        try:
            seconds = float(self.var_auto_sec.get() or 0)
        except ValueError:
            seconds = 0.0
        self.rt.set("rag", "auto_new_chat_after_silence_sec", value=seconds)
        self.rt.set_feature("auto_new_chat", bool(self.var_auto_chat.get()))
        self.lbl_chat_msg.config(
            text=f"✓ авто-новый чат: {'вкл' if self.var_auto_chat.get() else 'выкл'}, "
                 f"порог {seconds:g} сек",
            foreground=OK,
        )

    # ------------------------------------------------------------------
    # Вкладка журнала
    # ------------------------------------------------------------------
    def _build_log_tab(self):
        outer = self._panel(self.tab_log_inner)
        outer.pack(fill="both", expand=True, padx=10, pady=10)
        self.txt_log = tk.Text(outer, bg="#0d1116", fg=FG, insertbackground=FG,
                               font=("DejaVu Sans Mono", 9), wrap="word",
                               relief="flat", padx=8, pady=6)
        self.txt_log.pack(fill="both", expand=True)
        for tag, color in [("info", FG_DIM), ("ok", OK), ("error", ERR), ("accent", ACCENT)]:
            self.txt_log.tag_configure(tag, foreground=color)
        ttk.Button(outer, text="Очистить", command=lambda: self.txt_log.delete("1.0", "end")).pack(
            anchor="e", pady=(6, 0))

    # ------------------------------------------------------------------
    # Синхронизация с конфигом
    # ------------------------------------------------------------------
    def _load_config_into_widgets(self):
        cfg = self.rt.snapshot()
        for key, var in self.flags.items():
            var.set(bool(cfg["features"].get(key, True)))
        self.var_threshold.set(float(cfg["vad"]["threshold"]))
        self.var_silence.set(float(cfg["vad"]["silence_timeout"]))
        self.var_min_speech.set(float(cfg["vad"]["min_speech"]))
        self.var_robot.set(str(cfg.get("robot", {}).get("base_url", "")))
        # Авто-новый чат
        self.var_auto_chat.set(bool(cfg["features"].get("auto_new_chat", True)))
        self.var_auto_sec.set(str(cfg["rag"].get("auto_new_chat_after_silence_sec", 0)))
        self._refresh_triggers()

    def _on_flag_change(self, key):
        value = bool(self.flags[key].get())
        self.rt.set_feature(key, value)
        self.rt.log_event("info", f"{'ВКЛ' if value else 'ВЫКЛ'} {key}")

    def _on_slider(self, label):
        # В ручном режиме ползунки пишут прямо в конфиг — ассистент читает
        # их на каждом новом кандидате во фразу, перезапуск не нужен.
        self.rt.set("vad", "threshold", value=round(self.var_threshold.get(), 3))
        self.rt.set("vad", "silence_timeout", value=round(self.var_silence.get(), 3))
        self.rt.set("vad", "min_speech", value=round(self.var_min_speech.get(), 3))

    def _save_robot_url(self):
        self.rt.set("robot", "base_url", value=self.var_robot.get().strip())
        self.rt.log_event("info", f"Адрес API робота: {self.var_robot.get().strip()}")
        messagebox.showinfo("Валера", "Адрес API робота сохранён.")

    # ------------------------------------------------------------------
    # Триггеры
    # ------------------------------------------------------------------
    def _refresh_triggers(self, select_id: str = None):
        self.tree.delete(*self.tree.get_children())
        for trig in self.rt.triggers():
            item = self.tree.insert("", "end", iid=trig["id"], values=(
                SYM_OK if trig.get("enabled", True) else SYM_NO,
                trig.get("phrase", ""),
                trig.get("match", "contains"),
                trig.get("cooldown_sec", 0),
                trig.get("command", ""),
            ))
            if select_id and trig["id"] == select_id:
                self.tree.selection_set(item)

    def _selected_trigger(self):
        sel = self.tree.selection()
        return sel[0] if sel else None

    def _test_trigger(self):
        """Прогоняет команду выбранного правила без ожидания микрофона."""
        self._test_selected_trigger()

    # ------------------------------------------------------------------
    # Ассистент: старт/стоп
    # ------------------------------------------------------------------
    def _start_assistant(self):
        self._toggle_assistant(start=True)

    def _stop_assistant(self):
        self._toggle_assistant(start=False)

    def _toggle_assistant(self, start: bool):
        if start:
            if self.runner.running:
                return
            self.btn_start.config(state="disabled")
            self.btn_stop.config(state="normal")
            self.lbl_state.config(text="запускаю…", foreground=WARN)
            self.rt.set_status("Запуск…")
            self.runner.start(stt=self.var_stt.get(),
                              new_chat=bool(self.var_new_chat.get()),
                              no_clean=bool(self.var_no_clean.get()))
        else:
            self.runner.stop()
            self.btn_start.config(state="normal")
            self.btn_stop.config(state="disabled")
            self.lbl_state.config(text=f"{SYM_STOP} остановлен", foreground=WARN)
            self.rt.set_status("Остановлен")

    def _on_close(self):
        try:
            self.runner.stop()
        finally:
            self.root.destroy()

    # ------------------------------------------------------------------
    # Периодические обновления
    # ------------------------------------------------------------------
    def _tick_graphs(self):
        try:
            self._draw_graphs()
        except Exception:  # noqa: BLE001
            pass
        self.root.after(100, self._tick_graphs)

    def _draw_graphs(self):
        # ---- (1) три дорожки последней фразы ----
        self.ax_phrase.clear()
        self.ax_phrase.set_facecolor("#0d1116")
        self.ax_phrase.set_title(
            "Последняя фраза:  МИКРОФОН  -->  вырезанный шумодавом шум  -->  ИТОГ для Валеры",
            color=FG, fontsize=10, loc="left")
        self.ax_phrase.tick_params(colors=FG_DIM, labelsize=7)
        for spine in self.ax_phrase.spines.values():
            spine.set_color("#2a3441")

        phrase = self.rt.get_phrase()
        if phrase:
            n = len(phrase["raw"])
            t = np.linspace(0, phrase["duration"], n) if n else np.array([])
            if n:
                self.ax_phrase.plot(t, phrase["raw"], color="#7dd3fc", lw=0.7,
                                    label="с микрофона")
                self.ax_phrase.plot(t, phrase["removed"], color=ERR, lw=0.7,
                                    label="вырезано шумодавом (шум)")
                self.ax_phrase.plot(t, phrase["clean"], color=OK, lw=0.7,
                                    label="итог, ушёл Валере")
                # Смещаем дорожки по вертикали, чтобы они не накладывались
                rng = max(abs(min(phrase["raw"])), abs(max(phrase["raw"])), 1e-3)
                self.ax_phrase.set_ylim(-1.15 * rng, 1.15 * rng)
                legend = self.ax_phrase.legend(loc="upper right", fontsize=7,
                                               facecolor=BG_PANEL, edgecolor="#2a3441")
                for text in legend.get_texts():
                    text.set_color(FG)
        else:
            self.ax_phrase.text(0.5, 0.5, "скажите что-нибудь — здесь появится сравнение",
                                color=FG_DIM, ha="center", va="center",
                                transform=self.ax_phrase.transAxes, fontsize=9)

        # ---- (2) живой осциллограф ----
        raw_wave, gated_wave = self.rt.get_waves()
        self.ax_live.clear()
        self.ax_live.set_facecolor("#0d1116")
        self.ax_live.set_title("Живой сигнал с микрофона (2 сек)", color=FG,
                               fontsize=10, loc="left")
        self.ax_live.tick_params(colors=FG_DIM, labelsize=7)
        for spine in self.ax_live.spines.values():
            spine.set_color("#2a3441")
        if raw_wave:
            arr = np.asarray(raw_wave, dtype=np.float32)
            self.ax_live.plot(arr, color="#7dd3fc", lw=0.5)
            self.ax_live.set_ylim(-1.05, 1.05)
            self.ax_live.axhline(0, color="#2a3441", lw=0.5)
        else:
            self.ax_live.text(0.5, 0.5, "нет сигнала (ассистент остановлен?)",
                              color=FG_DIM, ha="center", va="center",
                              transform=self.ax_live.transAxes, fontsize=9)

        # ---- (3) шкала уровня + пороги ----
        levels = self.rt.get_levels()
        metrics = self.rt.get_metrics()
        self.ax_level.clear()
        self.ax_level.set_facecolor("#0d1116")
        self.ax_level.set_title("Уровень микрофона / после шумодава · порог гейта",
                                color=FG, fontsize=10, loc="left")
        self.ax_level.tick_params(colors=FG_DIM, labelsize=7)
        for spine in self.ax_level.spines.values():
            spine.set_color("#2a3441")
        if levels:
            mic = [x[1] for x in levels]
            gated = [x[2] for x in levels]
            self.ax_level.plot(mic, color="#7dd3fc", lw=1.0, label="микрофон (RMS)")
            self.ax_level.plot(gated, color=OK, lw=1.0, label="после шумодава")
            levels_max = max(max(mic), 1e-3)
            gate = metrics.get("gate_threshold", 0.0)
            noise = metrics.get("noise_rms", 0.0)
            if gate > 0:
                self.ax_level.axhline(gate, color=ERR, ls="--", lw=1.0,
                                      label=f"порог гейта {gate:.3f}")
            if noise > 0:
                self.ax_level.axhline(noise, color=WARN, ls=":", lw=1.0,
                                      label=f"шумовой пол {noise:.3f}")
            self.ax_level.set_ylim(0, max(levels_max * 1.25, gate * 1.25, 0.02))
            legend = self.ax_level.legend(loc="upper right", fontsize=7,
                                          facecolor=BG_PANEL, edgecolor="#2a3441")
            for text in legend.get_texts():
                text.set_color(FG)
        else:
            self.ax_level.text(0.5, 0.5, "уровни появятся при работе микрофона",
                               color=FG_DIM, ha="center", va="center",
                               transform=self.ax_level.transAxes, fontsize=9)

        self.canvas.draw_idle()

    def _tick_status(self):
        metrics = self.rt.get_metrics()
        fmt = lambda v: ("—" if v is None else f"{v:.3f}")  # noqa: E731
        self.tele_vars["mic"].set(fmt(metrics.get("mic_rms")))
        self.tele_vars["vad"].set(fmt(metrics.get("vad_prob")))
        self.tele_vars["thr"].set(fmt(metrics.get("vad_threshold")))
        self.tele_vars["noise"].set(fmt(metrics.get("noise_rms")))
        self.tele_vars["speech"].set(fmt(metrics.get("speech_rms")))
        self.tele_vars["gate"].set(fmt(metrics.get("gate_threshold")))
        self.tele_vars["gain"].set(fmt(metrics.get("gain")))
        self.tele_vars["status"].set(self.rt.get_status()[:40])

        # Активный чат (контекст диалога)
        chat = self.rt.get_current_chat()
        cid = chat.get("chat_id")
        self.lbl_chat.config(text=f"чат: {cid[:8] if cid else '—'}")

        if self.runner.error:
            self.lbl_state.config(text="! ошибка", foreground=ERR)
        elif self.runner.running:
            self.lbl_state.config(text=f"{SYM_DOT} работает", foreground=OK)
        elif self.btn_stop["state"] == "normal":
            # кнопка «стоп» активна, но поток уже завершился — покажем причину
            self.lbl_state.config(text=f"{SYM_STOP} остановлен", foreground=WARN)
            self.btn_start.config(state="normal")
            self.btn_stop.config(state="disabled")
        self.root.after(400, self._tick_status)

    def _tick_ui_queue(self):
        try:
            while True:
                kind, payload = self.ui_queue.get_nowait()
                if kind in ("trigger_result", "dialog_result"):
                    res = payload
                    text = _format_trigger_result(res)
                    if kind == "trigger_result":
                        self.lbl_trig_result.config(
                            text=text, foreground=OK if res.get("ok") else ERR)
                    else:
                        # Результат из диалога: показываем в том же месте и в журнале
                        self.rt.log_event(
                            "ok" if res.get("ok") else "error",
                            f"Триггер (диалог): {text}",
                        )
        except queue.Empty:
            pass

        # Переливаем события рантайма в текстовый журнал
        events = self.rt.get_events()
        shown = getattr(self, "_events_shown", 0)
        for _ts, level, text in events[shown:]:
            tag = "error" if level == "error" else ("ok" if level == "ok" else "info")
            stamp = time.strftime("%H:%M:%S", time.localtime(_ts))
            self.txt_log.insert("end", f"[{stamp}] {text}\n", tag)
            self.txt_log.see("end")
        self._events_shown = len(events)

        self.root.after(300, self._tick_ui_queue)


def main():
    parser = argparse.ArgumentParser(description="GUI-пульт робота Валеры")
    parser.add_argument("--stt", default="kairos",
                        choices=["kairos"],
                        help="Модель распознавания речи")
    parser.add_argument("--no-autostart", action="store_true",
                        help="Не запускать ассистента автоматически")
    # Общие флаги проекта: принимаем их, чтобы `run_all.sh gui ...` работал
    # так же, как `run_all.sh voice ...`, и сразу применяем к рантайму.
    parser.add_argument("--no-auto-vad", action="store_true",
                        help="Выключить авто-регулировку чувствительности")
    parser.add_argument("--barge-in", action="store_true",
                        help="Прерывать озвучку, если пользователь заговорил")
    parser.add_argument("--new-chat", action="store_true",
                        help="Начать новый чат (сбросить контекст)")
    parser.add_argument("--no-clean", action="store_true",
                        help="Не чистить текст вопроса (имена/мат сохраняются)")
    parser.add_argument("--no-confirm", action="store_true",
                        help="Не спрашивать подтверждение вопроса")
    parser.add_argument("--preload", action="store_true",
                        help="(совместимость) прогрев выполняется всегда")
    parser.add_argument("--proxy", default=None,
                        help="HTTP-прокси для исходящих запросов")
    # Толерантность к прочим флагам run_all.sh
    args, _unknown = parser.parse_known_args()

    if args.proxy:
        os.environ["HTTP_PROXY"] = args.proxy
        os.environ["HTTPS_PROXY"] = args.proxy
        os.environ["ALL_PROXY"] = args.proxy
        os.environ.setdefault(
            "NO_PROXY", "localhost,127.0.0.1,::1,searxng,qdrant,mongodb,app"
        )

    rt = get_runtime()
    # Флаги CLI перекрывают конфиг (и сохраняются в него — видны в галочках).
    if args.no_auto_vad:
        rt.set_feature("auto_vad", False)
    if args.barge_in:
        rt.set_feature("barge_in", True)
    if args.no_confirm:
        rt.set_feature("confirm_before_answer", False)
    if args.new_chat:
        rt.set_feature("_new_chat_requested", True)

    root = tk.Tk()
    gui = ValeraGUI(root, rt, stt=args.stt, autostart=not args.no_autostart)
    if args.new_chat:
        gui.var_new_chat.set(True)
    if args.no_clean:
        gui.var_no_clean.set(True)
    root.mainloop()


if __name__ == "__main__":
    main()
