# ═══════════════════════════════════════════════════════════════════════════════
# ОБНОВЛЕНИЕ JetPack 5.1.2 → 6.x НА Jetson AGX Orin (eMMC, 64 ГБ, без NVMe)
# ═══════════════════════════════════════════════════════════════════════════════

## ⚠️ 1. РЕЗЕРВНАЯ КОПИЯ — ОБЯЗАТЕЛЬНО, ПЕРЕД НАЧАЛОМ!

Перепрошивка JetPack **СТИРАЕТ ВСЁ** на eMMC. Сохраните свои файлы на
**внешний накопитель** (USB-флешка / SD-карта / другой ПК):

```bash
# СОХРАНИТЕ ПРОЕКТ (ОБЯЗАТЕЛЬНО) на USB/SD/другой ПК:
rsync -av ~/Desktop/QWEN-VALERA/ /media/usb/</any-temporary>/QWEN-VALERA/

# Проверьте, что скопировалось:
ls /media/usb/.../QWEN-VALERA/README.md
```

**Без внешнего накопителя вы не сможете перепрошить — это блокирующий шаг.**

---

## 2. ВАЖНО: Про прошивку

| Факт | Детали |
|------|--------|
| ❌ Jetson НЕ может перепрошить сам себя | Для этого нужен внешний ПК с Ubuntu |
| ✅ SDK Manager живёт на внешнем ПК | НЕ на Jetson (на Jetson он не прошьёт) |
| ✅ Для AGX Orin eMMC команда: | `sudo ./flash.sh jetson-agx-orin-devkit internal` |
| 🔌 Порт USB-C | Тот, что рядом с 40-pin header |
| ⚙️ Force Recovery Mode | Спец. комбинация кнопок (ниже) |

**Вам понадобится второй компьютер с Ubuntu 20.04/22.04 (x86_64)**
или Ubuntu VM. Это можно сделать двумя способами:

---

## 3. Способ A: SDK Manager (рекомендуется, на внешнем ПК)

### A.1. Установите SDK Manager на внешний ПК (Ubuntu x86_64)
```bash
# Скачайте .deb с https://developer.nvidia.com/sdk-manager (версия для x64)
sudo apt install ./sdkmanager_*.deb
sdkmanager
```

### A.2. Подключите Jetson к ПК (Force Recovery Mode)
1. **Выключите** Jetson (полное отключение питания).
2. Подключите USB-C кабель от ПК к **порту USB-C рядом с 40-pin header**.
3. Зажмите **кнопку Force Recovery** (по центру платы, рядом с USB-C).
4. Не отпуская, нажмите и отпустите кнопку **Power** (Reset).
5. Отпустите **Force Recovery**.
6. Проверьте на ПК:
```bash
lsusb
# Должно появиться: NVIDIA Corp. APX  (ID 0955:7023 для AGX Orin 64GB)
```

### A.3. Прошивка через SDK Manager
1. В SDK Manager выберите **Jetson AGX Orin Developer Kit**.
2. JetPack version: **6.2** (свежая, у вас уже r36.4 репозитории).
3. **Только Jetson Linux** (не отмечайте DeepStream и доп. пакеты — экономьте место).
4. **Storage Device**: `eMMC` (важно! у вас без NVMe).
5. Нажмите **Continue → Flash and Install**.
6. Дождитесь завершения (15–30 мин). Jetson перезагрузится с JetPack 6.

### A.4. После перезагрузки
- Задайте язык, Wi-Fi, имя пользователя и пароль.
- **Jetson 6.2 имеет Python 3.10 встроенный.**

---

## 4. Способ B: ручная прошивка через файлы L4T (без SDK Manager GUI)

Если SDK Manager не запускается — можно вручную, тоже с внешнего ПК:

```bash
# На внешнем ПК (Ubuntu), скачайте L4T R36.4:
#  - Jetson_Linux_R36.4.0_aarch64.tbz2
#  - Tegra_Linux_Sample-Root-Filesystem_R36.4.0_aarch64.tbz2
# с https://developer.nvidia.com/embedded/jetson-linux-r364

# Распакуйте:
tar xf Jetson_Linux_R36.4.0_aarch64.tbz2
cd Linux_for_Tegra/
sudo tar xpf ../Tegra_Linux_Sample-Root-Filesystem_R36.4.0_aarch64.tbz2 -C rootfs/
sudo ./tools/l4t_flash_prerequisites.sh
sudo ./apply_binaries.sh

# Подключите Jetson в Force Recovery Mode (как в A.2)
# Запишите JetPack 6.2 на eMMC:
sudo ./flash.sh jetson-agx-orin-devkit internal
```

---

## 5. ДИСК: модель на eMMC 59 ГБ

После чистой прошивки JetPack 6.2:

| Компонент | Размер | Итог |
|-----------|--------|------|
| JetPack 6.2 (eMMC) | ~15-18 ГБ | |
| **AWQ-4bit модель** ✅ | ~27 ГБ | Итого ~45 ГБ → влезает |
| ~~AWQ-8bit~~ | ~42 ГБ | Впритык (60 > 59) — не рекомендую |
| ~~NVFP4~~ | ~26 ГБ | ❌ Не работает на Jetson (Ampere) |

👉 **Используйте AWQ-4bit** — уже настроено по умолчанию в проекте.

---

## 6. После прошивки: установка проекта

```bash
# 1. Скопируйте проект с внешнего накопителя обратно:
rsync -av /media/usb/.../QWEN-VALERA/ ~/Desktop/QWEN-VALERA/
cd ~/Desktop/QWEN-VALERA

# 2. Запустите установку (JetPack 6 = Python 3.10, всё совместимо):
chmod +x setup_jetson6.sh
bash setup_jetson6.sh
#  - установит NVIDIA Jetson PyTorch 2.5.0
#  - скачает AWQ-4bit модель (~27 ГБ)

# 3. Проверка:
conda activate qwen-valera
python -c "import torch; print(torch.cuda.is_available())"   # True
python test_model.py                                          # все тесты

# 4. Запуск:
python main.py        # сервер
python client.py --mode voice   # голосовой клиент
```

---

## 7. Troubleshooting

| Проблема | Решение |
|----------|---------|
| `lsusb` не видит APX | CRITICAL: используйте USB-C порт рядом с 40-pin header, проверьте кабель (data, не charge-only) |
| SDK Manager не находит Jetson | Снова переведите в Recovery Mode, проверьте `lsusb` |
| Не хватает места под модель | Используйте AWQ-4bit (уже по умолчанию). Не ставьте доп. пакеты SDK Manager. |
| Потерян Wi-Fi после прошивки | Настройте заново при первом запуске |
| Пароль sudo | Задайте новый при установке JetPack 6 |
| SDK Manager просит аккаунт NVIDIA | Нужен бесплатный аккаунт developer.nvidia.com |

---

## 8. Итоговая архитектура (после JetPack 6.2)

```
┌───────────────────────────────────────────────────┐
│  Jetson AGX Orin 64GB, JetPack 6.2 (eMMC, CUDA 12)│
│                                                   │
│  ┌─────────────────────────────────────────────┐  │
│  │  FastAPI Server (:8765)                     │  │
│  │  /chat/voice  /chat/text  /documents (RAG)  │  │
│  └─────────────────────┬───────────────────────┘  │
│                        ▼                          │
│  ┌─────────────────────────────────────────┐      │
│  │  Qwen3-Omni AWQ-4bit (~27 ГБ)           │      │
│  │  Thinker + Talker (встроенные ASR + TTS)│      │
│  └─────────────────────────────────────────┘      │
│                                                   │
│  .doc/.docx → токенизация → ChromaDB → RAG        │
│  Интернет: DuckDuckGo (только это не локально)    │
└───────────────────────────────────────────────────┘
```
