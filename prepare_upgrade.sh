#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Подготовка к перепрошивке на JetPack 6.x
#
# 💡 ЭТОТ СКРИПТ ЗАПУСКАЕТСЯ НА JETSON (сейчас), ДО перепрошивки.
# Он делает 2 вещи:
#   1. Сохраняет проект на внешний накопитель (флешка/SD/другой ПК)
#   2. (опционально) Скачивает L4T-файлы для прошивки на внешнем ПК
#
# ВАЖНО: сам Jetson не может перепрошить себя. Нужен внешний ПК с Ubuntu.
# Этот скрипт лишь подготавливает всё необходимое.
# ═══════════════════════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "══════════════════════════════════════════════════════════"
echo "  Подготовка к перепрошивке Jetson AGX Orin → JetPack 6.x"
echo "══════════════════════════════════════════════════════════"
echo ""
echo "Текущая система: JetPack 5.1.2 (L4T 35.4.1)"
echo "Цель: JetPack 6.2 (L4T 36.4)"
echo ""

# ── Шаг 1: Сохранение проекта ───────────────────────────────────────────────

echo "[1/2] Сохранение проекта на внешний накопитель..."
echo ""
echo "Вставьте USB-флешку / SD-карту (НЕ eMMC) и скажите её путь."
echo "Текущие смонтированные разделы:"
lsblk -o NAME,SIZE,MOUNTPOINT,LABEL 2>/dev/null | grep -v "loop\|ram" | head -15
echo ""

read -p "Путь к внешнему накопителю (например /media/$USER/USB или оставьте пустым чтобы пропустить): " DEST
if [ -n "$DEST" ] && [ -d "$DEST" ]; then
    echo "  Копирую проект в $DEST/QWEN-VALERA ..."
    mkdir -p "$DEST/QWEN-VALERA"
    rsync -av --exclude='__pycache__' --exclude='.git' \
        "$SCRIPT_DIR/" "$DEST/QWEN-VALERA/" 2>&1 | tail -5
    echo "  ✅ Проект сохранён в: $DEST/QWEN-VALERA"
    echo "  Проверьте: ls '$DEST/QWEN-VALERA/README.md'"
else
    echo "  ⚠️  Пропущено (путь не указан или не существует)."
    echo "  Обязательно сохраните проект перед перепрошивкой!"
fi

# ── Шаг 2: (опционально) Скачать L4T для внешнего ПК ──────────────────────

echo ""
echo "[2/2] (опционально) Скачать L4T R36.4 для прошивки на внешнем ПК?"
echo "  Нужно, если будете прошивать вручную (Способ B)."
echo "  Размер: ~2 × 5-6 ГБ (пакеты tbz2)."
read -p "  Скачать сейчас? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "  Скачивание L4T R36.4..."
    echo "  ВАЖНО: файлы нужно скачать с developer.nvidia.com (нужен вход в аккаунт)."
    echo "  Ссылки:"
    echo "   - https://developer.nvidia.com/embedded/jetson-linux-r364"
    echo "   - файл: Jetson_Linux_R36.4.0_aarch64.tbz2 (~5 ГБ)"
    echo "   - файл: Tegra_Linux_Sample-Root-Filesystem_R36.4.0_aarch64.tbz2 (~6 ГБ)"
    echo ""
    echo "  ⚠️  Прямое скачивание здесь может не работать (требуется вход)."
    echo "  Лучше скачать эти 2 файла вручную в браузере и сохранить на USB:"
    read -p "  Нажмите Enter, когда скачаете и перенесёте файлы на USB..."
    echo "  Готово. Убедитесь, что оба .tbz2 лежат на USB для внешнего ПК."
else
    echo "  ⏭️  Пропущено."
fi

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  ДАЛЬШЕ: на ВНЕШНЕМ ПК с Ubuntu выполните:"
echo ""
echo "  Способ A (SDK Manager):"
echo "    sudo apt install ./sdkmanager_*.deb && sdkmanager"
echo "    JetPack 6.2 → Jetson AGX Orin → eMMC → Flash"
echo ""
echo "  Или Способ B (вручную):"
echo "    tar xf Jetson_Linux_R36.4.0_aarch64.tbz2"
echo "    cd Linux_for_Tegra/"
echo "    sudo tar xpf ../Tegra_Linux_Sample-Root-Filesystem_R36.4.0_aarch64.tbz2 -C rootfs/"
echo "    sudo ./tools/l4t_flash_prerequisites.sh && sudo ./apply_binaries.sh"
echo "    # Jetson в Force Recovery Mode →"
echo "    sudo ./flash.sh jetson-agx-orin-devkit internal"
echo ""
echo "  Полная инструкция: см. JETPACK_UPGRADE.md"
echo "══════════════════════════════════════════════════════════"
