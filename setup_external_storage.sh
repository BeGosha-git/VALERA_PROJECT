#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# QWEN-VALERA — перенос окружения / кэшей / модели на внешний диск (диск D)
#
# Зачем: внутренний eMMC (57 GB) заполнен на 99%, поэтому PyTorch (≈10 GB),
#        модель (27 GB) и кэши туда не влезают.
#
# ВАЖНО: внешний диск сейчас в формате exFAT, а exFAT НЕ умеет символические
#        ссылки и права доступа → conda/pip там работать не смогут.
#        Поэтому есть два режима:
#
#   VALERA_MODE=safe  (по умолчанию)
#       Ничего не форматируется. На диске D создаётся файл-образ ext4
#       (размер VALERA_IMG_SIZE, по умолчанию 100G), который монтируется
#       как /mnt/valera. Данные на диске D сохраняются.
#       Минус: создание образа один раз занимает ~5-10 минут.
#
#   VALERA_MODE=ext4  (быстрее, рекомендуется — диск D фактически пуст)
#       Раздел диска D переформатируется в ext4 и монтируется как /mnt/valera.
#       Полная скорость, никаких образов. ⚠️ данные на разделе УДАЛЯЮТСЯ
#       (сейчас там только «System Volume Information»).
#
# Что ещё делает скрипт:
#   • Монтирует хранилище и прописывает автозапуск при загрузке (systemd)
#   • Переносит модель (27 GB) на внешний диск → освобождает внутренний eMMC
#   • Переносит conda-окружения, настраивает PIP_CACHE_DIR / HF_HOME / TMPDIR
#   • Чистит старый кэш pip на внутреннем диске
#
# ЗАПУСК (нужен пароль sudo):
#     sudo bash setup_external_storage.sh                    # режим safe
#     sudo bash setup_external_storage.sh ext4               # переформатировать D
#
# Дополнительно:
#     sudo VALERA_IMG_SIZE=150G bash setup_external_storage.sh
#     sudo VALERA_DISK_UUID=XXXX-XXXX bash setup_external_storage.sh
# ═══════════════════════════════════════════════════════════════════════════════

set -uo pipefail

# ── Настройки ────────────────────────────────────────────────────────────────
MODE="${1:-${VALERA_MODE:-safe}}"               # safe | ext4
# UUID раздела меняется при mkfs, поэтому основной ориентир — метка (LABEL).
DISK_UUID="${VALERA_DISK_UUID:-20d575ea-e3a9-406b-b4a7-c8d9c47503f9}"
DISK_LABEL="${VALERA_DISK_LABEL:-VALERA}"        # метка, ставится mkfs.ext4 -L
IMG_MNT="${VALERA_IMG_MNT:-/mnt/valera}"         # точка монтирования хранилища
DISK_MNT="${VALERA_DISK_MNT:-/mnt/valera-disk}"  # (только safe) точка для exFAT
IMG_REL="VALERA/valera-env.img"
IMG_SIZE="${VALERA_IMG_SIZE:-100G}"
IMG_LABEL="VALERA"

REAL_USER="${SUDO_USER:-$(id -un)}"
REAL_HOME="$(getent passwd "$REAL_USER" | cut -d: -f6)"
REAL_UID_N="$(id -u "$REAL_USER")"
REAL_GID_N="$(id -g "$REAL_USER")"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELPER="/usr/local/bin/valera-storage.sh"

c_ok()   { echo "  ✓ $*"; }
c_info() { echo "  • $*"; }
c_warn() { echo "  ⚠️  $*"; }

echo "============================================"
echo " QWEN-VALERA — внешнее хранилище (диск D)"
echo "============================================"
echo ""

if [ "$(id -u)" -ne 0 ]; then
    echo "❌ Запустите с правами root:"
    echo "     sudo bash setup_external_storage.sh"
    exit 1
fi

case "$MODE" in
    safe|ext4) ;;
    *) echo "❌ VALERA_MODE должен быть 'safe' или 'ext4' (сейчас: $MODE)"; exit 1 ;;
esac

# ── [1/8] Поиск внешнего диска ───────────────────────────────────────────────
echo "[1/8] Поиск внешнего диска..."
DEV=""

# 1) по UUID (основной ориентир для уже настроенного диска)
if [ -n "$DISK_UUID" ]; then
    DEV="$(blkid -U "$DISK_UUID" 2>/dev/null || true)"
    [ -n "$DEV" ] && c_ok "Найден по UUID $DISK_UUID: $DEV"
fi

# 2) по метке — UUID меняется при mkfs, метка остаётся
if [ -z "$DEV" ]; then
    for lbl in "$DISK_LABEL" "MAIN VOLUME"; do
        DEV="$(blkid -L "$lbl" 2>/dev/null || true)"
        if [ -n "$DEV" ]; then
            c_ok "Найден по метке '$lbl': $DEV"
            break
        fi
    done
fi

# 3) автоматически: самый большой не-корневой раздел с файловой системой
if [ -z "$DEV" ]; then
    ROOT_SRC="$(findmnt -n -o SOURCE / 2>/dev/null | head -n1)"
    ROOT_PK="$(lsblk -no PKNAME "$ROOT_SRC" 2>/dev/null | head -n1)"
    DETECTED="$(lsblk -rno NAME,SIZE,FSTYPE,PKNAME,TYPE 2>/dev/null \
        | awk -v rootpk="$ROOT_PK" \
            '$5 == "part" && $3 != "" && $3 != "squashfs" && $4 != rootpk {print $1, $2}' \
        | sort -k2 -h -r | head -n1)"
    if [ -n "$DETECTED" ]; then
        DEV="/dev/$(echo "$DETECTED" | awk '{print $1}')"
        c_warn "Диск определён автоматически: $DEV"
    fi
fi

if [ -z "$DEV" ]; then
    echo "❌ Внешний диск не найден."
    echo "   Подключите диск D. Доступные диски:"
    lsblk -o NAME,SIZE,FSTYPE,UUID,LABEL,MOUNTPOINT
    echo ""
    echo "   Можно указать вручную:"
    echo "     sudo VALERA_DISK_UUID=XXXX-XXXX bash setup_external_storage.sh"
    echo "     sudo VALERA_DISK_LABEL=МЕТКА    bash setup_external_storage.sh"
    exit 1
fi
DEV_FS="$(lsblk -no FSTYPE "$DEV" 2>/dev/null | head -n1)"
DEV_SIZE="$(lsblk -no SIZE "$DEV" 2>/dev/null | head -n1)"
c_ok "Диск найден: $DEV ($DEV_FS, $DEV_SIZE)"

# ── [2/8] Подготовка раздела ─────────────────────────────────────────────────
echo "[2/8] Подготовка раздела (режим: $MODE)..."

# Отмонтируем всё, что уже примонтировал udisks
while read -r tgt; do
    [ -n "$tgt" ] || continue
    umount "$tgt" 2>/dev/null && c_info "отключено: $tgt"
done < <(findmnt -n -o TARGET -S "$DEV" 2>/dev/null || true)

if [ "$MODE" = "ext4" ]; then
    if [ "$DEV_FS" != "ext4" ]; then
        echo ""
        echo "  ⚠️  ВНИМАНИЕ: раздел $DEV будет ПЕРЕФОРМАТИРОВАН в ext4."
        echo "     Все данные на нём будут уничтожены: $DEV ($DEV_SIZE)"
        echo ""
        read -r -p "     Продолжить? Введите 'yes' для подтверждения: " ANSWER
        if [ "$ANSWER" != "yes" ]; then
            echo "     Отменено."
            exit 1
        fi
        c_info "Форматирую $DEV в ext4..."
        mkfs.ext4 -F -L "$IMG_LABEL" -m 0 -E lazy_itable_init=1 "$DEV" >/dev/null 2>&1 \
            || { echo "❌ Ошибка mkfs.ext4"; exit 1; }
        udevadm settle 2>/dev/null || true
        c_ok "Раздел отформатирован в ext4"
    else
        c_ok "Раздел уже в ext4"
    fi

    mkdir -p "$IMG_MNT"
    if mountpoint -q "$IMG_MNT"; then
        c_ok "Уже подключён в $IMG_MNT"
    else
        mount -o rw,noatime "$DEV" "$IMG_MNT" || { echo "❌ Не удалось подключить $DEV"; exit 1; }
        c_ok "Подключён: $DEV → $IMG_MNT"
    fi
else
    # ── safe: exFAT-диск + ext4-образ внутри него ────────────────────────────
    mkdir -p "$DISK_MNT"
    if mountpoint -q "$DISK_MNT"; then
        c_ok "Диск уже подключён в $DISK_MNT"
    else
        if [ "$DEV_FS" = "exfat" ] || [ "$DEV_FS" = "vfat" ]; then
            mount -t "$DEV_FS" -o "rw,uid=$REAL_UID_N,gid=$REAL_GID_N,umask=000,noatime" "$DEV" "$DISK_MNT" \
                || { echo "❌ Не удалось подключить $DEV"; exit 1; }
        else
            mount -o rw,noatime "$DEV" "$DISK_MNT" \
                || { echo "❌ Не удалось подключить $DEV"; exit 1; }
        fi
        c_ok "Подключён: $DEV → $DISK_MNT"
    fi

    AVAIL_KB="$(df -Pk "$DISK_MNT" | awk 'NR==2 {print $4}')"
    c_info "Свободно на диске D: $((AVAIL_KB / 1024 / 1024)) GB"

    mkdir -p "$DISK_MNT/VALERA"
    IMG="$DISK_MNT/$IMG_REL"

    if [ ! -f "$IMG" ]; then
        SIZE_MB=$(( ${IMG_SIZE%G} * 1024 ))
        if [ "$SIZE_MB" -gt "$((AVAIL_KB / 1024))" ]; then
            echo "❌ На диске D недостаточно места для образа $IMG_SIZE"
            exit 1
        fi
        echo "  • Выделяю образ $IMG_SIZE — на exFAT это может занять несколько минут..."
        truncate -s "$IMG_SIZE" "$IMG" || dd if=/dev/zero of="$IMG" bs=1M count="$SIZE_MB" status=progress || {
            echo "❌ Не удалось создать образ (мало места?)"; exit 1; }
        c_info "Форматирую образ в ext4..."
        mkfs.ext4 -F -L "$IMG_LABEL" -m 0 -E lazy_itable_init=1 "$IMG" >/dev/null 2>&1 \
            || { echo "❌ Ошибка mkfs.ext4"; exit 1; }
        c_ok "Образ создан: $IMG"
    else
        c_ok "Образ уже существует: $IMG"
    fi

    mkdir -p "$IMG_MNT"
    if mountpoint -q "$IMG_MNT"; then
        c_ok "Образ уже подключён в $IMG_MNT"
    else
        mount -o loop,noatime "$IMG" "$IMG_MNT" || { echo "❌ Не удалось подключить образ"; exit 1; }
        c_ok "Образ подключён: $IMG → $IMG_MNT"
    fi
fi

# ── UUID/метка после mkfs ────────────────────────────────────────────────────
# mkfs.ext4 создаёт НОВЫЙ UUID, поэтому читаем его с диска заново — иначе
# автоподключение при загрузке будет искать старый UUID и диск не поднимется.
NEW_UUID="$(blkid -s UUID -o value "$DEV" 2>/dev/null || true)"
NEW_LABEL="$(blkid -s LABEL -o value "$DEV" 2>/dev/null || true)"
[ -n "$NEW_UUID" ] && DISK_UUID="$NEW_UUID"
[ -n "$NEW_LABEL" ] && DISK_LABEL="$NEW_LABEL"
c_info "UUID=$DISK_UUID  LABEL=${DISK_LABEL:-—}"

# ── Каталоги хранилища ───────────────────────────────────────────────────────
mkdir -p "$IMG_MNT"/{conda-envs,conda-pkgs,pip-cache,hf-cache,tmp,models}
chown -R "$REAL_UID_N:$REAL_GID_N" "$IMG_MNT"
chmod 755 "$IMG_MNT"
c_ok "Каталоги: conda-envs, conda-pkgs, pip-cache, hf-cache, tmp, models"
echo ""

# ── [3/8] Перенос модели (27 GB) ─────────────────────────────────────────────
echo "[3/8] Перенос модели на внешний диск..."
MODELS_LINK="$PROJECT_DIR/models"
if [ -L "$MODELS_LINK" ]; then
    c_ok "models/ уже ссылка → $(readlink "$MODELS_LINK")"
elif [ -d "$MODELS_LINK" ]; then
    c_info "Копирую models/ (~27 GB, подождите)..."
    rsync -a --info=progress2 "$MODELS_LINK/" "$IMG_MNT/models/" \
        || { echo "❌ Ошибка копирования модели"; exit 1; }
    chown -R "$REAL_UID_N:$REAL_GID_N" "$IMG_MNT/models"
    rm -rf "$MODELS_LINK"
    ln -s "$IMG_MNT/models" "$MODELS_LINK"
    chown -h "$REAL_UID_N:$REAL_GID_N" "$MODELS_LINK"
    c_ok "Модель перенесена: models/ → $IMG_MNT/models"
else
    ln -s "$IMG_MNT/models" "$MODELS_LINK"
    chown -h "$REAL_UID_N:$REAL_GID_N" "$MODELS_LINK"
    c_ok "Ссылка models/ → $IMG_MNT/models создана"
fi
echo ""

# ── [4/8] Перенос conda-окружений ────────────────────────────────────────────
echo "[4/8] Перенос conda-окружений..."
OLD_ENVS="$REAL_HOME/miniconda3/envs"
if [ -d "$OLD_ENVS" ] && [ -n "$(ls -A "$OLD_ENVS" 2>/dev/null)" ]; then
    for env_dir in "$OLD_ENVS"/*; do
        [ -e "$env_dir" ] || continue
        name="$(basename "$env_dir")"
        if [ -e "$IMG_MNT/conda-envs/$name" ]; then
            c_warn "окружение '$name' уже есть на внешнем диске — пропускаю"
        else
            mv "$env_dir" "$IMG_MNT/conda-envs/" && c_ok "перенесено: $name"
        fi
    done
    chown -R "$REAL_UID_N:$REAL_GID_N" "$IMG_MNT/conda-envs"
else
    c_info "Старых окружений нет — новые создадутся сразу на внешнем диске"
fi
echo ""

# ── [5/8] Автоподключение при старте (systemd) ───────────────────────────────
echo "[5/8] Настройка автоподключения при старте..."

cat > "$HELPER" <<HELPER_EOF
#!/bin/bash
# VALERA — подключение внешнего хранилища (сгенерировано setup_external_storage.sh)
set -u

MODE="$MODE"
DISK_UUID="$DISK_UUID"
DISK_LABEL="$DISK_LABEL"
IMG_MNT="$IMG_MNT"
DISK_MNT="$DISK_MNT"
IMG_REL="$IMG_REL"
DISK_UID="$REAL_UID_N"
DISK_GID="$REAL_GID_N"

log() { echo "[valera-storage] \$*"; return 0; }

up() {
    local wait="\${1:-}"
    local dev="" tries=1 i
    [ "\$wait" = "--wait" ] && tries=15

    # UUID меняется при mkfs, поэтому если по UUID не нашли — ищем по метке
    for ((i=0; i<tries; i++)); do
        dev="\$(blkid -U "\$DISK_UUID" 2>/dev/null || true)"
        if [ -z "\$dev" ] && [ -n "\$DISK_LABEL" ]; then
            dev="\$(blkid -L "\$DISK_LABEL" 2>/dev/null || true)"
        fi
        [ -n "\$dev" ] && break
        sleep 2
    done
    if [ -z "\$dev" ]; then
        log "внешний диск (UUID=\$DISK_UUID / LABEL=\$DISK_LABEL) не найден — пропускаю"
        return 0
    fi

    if mountpoint -q "\$IMG_MNT"; then
        log "\$IMG_MNT уже подключён"
        return 0
    fi

    if [ "\$MODE" = "ext4" ]; then
        # udisks мог уже смонтировать раздел в /media/... — сначала отключаем,
        # иначе один и тот же раздел окажется смонтирован дважды (это опасно).
        local tgt
        while read -r tgt; do
            [ -n "\$tgt" ] || continue
            [ "\$tgt" = "\$IMG_MNT" ] && continue
            umount "\$tgt" 2>/dev/null && log "отключено: \$tgt"
        done < <(findmnt -n -o TARGET -S "\$dev" 2>/dev/null || true)

        if findmnt -n -o TARGET -S "\$dev" >/dev/null 2>&1; then
            log "раздел \$dev всё ещё смонтирован в другом месте — пропускаю"
            return 0
        fi

        mkdir -p "\$IMG_MNT"
        if mount -o rw,noatime "\$dev" "\$IMG_MNT"; then
            log "диск \$dev подключён к \$IMG_MNT"
        else
            log "не удалось подключить \$dev"
        fi
        return 0
    fi

    # safe: exFAT-диск, внутри него — ext4-образ
    local existing
    existing="\$(findmnt -n -o TARGET -S "\$dev" 2>/dev/null | head -n1 || true)"
    if [ -n "\$existing" ]; then
        DISK_MNT="\$existing"
    else
        mkdir -p "\$DISK_MNT"
        if ! mount -t exfat -o "rw,uid=\$DISK_UID,gid=\$DISK_GID,umask=000,noatime" "\$dev" "\$DISK_MNT"; then
            log "не удалось подключить \$dev"
            return 0
        fi
        log "диск \$dev подключён к \$DISK_MNT"
    fi

    local img="\$DISK_MNT/\$IMG_REL"
    if [ ! -f "\$img" ]; then
        log "образ \$img не найден — пропускаю"
        return 0
    fi

    mkdir -p "\$IMG_MNT"
    if mount -o loop,noatime "\$img" "\$IMG_MNT"; then
        log "образ подключён к \$IMG_MNT"
    else
        log "не удалось подключить образ \$img"
    fi
    return 0
}

down() {
    if mountpoint -q "\$IMG_MNT"; then
        umount "\$IMG_MNT" && log "отключён \$IMG_MNT"
    fi
    if [ "\$MODE" = "safe" ]; then
        local dev tgt
        dev="\$(blkid -U "\$DISK_UUID" 2>/dev/null || true)"
        if [ -n "\$dev" ]; then
            tgt="\$(findmnt -n -o TARGET -S "\$dev" 2>/dev/null | head -n1 || true)"
            if [ -n "\$tgt" ]; then
                umount "\$tgt" 2>/dev/null && log "отключён \$tgt"
            fi
        fi
    fi
    return 0
}

case "\${1:-up}" in
    up)   up "\${2:-}" ;;
    down) down ;;
    *)    echo "usage: \$0 {up [--wait]|down}"; exit 2 ;;
esac
HELPER_EOF

chmod +x "$HELPER"
c_ok "Скрипт-помощник: $HELPER"

cat > /etc/systemd/system/valera-storage.service <<'UNIT_EOF'
[Unit]
Description=VALERA: внешнее хранилище (диск D) — окружение, кэши, модель
After=local-fs.target
Before=multi-user.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/bin/valera-storage.sh up --wait
ExecStop=/usr/local/bin/valera-storage.sh down

[Install]
WantedBy=multi-user.target
UNIT_EOF

systemctl daemon-reload
systemctl enable valera-storage.service >/dev/null 2>&1
c_ok "Служба valera-storage.service включена (запуск при загрузке)"
echo ""

# ── [6/8] Переменные окружения ───────────────────────────────────────────────
echo "[6/8] Настройка переменных окружения..."

cat > /etc/profile.d/valera-storage.sh <<PROFILE_EOF
# VALERA — внешнее хранилище (диск D)
# Переменные выставляются только если хранилище реально подключено
if [ -d "$IMG_MNT/conda-envs" ]; then
    export CONDA_ENVS_PATH="$IMG_MNT/conda-envs"
    export CONDA_PKGS_DIRS="$IMG_MNT/conda-pkgs"
    export PIP_CACHE_DIR="$IMG_MNT/pip-cache"
    export HF_HOME="$IMG_MNT/hf-cache"
    export VALERA_STORAGE="$IMG_MNT"
    if [ -w "$IMG_MNT/tmp" ]; then
        export TMPDIR="$IMG_MNT/tmp"
    fi
fi
PROFILE_EOF
chmod 644 /etc/profile.d/valera-storage.sh
c_ok "/etc/profile.d/valera-storage.sh"

# ~/.condarc — conda по умолчанию ставит окружения на внешний диск
CONDARC="$REAL_HOME/.condarc"
if ! grep -q "$IMG_MNT/conda-envs" "$CONDARC" 2>/dev/null; then
    cp -a "$CONDARC" "$CONDARC.bak" 2>/dev/null || true
    cat > "$CONDARC" <<CONDARC_EOF
envs_dirs:
  - $IMG_MNT/conda-envs
pkgs_dirs:
  - $IMG_MNT/conda-pkgs
CONDARC_EOF
fi
chown "$REAL_UID_N:$REAL_GID_N" "$CONDARC"
c_ok "$CONDARC (envs_dirs / pkgs_dirs)"

# ~/.bashrc — переменные и в не-логин шеллах
BASHRC="$REAL_HOME/.bashrc"
if ! grep -q "profile.d/valera-storage.sh" "$BASHRC" 2>/dev/null; then
    cat >> "$BASHRC" <<'BASHRC_EOF'

# VALERA — внешнее хранилище (диск D)
if [ -f /etc/profile.d/valera-storage.sh ]; then
    . /etc/profile.d/valera-storage.sh
fi
BASHRC_EOF
    chown "$REAL_UID_N:$REAL_GID_N" "$BASHRC"
    c_ok "~/.bashrc обновлён"
fi

# Старый кэш pip на внутреннем диске больше не нужен
if [ -d "$REAL_HOME/.cache/pip" ]; then
    OLD_PIP="$(du -sh "$REAL_HOME/.cache/pip" 2>/dev/null | cut -f1)"
    rm -rf "$REAL_HOME/.cache/pip"
    c_ok "Удалён старый кэш pip на внутреннем диске ($OLD_PIP)"
fi
echo ""

# ── [7/8] Проверка ───────────────────────────────────────────────────────────
echo "[7/8] Проверка..."
# shellcheck disable=SC1091
. /etc/profile.d/valera-storage.sh
c_ok "CONDA_ENVS_PATH = ${CONDA_ENVS_PATH:-(не задано)}"
c_ok "PIP_CACHE_DIR   = ${PIP_CACHE_DIR:-(не задано)}"
c_ok "HF_HOME         = ${HF_HOME:-(не задано)}"
c_ok "TMPDIR          = ${TMPDIR:-(не задано)}"
echo ""

# ── [8/8] Итог ───────────────────────────────────────────────────────────────
echo "[8/8] Итоговое состояние:"
echo ""
df -h / "$IMG_MNT" | sed 's/^/  /'
echo ""
echo "  Хранилище:   $IMG_MNT"
echo "  Модели:      $IMG_MNT/models  (ссылка: $PROJECT_DIR/models)"
echo "  Управление:  sudo valera-storage.sh up | down"
echo "  Служба:      sudo systemctl status valera-storage.service"
echo ""
echo "============================================"
echo " Готово! Перезапустите терминал (или: source ~/.bashrc)"
echo " и запустите установку заново:"
echo ""
echo "   bash setup_jetson6.sh"
echo "============================================"
echo ""
