"""Исполнитель жестов для H1. На роботе файл размещается в ~/gesture_runner.py."""

import argparse
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="Запустить жест Валеры на H1")
    parser.add_argument("gesture", choices=("wave_hand", "shake_hand"))
    gesture = parser.parse_args(argv).gesture
    client = Path.home() / "unitree_sdk2/build/bin/h1_loco_client"
    # Клиент H1 сам выполняет обе стадии --shake_hand с паузой между ними.
    try:
        return subprocess.run([str(client), "--network_interface=eth0", "--" + gesture]).returncode
    except OSError as error:
        print(f"Не удалось запустить клиент H1: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
