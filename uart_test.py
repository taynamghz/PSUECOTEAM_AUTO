"""
UART Manual Test — PSU Eco Racing
Run from AdhamTeam/:  python3 uart_test.py

Keys:
  s → enter steering angle in degrees  (-STEER_MAX to +STEER_MAX, 0 = centre)
  t → enter speed in km/h
  1 → MAX BRAKE   (CMD=0x02, val=255)
  0 → IDLE        (CMD=0x00, val=0)
  q → quit
"""

import sys, os, tty, termios, time, logging

ACK_TIMEOUT = 0.05   # seconds to wait for MCU echo

# ── Allow running without ZED SDK ─────────────────────────────────────────────
from unittest.mock import MagicMock
sys.modules.setdefault("pyzed",    MagicMock())
sys.modules.setdefault("pyzed.sl", MagicMock())

sys.path.insert(0, os.path.dirname(__file__))

from perception_stack.control.uart import (
    UARTController, CMD_IDLE, CMD_THROTTLE, CMD_BRAKE, CMD_STEER, _build_frame,
)
from perception_stack.config import UART_PORT, UART_BAUD, STEER_MAX_DEG

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("uart_test")

# ── Frame preview helper ───────────────────────────────────────────────────────
def show_frame(cmd, val):
    frame = _build_frame(cmd, val)
    hex_str = " ".join(f"{b:02X}" for b in frame)
    log.debug("  Raw frame bytes: %s", hex_str)

# ── ACK check ─────────────────────────────────────────────────────────────────
def check_ack(uart, cmd):
    ser = uart._ser
    ser.timeout = ACK_TIMEOUT
    ack = ser.read(1)
    if ack:
        log.info("    ✔ ACK received:  0x%02X  (expected CMD echo: 0x%02X)  %s",
                 ack[0], cmd,
                 "MATCH" if ack[0] == cmd else "MISMATCH — wrong byte echoed")
    else:
        log.warning("    ✘ NO ACK — LLC did not echo within %dms", int(ACK_TIMEOUT * 1000))

# ── Single-char read (no Enter needed) ────────────────────────────────────────
def getch():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

# ── Prompt for a value (needs Enter) ──────────────────────────────────────────
def prompt(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()
    return input()

# ── Degrees → steer byte ──────────────────────────────────────────────────────
def deg_to_byte(deg: float) -> int:
    deg = max(-STEER_MAX_DEG, min(STEER_MAX_DEG, deg))
    return int(round(127.0 - deg * 127.0 / STEER_MAX_DEG))

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*50}")
    print(f"  UART Manual Test")
    print(f"  Port : {UART_PORT}")
    print(f"  Baud : {UART_BAUD}")
    print(f"  Steer range: ±{STEER_MAX_DEG}°  (0 = centre)")
    print(f"{'='*50}")
    print("  [s] steer  — enter angle in degrees")
    print("  [t] speed  — enter speed in km/h")
    print("  [1] MAX BRAKE")
    print("  [0] IDLE")
    print("  [q] quit")
    print(f"{'='*50}\n")

    uart = UARTController()
    log.info("Opening %s @ %d baud ...", UART_PORT, UART_BAUD)

    if not uart.open():
        log.error("FAILED to open UART port.")
        log.error("  → Is %s present?  Run: ls /dev/ttyTHS*", UART_PORT)
        log.error("  → Permission?     Run: sudo usermod -aG dialout $USER")
        sys.exit(1)

    log.info("UART opened OK.  Sending IDLE as safe start...")
    show_frame(CMD_IDLE, 0)
    uart.idle()
    time.sleep(0.1)

    try:
        while True:
            log.info("Waiting for key press  (s=steer  t=speed  1=brake  0=idle  q=quit)...")
            key = getch()

            if key in ("s", "S"):
                sys.stdout.write("\n")
                raw = prompt(f"  Steering angle (degrees, ±{STEER_MAX_DEG}°, 0=centre): ")
                try:
                    deg = float(raw.strip())
                    byte = deg_to_byte(deg)
                    log.info(">>> STEER  %.1f°  → byte=%d", deg, byte)
                    show_frame(CMD_STEER, byte)
                    uart._last_cmd = -1
                    ok = uart.steer(byte)
                    log.info("    send() returned: %s", ok)
                    check_ack(uart, CMD_STEER)
                except ValueError:
                    log.warning("    Invalid input: %r — ignored", raw)

            elif key in ("t", "T"):
                sys.stdout.write("\n")
                raw = prompt("  Speed (km/h): ")
                try:
                    kmh = float(raw.strip())
                    log.info(">>> THROTTLE  %.1f km/h", kmh)
                    uart._last_cmd = -1
                    ok = uart.set_speed(kmh)
                    log.info("    send() returned: %s", ok)
                    check_ack(uart, CMD_THROTTLE)
                except ValueError:
                    log.warning("    Invalid input: %r — ignored", raw)

            elif key == "1":
                log.info(">>> KEY 1 — MAX BRAKE (val=255)")
                show_frame(CMD_BRAKE, 255)
                uart._last_cmd = -1
                ok = uart.send(CMD_BRAKE, 255)
                log.info("    send() returned: %s", ok)
                check_ack(uart, CMD_BRAKE)

            elif key == "0":
                log.info(">>> KEY 0 — IDLE")
                show_frame(CMD_IDLE, 0)
                uart._last_cmd = -1
                ok = uart.send(CMD_IDLE, 0)
                log.info("    send() returned: %s", ok)
                check_ack(uart, CMD_IDLE)

            elif key in ("q", "Q", "\x03"):
                log.info("Quit requested.")
                break

            else:
                log.debug("Unknown key: %r (ignored)", key)

    finally:
        log.info("Sending IDLE before exit...")
        uart._last_cmd = -1
        uart.send(CMD_IDLE, 0)
        time.sleep(0.05)
        uart.close()
        log.info("UART closed. Bye.")


if __name__ == "__main__":
    main()
