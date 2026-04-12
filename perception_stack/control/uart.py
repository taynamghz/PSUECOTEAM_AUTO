"""
PSU Eco Racing — Perception Stack
control/uart.py  |  UART transport layer to the low-level controller (Nucleo).

TX protocol — 5-byte binary frame sent Jetson → Nucleo:
    [0xAA]  start byte
    [LEN ]  payload length = 2
    [CMD ]  command  0x00=IDLE  0x01=THROTTLE  0x02=BRAKE  0x03=STEER
    [DATA]  value    0-255
    [CRC8]  CRC-8/SMBUS over [LEN, CMD, DATA]  (poly=0x07, init=0x00)

RX protocol — speed telemetry sent Nucleo → Jetson (background reader thread):
    [0xBB]  start byte
    [0x02]  payload length = 2
    [0x10]  CMD_SPEED_REPORT
    [DATA]  speed in tenths of km/h  (e.g. 153 = 15.3 km/h, max 25.5 km/h)
    [CRC8]  CRC-8/SMBUS over [0x02, 0x10, DATA]

The reader thread continuously parses incoming bytes.  main thread reads
self.speed_kmh at any time — no blocking, no polling.

Watchdog: Nucleo expects a valid TX packet every 200 ms or falls back to manual.
De-duplication: identical (cmd, val) pairs are not retransmitted.
"""

import struct
import logging
import threading
import time
import serial

from perception_stack.config import (
    UART_PORT, UART_BAUD, UART_TIMEOUT_S, UART_ACK_TIMEOUT_S,
)

log = logging.getLogger(__name__)

# ── Command constants ───────────────────────────────────────────────────────────
CMD_IDLE     = 0x00
CMD_THROTTLE = 0x01
CMD_BRAKE    = 0x02
CMD_STEER    = 0x03   # DATA: 0=full-left, 127=centre, 255=full-right

CMD_SPEED_REPORT = 0x10   # RX-only: speed from Nucleo

_TX_START = 0xAA
_RX_START = 0xBB

_CMD_NAME = {CMD_IDLE: "IDLE", CMD_THROTTLE: "THROTTLE",
             CMD_BRAKE: "BRAKE", CMD_STEER: "STEER"}


# ── CRC-8/SMBUS ────────────────────────────────────────────────────────────────

def _crc8(data: bytes) -> int:
    crc = 0x00
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ 0x07
            else:
                crc <<= 1
        crc &= 0xFF
    return crc


def _build_frame(cmd: int, value: int) -> bytes:
    """Build 5-byte TX frame: [0xAA][LEN=2][CMD][DATA][CRC8]."""
    payload = bytes([2, cmd & 0xFF, value & 0xFF])
    return struct.pack("BBBBB", _TX_START, payload[0], payload[1], payload[2],
                       _crc8(payload))


class UARTController:
    """
    Thread-safe, non-blocking UART wrapper.

    TX: send(), set_speed(), brake(), steer()
    RX: speed_kmh property — updated continuously by the reader thread.

    Usage:
        uart = UARTController()
        uart.open()
        uart.set_speed(15.0)     # tell Nucleo PID: target = 15.0 km/h
        uart.steer(140)
        print(uart.speed_kmh)    # latest speed received from Nucleo
        uart.close()
    """

    def __init__(self):
        self._ser:        serial.Serial | None = None
        self._lock        = threading.Lock()        # protects _ser writes
        self._last_cmd:   int   = -1
        self._last_val:   int   = -1
        self.connected:   bool  = False

        # RX speed (written by reader thread, read by main thread)
        self._speed_kmh:  float = 0.0
        self._speed_lock  = threading.Lock()
        self._reader_thread: threading.Thread | None = None

    # ── Lifecycle ───────────────────────────────────────────────────────────────

    def open(self) -> bool:
        try:
            self._ser = serial.Serial(
                port     = UART_PORT,
                baudrate = UART_BAUD,
                bytesize = serial.EIGHTBITS,
                parity   = serial.PARITY_NONE,
                stopbits = serial.STOPBITS_ONE,
                timeout  = UART_TIMEOUT_S,
            )
            self.connected = True
            log.info("[UART] Opened %s @ %d baud", UART_PORT, UART_BAUD)
        except serial.SerialException as e:
            log.error("[UART] Open failed: %s", e)
            return False

        # Start background reader for Nucleo → Jetson telemetry
        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True, name="UARTReader")
        self._reader_thread.start()
        return True

    def close(self):
        if self._ser and self._ser.is_open:
            self.send(CMD_IDLE, 0)
            time.sleep(0.05)
            self._ser.close()
        self.connected = False
        log.info("[UART] Closed")

    # ── Speed property (thread-safe read) ───────────────────────────────────────

    @property
    def speed_kmh(self) -> float:
        with self._speed_lock:
            return self._speed_kmh

    # ── TX ──────────────────────────────────────────────────────────────────────

    def send(self, cmd: int, value: int = 0) -> bool:
        """
        Transmit one command frame.
        Skips retransmit of identical (cmd, value) pairs.
        """
        if not self.connected or self._ser is None:
            return False

        if cmd == self._last_cmd and value == self._last_val:
            return True

        frame = _build_frame(cmd, value)
        with self._lock:
            try:
                self._ser.write(frame)
                self._ser.flush()
            except serial.SerialException as e:
                log.error("[UART] Write error: %s", e)
                self.connected = False
                return False

        self._last_cmd  = cmd
        self._last_val  = value
        log.debug("[UART] TX %s val=%d", _CMD_NAME.get(cmd, f"0x{cmd:02X}"), value)
        return True

    def set_speed(self, kmh: float) -> bool:
        """
        Send a target speed setpoint to the Nucleo PID controller.
        Encoded as DATA = int(kmh * 10)  →  e.g. 150 = 15.0 km/h.
        Max representable speed: 25.5 km/h (DATA=255).
        The Nucleo drives throttle/brake internally to reach this setpoint.
        """
        data = int(max(0, min(255, round(kmh * 10.0))))
        return self.send(CMD_THROTTLE, data)

    def brake(self, value: int = 255) -> bool:
        """Send emergency-stop command (overrides Nucleo PID)."""
        return self.send(CMD_BRAKE, value)

    def idle(self) -> bool:
        return self.send(CMD_IDLE, 0)

    def steer(self, value: int = 127) -> bool:
        """
        Send a steering angle setpoint immediately.
        0 = full left (-STEER_MAX_DEG), 127 = straight (0°), 255 = full right (+STEER_MAX_DEG).
        """
        if not self.connected or self._ser is None:
            return False
        frame = _build_frame(CMD_STEER, value)
        with self._lock:
            try:
                self._ser.write(frame)
                self._ser.flush()
            except serial.SerialException as e:
                log.error("[UART] Write error: %s", e)
                self.connected = False
                return False
        log.debug("[UART] TX STEER val=%d", value)
        return True

    # ── RX reader thread ────────────────────────────────────────────────────────

    def _reader_loop(self) -> None:
        """
        Continuously parse incoming bytes from the Nucleo.

        Packet format:  [0xBB][0x02][0x10][DATA][CRC8]
        DATA = speed in tenths of km/h (e.g. 153 → 15.3 km/h).

        Tolerant to byte-level noise: scans for the 0xBB start byte, validates
        CRC before accepting any value.  Malformed packets are silently dropped.
        """
        buf = bytearray()
        PACKET_LEN = 5   # [start][len][cmd][data][crc]

        while self.connected and self._ser and self._ser.is_open:
            try:
                raw = self._ser.read(PACKET_LEN)
            except serial.SerialException:
                break
            if not raw:
                continue
            buf.extend(raw)

            # Consume all complete packets from buf
            while len(buf) >= PACKET_LEN:
                # Scan for start byte
                if buf[0] != _RX_START:
                    buf.pop(0)
                    continue

                packet = buf[:PACKET_LEN]
                _, plen, cmd, data, crc = packet
                expected_crc = _crc8(bytes([plen, cmd, data]))

                if crc != expected_crc or cmd != CMD_SPEED_REPORT:
                    buf.pop(0)   # bad packet — advance one byte and retry
                    continue

                # Valid speed packet
                speed = data / 10.0   # tenths → km/h
                with self._speed_lock:
                    self._speed_kmh = speed
                log.debug("[UART] RX speed=%.1f km/h", speed)
                del buf[:PACKET_LEN]
