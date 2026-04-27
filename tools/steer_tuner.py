"""
PSU Eco Racing — Steering Tuner
tools/steer_tuner.py

Real-time live plot of speed and steering from the Nucleo LLC over USB serial.
Run on any laptop connected to the Nucleo debug USB port.

Usage:
    python tools/steer_tuner.py

Requirements:
    pip install pyserial matplotlib

The Nucleo transmits at 500 ms intervals:
    Target: X.XX | Actual: X.XX km/h | Angle: N deg | SteerTarget: N | tick:NNNN
"""

import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import re

PORT = '/dev/tty.usbmodem11103'   # macOS — change to COMx on Windows or /dev/ttyACMx on Linux
BAUD = 115200
MAX_POINTS = 200

target_speed_data = deque([0.0] * MAX_POINTS, maxlen=MAX_POINTS)
actual_speed_data = deque([0.0] * MAX_POINTS, maxlen=MAX_POINTS)
actual_angle_data = deque([0.0] * MAX_POINTS, maxlen=MAX_POINTS)
target_angle_data = deque([0.0] * MAX_POINTS, maxlen=MAX_POINTS)

ser = serial.Serial(PORT, BAUD, timeout=1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Speed plot
line_target_speed, = ax1.plot([], [], 'r-', label='Target km/h', linewidth=2)
line_actual_speed, = ax1.plot([], [], 'b-', label='Actual km/h', linewidth=2)
ax1.set_ylim(-1, 35)
ax1.set_xlim(0, MAX_POINTS)
ax1.legend(loc='upper left')
ax1.set_title('Speed Control — Target vs Actual')
ax1.set_ylabel('Speed (km/h)')
ax1.grid(True)

# Steering plot
LEFT_LIMIT  = -49.0 - (-10.0)   # -39 deg
RIGHT_LIMIT =  16.0 - (-10.0)   # +26 deg
ax2.axhline(y=LEFT_LIMIT,  color='red',  linestyle='--', linewidth=1, label=f'Left limit ({LEFT_LIMIT:.0f}°)')
ax2.axhline(y=RIGHT_LIMIT, color='red',  linestyle='--', linewidth=1, label=f'Right limit ({RIGHT_LIMIT:.0f}°)')
ax2.axhline(y=0,           color='gray', linestyle=':', linewidth=1,  label='Center (0°)')
line_actual_angle, = ax2.plot([], [], 'b-', label='Actual angle', linewidth=2)
line_target_angle, = ax2.plot([], [], 'r--', label='Target angle', linewidth=1.5)
ax2.set_ylim(LEFT_LIMIT - 5, RIGHT_LIMIT + 5)
ax2.set_xlim(0, MAX_POINTS)
ax2.legend(loc='upper left')
ax2.set_title('Steering Angle — Target vs Actual')
ax2.set_ylabel('Angle (degrees)')
ax2.set_xlabel('Samples')
ax2.grid(True)

angle_text  = ax2.text(0.02, 0.85, '', transform=ax2.transAxes,
                       fontsize=12, fontweight='bold', color='darkblue')
target_text = ax2.text(0.02, 0.70, '', transform=ax2.transAxes,
                       fontsize=12, fontweight='bold', color='red')


def update(frame):
    try:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        match = re.search(
            r'Target:\s*([\d.]+).*Actual:\s*([\d.]+).*Angle:\s*(-?[\d]+).*SteerTarget:\s*(-?[\d]+)',
            line
        )
        if match:
            target_speed = float(match.group(1))
            actual_speed = float(match.group(2))
            actual_angle = int(match.group(3))
            target_angle = int(match.group(4))

            target_speed_data.append(target_speed)
            actual_speed_data.append(actual_speed)
            actual_angle_data.append(actual_angle)
            target_angle_data.append(target_angle)

            line_target_speed.set_data(range(MAX_POINTS), list(target_speed_data))
            line_actual_speed.set_data(range(MAX_POINTS), list(actual_speed_data))
            line_actual_angle.set_data(range(MAX_POINTS), list(actual_angle_data))
            line_target_angle.set_data(range(MAX_POINTS), list(target_angle_data))

            color = ('red'    if actual_angle <= LEFT_LIMIT + 3  or actual_angle >= RIGHT_LIMIT - 3  else
                     'orange' if actual_angle <= LEFT_LIMIT + 8  or actual_angle >= RIGHT_LIMIT - 8  else
                     'darkblue')
            angle_text.set_text(f'Actual: {actual_angle}°')
            angle_text.set_color(color)
            target_text.set_text(f'Target: {target_angle}°')

    except Exception:
        pass

    return (line_target_speed, line_actual_speed,
            line_actual_angle, line_target_angle,
            angle_text, target_text)


ani = animation.FuncAnimation(fig, update, interval=50, blit=True)
plt.tight_layout()
plt.show()
ser.close()
