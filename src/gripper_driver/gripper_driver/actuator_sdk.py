
"""
Actuator SDK (Modbus RTU over RS-485)--平动手，协作手通用
-------------------------------------
A lightweight Python SDK for controlling the actuator using minimalmodbus,
with clean read/write helpers, thread safety, and convenience methods for
both "temporary-zone" moves and multi-point moves.

Dependencies:
    pip install minimalmodbus pyserial

Author: ChatGPT
"""

import time
import threading
import minimalmodbus
import serial

# -----------------------------
# Register Map (Holding, 0x03/0x06/0x10)
# -----------------------------
# Enable / Disable
REG_ENABLE                = 0x0100

# Temporary-zone motion (write)
REG_TMP_POS_H             = 0x0102  # high 16
REG_TMP_POS_L             = 0x0103  # low  16
REG_TMP_SPEED             = 0x0104  # 0~100 (% of max speed @ 0x0305)
REG_TMP_FORCE             = 0x0105  # 0~100 (% of max torque @ 0x0306)
REG_TMP_ACCEL             = 0x0106
REG_TMP_DECEL             = 0x0107
REG_TMP_TRIGGER           = 0x0108  # 0:idle, 1:trigger

# Multipoint (write)
REG_CMD_UPDATE_MODE       = 0x010F  # 0: immediate, 1: ignore updates until motion ends
REG_MULTI_MODE            = 0x0110  # 0: sequence, 1: loop, 2: select
REG_MULTI_START_SEG       = 0x0111
REG_MULTI_END_SEG         = 0x0112
REG_MULTI_RESUME_POLICY   = 0x0113  # 0: resume remaining, 1: restart from start
REG_MULTI_LOOP_COUNT      = 0x0114  # 0xFFFF for infinite
REG_MULTI_SELECT_SEG      = 0x0116  # valid when REG_MULTI_MODE == 2
REG_MULTI_TRIGGER         = 0x0117  # 0:idle, 1:trigger
REG_MULTI_PAUSE           = 0x0118  # 0:idle, 1:pause

# -----------------------------
# Status / Feedback (Read Only via 0x03)
# -----------------------------
REG_TORQUE_REACHED        = 0x0601  # 0/1
REG_POS_REACHED           = 0x0602  # 0/1
REG_SPEED_MAX_REACHED     = 0x0603  # 0/1
REG_READY                 = 0x0604  # 0/1 (torque OR position reached)
REG_CURR_LOOP_COUNT       = 0x0606  # current loop count in multipoint motion
REG_CURR_SEG              = 0x0607  # current running segment
REG_POS_FB_H              = 0x0609  # high 16 of 32-bit pos feedback
REG_POS_FB_L              = 0x060A  # low  16 of 32-bit pos feedback
REG_SPEED_FB              = 0x060B
REG_CURRENT_FB            = 0x060C  # torque current feedback
REG_ALARM                 = 0x0612  # bitmask: 0x01 overtemp, 0x02 stall, 0x04 overspeed, 0x08 init fault, 0x10 limit, 0x20 drop
REG_PARAM_CHANGED         = 0x0614  # 0/1: unsaved parameters exist

class ActuatorSDK:
    def __init__(self, port: str, slave_id: int = 1, baudrate: int = 115200, timeout: float = 0.3):
        """Create a Modbus RTU instrument.

        Args:
            port: Serial port, e.g., 'COM4' on Windows or '/dev/ttyUSB0' on Linux.
            slave_id: Modbus slave address (1 by default).
            baudrate: Serial baudrate, default 115200.
            timeout: Read/write timeout in seconds.
        """
        self.instrument = minimalmodbus.Instrument(port, slave_id)
        self.instrument.serial.baudrate = baudrate
        self.instrument.serial.bytesize = 8
        self.instrument.serial.parity   = serial.PARITY_NONE
        self.instrument.serial.stopbits = 1
        self.instrument.serial.timeout  = timeout
        self.instrument.mode = minimalmodbus.MODE_RTU
        self._lock = threading.Lock()

    # ------------- Low-Level Helpers -------------
    def _w1(self, addr: int, value: int):
        """Write a single 16-bit holding register (function 0x06)."""
        with self._lock:
            return self.instrument.write_register(addr, value, functioncode=6)

    def _wn(self, addr: int, values):
        """Write multiple 16-bit holding registers starting at addr (function 0x10)."""
        with self._lock:
            return self.instrument.write_registers(addr, list(values))

    def _r(self, addr: int, count: int = 1):
        """Read one or more 16-bit holding registers (function 0x03)."""
        with self._lock:
            return self.instrument.read_registers(addr, count, functioncode=3)

    # ------------- Enable / Disable -------------
    def enable(self, enable: bool = True):
        """Enable or disable the actuator."""
        return self._w1(REG_ENABLE, 1 if enable else 0)

    # ------------- Temporary-Zone Motion -------------
    def set_temp_position_mm(self, position_mm: int):
        """Set temporary target position (32-bit) via high/low registers.
        position_mm: integer position in mm (device units per documentation).
        """
        if position_mm < 0 or position_mm > 0xFFFFFFFF:
            raise ValueError("position_mm must be within 0..0xFFFFFFFF")
        hi = (position_mm >> 16) & 0xFFFF
        lo = position_mm & 0xFFFF
        # Write two registers via function 0x10
        return self._wn(REG_TMP_POS_H, [hi, lo])

    def set_temp_speed_pct(self, speed_pct: int):
        """Set temporary speed as percent of max [0..100]."""
        if not (0 <= speed_pct <= 100):
            raise ValueError("speed_pct must be 0..100")
        return self._w1(REG_TMP_SPEED, speed_pct)

    def set_temp_force_pct(self, force_pct: int):
        """Set temporary torque/force as percent of max [0..100]."""
        if not (0 <= force_pct <= 100):
            raise ValueError("force_pct must be 0..100")
        return self._w1(REG_TMP_FORCE, force_pct)

    def set_temp_accel(self, accel: int):
        return self._w1(REG_TMP_ACCEL, accel)

    def set_temp_decel(self, decel: int):
        return self._w1(REG_TMP_DECEL, decel)

    def trigger_temp_move(self):
        """Trigger motion based on temporary-zone registers."""
        return self._w1(REG_TMP_TRIGGER, 1)

    def temp_move(self, position_mm: int, speed_pct: int = 100, force_pct: int = 60, accel: int = 2000, decel: int = 2000, trigger: bool = True):
        """Convenience: set all temp-zone params and (optionally) trigger."""
        self.set_temp_position_mm(position_mm)
        self.set_temp_speed_pct(speed_pct)
        self.set_temp_force_pct(force_pct)
        self.set_temp_accel(accel)
        self.set_temp_decel(decel)
        if trigger:
            self.trigger_temp_move()

    # ------------- Multipoint Motion -------------
    def set_cmd_update_mode(self, mode: int):
        """0: immediate updates; 1: ignore updates until motion ends."""
        if mode not in (0, 1):
            raise ValueError("mode must be 0 or 1")
        return self._w1(REG_CMD_UPDATE_MODE, mode)

    def set_multi_mode(self, mode: int):
        """0: sequence, 1: loop, 2: select."""
        if mode not in (0, 1, 2):
            raise ValueError("mode must be 0, 1 or 2")
        return self._w1(REG_MULTI_MODE, mode)

    def set_multi_range(self, start_seg: int, end_seg: int):
        self._w1(REG_MULTI_START_SEG, start_seg)
        return self._w1(REG_MULTI_END_SEG, end_seg)

    def set_multi_resume_policy(self, policy: int):
        """0: resume remaining; 1: restart from start."""
        if policy not in (0, 1):
            raise ValueError("policy must be 0 or 1")
        return self._w1(REG_MULTI_RESUME_POLICY, policy)

    def set_multi_loop_count(self, count: int):
        """Set loop count; 0xFFFF means infinite loop."""
        if not (0 <= count <= 0xFFFF):
            raise ValueError("count must be 0..0xFFFF")
        return self._w1(REG_MULTI_LOOP_COUNT, count)

    def set_multi_select_segment(self, seg: int):
        return self._w1(REG_MULTI_SELECT_SEG, seg)

    def trigger_multi(self):
        return self._w1(REG_MULTI_TRIGGER, 1)

    def pause_multi(self):
        return self._w1(REG_MULTI_PAUSE, 1)

    # ------------- Status / Feedback -------------
    def _read_bool(self, addr: int) -> bool:
        return bool(self._r(addr, 1)[0])

    def torque_reached(self) -> bool:
        return self._read_bool(REG_TORQUE_REACHED)

    def position_reached(self) -> bool:
        return self._read_bool(REG_POS_REACHED)

    def speed_max_reached(self) -> bool:
        return self._read_bool(REG_SPEED_MAX_REACHED)

    def ready(self) -> bool:
        return self._read_bool(REG_READY)

    def current_loop_count(self) -> int:
        return self._r(REG_CURR_LOOP_COUNT, 1)[0]

    def current_segment(self) -> int:
        return self._r(REG_CURR_SEG, 1)[0]

    def feedback_position(self) -> int:
        hi, lo = self._r(REG_POS_FB_H, 2)
        return ((hi & 0xFFFF) << 16) | (lo & 0xFFFF)

    def feedback_speed(self) -> int:
        return self._r(REG_SPEED_FB, 1)[0]

    def feedback_current(self) -> int:
        return self._r(REG_CURRENT_FB, 1)[0]

    def read_alarm(self) -> int:
        return self._r(REG_ALARM, 1)[0]

    def param_changed(self) -> bool:
        return self._read_bool(REG_PARAM_CHANGED)

    # ------------- Utilities -------------
    def wait_until_ready(self, timeout: float = 5.0, poll: float = 0.02) -> bool:
        """Poll REG_READY until True or timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if self.ready():
                    return True
            except (minimalmodbus.NoResponseError, serial.SerialException):
                # transient: ignore and retry within timeout
                pass
            time.sleep(poll)
        return False

    def wait_until_pos_or_torque(self, timeout: float = 5.0, poll: float = 0.02) -> str:
        """Wait until either position_reached or torque_reached, or timeout.
        Returns: 'position', 'torque', or 'timeout'.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if self.position_reached():
                    return 'position'
                if self.torque_reached():
                    return 'torque'
            except (minimalmodbus.NoResponseError, serial.SerialException):
                pass
            time.sleep(poll)
        return 'timeout'


if __name__ == "__main__":
    # Example usage (adjust port/slave to your setup)
    PORT = "/dev/ttyUSB0"            # e.g., '/dev/ttyUSB0' on Linux
    SLAVE_ID = 1
    sdk = ActuatorSDK(PORT, SLAVE_ID, baudrate=115200, timeout=0.5)

    try:
        sdk.enable(True)
        # Temporary-zone move to 20.0 mm (example), 100% speed, 60% torque
        sdk.temp_move(position_mm=2000, speed_pct=100, force_pct=60, accel=2000, decel=2000, trigger=True)
        result = sdk.wait_until_pos_or_torque(timeout=10.0)
        print("Move result:", result)
        print("Feedback pos:", sdk.feedback_position())
        print("Alarm bits  :", hex(sdk.read_alarm()))

        # Multipoint skeleton (segments taught in HMI)
        # sdk.set_multi_mode(0)              # 0: sequence
        # sdk.set_multi_range(1, 3)          # run segment 1..3
        # sdk.set_multi_loop_count(1)        # one pass (use 0xFFFF for infinite)
        # sdk.trigger_multi()
        # sdk.wait_until_ready(timeout=10.0)

    except Exception as e:
        print("Error:", e)