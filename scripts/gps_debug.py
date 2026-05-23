#!/usr/bin/env python3
"""
gps_debug.py  —  Diagnose SIM7600/SIM7670 GPS module on Raspberry Pi
=====================================================================
Usage:
    python gps_debug.py                     # scan all likely ports
    python gps_debug.py --port /dev/ttyUSB2 # test a specific port
    python gps_debug.py --raw               # print raw serial bytes
"""
import argparse, time, re, sys, os
import glob

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    sys.exit("ERROR: pyserial not installed.\n  pip install pyserial --break-system-packages")


def find_serial_ports():
    """Return all likely USB serial ports on this system."""
    candidates = (
        glob.glob("/dev/ttyUSB*") +
        glob.glob("/dev/ttyACM*") +
        glob.glob("/dev/ttyS*")
    )
    # Also list what pyserial sees
    detected = [p.device for p in serial.tools.list_ports.comports()]
    all_ports = sorted(set(candidates + detected))
    return [p for p in all_ports if os.path.exists(p)]


def send_at(ser, cmd, wait=1.0, read_bytes=512):
    """Send AT command and return response string."""
    ser.reset_input_buffer()
    ser.write((cmd + "\r\n").encode())
    time.sleep(wait)
    raw = ser.read(read_bytes)
    return raw.decode(errors="replace")


def test_port(port, baud=115200, raw_mode=False):
    print(f"\n{'='*55}")
    print(f"  Testing: {port}  @ {baud} baud")
    print(f"{'='*55}")

    try:
        ser = serial.Serial(port, baud, timeout=3)
    except Exception as e:
        print(f"  ✗  Cannot open: {e}")
        return False

    # Force a clean GPS start
    # send_at(ser, "AT+CGPS=0", wait=1)
    # send_at(ser, "AT+CGPS=1", wait=2)

    # ── Step 1: Basic AT ──────────────────────────────────────────────
    resp = send_at(ser, "AT")
    if "OK" in resp:
        print("  ✓  AT → OK  (module is responding)")
    else:
        print(f"  ✗  AT → no OK.  Raw: {repr(resp[:80])}")
        ser.close()
        return False

    if raw_mode:
        print("\n  [Raw mode — Ctrl-C to stop]")
        try:
            while True:
                if ser.in_waiting:
                    print(repr(ser.read(ser.in_waiting)))
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        ser.close()
        return True

    # ── Step 2: Module ID ─────────────────────────────────────────────
    resp = send_at(ser, "ATI")
    model = resp.strip().splitlines()
    print(f"  Module: {' / '.join(l.strip() for l in model if l.strip())[:80]}")

    # ── Step 3: Enable GPS ────────────────────────────────────────────
    print("\n  Enabling GPS (AT+CGPS=1)...")
    resp = send_at(ser, "AT+CGPS=1", wait=1.5)
    if "OK" in resp or "ERROR" in resp:
        status = "OK" if "OK" in resp else f"ERROR ({resp.strip()[:40]})"
        print(f"  AT+CGPS=1 → {status}")
    else:
        print(f"  AT+CGPS=1 → no response: {repr(resp[:60])}")

    # ── Step 4: Check GPS power state ────────────────────────────────
    resp = send_at(ser, "AT+CGPS?")
    print(f"  GPS power state: {resp.strip()[:60]}")

    # ── Step 5: Poll CGPSINFO several times ───────────────────────────
    print("\n  Polling AT+CGPSINFO (10 attempts, 2s apart)...")
    print("  (Note: a cold-start GPS can take 1-10 minutes to get a fix)")
    got_fix = False
    for i in range(10):
        resp = send_at(ser, "AT+CGPSINFO", wait=2.0)
        for line in resp.splitlines():
            line = line.strip()
            if line.startswith("+CGPSINFO:"):
                if ",,,,,,,," in line or line == "+CGPSINFO: ":
                    print(f"  [{i+1:2d}] No fix yet — waiting for satellites...")
                else:
                    print(f"  [{i+1:2d}] ✓  RAW: {line}")
                    fix = parse_cgpsinfo(line)
                    if fix:
                        print(f"       lat={fix['lat']:.6f}  lon={fix['lon']:.6f}"
                              f"  alt={fix['alt_m']}m  speed={fix['speed_km']} km/h")
                        got_fix = True
                break
        else:
            print(f"  [{i+1:2d}] No +CGPSINFO line in response: {repr(resp[:60])}")
        time.sleep(1)

    # ── Step 6: Try NMEA output mode as fallback ──────────────────────
    # if not got_fix:
    #    print("\n  Trying NMEA stream (AT+CGPSINFOCFG) as fallback...")
    #   resp = send_at(ser, "AT+CGPSINFOCFG=1,31", wait=2.0, read_bytes=1024)
    #   lines = [l.strip() for l in resp.splitlines() if l.startswith("$GP")]
    #    if lines:
    #        print(f"  ✓  NMEA received: {lines[0]}")
    #    else:
    #        print(f"  ✗  No NMEA lines.  Raw: {repr(resp[:120])}")
    #    send_at(ser, "AT+CGPSINFOCFG=0,31")  # stop NMEA

    ser.close()

    if got_fix:
        print("\n  ✅  GPS working — fix obtained!")
    else:
        print("\n  ⚠️  No fix yet.  See checklist below.")

    return got_fix


def parse_cgpsinfo(line):
    m = re.search(
        r'\+CGPSINFO:\s*(\d+\.\d+),([NS]),(\d+\.\d+),([EW]),'
        r'(\d*),([\d\.]*),([\d\.]*),([\d\.]*)', line)
    if not m:
        return None
    lr, ld, lor, lod, date, utc, alt, spd = m.groups()
    lat = float(lr[:2]) + float(lr[2:]) / 60
    lon = float(lor[:3]) + float(lor[3:]) / 60
    if ld == "S": lat = -lat
    if lod == "W": lon = -lon
    return {"lat": round(lat,6), "lon": round(lon,6),
            "alt_m": float(alt) if alt else None,
            "speed_km": round(float(spd)*1.852, 1) if spd else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default=None,
                   help="Serial port to test (default: scan all likely ports)")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--raw", action="store_true",
                   help="Print raw bytes from the port (for deep debugging)")
    args = p.parse_args()

    print("\n=== GPS Diagnostic ===\n")

    # Show all serial ports
    ports = find_serial_ports()
    print(f"Available serial ports: {ports if ports else 'none found'}")

    if args.port:
        test_port(args.port, args.baud, args.raw)
    else:
        # Try likely GPS ports first
        priority = ["/dev/ttyUSB2", "/dev/ttyUSB1", "/dev/ttyUSB0"]
        ordered  = priority + [p for p in ports if p not in priority]
        found_any = False
        for port in ordered:
            if os.path.exists(port):
                ok = test_port(port, args.baud, args.raw)
                if ok:
                    found_any = True
                    break   # found a working GPS port, stop scanning
        if not found_any:
            print("\n" + "="*55)
            print("  No GPS found.  Checklist:")
            print("  1. Is the SIM7600 physically connected?")
            print("     lsusb  ← should show 'SIMCom' or similar")
            print("  2. Check which port it's on:")
            print("     ls -la /dev/ttyUSB*")
            print("  3. Do you have permission?")
            print("     sudo usermod -aG dialout $USER  (then log out/in)")
            print("  4. Is another process using the port?")
            print("     sudo fuser /dev/ttyUSB2")
            print("  5. Is the GPS module powered? Check 5V supply.")
            print("  6. Outdoors / clear sky view needed for satellite fix.")
            print("="*55)


if __name__ == "__main__":
    main()
