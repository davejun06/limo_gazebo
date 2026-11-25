#!/usr/bin/env python3
"""Sends SIGINT to window 6 about 4 s after motion, as Ctrl-C would. usage (in the container, windows 1-5 up): stop_test.py"""
import os, signal, subprocess, time
p = subprocess.Popen(['bash', 'tools/run/6_start.sh', '1'], start_new_session=True, preexec_fn=lambda: [signal.signal(signal.SIGINT, signal.SIG_DFL), signal.signal(signal.SIGTERM, signal.SIG_DFL)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
import threading
log = []; seen = {}
def rd():
    for l in p.stdout:
        log.append(l); print(l.rstrip()[:130], flush=True)
        if 'motion at' in l: seen['m'] = time.time()
        if 'stopping' in l: seen['s'] = time.time()
threading.Thread(target=rd, daemon=True).start()
while 'm' not in seen and p.poll() is None: time.sleep(0.1)
if p.poll() is not None: raise SystemExit(f'ended before motion, rc {p.returncode}')
time.sleep(4.0)
ign = [l for l in open(f'/proc/{p.pid}/status') if l.startswith('SigIgn')][0].strip(); bit = int(ign.split()[1], 16) >> 1 & 1
print('script', ign, 'INT ignored', bool(bit), flush=True)
if bit: os.killpg(p.pid, signal.SIGTERM); raise SystemExit('INT ignored, test void')
t0 = time.time(); os.killpg(p.pid, signal.SIGINT); print('SIGINT sent', flush=True)
try: rc = p.wait(30)
except subprocess.TimeoutExpired: rc = 'still running after 30 s'
time.sleep(0.5)
print('rc', rc, 'stopping printed after s', round(seen['s'] - t0, 1) if 's' in seen else None, flush=True)
