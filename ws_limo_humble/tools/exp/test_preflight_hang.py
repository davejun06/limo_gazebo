#!/usr/bin/env python3
"""Run preflight with every `ros2` call hanging; it must end by itself and report a failure."""
import runpy
import subprocess
import sys

real = subprocess.run


def fake(cmd, **kw):
    if isinstance(cmd, list) and cmd[0] == 'ros2':
        cmd = ['sleep', '100000']
    return real(cmd, **kw)


subprocess.run = fake
sys.argv = ['preflight.py', '--no-control', 'alpha1=0.2']
runpy.run_path('tools/exp/preflight.py', run_name='__main__')
