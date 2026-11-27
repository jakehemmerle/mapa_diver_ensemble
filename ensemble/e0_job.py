"""Unit body for ``launch/e0.sh``: export a wave's members, then run the E0 check.

Takes two argument lists for ``ensemble.run`` (export, then e0), each joined by the
ASCII unit separator, so that one systemd unit covers both steps in order.
"""
from __future__ import annotations

import sys

from ensemble import run

if __name__ == "__main__":
    for packed in sys.argv[1:3]:
        run.main([part for part in packed.split("\x1f") if part])
