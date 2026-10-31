"""live_gear36.py with every frame's pose sent over UDP to run_step1.py (the robot loop).

Each datagram is JSON {"sequence", "timestamp_monotonic" (frame receive time, this host's monotonic clock),
"status", "object_in_camera" (4x4, metres, camera optical frame; null when tracking is lost or rejected)}.
All live_gear36.py options pass through:

  bash tasks/dexlong_gear3/perception/run_publish.sh --target 127.0.0.1:5701 --no-display
"""
from __future__ import annotations

import argparse
import json
import socket

from live_gear36 import main as live_main


class UdpPosePublisher:
    def __init__(self, host, port):
        self.address = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def set_camera_metadata(self, metadata):
        pass

    def update(self, record, rgb=None, K=None):
        message = {k: record.get(k) for k in ('sequence', 'timestamp_monotonic', 'status', 'object_in_camera')}
        self.sock.sendto(json.dumps(message).encode(), self.address)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--target', default='127.0.0.1:5701', help='host:port of run_step1.py --pose-port')
    args, rest = parser.parse_known_args(argv)
    host, port = args.target.rsplit(':', 1)
    live_main(rest, observer=UdpPosePublisher(host, int(port)))


if __name__ == '__main__':
    main()
