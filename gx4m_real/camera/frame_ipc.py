"""Messages of a JSON header plus raw numpy arrays over a multiprocessing Connection.

No pickle: the two processes may run different conda environments and numpy versions.
"""
import json
import struct

import numpy as np


def send(connection, header: dict, *arrays):
    arrays = [np.ascontiguousarray(a) for a in arrays]
    head = json.dumps({**header, 'arrays': [[a.dtype.str, list(a.shape)] for a in arrays]}).encode()
    connection.send_bytes(b''.join([struct.pack('<I', len(head)), head, *(a.tobytes() for a in arrays)]))


def receive(connection):
    """(header, arrays); the arrays are writable copies."""
    data = connection.recv_bytes()
    (size,) = struct.unpack_from('<I', data)
    header = json.loads(data[4:4 + size])
    offset, arrays = 4 + size, []
    for dtype, shape in header.pop('arrays'):
        dtype, count = np.dtype(dtype), int(np.prod(shape))
        arrays.append(np.frombuffer(data, dtype, count, offset).reshape(shape).copy())
        offset += count * dtype.itemsize
    if offset != len(data):
        raise ValueError('Message size does not match its header')
    return header, arrays
