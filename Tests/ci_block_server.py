"""CI: real WebSocket server (Windows code) against the real iOS ChunkStreamer.

The first connection stores block 3 durably but drops the socket before its
ACK, emulating Wi-Fi loss after fsync. The phone must reconnect, resend that
block (idempotent, no duplicate) and only then delete it locally.
"""
import json
import sys
import threading
import time
from pathlib import Path

from chunk_protocol import BlockServer, decode_lossless_packet, open_reply

port = int(sys.argv[1]); store = Path(sys.argv[2]); raw_dir = Path(sys.argv[3])
done = threading.Event(); result = {}
connections = {'count': 0}


class FlakyServer(BlockServer):
    def handle(self, socket):
        connections['count'] += 1
        if connections['count'] > 1:
            return super().handle(socket)
        original, session, reply = open_reply(json.loads(socket.recv(timeout=15)))
        socket.send(json.dumps(reply))
        received = 0
        for message in socket:
            if isinstance(message, bytes):
                ack = self.store.accept(session, message)
                if received == 3:
                    socket.close()  # stored on disk, ACK lost
                    return
                socket.send(json.dumps({**ack, 'session': original}))
                received += 1


def committed(path, receipt):
    result.update(receipt); result['path'] = str(path); done.set()


server = FlakyServer(store, committed, host='127.0.0.1', port=port,
                     on_error=lambda session, error: print('server error:', session, error, flush=True))
server.start()
print('READY', flush=True)
if not done.wait(120):
    print('TIMEOUT: session never committed', flush=True); sys.exit(1)
session = result['session']
folder = store / 'Chunks' / session
blocks = sorted(folder.glob('*.b3dc'))
expected = sorted(raw_dir.glob('ws*.raw'))
assert result['frames'] == len(expected) == len(blocks), (result, len(expected), len(blocks))
previous = None
for block, raw in zip(blocks, expected):
    previous = decode_lossless_packet(block.read_bytes(), previous)
    assert previous == raw.read_bytes(), f'{block.name} not bit-exact'
assert connections['count'] >= 2, 'reconnection path was not exercised'
print(f'WS_OK session={session} frames={result["frames"]} connections={connections["count"]} bit-exact', flush=True)
server.close()
