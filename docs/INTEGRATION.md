# Integrate

The daemon pushes every classification result as a protobuf `Envelope` onto a Unix socket, so any local program can consume inference without HTTP, a browser, or a client library.

|           |                                                               |
| --------- | ------------------------------------------------------------- |
| Socket    | `/run/acousticslab/result.sock` - mode `0666`, any local user |
| Framing   | `[u32 little-endian length][Envelope bytes]`, length <= 65536 |
| Payload   | `Envelope` carrying an `InferenceFrame`                       |
| Rate      | one frame per inference stride - ~1 Hz by default             |
| Direction | push-only; never write to the socket                          |

The socket comes from `[output.inference]` in `/etc/acousticslab/launch.toml`; drop that section to disable it. Stride and class count (`hop_samples`, `top_k`) live in the workspace config and are tunable from the console.

## Generate bindings

Three schema files, importing only each other:

```
modules/proto/envelope.proto          # Envelope { oneof payload }
modules/proto/inference_stream.proto  # InferenceFrame, TopK
modules/proto/audio_stream.proto      # AudioFrame (WebSocket only)
```

```bash
protoc -I modules/proto --python_out=. modules/proto/*.proto
# --cpp_out / --go_out / --java_out work the same way
```

## Read the stream

Read 4 bytes, decode the little-endian length, read exactly that many bytes, parse them as an `Envelope`, dispatch on the payload - then repeat. The length prefix is the only sync point, so every error is terminal: close the socket and reconnect rather than trying to resynchronise.

```python
import socket, struct, sys
from envelope_pb2 import Envelope

SOCKET, MAX_FRAME = "/run/acousticslab/result.sock", 64 * 1024

def read_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None          # daemon closed the connection
        buf += chunk
    return buf

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect(SOCKET)

while True:
    header = read_exact(sock, 4)
    if header is None:
        break
    (size,) = struct.unpack("<I", header)
    if size > MAX_FRAME:
        sys.exit("oversized frame; stream is corrupt")
    body = read_exact(sock, size)
    if body is None:
        break

    env = Envelope()
    env.ParseFromString(body)
    if env.HasField("inference"):
        f = env.inference
        best = f.top_k[0]        # sorted descending by prob
        print(f"seq={f.seq} {best.label} {best.prob:.3f}")
```

```console
$ python3 consume.py
seq=6 _unknown_ 0.995
seq=7 _unknown_ 0.995
```

## Messages

`Envelope` holds `oneof payload { AudioFrame audio = 10; InferenceFrame inference = 11; }`. Only `inference` appears on this socket, but dispatch on the oneof anyway: new variants are additive, and unknown ones should be skipped.

`InferenceFrame`:

| Field                     | Meaning                                                                                 |
| ------------------------- | --------------------------------------------------------------------------------------- |
| `seq`                     | monotonic counter; restarts when the daemon restarts                                     |
| `top_k`                   | best classes, sorted descending by `prob` - `top_k[0]` is the prediction                 |
| `t_us_capture_monotonic`  | `CLOCK_MONOTONIC` µs of the first sample of the classified window; per-process           |
| `t_us_publish_unix`       | wall-clock µs at emit - correlates across processes, not for aligning audio              |
| `head_id`, `head_version` | which weights, and which generation of them, produced this frame; changes on a model swap |

`TopK` is `class_idx`, `label`, `prob`. Timestamp and head fields are `optional`: check presence rather than treating `0` as a value.

## Rules

- Read promptly: a consumer that stalls for 5 s, or falls behind the broadcast, is disconnected - a lagged reader cannot resync a length-prefixed stream, so the daemon closes it rather than corrupting it. Reconnect and carry on; gaps in `seq` say what you missed.
- Frames keep flowing during silence: an all-zero (muted/quiet) input classifies like any other audio and emits at the normal cadence, so "a frame arrived" does not mean "a sound happened" - read `top_k`, don't infer activity from frame arrival. A stream that stops emitting means no audio is reaching the daemon (device loss) or the engine is down - check `/api/v1/status` - never a quiet room.
- 16 consumers max on this socket; further connections are refused.
- Restarts reset `seq` and `t_us_capture_monotonic` - both are per-process, only comparable within one daemon run. `t_us_publish_unix` survives restarts.
- Never write to the socket; it is a one-way push.

## WebSocket consumers

The web console front (`acousticslab-webd`) streams the same `Envelope` over WebSocket - with no length prefix, since WebSocket frames the messages itself:

| Endpoint                        | Payload                           |
| ------------------------------- | --------------------------------- |
| `ws://<host>:8080/stream/infer` | `Envelope` -> `InferenceFrame`    |
| `ws://<host>:8080/stream/audio` | `Envelope` -> `AudioFrame` (Opus) |

Offer `Sec-WebSocket-Protocol: acousticslab.v1` on the upgrade. The console listens on loopback by default - see [INSTALL.md](INSTALL.md).
