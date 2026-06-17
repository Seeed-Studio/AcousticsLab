// Hand-written decoder mirroring the {envelope,audio_stream,inference_stream}.proto schemas;
// field numbers and wire types must stay in lockstep when those schemas change.
// tag = (field_number << 3) | wire_type; wire 0=varint, 1=fixed64, 2=length-delimited, 5=fixed32.
// Unknown tags are silently skipped per proto3 unknown-field semantics.

export interface TopK {
  class_idx: number;
  label: string;
  prob: number;
}

export interface AudioFrame {
  seq: number;
  t_us_capture_monotonic: number | null;
  t_us_publish_unix: number | null;
  sample_rate: number | null;
  frame_duration_ms: number | null;
  codec: 'opus' | 'pcm_s16' | 'flac' | null;
  payload: Uint8Array | null;
}

export interface InferenceFrame {
  seq: number;
  t_us_capture_monotonic: number | null;
  t_us_publish_unix: number | null;
  top_k: TopK[];
  head_id: string | null;
  head_version: number | null;
}

export type EnvelopePayload =
  | { kind: 'audio'; audio: AudioFrame }
  | { kind: 'inference'; inference: InferenceFrame }
  | { kind: 'unknown' };

// A non-streaming TextDecoder is stateless across .decode() calls, so one shared instance avoids per-call allocation.
const UTF8_DECODER = new TextDecoder('utf-8');

class Reader {
  private readonly buf: Uint8Array;
  private readonly view: DataView;
  private pos: number;
  private readonly end: number;

  constructor(buf: Uint8Array, start = 0, end = buf.length) {
    this.buf = buf;
    this.view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    this.pos = start;
    this.end = end;
  }

  remaining(): boolean {
    return this.pos < this.end;
  }

  // JS number loses precision above 2^53, but the microsecond timestamp fields stay below that for any realistic process lifetime.
  varint(): number {
    let result = 0;
    let shift = 0;
    while (this.pos < this.end) {
      const byte = this.buf[this.pos++];
      result += (byte & 0x7f) * 2 ** shift;
      if ((byte & 0x80) === 0) return result;
      shift += 7;
      if (shift > 63) throw new Error('varint overflow');
    }
    throw new Error('varint truncated');
  }

  bytesView(): Uint8Array {
    const len = this.varint();
    if (this.pos + len > this.end) throw new Error('length-delimited truncated');
    const out = this.buf.subarray(this.pos, this.pos + len);
    this.pos += len;
    return out;
  }

  bytesCopy(): Uint8Array {
    return new Uint8Array(this.bytesView());
  }

  string(): string {
    return UTF8_DECODER.decode(this.bytesView());
  }

  float32(): number {
    if (this.pos + 4 > this.end) throw new Error('float truncated');
    const v = this.view.getFloat32(this.pos, true);
    this.pos += 4;
    return v;
  }

  skip(wire: number): void {
    switch (wire) {
      case 0:
        this.varint();
        return;
      case 1:
        if (this.pos + 8 > this.end) throw new Error('fixed64 truncated');
        this.pos += 8;
        return;
      case 2: {
        const len = this.varint();
        if (this.pos + len > this.end) throw new Error('len truncated');
        this.pos += len;
        return;
      }
      case 5:
        if (this.pos + 4 > this.end) throw new Error('fixed32 truncated');
        this.pos += 4;
        return;
      default:
        throw new Error(`unsupported wire type ${wire}`);
    }
  }
}

export function decodeEnvelope(bytes: Uint8Array): EnvelopePayload {
  const r = new Reader(bytes);
  while (r.remaining()) {
    const tag = r.varint();
    const field = tag >>> 3;
    const wire = tag & 7;
    if (field === 10 && wire === 2) {
      return { kind: 'audio', audio: decodeAudioFrame(r.bytesView()) };
    }
    if (field === 11 && wire === 2) {
      return { kind: 'inference', inference: decodeInferenceFrame(r.bytesView()) };
    }
    r.skip(wire);
  }
  return { kind: 'unknown' };
}

function decodeAudioFrame(bytes: Uint8Array): AudioFrame {
  const r = new Reader(bytes);
  const f: AudioFrame = {
    seq: 0,
    t_us_capture_monotonic: null,
    t_us_publish_unix: null,
    sample_rate: null,
    frame_duration_ms: null,
    codec: null,
    payload: null
  };
  while (r.remaining()) {
    const tag = r.varint();
    const field = tag >>> 3;
    const wire = tag & 7;
    switch (field) {
      case 1:
        f.seq = r.varint();
        break;
      case 2:
        f.t_us_capture_monotonic = r.varint();
        break;
      case 4:
        f.sample_rate = r.varint();
        break;
      case 5:
        f.frame_duration_ms = r.varint();
        break;
      case 6:
        f.t_us_publish_unix = r.varint();
        break;
      case 10:
        f.codec = 'opus';
        f.payload = r.bytesCopy();
        break;
      case 11:
        f.codec = 'pcm_s16';
        f.payload = r.bytesCopy();
        break;
      case 12:
        f.codec = 'flac';
        f.payload = r.bytesCopy();
        break;
      default:
        r.skip(wire);
    }
  }
  return f;
}

function decodeInferenceFrame(bytes: Uint8Array): InferenceFrame {
  const r = new Reader(bytes);
  const f: InferenceFrame = {
    seq: 0,
    t_us_capture_monotonic: null,
    t_us_publish_unix: null,
    top_k: [],
    head_id: null,
    head_version: null
  };
  while (r.remaining()) {
    const tag = r.varint();
    const field = tag >>> 3;
    const wire = tag & 7;
    switch (field) {
      case 1:
        f.seq = r.varint();
        break;
      case 2:
        f.t_us_capture_monotonic = r.varint();
        break;
      case 4:
        f.top_k.push(decodeTopK(r.bytesView()));
        break;
      case 5:
        f.head_id = r.string();
        break;
      case 7:
        f.t_us_publish_unix = r.varint();
        break;
      case 9:
        f.head_version = r.varint();
        break;
      default:
        r.skip(wire);
    }
  }
  return f;
}

function decodeTopK(bytes: Uint8Array): TopK {
  const r = new Reader(bytes);
  const t: TopK = { class_idx: 0, label: '', prob: 0 };
  while (r.remaining()) {
    const tag = r.varint();
    const field = tag >>> 3;
    const wire = tag & 7;
    switch (field) {
      case 1:
        t.class_idx = r.varint();
        break;
      case 2:
        t.label = r.string();
        break;
      case 3:
        t.prob = r.float32();
        break;
      default:
        r.skip(wire);
    }
  }
  return t;
}
