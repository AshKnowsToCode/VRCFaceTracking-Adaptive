"""Reader for VRCFaceTracking-Adaptive session recordings (.vftrec, schema 1).

The format is defined in VRCFaceTracking.Core/Adaptive/Recording/RecordingFormat.cs: one UTF-8 JSON header line,
then a raw deflate stream of little-endian records, each a type byte + int64 QPC tick + payload:

    F  float32[2 * block_length]   raw block (module output) then out block (after all mutations)
    A  float32                     mic RMS dBFS of one 10 ms block (tick = end of block)
    M  uint16 n, utf8[n]           marker label
    G  uint16 n, utf8[n]           event "kind:detail" (tracking, focus, modules, audio, recorder)

A block is header["frame"]["channels"] (eye + head) followed by header["expressions"].

    from vftrec import read
    rec = read("20261007-201500.vftrec")
    rec.t, rec.raw_col("JawOpen"), rec.out_col("JawOpen")

Command line: python vftrec.py info FILE [--session] [--shapes]
"""

from __future__ import annotations

import argparse
import json
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SUPPORTED_SCHEMAS = {1}

_FRAME, _AUDIO, _MARKER, _GATE = b"F"[0], b"A"[0], b"M"[0], b"G"[0]
_TICK = struct.Struct("<q")
_TEXT_LEN = struct.Struct("<H")
_AUDIO_DTYPE = np.dtype([("tick", "<i8"), ("db", "<f4")])


@dataclass
class Recording:
    """One .vftrec file, or several files of one session joined by read_session()."""

    paths: list[Path]
    header: dict
    tick: np.ndarray  # int64 (n,) QPC tick of each frame
    raw: np.ndarray  # float32 (n, block_length) module output, before any mutation
    out: np.ndarray  # float32 (n, block_length) after all mutations
    audio_tick: np.ndarray  # int64 (m,)
    audio_db: np.ndarray  # float32 (m,) mic RMS level in dBFS
    markers: list[tuple[int, str]] = field(default_factory=list)  # (tick, label)
    events: list[tuple[int, str, str]] = field(default_factory=list)  # (tick, kind, detail)
    truncated: bool = False  # file ended mid-stream (app killed or still recording)

    @property
    def frequency(self) -> int:
        return int(self.header["qpc_frequency"])

    @property
    def origin(self) -> int:
        """Tick of the session's first frame; times are seconds since then."""
        return int(self.header["session"]["qpc_start"])

    def seconds(self, ticks) -> np.ndarray:
        return (np.asarray(ticks, dtype=np.int64) - self.origin) / self.frequency

    @property
    def t(self) -> np.ndarray:
        return self.seconds(self.tick)

    @property
    def audio_t(self) -> np.ndarray:
        return self.seconds(self.audio_tick)

    @property
    def columns(self) -> list[str]:
        return list(self.header["frame"]["channels"]) + list(self.header["expressions"])

    @property
    def expressions(self) -> list[str]:
        return list(self.header["expressions"])

    def index(self, name: str) -> int:
        try:
            return self.columns.index(name)
        except ValueError:
            raise KeyError(f"no channel or expression named {name!r}") from None

    def raw_col(self, name: str) -> np.ndarray:
        return self.raw[:, self.index(name)]

    def out_col(self, name: str) -> np.ndarray:
        return self.out[:, self.index(name)]

    def group(self, name: str, block: str = "raw") -> np.ndarray:
        """Mean of the expressions whose name starts with `name` (e.g. "LipPucker" -> its 4 shapes), or the
        single channel/expression called `name`."""
        data = self.raw if block == "raw" else self.out
        if name in self.columns:
            return data[:, self.index(name)]
        members = [i for i, c in enumerate(self.columns) if c.startswith(name)]
        if not members:
            raise KeyError(f"no channel or expression starting with {name!r}")
        return data[:, members].mean(axis=1)

    def events_of(self, kind: str) -> list[tuple[float, str]]:
        """(seconds, detail) for every event of one kind, e.g. "tracking" or "focus"."""
        return [(float(self.seconds(tick)), detail) for tick, k, detail in self.events if k == kind]


def read_header(path) -> dict:
    with open(path, "rb") as f:
        line = f.readline()
    return _parse_header(line, path)


def _parse_header(line: bytes, path) -> dict:
    try:
        header = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"{path}: not a .vftrec file (bad header line)") from e
    if header.get("format") != "vftrec":
        raise ValueError(f"{path}: not a .vftrec file")
    if header.get("schema") not in SUPPORTED_SCHEMAS:
        raise ValueError(f"{path}: schema {header.get('schema')} not supported (reader knows {sorted(SUPPORTED_SCHEMAS)})")
    return header


def read(path) -> Recording:
    """Read one file. A truncated file (killed app, or one still being written) yields every complete record."""
    path = Path(path)
    data = path.read_bytes()
    newline = data.find(b"\n")
    if newline < 0:
        raise ValueError(f"{path}: no header line")
    header = _parse_header(data[:newline], path)

    inflater = zlib.decompressobj(-15)  # raw deflate
    try:
        body = inflater.decompress(data[newline + 1 :]) + inflater.flush()
    except zlib.error:
        body = b""
    truncated = not inflater.eof

    k = int(header["frame"]["block_length"])
    frame_dtype = np.dtype([("tick", "<i8"), ("values", "<f4", (2 * k,))])
    frame_size = 1 + frame_dtype.itemsize
    audio_size = 1 + _AUDIO_DTYPE.itemsize

    view = memoryview(body)
    frames, levels = [], []
    markers, events = [], []
    pos, end = 0, len(body)
    while pos < end:
        kind = body[pos]
        if kind == _FRAME:
            if pos + frame_size > end:
                truncated = True
                break
            frames.append(view[pos + 1 : pos + frame_size])
            pos += frame_size
        elif kind == _AUDIO:
            if pos + audio_size > end:
                truncated = True
                break
            levels.append(view[pos + 1 : pos + audio_size])
            pos += audio_size
        elif kind in (_MARKER, _GATE):
            if pos + 11 > end:
                truncated = True
                break
            (tick,) = _TICK.unpack_from(body, pos + 1)
            (length,) = _TEXT_LEN.unpack_from(body, pos + 9)
            if pos + 11 + length > end:
                truncated = True
                break
            text = body[pos + 11 : pos + 11 + length].decode("utf-8", "replace")
            if kind == _MARKER:
                markers.append((tick, text))
            else:
                event_kind, _, detail = text.partition(":")
                events.append((tick, event_kind, detail))
            pos += 11 + length
        else:
            raise ValueError(f"{path}: unknown record type {kind:#04x} at body offset {pos}")

    frame_array = np.frombuffer(b"".join(frames), dtype=frame_dtype)
    audio_array = np.frombuffer(b"".join(levels), dtype=_AUDIO_DTYPE)
    values = frame_array["values"].reshape(-1, 2 * k)
    return Recording(
        paths=[path],
        header=header,
        tick=frame_array["tick"].copy(),
        raw=values[:, :k].copy(),
        out=values[:, k:].copy(),
        audio_tick=audio_array["tick"].copy(),
        audio_db=audio_array["db"].copy(),
        markers=markers,
        events=events,
        truncated=truncated,
    )


def session_files(path) -> list[Path]:
    """All files in the same folder that belong to the same session (app run) as `path`, in order."""
    path = Path(path)
    session = read_header(path)["session"]["id"]
    found = []
    for candidate in path.parent.glob("*.vftrec"):
        try:
            header = read_header(candidate)
        except (OSError, ValueError):
            continue
        if header["session"]["id"] == session:
            found.append((header["session"]["file_index"], candidate))
    return [p for _, p in sorted(found)]


def read_session(path) -> Recording:
    """Read every file of the session that `path` belongs to (hourly rotation, idle gaps) as one Recording."""
    parts = [read(p) for p in session_files(path)]
    first = parts[0]
    return Recording(
        paths=[p for r in parts for p in r.paths],
        header=first.header,
        tick=np.concatenate([r.tick for r in parts]),
        raw=np.concatenate([r.raw for r in parts]),
        out=np.concatenate([r.out for r in parts]),
        audio_tick=np.concatenate([r.audio_tick for r in parts]),
        audio_db=np.concatenate([r.audio_db for r in parts]),
        markers=[m for r in parts for m in r.markers],
        events=[e for r in parts for e in r.events],
        truncated=any(r.truncated for r in parts),
    )


def _print_info(rec: Recording, shapes: bool) -> None:
    h = rec.header
    print(f"files:      {', '.join(p.name for p in rec.paths)}{'  (truncated)' if rec.truncated else ''}")
    print(f"app:        {h['app'].get('name')} {h['app'].get('version')}  schema {h['schema']}")
    print(f"started:    {h['clock']['utc']} UTC  session {h['session']['id'][:8]} file #{h['session']['file_index']}")
    modules = [m["name"] for m in h.get("modules", []) if m.get("active")]
    print(f"modules:    {', '.join(modules) or '(none active)'}")
    active = [m["name"] for m in h.get("mutations", []) if m.get("active")]
    print(f"mutations:  {' > '.join(active) or '(none active)'}")

    n = len(rec.tick)
    if n:
        duration = rec.t[-1] - rec.t[0]
        dt = np.diff(rec.tick) / rec.frequency * 1000
        rate = (n - 1) / duration if duration > 0 else float("nan")
        print(f"frames:     {n} over {duration:.1f} s ({rate:.1f} Hz)", end="")
        if len(dt):
            p50, p95, p99 = np.percentile(dt, [50, 95, 99])
            print(f"; interval ms p50 {p50:.1f} p95 {p95:.1f} p99 {p99:.1f} max {dt.max():.1f}")
        else:
            print()
    else:
        print("frames:     0")

    if len(rec.audio_db):
        loud = np.percentile(rec.audio_db, [10, 50, 90])
        print(f"mic level:  {len(rec.audio_db)} blocks; dBFS p10 {loud[0]:.1f} p50 {loud[1]:.1f} p90 {loud[2]:.1f}")
    else:
        print("mic level:  none")

    print(f"markers:    {len(rec.markers)}")
    for tick, label in rec.markers:
        print(f"  {rec.seconds(tick):9.2f} s  {label}")
    print(f"events:     {len(rec.events)}")
    for tick, kind, detail in rec.events:
        if kind == "modules":
            detail = ", ".join(m["name"] for m in json.loads(detail) if m.get("active")) or "(none active)"
        elif kind == "mutations":
            detail = " > ".join(json.loads(detail)) or "(none active)"
        print(f"  {rec.seconds(tick):9.2f} s  {kind}: {detail}")

    if shapes and n:
        print(f"\n{'expression':28s} {'raw p5':>7s} {'p50':>6s} {'p95':>6s} {'max':>6s}   {'out p5':>7s} {'p50':>6s} {'p95':>6s} {'max':>6s}")
        offset = len(h["frame"]["channels"])
        for i, name in enumerate(rec.expressions):
            r = rec.raw[:, offset + i]
            o = rec.out[:, offset + i]
            if np.nanmax(np.abs(r)) == 0 and np.nanmax(np.abs(o)) == 0:
                continue
            rp = np.nanpercentile(r, [5, 50, 95])
            op = np.nanpercentile(o, [5, 50, 95])
            print(f"{name:28s} {rp[0]:7.3f} {rp[1]:6.3f} {rp[2]:6.3f} {np.nanmax(r):6.3f}   "
                  f"{op[0]:7.3f} {op[1]:6.3f} {op[2]:6.3f} {np.nanmax(o):6.3f}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    info = sub.add_parser("info", help="summarise a recording")
    info.add_argument("file")
    info.add_argument("--session", action="store_true", help="include the session's other files")
    info.add_argument("--shapes", action="store_true", help="per-expression raw/out percentiles")
    args = parser.parse_args(argv)

    rec = read_session(args.file) if args.session else read(args.file)
    _print_info(rec, args.shapes)


if __name__ == "__main__":
    main()
