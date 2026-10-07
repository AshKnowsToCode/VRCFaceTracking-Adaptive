"""Plot expressions over time from a .vftrec recording: raw module output vs final output, mic level, markers and
tracking-loss spans.

    python plot_vftrec.py 20261007-201500.vftrec                     # JawOpen and LipPucker, interactive
    python plot_vftrec.py FILE --shapes JawOpen MouthClosed --out jaw.png --start 60 --end 120

A name that is not an expression is averaged over the expressions starting with it (LipPucker -> 4 shapes).
"""

from __future__ import annotations

import argparse

import numpy as np

from vftrec import Recording, read, read_session


def tracking_loss_spans(rec: Recording) -> list[tuple[float, float, str]]:
    """(start s, end s, state) for every stretch where the recorder saw all-zero or frozen tracking."""
    spans, start, state = [], None, None
    for t, detail in rec.events_of("tracking"):
        if detail != "ok" and start is None:
            start, state = t, detail
        elif detail == "ok" and start is not None:
            spans.append((start, t, state))
            start = None
    if start is not None and len(rec.tick):
        spans.append((start, float(rec.t[-1]), state))
    return spans


def plot(rec: Recording, shapes: list[str], start: float | None, end: float | None):
    import matplotlib.pyplot as plt

    t = rec.t
    mask = np.ones(len(t), dtype=bool)
    if start is not None:
        mask &= t >= start
    if end is not None:
        mask &= t <= end

    has_audio = len(rec.audio_db) > 0
    rows = len(shapes) + (1 if has_audio else 0)
    fig, axes = plt.subplots(rows, 1, sharex=True, figsize=(14, 2.4 * rows + 0.8), squeeze=False)
    axes = axes[:, 0]

    for ax, name in zip(axes, shapes):
        ax.plot(t[mask], rec.group(name, "raw")[mask], lw=0.8, color="#3a6ea5", label="raw (module)")
        ax.plot(t[mask], rec.group(name, "out")[mask], lw=0.8, color="#d1495b", alpha=0.85, label="out (after mutations)")
        ax.set_ylabel(name)
        ax.set_ylim(-0.05, max(1.05, float(np.nanmax(rec.group(name, "out")[mask], initial=0)) + 0.05))
        ax.grid(alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=8)

    if has_audio:
        at = rec.audio_t
        amask = np.ones(len(at), dtype=bool)
        if start is not None:
            amask &= at >= start
        if end is not None:
            amask &= at <= end
        ax = axes[-1]
        ax.plot(at[amask], rec.audio_db[amask], lw=0.5, color="#5b8c5a")
        ax.set_ylabel("mic dBFS")
        ax.set_ylim(-125, 0)   # silence is clamped to -120
        ax.grid(alpha=0.25)

    for ax in axes:
        for s, e, _ in tracking_loss_spans(rec):
            ax.axvspan(s, e, color="0.5", alpha=0.2, lw=0)
        for tick, _ in rec.markers:
            ax.axvline(rec.seconds(tick), color="0.2", ls="--", lw=0.7)
    for tick, label in rec.markers:
        axes[0].annotate(label, (rec.seconds(tick), 1.0), xycoords=("data", "axes fraction"),
                         rotation=90, va="top", ha="right", fontsize=7)

    axes[-1].set_xlabel("seconds since session start (grey = tracking lost/frozen, dashed = marker)")
    fig.suptitle(", ".join(p.name for p in rec.paths) + (" (truncated)" if rec.truncated else ""), fontsize=10)
    fig.tight_layout()
    return fig


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("file")
    parser.add_argument("--shapes", nargs="+", default=["JawOpen", "LipPucker"])
    parser.add_argument("--session", action="store_true", help="include the session's other files")
    parser.add_argument("--start", type=float, help="seconds")
    parser.add_argument("--end", type=float, help="seconds")
    parser.add_argument("--out", help="save to this image file instead of showing a window")
    args = parser.parse_args(argv)

    if args.out:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rec = read_session(args.file) if args.session else read(args.file)
    fig = plot(rec, args.shapes, args.start, args.end)
    if args.out:
        fig.savefig(args.out, dpi=110)
        print(f"wrote {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
