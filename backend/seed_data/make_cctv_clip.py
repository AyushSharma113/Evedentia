"""Builds junction-camera-2.mp4, the CCTV clip of the seed case (evidence 4).

The clip is a schematic, top-down stand-in for the real footage: four minutes
(21:38:00 to 21:42:00) at 5 frames a second, following the times in the clip's
description in app/seed.py. The seed loads the committed file; run this only to
change it. Needs ffmpeg with libx264 on the PATH:

    python seed_data/make_cctv_clip.py
"""

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("junction-camera-2.mp4")
WIDTH, HEIGHT, FPS = 640, 360, 5
START = datetime(2026, 10, 3, 21, 38, tzinfo=UTC)  # shown as camera time, so read as naive
SECONDS = 240

# Ten pixels a metre; the junction centre is at (320, 180). x east, y south.
ROAD, LINE, GROUND = (62, 76, 89), (228, 231, 235), (31, 41, 51)
VAN, HATCHBACK, CYCLIST, PEDESTRIAN = (33, 134, 235), (245, 247, 250), (247, 140, 56), (62, 189, 147)


def path(t: float, keys: list[tuple[float, float, float]]) -> tuple[float, float]:
    """Position at time t, linear between (time, x, y) keyframes and held outside them."""
    if t <= keys[0][0]:
        return keys[0][1], keys[0][2]
    for (t0, x0, y0), (t1, x1, y1) in zip(keys, keys[1:]):
        if t <= t1:
            f = (t - t0) / (t1 - t0)
            return x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
    return keys[-1][1], keys[-1][2]


# Seconds from 21:38:00. Centres of each road user.
VAN_PATH = [(100, 302, 400), (108, 302, 260), (126, 302, 260), (127, 302, 250), (131, 302, 250), (132, 302, 191), (133.5, 305, 178)]
HATCHBACK_PATH = [(130, -19, 162), (132, 302, 162), (133.5, 385, 140)]
CYCLIST_PATH = [(0, 388, 112), (140, 388, 112), (141, 362, 104), (146, 362, -20)]
PEDESTRIAN_PATH = [(0, 402, 118)]


def box(frame: np.ndarray, cx: float, cy: float, w: int, h: int, colour: tuple[int, int, int]) -> None:
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    frame[max(y0, 0) : max(min(y0 + h, HEIGHT), 0), max(x0, 0) : max(min(x0 + w, WIDTH), 0)] = colour


def background() -> np.ndarray:
    frame = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:] = GROUND
    frame[130:230, :] = ROAD  # Ring Road, east-west
    frame[:, 270:370] = ROAD  # 5th Avenue, north-south
    for x in range(0, WIDTH, 40):  # Ring Road centre line, outside the junction
        if not 250 <= x <= 370:
            frame[179:181, x : x + 22] = LINE
    for y in range(0, HEIGHT, 40):  # 5th Avenue centre line
        if not 110 <= y <= 230:
            frame[y : y + 22, 319:321] = LINE
    frame[239:241, 270:320] = LINE  # northbound stop line
    return frame


def frames():
    base = background()
    for index in range(SECONDS * FPS):
        t = index / FPS
        frame = base.copy()
        box(frame, *path(t, PEDESTRIAN_PATH), 8, 8, PEDESTRIAN)
        box(frame, *path(t, CYCLIST_PATH), 6, 14, CYCLIST)
        box(frame, *path(t, VAN_PATH), 18, 40, VAN)
        box(frame, *path(t, HATCHBACK_PATH), 38, 17, HATCHBACK)
        yield frame.tobytes()


def main() -> None:
    epoch = int(START.timestamp())
    clock = f"%{{pts\\:gmtime\\:{epoch}\\:%Y-%m-%d %H\\\\\\:%M\\\\\\:%S}}"
    text = "fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf:fontsize=15:fontcolor=white"
    filters = ",".join(
        [
            f"drawtext={text}:text='CAM 2  RING RD / 5TH AVE  FACING N':x=12:y=10",
            f"drawtext={text}:text='{clock}':x=w-tw-12:y=10",
            f"drawtext={text}:fontsize=12:fontcolor=0x9aa5b1"
            ":text='Schematic stand-in for CCTV footage, generated for the demo case':x=12:y=h-22",
        ]
    )
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
        "-vf", filters,
        "-c:v", "libx264", "-preset", "veryslow", "-crf", "30", "-tune", "animation",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-map_metadata", "-1", "-fflags", "+bitexact",
        str(OUT),
    ]  # fmt: skip
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    for frame in frames():
        process.stdin.write(frame)
    process.stdin.close()
    if process.wait() != 0:
        raise SystemExit("ffmpeg failed")
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
