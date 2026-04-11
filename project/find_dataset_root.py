from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}


def _is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXTS


def _count_videos(dir_path: str, max_scan: int = 5000) -> int:
    """
    Counts video files directly under dir_path (non-recursive).
    max_scan limits enumeration to keep it fast on huge folders.
    """
    try:
        n = 0
        for i, name in enumerate(os.listdir(dir_path)):
            if i >= max_scan:
                break
            if _is_video(name):
                n += 1
        return n
    except Exception:
        return 0


def _norm(s: str) -> str:
    return s.strip().lower().replace("-", "_").replace(" ", "_")


FIGHT_NAMES = {"fight", "fights", "violent", "violence"}
NONFIGHT_NAMES = {"nonfight", "non_fight", "no_fight", "nofight", "normal", "non_violent", "nonviolent"}


@dataclass(frozen=True)
class Candidate:
    root: str
    fight_dir: str
    nonfight_dir: str
    fight_videos: int
    nonfight_videos: int


def find_candidates(
    search_root: str,
    *,
    min_videos_per_class: int = 10,
    limit: int = 20,
) -> List[Candidate]:
    """
    Heuristic scan:
    - Looks for sibling folders named like fight/nonfight/normal under the same parent.
    - Counts video files directly under those class folders.
    Returns parent folder paths that can be used as --data_root for train_rwf2000.py.
    """
    search_root = os.path.abspath(search_root)
    out: List[Candidate] = []

    for dirpath, dirnames, _filenames in os.walk(search_root):
        # Light pruning for speed
        low = dirpath.lower()
        if any(x in low for x in [r"\.git", r"\__pycache__", r"\.venv", r"\venv", r"\site-packages"]):
            dirnames[:] = []
            continue

        normed = {_norm(d): d for d in dirnames}
        fight_key = next((k for k in normed.keys() if k in FIGHT_NAMES), None)
        nonfight_key = next((k for k in normed.keys() if k in NONFIGHT_NAMES), None)
        if not fight_key or not nonfight_key:
            continue

        fight_dir = os.path.join(dirpath, normed[fight_key])
        nonfight_dir = os.path.join(dirpath, normed[nonfight_key])
        fv = _count_videos(fight_dir)
        nv = _count_videos(nonfight_dir)
        if fv < min_videos_per_class or nv < min_videos_per_class:
            continue

        out.append(
            Candidate(
                root=dirpath,
                fight_dir=fight_dir,
                nonfight_dir=nonfight_dir,
                fight_videos=fv,
                nonfight_videos=nv,
            )
        )
        if len(out) >= limit:
            break

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Find likely fight/nonfight dataset root folders on disk.")
    ap.add_argument("--search_root", type=str, default=os.path.expanduser("~"), help="Where to search (default: user home)")
    ap.add_argument("--min_videos", type=int, default=10, help="Min videos per class folder (default: 10)")
    ap.add_argument("--limit", type=int, default=20, help="Max candidates to print (default: 20)")
    args = ap.parse_args()

    cands = find_candidates(args.search_root, min_videos_per_class=args.min_videos, limit=args.limit)
    if not cands:
        print("No candidates found.")
        print("Tip: If you haven't extracted the dataset yet, extract it so you have e.g. <root>/fight and <root>/nonfight folders.")
        return

    print("Candidates (use root as --data_root for train_rwf2000.py):")
    for i, c in enumerate(cands, start=1):
        print(
            f"{i:2d}) root={c.root}\n"
            f"    fight={c.fight_dir} ({c.fight_videos} videos)\n"
            f"    nonfight={c.nonfight_dir} ({c.nonfight_videos} videos)\n"
        )


if __name__ == "__main__":
    main()

