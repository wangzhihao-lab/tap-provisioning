"""Convert the in-browser sign-in extraction (xxt_signins_part*.json) into one CSV per sign-in.

Each JSON record holds one sign-in activity with the sorted check-in epoch times (ms) and no
student identifiers. The CSVs carry only 签到时间 and 发起时间 so that the standard
``xxt_signin`` loader and ``signin_sessions.py`` can consume them unchanged.
"""

import argparse
import json
import logging
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
TZ = timezone(timedelta(hours=8))  # Asia/Shanghai, the platform's wall clock


def _slug(text: str) -> str:
    return re.sub(r"[^\w一-鿿-]+", "_", text).strip("_")


def convert(json_paths: list[Path], out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for jp in json_paths:
        data = json.loads(jp.read_text(encoding="utf-8"))
        for rec in data["records"]:
            times = [datetime.fromtimestamp(t / 1000.0, TZ) for t in rec["times"]]
            if not times:
                continue
            start = rec.get("startTime") or ""
            stem = f"{_slug(rec['course'])}__{_slug(rec['className'])}__{times[0]:%Y%m%d-%H%M}__{rec['activeId']}"
            df = pd.DataFrame({"签到时间": [t.strftime("%Y-%m-%d %H:%M:%S") for t in times],
                               "发起时间": [start] * len(times)})
            df.to_csv(out_dir / f"{stem}.csv", index=False, encoding="utf-8-sig")
            rows.append({"file": stem, "course": rec["course"], "class": rec["className"],
                         "students": rec.get("students"), "signed": rec["signed"],
                         "unsigned": rec.get("unsigned"), "start": start})
    index = pd.DataFrame(rows)
    index.to_csv(out_dir / "_index.csv", index=False, encoding="utf-8-sig")
    logger.info("%d activities written to %s", len(index), out_dir)
    return index


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    convert([Path(p) for p in args.json], Path(args.out))


if __name__ == "__main__":
    main()
