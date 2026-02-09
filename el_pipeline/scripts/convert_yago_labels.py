import argparse
import json
import re
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(
        description="Convert YAGO labels TSV to KB JSONL (id, title, description)."
    )
    ap.add_argument("tsv", help="Path to yagoLabels.tsv")
    ap.add_argument("output", help="Output JSONL path")
    ap.add_argument("--lang", default="eng", help="Language tag filter (default: eng)")
    ap.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Optional limit on number of entries (default: all)",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    src = Path(args.tsv)
    out = Path(args.output)

    label_re = re.compile(r'^"(.*)"@')

    def clean(val: str) -> str:
        if val.startswith("<") and val.endswith(">"):
            return val[1:-1]
        return val

    count = 0
    with src.open("r", encoding="utf-8", errors="ignore") as f, out.open(
        "w", encoding="utf-8"
    ) as w:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            subj, title_raw, pred, label_raw = parts[:4]
            if not pred.endswith("prefLabel"):
                continue
            if f"@{args.lang}" not in label_raw:
                continue
            m = label_re.match(label_raw.strip())
            if not m:
                continue
            label = m.group(1)
            entity_id = clean(subj)
            rec = {"id": entity_id, "title": label, "description": ""}
            w.write(json.dumps(rec) + "\n")
            count += 1
            if args.max_entries and count >= args.max_entries:
                break
    print(f"Wrote {count} entries to {out}")


if __name__ == "__main__":
    main()

