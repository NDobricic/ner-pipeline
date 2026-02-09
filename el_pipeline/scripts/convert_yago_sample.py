"""
Convert YAGO sample TSV (RDF-style) to KB JSONL format.

The TSV format has lines like:
    yago:Entity_Name    rdfs:label      "Label"
    yago:Entity_Name    rdf:type        yago:Type

This script extracts rdfs:label as title and rdf:type as description.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(
        description="Convert YAGO sample TSV to KB JSONL."
    )
    ap.add_argument("tsv", help="Path to yago-sample.tsv")
    ap.add_argument("output", help="Output JSONL path")
    ap.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Optional limit on number of entries",
    )
    return ap.parse_args()


def clean_entity_id(raw: str) -> str:
    """Clean entity ID, decoding URL-encoded characters."""
    entity_id = raw.strip()
    if entity_id.startswith("yago:"):
        entity_id = entity_id[5:]
    entity_id = entity_id.replace("__u0028_", "(").replace("_u0029_", ")")
    entity_id = entity_id.replace("_", " ")
    return entity_id


def clean_label(raw: str) -> str:
    """Extract label from quoted string."""
    raw = raw.strip()
    if raw.startswith('"') and '"' in raw[1:]:
        end_quote = raw.index('"', 1)
        return raw[1:end_quote]
    return raw.strip('"')


def clean_type(raw: str) -> str:
    """Clean type value."""
    raw = raw.strip()
    for prefix in ["yago:", "schema:"]:
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
    raw = raw.replace("_", " ")
    if raw.startswith("Q") and raw[1:].split("_")[0].isdigit():
        parts = raw.split(" ", 1)
        if len(parts) > 1:
            raw = parts[1]
    return raw


def main():
    args = parse_args()
    src = Path(args.tsv)
    out = Path(args.output)
    
    entities = defaultdict(lambda: {"labels": [], "types": [], "alt_names": []})
    
    print(f"Reading {src}...")
    with src.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            
            subj, pred, obj = parts[0], parts[1], parts[2]
            entity_id = subj.strip()
            
            if pred == "rdfs:label":
                label = clean_label(obj)
                entities[entity_id]["labels"].append(label)
            elif pred == "rdf:type":
                type_val = clean_type(obj)
                if type_val and type_val not in entities[entity_id]["types"]:
                    entities[entity_id]["types"].append(type_val)
            elif pred == "schema:alternateName":
                alt = clean_label(obj)
                if alt:
                    entities[entity_id]["alt_names"].append(alt)
    
    print(f"Found {len(entities)} unique entities")
    
    count = 0
    with out.open("w", encoding="utf-8") as w:
        for entity_id, data in entities.items():
            if not data["labels"]:
                continue
            
            title = data["labels"][0]
            
            desc_parts = []
            if data["types"]:
                desc_parts.append(f"Type: {', '.join(data['types'][:3])}")
            if data["alt_names"]:
                desc_parts.append(f"Also known as: {', '.join(data['alt_names'][:3])}")
            description = ". ".join(desc_parts) if desc_parts else ""
            
            clean_id = clean_entity_id(entity_id)
            
            rec = {
                "id": entity_id,
                "title": title,
                "description": description,
                "clean_id": clean_id,
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
            
            if args.max_entries and count >= args.max_entries:
                break
    
    print(f"Wrote {count} entries to {out}")


if __name__ == "__main__":
    main()

