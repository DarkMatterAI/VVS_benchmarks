#!/usr/bin/env python
"""
Create (or delete) score-plugin rows on the backend.

• Reads backend URL and auth from .env
• Skips creation if a plugin with the same **name + group_key** already exists.
• Persists {name: id} in plugin_ids.json for the consumer.
"""
import os, json, requests, sys
from pathlib import Path
from dotenv import load_dotenv; load_dotenv()
from rich import print

BACKEND   = os.getenv("BACKEND_URL", "http://backend:8000")
MAP_PATH  = Path(os.getenv("PLUGIN_MAP_PATH", "/plugin_map/plugin_ids.json"))
GROUP_KEY = "benchmark_score"

# ---------------------------------------------------------------
PLUGINS = [
    ("synthemol_rf", {"timeout": 100, "max_concurrency": 64, "batch_size": 8}),
    ("docking_2zdt", {"timeout": 400, "max_concurrency": 64}),
    ("docking_6lud", {"timeout": 400, "max_concurrency": 64}),
    ("rocs_2chw",    {"timeout": 400, "max_concurrency": 64}),
    ("bench_dummy",  {"timeout": 15,  "max_concurrency": 64}),
    ("erbb1_mlp",    {"timeout": 20, "batch_size": 1024, 
                      "max_concurrency": 64, "group_key": "benchmark_score_gpu"}),
]

# ---------------------------------------------------------------
def list_existing() -> dict[str, int]:
    """Return {name: id} for existing score plugins in the group."""
    url = f"{BACKEND}/api/v1/plugins/"
    params = {
        "plugin_type": "score",
        "group_key": GROUP_KEY,
        "skip": 0,
        "limit": 1000,
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status(), r.text
    return {p["name"]: p["id"] for p in r.json()}

def create_plugins():
    existing = list_existing()
    mapping  = {}

    for name, cfg in PLUGINS:
        if name in existing:
            pid = existing[name]
            print(f"[yellow]• {name} already exists → id {pid} - skipped")
        else:
            payload = {
                "name": name,
                "type": "score",
                "plugin_class": "generic",
                "execution_type": "queue",
                "group_key": GROUP_KEY,
            }
            payload.update(cfg)
            r = requests.post(f"{BACKEND}/api/v1/plugins", json=payload, timeout=15)
            print(r.text)
            r.raise_for_status()
            pid = r.json()["id"]
            print(f"[green]✓ created {name} → id {pid}")
        mapping[name] = pid

    MAP_PATH.write_text(json.dumps(mapping))
    print(f"[bold]Saved mapping → {MAP_PATH}")

def delete_plugins():
    if not MAP_PATH.exists():
        print("[yellow]No plugin_ids.json found - nothing to delete")
        return

    mapping = json.loads(MAP_PATH.read_text())
    for name, pid in mapping.items():
        url = f"{BACKEND}/api/v1/plugins/{pid}"
        r = requests.delete(url, timeout=10)
        if r.status_code == 204:
            print(f"[red]✗ deleted {name} (id {pid})")
        else:
            print(f"[yellow]• could not delete {name} (id {pid}) - status {r.status_code}")
    MAP_PATH.unlink(missing_ok=True)

# ---------------------------------------------------------------
if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "create"
    if action == "create":
        create_plugins()
    elif action == "delete":
        delete_plugins()
    else:
        print("Usage: create_records.py [create|delete]")
        sys.exit(1)
