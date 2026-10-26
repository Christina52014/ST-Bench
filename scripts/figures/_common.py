from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "build" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
