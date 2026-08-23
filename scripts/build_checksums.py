from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "SHA256SUMS.txt"


def main() -> None:
    lines = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == OUTPUT or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"Wrote {len(lines)} checksums to {OUTPUT.name}")


if __name__ == "__main__":
    main()
