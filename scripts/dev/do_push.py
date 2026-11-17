"""Push this repository to GitHub.

Security: the token is read from GH_TOKEN_MEM (environment) or a hidden prompt.
It is never written to disk, never stored in git config, and never persisted in a
remote URL -- it only exists in this process and in the ephemeral push argv.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def main() -> int:
    token = os.environ.get("GH_TOKEN_MEM", "")
    if not token:
        print("missing GH_TOKEN_MEM", file=sys.stderr)
        return 2

    login = "zzc1-nz"
    repo = "text2sql-fc-pipeline"
    url = f"https://{login}:{token}@github.com/{login}/{repo}.git"

    print("pushing HEAD:main ...")
    proc = subprocess.run(
        ["git", "-c", "credential.helper=",
         "-c", "http.proxy=http://127.0.0.1:7897",
         "push", url, "HEAD:main"],
        cwd=str(ROOT), capture_output=True, text=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    out = (proc.stdout + proc.stderr).replace(token, "***").strip()
    print(out[-1500:])
    if proc.returncode != 0:
        print(f"push FAILED (exit {proc.returncode})", file=sys.stderr)
        return 1
    print(f"\nOK -> https://github.com/{login}/{repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
