"""Compare the remote tree against the local committed tree, file by file.

Answers one question: is what is on GitHub actually identical to what is in the
local repository? Uses the git trees API (recursive), so it compares blob SHAs --
an exact content check, not a filename check.
"""

from __future__ import annotations

import json
import os
import socket
import ssl
import subprocess
import sys
from pathlib import Path

PROXY = ("127.0.0.1", 7897)
API = "api.github.com"
REPO = "text2sql-fc-pipeline"
ROOT = Path(__file__).resolve().parent.parent.parent


def call(path: str, token: str):
    raw = socket.create_connection(PROXY, timeout=30)
    raw.sendall(f"CONNECT {API}:443 HTTP/1.1\r\nHost: {API}:443\r\n\r\n".encode())
    raw.recv(200)
    tls = ssl.create_default_context().wrap_socket(raw, server_hostname=API)
    req = (
        f"GET {path} HTTP/1.1\r\nHost: {API}\r\n"
        f"Authorization: Bearer {token}\r\n"
        "Accept: application/vnd.github+json\r\n"
        "X-GitHub-Api-Version: 2022-11-28\r\n"
        "User-Agent: fcdata-diff\r\nConnection: close\r\n\r\n"
    ).encode()
    tls.sendall(req)
    data = b""
    while True:
        chunk = tls.recv(65536)
        if not chunk:
            break
        data += chunk
    tls.close()
    head, _, bod = data.partition(b"\r\n\r\n")
    return int(head.split()[1]), json.loads(bod.decode("utf-8", "replace") or "{}")


def main() -> int:
    token = os.environ.get("GH_TOKEN_MEM", "")
    if not token:
        print("missing GH_TOKEN_MEM", file=sys.stderr)
        return 2

    st, me = call("/user", token)
    if st != 200:
        print(f"auth failed {st}", file=sys.stderr)
        return 1
    login = me["login"]

    st, repo = call(f"/repos/{login}/{REPO}", token)
    branch = repo.get("default_branch", "main")
    st, tree = call(f"/repos/{login}/{REPO}/git/trees/{branch}?recursive=1", token)
    if st != 200:
        print(f"tree lookup failed {st}: {tree.get('message')}", file=sys.stderr)
        return 1

    remote_blobs = {
        item["path"]: item["sha"]
        for item in tree.get("tree", [])
        if item["type"] == "blob"
    }
    print(f"remote blobs: {len(remote_blobs)}  (truncated={tree.get('truncated')})")

    # local: git ls-tree -r HEAD gives "<mode> blob <sha>\t<path>"
    out = subprocess.run(["git", "ls-tree", "-r", "HEAD"], cwd=str(ROOT),
                         capture_output=True, text=True, check=True).stdout
    local_blobs: dict[str, str] = {}
    for line in out.splitlines():
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if len(parts) >= 3 and parts[1] == "blob":
            local_blobs[path] = parts[2]
    print(f"local blobs : {len(local_blobs)}")

    only_remote = sorted(set(remote_blobs) - set(local_blobs))
    only_local = sorted(set(local_blobs) - set(remote_blobs))
    differ = sorted(p for p in set(local_blobs) & set(remote_blobs)
                    if local_blobs[p] != remote_blobs[p])

    print()
    if not (only_remote or only_local or differ):
        print("RESULT: remote content is IDENTICAL to local (all blob SHAs match)")
        print("        -> nothing to push; the repo is already up to date")
        return 0

    if only_remote:
        print(f"only on remote ({len(only_remote)}):")
        for p in only_remote[:20]:
            print(f"   + {p}")
    if only_local:
        print(f"only local ({len(only_local)}):")
        for p in only_local[:20]:
            print(f"   - {p}")
    if differ:
        print(f"content differs ({len(differ)}):")
        for p in differ[:20]:
            print(f"   ~ {p}")
    print("\nRESULT: remote and local DIFFER")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
