"""Inspect the target GitHub repository's current contents via the REST API.

Read-only: reports what is already on the remote so we can decide between
force-push (overwrite) and delete+recreate. Never prints the token.
"""

from __future__ import annotations

import json
import os
import socket
import ssl
import sys

PROXY = ("127.0.0.1", 7897)
API = "api.github.com"
REPO = "text2sql-fc-pipeline"


def call(path: str, token: str, method: str = "GET", body: dict | None = None):
    raw = socket.create_connection(PROXY, timeout=30)
    raw.sendall(f"CONNECT {API}:443 HTTP/1.1\r\nHost: {API}:443\r\n\r\n".encode())
    raw.recv(200)
    tls = ssl.create_default_context().wrap_socket(raw, server_hostname=API)
    payload = b"" if body is None else json.dumps(body).encode()
    req = (
        f"{method} {path} HTTP/1.1\r\nHost: {API}\r\n"
        f"Authorization: Bearer {token}\r\n"
        "Accept: application/vnd.github+json\r\n"
        "X-GitHub-Api-Version: 2022-11-28\r\n"
        "User-Agent: fcdata-inspect\r\nContent-Type: application/json\r\n"
        f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n"
    ).encode() + payload
    tls.sendall(req)
    data = b""
    while True:
        chunk = tls.recv(65536)
        if not chunk:
            break
        data += chunk
    tls.close()
    head, _, bod = data.partition(b"\r\n\r\n")
    status = int(head.split()[1])
    try:
        parsed = json.loads(bod.decode("utf-8", "replace") or "{}")
    except json.JSONDecodeError:
        parsed = {"raw": bod.decode("utf-8", "replace")[:300]}
    return status, parsed


def main() -> int:
    token = os.environ.get("GH_TOKEN_MEM", "")
    if not token:
        print("missing GH_TOKEN_MEM", file=sys.stderr)
        return 2

    st, me = call("/user", token)
    if st != 200:
        print(f"AUTH FAILED {st}: {me.get('message')}", file=sys.stderr)
        return 1
    login = me["login"]
    print(f"authenticated: {login}")

    st, repo = call(f"/repos/{login}/{REPO}", token)
    if st == 404:
        print(f"repo {login}/{REPO}: DOES NOT EXIST (will create)")
        return 0
    if st != 200:
        print(f"repo lookup failed {st}: {repo.get('message')}", file=sys.stderr)
        return 1

    print(f"repo exists: {repo.get('html_url')}")
    print(f"  private={repo.get('private')}  default_branch={repo.get('default_branch')}")
    print(f"  size={repo.get('size')}KB  pushed_at={repo.get('pushed_at')}")

    st, root = call(f"/repos/{login}/{REPO}/contents/", token)
    if st == 200 and isinstance(root, list):
        print(f"  root entries ({len(root)}):")
        for item in root:
            print(f"    - {item['name']:32s} {item['type']}")
    elif st == 404:
        print("  repo is EMPTY (no commits yet)")
    else:
        print(f"  contents lookup: {st} {root.get('message')}")

    st, commits = call(f"/repos/{login}/{REPO}/commits?per_page=10", token)
    if st == 200 and isinstance(commits, list):
        print(f"  commits on remote: {len(commits)}")
        for c in commits[:5]:
            msg = c["commit"]["message"].split("\n")[0][:70]
            print(f"    {c['sha'][:7]} {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
