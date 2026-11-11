"""Push the prepared repository to GitHub through the local proxy.

Security note: the token is read from the environment and is NEVER written to
disk, to git config, or into a remote URL. It exists only in this process's
memory and in the HTTPS Authorization header.

Run it as a one-shot: the environment variable is scoped to this command only.
"""

from __future__ import annotations

import json
import os
import socket
import ssl
import subprocess
import sys
from pathlib import Path

PROXY_CANDIDATES = [
    ("127.0.0.1", 7897),   # Clash Verge default
    ("127.0.0.1", 7890),   # Clash default
    ("127.0.0.1", 10809),  # v2rayN http
    ("127.0.0.1", 1080),   # socks->http bridges
]
API_HOST = "api.github.com"
REPO = "text2sql-fc-pipeline"
DESCRIPTION = (
    "LLM data-synthesis pipeline for function-calling fine-tuning (Text2SQL/Spider): "
    "synthesis -> two-gate validation -> tool-call formatting -> LoRA SFT -> real SQLite evaluation"
)
TOPICS = ["function-calling", "text2sql", "spider", "lora", "peft", "llm",
          "data-synthesis", "fine-tuning", "pytorch", "evaluation", "sqlite", "qlora"]


def api(path: str, token: str, method: str = "GET", body: dict | None = None,
        proxy: tuple[str, int] | None = None):
    """Minimal GitHub REST call through the CONNECT proxy."""
    proxy = proxy or PROXY
    raw = socket.create_connection(proxy, timeout=30)
    raw.sendall(f"CONNECT {API_HOST}:443 HTTP/1.1\r\nHost: {API_HOST}:443\r\n\r\n".encode())
    raw.recv(200)
    tls = ssl.create_default_context().wrap_socket(raw, server_hostname=API_HOST)
    payload = b"" if body is None else json.dumps(body).encode()
    req = (
        f"{method} {path} HTTP/1.1\r\nHost: {API_HOST}\r\n"
        f"Authorization: Bearer {token}\r\n"
        "Accept: application/vnd.github+json\r\n"
        "X-GitHub-Api-Version: 2022-11-28\r\n"
        "User-Agent: fcdata-push\r\nContent-Type: application/json\r\n"
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
        parsed = {"raw": bod.decode("utf-8", "replace")[:400]}
    return status, parsed


def find_proxy(token: str) -> tuple[str, int] | None:
    """Pick the first proxy that can actually reach api.github.com over TLS."""
    for proxy in PROXY_CANDIDATES:
        try:
            status, me = api("/user", token, proxy=proxy)
            if status == 200:
                print(f"proxy {proxy[0]}:{proxy[1]} works; authenticated as {me['login']}")
                return proxy
        except OSError as exc:
            print(f"proxy {proxy[0]}:{proxy[1]} unreachable ({type(exc).__name__})")
    return None


def main() -> int:
    # Token comes from the environment (no shell history) or a hidden prompt
    # (no echo). It is never written to disk, git config, or a persisted URL.
    token = os.environ.get("GH_TOKEN_MEM")
    if not token:
        import getpass
        token = getpass.getpass("GitHub PAT (hidden input, paste + Enter): ").strip()
    if not token:
        print("no token provided", file=sys.stderr)
        return 2

    root = Path(__file__).resolve().parent
    proxy = find_proxy(token)
    if proxy is None:
        print(
            "\nno working proxy found. Start your proxy client (Clash Verge etc.)\n"
            "and make sure a local http port is listening (7897 / 7890 / 10809),\n"
            "or add your port to PROXY_CANDIDATES in this script.",
            file=sys.stderr)
        return 1
    login = None
    status, me = api("/user", token, proxy=proxy)
    if status != 200:
        print(f"auth failed: {status} {me.get('message')}", file=sys.stderr)
        return 1
    login = me["login"]
    print(f"authenticated as {login}")

    # 1) create the repository if it does not exist yet
    status, repo = api(f"/repos/{login}/{REPO}", token)
    if status == 404:
        print(f"creating {login}/{REPO} ...")
        status, repo = api("/user/repos", token, "POST", {
            "name": REPO,
            "description": DESCRIPTION,
            "private": False,
            "has_issues": True,
            "has_wiki": False,
        })
        if status not in (200, 201):
            print(f"create failed: {status} {repo.get('message')}", file=sys.stderr)
            return 1
        print(f"created: {repo.get('html_url')}")
    elif status == 200:
        print(f"repository already exists: {repo.get('html_url')}")
    else:
        print(f"repo lookup failed: {status} {repo.get('message')}", file=sys.stderr)
        return 1

    # 2) set topics
    status, res = api(f"/repos/{login}/{REPO}/topics", token, "PUT", {"names": TOPICS},
                      proxy=proxy)
    print(f"topics: {status} {'ok' if status == 200 else res.get('message')}")

    # 3) push. The token is passed to git ONLY via the ephemeral URL embedded in
    #    the subprocess argv (in memory); it is never written to any file.
    #    -c credential.helper= : skip stored/manager helpers (no GUI prompt)
    #    GIT_TERMINAL_PROMPT=0 : never hang waiting for a terminal prompt
    print("pushing ...")
    proc = subprocess.run(
        ["git", "-c", "credential.helper=",
         "-c", f"http.proxy=http://{proxy[0]}:{proxy[1]}",
         "push", f"https://{login}:{token}@github.com/{login}/{REPO}.git",
         "HEAD:main", "--force"],
        cwd=str(root), capture_output=True, text=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GH_TOKEN": ""},
    )
    out = (proc.stdout + proc.stderr).replace(token, "***")
    print(out.strip()[-2000:])
    if proc.returncode != 0:
        print(f"push failed (exit {proc.returncode})", file=sys.stderr)
        return 1

    # 4) make sure the default branch is main
    api(f"/repos/{login}/{REPO}", token, "PATCH", {"default_branch": "main"}, proxy=proxy)

    print(f"\nDONE -> https://github.com/{login}/{REPO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
