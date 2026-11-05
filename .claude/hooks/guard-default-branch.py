#!/usr/bin/env python3
"""PreToolUse guard: no file edits while this repo sits on master/main.

Installed by the workflow kit (managed: `workflow-kit sync` updates it) and wired to
Edit|Write|NotebookEdit in .claude/settings.json. Reads the tool call's JSON from stdin and blocks
(exit 2, stderr fed back to Claude) when the target file lives inside THIS repo and the checkout
containing it (the main checkout or a worktree under .claude/worktrees/) is on master or main:
every change starts on a feature branch.

Files outside this repo (scratchpad, other checkouts, memory) are never blocked, nor are the
gitignored scratch directories listed under "branch_guard_exempt" in .claude/workflow.json. Any
unexpected failure exits 0: a broken guard must degrade to "no guard", never to "no edits".

Known gap: only the file-editing TOOLS pass through this hook. Shell writes (sed -i, cat >>) do
not; CLAUDE.md's "branch first" rule covers those by instruction.
"""

import json
import os
import subprocess
import sys

def _current_branch(directory: str) -> str:
    """Branch checked out in the git checkout/worktree that contains `directory`."""
    branch = subprocess.run(
        ["git", "-C", directory, "branch", "--show-current"],
        capture_output=True, text=True, timeout=10,
    ).stdout.strip()
    if not branch:  # e.g. detached HEAD or an older git without --show-current
        branch = subprocess.run(
            ["git", "-C", directory, "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    return branch


def _exempt_dirs(repo: str) -> list:
    try:
        with open(os.path.join(repo, ".claude", "workflow.json")) as f:
            return [d.strip("/") for d in json.load(f).get("branch_guard_exempt", []) if d.strip("/")]
    except Exception:
        return []


def main() -> int:
    try:
        data = json.load(sys.stdin)
        tool_input = data.get("tool_input") or {}
        file_path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
        if not file_path:
            return 0

        # The repo this guard protects = the repo the script lives in.
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        real_repo = os.path.realpath(repo)
        real_file = os.path.realpath(file_path)
        if not real_file.startswith(real_repo + os.sep):
            return 0
        # Gitignored scratch the repo lists in .claude/workflow.json ("branch_guard_exempt"):
        # it can never reach a commit, so the branch says nothing about it.
        for exempt in _exempt_dirs(real_repo):
            if real_file.startswith(os.path.join(real_repo, exempt) + os.sep):
                return 0

        # Resolve the branch from the TARGET FILE's directory, not the repo root.
        # Git worktrees live under .claude/worktrees/<name>/ inside this repo, each
        # on its own branch; a subagent editing there while the main checkout sits
        # on master must be judged by the worktree's branch, not the main one's.
        # For a not-yet-existing file, walk up to the nearest existing directory.
        start_dir = os.path.dirname(real_file)
        while start_dir and not os.path.isdir(start_dir) and start_dir != real_repo:
            start_dir = os.path.dirname(start_dir)
        if not os.path.isdir(start_dir):
            start_dir = real_repo
        branch = _current_branch(start_dir)
        if branch in ("master", "main"):
            print(
                f"BLOCKED: this repo is on '{branch}', and the working agreement is to never "
                "make changes directly on it. Create or switch to a feature branch first "
                "(e.g. `git checkout -b <descriptive-name>`), then retry this edit.",
                file=sys.stderr,
            )
            return 2
        return 0
    except Exception:
        return 0  # fail open — a broken guard must never block all editing

if __name__ == "__main__":
    sys.exit(main())
