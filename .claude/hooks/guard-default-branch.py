#!/usr/bin/env python3
"""PreToolUse guard: no changes while this repo sits on master/main.

Installed by the workflow kit (managed: `workflow-kit sync` updates it) and wired to
Edit|Write|NotebookEdit|Bash in .claude/settings.json. Reads the tool call's JSON from stdin and
blocks (exit 2, stderr fed back to Claude) when:

- an Edit, Write or NotebookEdit targets a file inside THIS repo and the checkout containing it
  (the main checkout or a worktree under .claude/worktrees/) is on master or main; or
- a Bash command runs `git commit` in a checkout of this repo that is on master or main, or
  `git push` that would update master or main (a refspec naming it, or a bare push from it).

Every change starts on a feature branch, and only the owner updates the default branch.

Files outside this repo (scratchpad, other checkouts, memory) are never blocked, nor are the
gitignored scratch directories listed under "branch_guard_exempt" in .claude/workflow.json. Any
unexpected failure exits 0: a broken guard must degrade to "no guard", never to "no edits".

Known gaps: shell writes (sed -i, cat >>) are not judged, only commits and pushes are, so a shell
write on main is caught when it is committed. The command parser follows `git -C <dir>` and a
leading `cd <dir> &&`; anything cleverer (eval, scripts, aliases) passes. This hook is a backstop
for an agent following CLAUDE.md, not a security boundary: branch protection on the remote is.
"""

import json
import os
import re
import shlex
import subprocess
import sys

DEFAULT_BRANCHES = ("master", "main")
# Split a shell command into simple commands. Good enough for agent-written one-liners; a
# separator inside quotes can mis-split, which only ever makes the guard miss, never over-block
# on its own (each piece still has to parse as a git commit or push).
SEPARATORS = re.compile(r"&&|\|\||;|\||\n")
# git's global options that take a separate value, before the subcommand.
GIT_OPTS_WITH_VALUE = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path"}


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


def _inside(path: str, repo: str) -> bool:
    return path == repo or path.startswith(repo + os.sep)


def _block(message: str) -> int:
    print(f"BLOCKED: {message}", file=sys.stderr)
    return 2


def _check_edit(tool_input: dict, repo: str) -> int:
    file_path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
    if not file_path:
        return 0
    real_file = os.path.realpath(file_path)
    if not real_file.startswith(repo + os.sep):
        return 0
    # Gitignored scratch the repo lists in .claude/workflow.json ("branch_guard_exempt"):
    # it can never reach a commit, so the branch says nothing about it.
    for exempt in _exempt_dirs(repo):
        if real_file.startswith(os.path.join(repo, exempt) + os.sep):
            return 0

    # Resolve the branch from the TARGET FILE's directory, not the repo root.
    # Git worktrees live under .claude/worktrees/<name>/ inside this repo, each
    # on its own branch; a subagent editing there while the main checkout sits
    # on master must be judged by the worktree's branch, not the main one's.
    # For a not-yet-existing file, walk up to the nearest existing directory.
    start_dir = os.path.dirname(real_file)
    while start_dir and not os.path.isdir(start_dir) and start_dir != repo:
        start_dir = os.path.dirname(start_dir)
    if not os.path.isdir(start_dir):
        start_dir = repo
    branch = _current_branch(start_dir)
    if branch in DEFAULT_BRANCHES:
        return _block(
            f"this repo is on '{branch}', and the working agreement is to never make changes "
            "directly on it. Create or switch to a feature branch first "
            "(e.g. `git checkout -b <descriptive-name>`), then retry this edit."
        )
    return 0


def _git_invocations(command: str, cwd: str):
    """Yield (directory, subcommand, args) for each `git ...` simple command in `command`."""
    directory = cwd
    for piece in SEPARATORS.split(command):
        try:
            words = shlex.split(piece)
        except ValueError:
            continue
        if not words:
            continue
        if words[0] == "cd" and len(words) >= 2:  # `cd <dir> && git ...`
            directory = os.path.join(directory, os.path.expanduser(words[1]))
            continue
        # Skip leading environment assignments (FOO=bar git ...).
        while words and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", words[0]):
            words = words[1:]
        if not words or os.path.basename(words[0]) != "git":
            continue
        here = directory
        i = 1
        while i < len(words) and words[i].startswith("-"):
            opt = words[i]
            if opt in GIT_OPTS_WITH_VALUE and i + 1 < len(words):
                if opt == "-C":
                    here = os.path.join(here, os.path.expanduser(words[i + 1]))
                i += 2
            else:
                i += 1
        if i < len(words):
            yield here, words[i], words[i + 1:]


def _push_targets_default(args: list, branch: str) -> bool:
    """Whether `git push <args>`, run on `branch`, would update master or main."""
    positional = [a for a in args if not a.startswith("-")]
    refspecs = positional[1:]  # positional[0] is the remote
    if not refspecs:
        # A bare push sends the current branch (push.default simple/current/upstream).
        return branch in DEFAULT_BRANCHES and "--tags" not in args
    for spec in refspecs:
        spec = spec.lstrip("+")
        dest = spec.split(":", 1)[1] if ":" in spec else spec
        if dest in ("HEAD", "@") and branch in DEFAULT_BRANCHES:
            return True
        if dest.removeprefix("refs/heads/") in DEFAULT_BRANCHES:
            return True
    return False


def _check_bash(tool_input: dict, cwd: str, repo: str) -> int:
    command = tool_input.get("command") or ""
    if "git" not in command or ("commit" not in command and "push" not in command):
        return 0  # fast path: most shell commands never reach git
    for directory, sub, args in _git_invocations(command, cwd):
        if sub not in ("commit", "push"):
            continue
        real_dir = os.path.realpath(directory)
        if not _inside(real_dir, repo) or not os.path.isdir(real_dir):
            continue
        branch = _current_branch(real_dir)
        if sub == "commit" and branch in DEFAULT_BRANCHES:
            return _block(
                f"this checkout is on '{branch}', and the working agreement is to never commit "
                "directly on it. Create a feature branch (`git checkout -b <descriptive-name>`; "
                "your uncommitted changes come with you), then commit there."
            )
        if sub == "push" and _push_targets_default(args, branch):
            return _block(
                "this push would update the default branch, which only the owner does "
                "(through a pull request). Push a feature branch and open a pull request instead."
            )
    return 0


def main() -> int:
    try:
        data = json.load(sys.stdin)
        tool_input = data.get("tool_input") or {}
        # The repo this guard protects = the repo the script lives in.
        repo = os.path.realpath(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        if data.get("tool_name") == "Bash":
            return _check_bash(tool_input, data.get("cwd") or os.getcwd(), repo)
        return _check_edit(tool_input, repo)
    except Exception:
        return 0  # fail open — a broken guard must never block all editing


if __name__ == "__main__":
    sys.exit(main())
