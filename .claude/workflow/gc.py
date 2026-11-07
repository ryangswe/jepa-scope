#!/usr/bin/env python3
"""Report, and on request remove, the worktrees and branches that agent work has left behind.

    python3 .claude/workflow/gc.py                 # report only; changes nothing
    python3 .claude/workflow/gc.py --apply         # remove what is SAFE, report the rest
    python3 .claude/workflow/gc.py --stale-days 30 # when an unmerged branch counts as stale

Copied into the repository by the workflow kit (managed: `workflow-kit sync` updates it).
Standard library only; needs git 2.38+ to recognise squash-merged branches (older git reports
them as unmerged, which only means less gets removed).

Every local branch other than the default one is put in one class:

The default branch is read as origin/<default> when that ref exists (fetch first: the skill does),
so a local default branch that is behind does not make merged work look unmerged.

- MERGED: its tip is in the default branch's history (a merge commit or a fast-forward).
- IN DEFAULT: merging it into the default branch would change nothing (`git merge-tree`), so its
  content landed some other way, usually a squash merge.
- STALE: neither, and its last commit is older than --stale-days.
- ACTIVE: neither, and committed to recently.

SAFE means MERGED or IN DEFAULT, and, if a worktree has it checked out, that worktree has no
uncommitted or untracked changes and is not locked. `--apply` removes SAFE worktrees
(`git worktree remove`, never forced) and deletes SAFE branches, after `git worktree prune`
clears records of worktree folders that no longer exist. It refuses to run at all when every
worktree's folder looks missing: that means this shell can't see the recorded paths, and a
prune would drop the records of folders that are still there. It never touches STALE or ACTIVE
branches, the default branch, the main checkout, or anything with uncommitted work: those are
the owner's call, and the report lists them with their last commit.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def git(root: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout


def default_branch(root: Path) -> str:
    head = git(root, "symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD", check=False)
    if head.strip():
        name = head.strip().split("/", 1)[-1]
        if git(root, "rev-parse", "--verify", "--quiet", f"refs/heads/{name}", check=False).strip():
            return name
    for name in ("main", "master"):
        if git(root, "rev-parse", "--verify", "--quiet", f"refs/heads/{name}", check=False).strip():
            return name
    raise RuntimeError("no main or master branch found")


def worktrees(root: Path) -> list[dict]:
    """Every worktree but the main checkout: path, branch, locked, prunable."""
    out, current = [], {}
    for line in git(root, "worktree", "list", "--porcelain").splitlines() + [""]:
        if not line:
            if current:
                out.append(current)
            current = {}
            continue
        key, _, value = line.partition(" ")
        if key == "worktree":
            current = {"path": value, "branch": None, "locked": False, "prunable": False}
        elif key == "branch":
            current["branch"] = value.removeprefix("refs/heads/")
        elif key == "locked":
            current["locked"] = True
        elif key == "prunable":
            current["prunable"] = True
    return out[1:]  # the first entry is the main checkout


def in_default(root: Path, branch: str, default: str) -> bool:
    """True when merging `branch` into `default` would leave default's tree unchanged."""
    merged = subprocess.run(
        ["git", "-C", str(root), "merge-tree", "--write-tree", default, branch],
        capture_output=True, text=True)
    if merged.returncode != 0:  # a conflict, or git older than 2.38
        return False
    tree = merged.stdout.split("\n", 1)[0].strip()
    return tree == git(root, "rev-parse", f"{default}^{{tree}}").strip()


def dirty(path: str) -> int:
    if not os.path.isdir(path):
        return 0
    result = subprocess.run(["git", "-C", path, "status", "--porcelain"],
                            capture_output=True, text=True)
    return len(result.stdout.splitlines()) if result.returncode == 0 else -1


def reference(root: Path, default: str) -> str:
    """origin/<default> when it exists (it is usually newer than the local branch), else <default>."""
    remote = f"refs/remotes/origin/{default}"
    if git(root, "rev-parse", "--verify", "--quiet", remote, check=False).strip():
        return f"origin/{default}"
    return default


def classify(root: Path, stale_days: int) -> tuple[str, list[dict]]:
    default = reference(root, default_branch(root))
    trees = {w["branch"]: w for w in worktrees(root) if w["branch"]}
    main_branch = git(root, "branch", "--show-current").strip()
    now = time.time()
    rows = []
    refs = git(root, "for-each-ref", "--format=%(refname:short)\t%(committerdate:unix)\t%(subject)",
               "refs/heads/")
    for line in refs.splitlines():
        name, stamp, subject = (line.split("\t", 2) + ["", ""])[:3]
        if name == default.removeprefix("origin/"):
            continue
        age = int((now - int(stamp or now)) // 86400)
        ancestor = subprocess.run(
            ["git", "-C", str(root), "merge-base", "--is-ancestor", name, default],
            capture_output=True)
        if ancestor.returncode == 0:
            cls = "MERGED"
        elif in_default(root, name, default):
            cls = "IN DEFAULT"
        else:
            cls = "STALE" if age >= stale_days else "ACTIVE"
        tree = trees.get(name)
        reason = ""
        if name == main_branch:
            reason = "checked out in the main checkout"
        elif tree and tree["locked"]:
            reason = "worktree is locked"
        elif tree and dirty(tree["path"]) != 0:
            reason = "worktree has uncommitted changes"
        safe = cls in ("MERGED", "IN DEFAULT") and not reason
        rows.append({"branch": name, "class": cls, "age": age, "subject": subject[:60],
                     "worktree": tree["path"] if tree else "", "safe": safe, "reason": reason})
    return default, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", default=".", help="repository root (default: .)")
    parser.add_argument("--apply", action="store_true", help="remove SAFE worktrees and branches")
    parser.add_argument("--stale-days", type=int, default=14)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    try:
        if args.apply:
            trees = worktrees(root)
            if len(trees) > 1 and all(w["prunable"] for w in trees):
                print("gc: every worktree's folder looks missing, which usually means this shell "
                      "can't see the paths git recorded (a VM or container). Run gc on the machine "
                      "that created them; nothing was changed.", file=sys.stderr)
                return 1
            git(root, "worktree", "prune")
        default, rows = classify(root, args.stale_days)
    except RuntimeError as e:
        print(f"gc: {e}", file=sys.stderr)
        return 1

    order = {"MERGED": 0, "IN DEFAULT": 1, "STALE": 2, "ACTIVE": 3}
    rows.sort(key=lambda r: (not r["safe"], order[r["class"]], -r["age"]))
    print(f"default branch: {default}")
    removed = failed = 0
    for r in rows:
        mark = "safe" if r["safe"] else "keep"
        note = f"  ({r['reason']})" if r["reason"] else ""
        where = f"  [{os.path.relpath(r['worktree'], root)}]" if r["worktree"] else ""
        print(f"{mark}  {r['class']:10} {r['age']:4}d  {r['branch']}{where}  {r['subject']}{note}")
        if not (args.apply and r["safe"]):
            continue
        try:
            if r["worktree"] and os.path.isdir(r["worktree"]):
                git(root, "worktree", "remove", r["worktree"])
            # -D: the classification above already proved the work is in the default branch,
            # and -d would refuse a branch merged into origin/<default> but not the local one.
            git(root, "branch", "-D", r["branch"])
            removed += 1
        except RuntimeError as e:
            print(f"      could not remove: {e}")
            failed += 1

    counts = {c: sum(1 for r in rows if r["class"] == c) for c in order}
    safe = sum(1 for r in rows if r["safe"])
    print(f"\n{len(rows)} branches: " + ", ".join(f"{n} {c.lower()}" for c, n in counts.items())
          + f"; {safe} safe to remove.")
    if args.apply:
        print(f"removed {removed}" + (f", {failed} failed" if failed else "")
              + ". STALE and ACTIVE branches are the owner's call.")
    elif safe:
        print("report only: re-run with --apply to remove the safe ones.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
