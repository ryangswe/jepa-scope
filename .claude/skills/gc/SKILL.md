---
name: gc
description: Clean up what agent work leaves behind — worktrees and local branches whose work is already in the default branch — and list the rest for the owner. Use weekly, at a handoff, when `.claude/worktrees/` has more than a few entries, or when the owner asks to tidy branches or worktrees.
---

# Clean up worktrees and branches

`.claude/workflow/gc.py` decides what is safe; you only run it and report. Never remove anything
by hand that the script did not mark `safe`.

1. **Refresh what the default branch contains**, so merged and squash-merged work is recognised:
   `git fetch origin`. The script compares against `origin/<default>`. If the fetch fails, say so
   and go on: the report will just call more branches unmerged. Run gc on the machine whose
   worktrees these are, not from a VM or container that sees the repository under another path.

2. **Report:** `python3 .claude/workflow/gc.py`. It changes nothing. Each line is `safe` or `keep`,
   a class (MERGED, IN DEFAULT, STALE, ACTIVE), the age of the last commit, the branch and its
   worktree.

3. **Remove the safe ones:** `python3 .claude/workflow/gc.py --apply`. It removes only branches whose
   work is already in the default branch, and their worktrees only when they hold no uncommitted
   changes; it never forces. This needs no approval: nothing it removes holds work that isn't in
   the default branch. If the owner has said to ask first, show step 2's report and wait.

4. **Hand the rest to the owner** in a short list: each STALE branch with its age and last commit
   subject, and each `keep` line with a reason (uncommitted changes, locked). Ask which to delete.
   Delete only what they name, with `git worktree remove` and `git branch -D`. Never use
   `--force` on a worktree with uncommitted changes without the owner's explicit say.

Report the counts from the script's last lines and the disk freed (`du -sh .claude/worktrees`
before and after), in two or three lines.
