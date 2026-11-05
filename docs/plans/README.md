# Plans

Durable implementation plans, so a **fresh session** — often on a different model from the one
that wrote the plan — can execute the work from the file alone.

## Naming

`YYYY-MM-DD-<short-slug>.md`, dated when the plan was written. One file per milestone or effort.

## Active plans

Add a plan as soon as its file is committed (a Draft counts); remove it when its status becomes
Done, Closed, Superseded or Shelved. Every other file here is a record. `.claude/workflow/check.py` fails when this list and a
plan's `Status:` line disagree.

## Lifecycle

- **Draft** while planning; reviewed by the owner before any code is touched.
- **In progress**: the implementing session marks tasks as it lands them.
- **Done** when every step is verified and committed; **Superseded** (name the successor) or
  **Shelved** when it stops. Update the `Status:` line and the list above in the same commit.
- **The execution log lives beside the plan**, in `<plan>-log.md`, appended as work proceeds, so
  the plan stays a lean spec.
- **On Done**, compress the log into the completion summary (outcome, deviations, follow-ups,
  PRs), then delete the `-log.md` file. Git keeps the full log.
- **Write the lesson field** (at most one, or `NO_ENTRY`), then list earlier ones with
  `rg -n '^- \*\*Lesson:\*\* cue' docs/plans`. When a lesson repeats one from an earlier plan,
  promote it: a line in `CLAUDE.md` or an agent file, or better a hook or a check, which cannot be
  skipped the way prose can.

Small fixes and one-sitting features don't need a plan file.
