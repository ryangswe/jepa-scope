---
name: checkpoint
description: Verify a completed step before calling it done — typecheck, tests, build and the workflow check, then a live check of anything observable. Use after integrating a step, at a milestone boundary, and before asking for review or a merge.
---

# Checkpoint

Run every stage; do not skip one because the change "looks safe". Take the commands from the
**Commands** section of `CLAUDE.md`. If a stage has no command there, say so and skip it.

1. **Typecheck / lint**, if the project has one.
2. **Tests**: the full suite.
3. **Build**, if the project has a build step.
4. **Workflow check**: `python3 .claude/workflow/check.py` (CLAUDE.md budget, skills, rules,
   decisions, plans).

If any stage fails, stop and report it with the actual output. Do not move to the next stage, and
do not describe the step as done.

5. **Live check**, only if the step changed something observable in a running surface (a page, a
   server, a CLI's output). Start it the way `CLAUDE.md` describes, exercise the change, and check
   logs and the console for errors. For changes nothing can render yet, skip this and say why.

Report each stage as pass or fail, and end with one line: is the step verified? If it is, say this
is a good point to commit.
