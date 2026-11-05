---
name: handoff
description: End a session so a fresh one can pick the work up — record where things stand in the repository (the plan, decisions, a commit), push, and print a short starting prompt that points there. Use when the owner asks for a handoff, is stopping for the day mid-task, is switching topic, or the session has grown long (around 300–400k tokens).
argument-hint: "What will the next session be used for?"
---

# Handoff

The repository is the handoff; the prompt only points at it. A file in a temp folder vanishes
with a cloud sandbox and never reaches another machine, and a summary that restates the plan
drifts from it. So write state where the next session will look anyway, then print a few lines
that send it there. If the owner passed an argument, it says what the next session is for:
shape steps 3 and 5 around that.

1. **Find where the work lives.** The current branch (never the default branch: if you are on it
   with changes, create a branch first), and the plan for this work if there is one: the dated
   plan in `docs/plans/` whose task this session worked on, from the active list in
   `docs/plans/README.md`. No plan is normal for small work.

2. **Take stock**, checking rather than recalling:
   - `git status` and `git log --oneline @{u}..` (or since the branch point): uncommitted and
     unpushed work.
   - Anything still running: background tasks or agents, servers, watchers, worktrees you made.
     Stop what should not outlive the session, and note what keeps running.
   - What is waiting on the owner: a question you asked, an approval, a step only they can do.

3. **Record it where it belongs.** Write each fact once, in the place that matches how long it
   stays true; point at existing plans, issues, PRs and commits instead of copying them.
   - **A decision a later session could reopen** that is not recorded yet: add it now, per
     `CLAUDE.md` (index line and detail entry).
   - **With a plan:** update its `Status:` line, tick finished tasks, and add one log entry: what
     was done, what was learned, and the **next step** as one concrete action (a command, a file
     and function, a question to settle). Keep the active list in `docs/plans/README.md` in step.
   - **Without a plan:** the commit message carries the state. Its body says what is done, what
     is not, and the next step.

4. **Commit and push.** Stage explicit paths, never `git add .`. Work in progress is fine to
   commit on a feature branch; say so in the subject (`wip: …`). If nothing changed but the state
   is only in your head, an empty commit (`git commit --allow-empty`) holding the step 3 text is
   better than losing it. Push the branch. If you cannot push, say so plainly: the next session
   will not see unpushed work, and the prompt must say where it is.

5. **Print the starting prompt** in one code block, at most ten lines, for the owner to paste into
   a new session:
   - repository and branch;
   - where to read: the plan and its section, or the handoff commit by its short hash;
   - the next step, in one line;
   - anything waiting on the owner, and anything left running;
   - which skills to load, by name, if the next step needs them.

   Leave out everything the plan or commit already says. Never include secrets, tokens or
   personal data, even if they appeared in the session.

6. **Finish with one line:** what was committed and pushed, or what could not be.

Do not hand off what needs no handoff. If the work is finished and merged, or nothing is in
flight, say so: the plan and the history already show where things stand.
