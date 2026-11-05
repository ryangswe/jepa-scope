#!/usr/bin/env python3
"""Check that the workflow's docs and config agree with themselves.

    python3 .claude/workflow/check.py            # from the repository root
    python3 .claude/workflow/check.py --root DIR

Copied into the repository by the workflow kit (managed: `workflow-kit sync` updates it). Run it
in CI or a test so a broken rule fails a build rather than relying on anyone to remember it.
Budgets come from `.claude/workflow.json`. Standard library only; exit 1 on any failure.

Checks:
- CLAUDE.md stays under its byte budget (it loads in every session).
- Every skill in .claude/skills/ has a name matching its directory and a description, and is
  named in backticks in CLAUDE.md or an agent file (an unnamed skill is never opened).
- Every path-scoped rule in .claude/rules/ has a strict `paths:` header whose fixed prefixes exist
  (a header that fails to parse makes the rule load in every session instead).
- docs/decisions.md, if present: under its byte budget, lines within the length limit, each theme
  heading names an existing detail file; every detail entry opens with a status line; every index
  line is claimed by an entry marked `live, indexed`, and every claim names an index line.
- docs/plans/README.md, if present: its "Active plans" list agrees with each dated plan's
  `Status:` line (a closed plan is not listed; an open one is).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

DEFAULTS = {
    "claude_md_max_bytes": 14336,
    "decisions_index_max_bytes": 20000,
    "decisions_line_max_chars": 250,
}
STATUS = re.compile(
    r"^\*\*Status:\*\* (live, indexed|live, not indexed|superseded by \d{4}-\d{2}-\d{2}"
    r"|superseded|withdrawn|record)(?: — (.+))?$"
)
INDEX_LINE = re.compile(r"^- (\d{4}-\d{2}(?:-\d{2})?) — ")
HEADING = re.compile(r"^## (\d{4}-\d{2}(?:-\d{2})?) — ")
TERMINAL = re.compile(r"^(Done|Closed|Superseded|Shelved)\b", re.I)


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def fail(self, check: str, detail: str) -> None:
        self.failures.append(f"{check}: {detail}")


def frontmatter(text: str) -> str | None:
    text = text.replace("\r\n", "\n")
    m = re.match(r"^---\n(.*?)\n---(?:\n|$)", text, re.S)
    return m.group(1) if m else None


def check_claude_md(root: Path, cfg: dict, r: Report) -> None:
    path = root / "CLAUDE.md"
    if not path.exists():
        r.fail("CLAUDE.md", "missing")
        return
    size = path.stat().st_size
    if size > cfg["claude_md_max_bytes"]:
        r.fail("CLAUDE.md", f"{size} bytes, over the {cfg['claude_md_max_bytes']} budget: "
                            "cut always-on text, or move procedures to a skill")


def check_skills(root: Path, r: Report) -> None:
    skills_dir = root / ".claude/skills"
    if not skills_dir.is_dir():
        return
    always_on = [(root / "CLAUDE.md").read_text(encoding="utf-8-sig")] if (root / "CLAUDE.md").exists() else []
    agents = root / ".claude/agents"
    if agents.is_dir():
        always_on += [p.read_text(encoding="utf-8-sig") for p in sorted(agents.glob("*.md"))]
    text = "\n".join(always_on)
    for skill in sorted(p for p in skills_dir.iterdir() if p.is_dir()):
        file = skill / "SKILL.md"
        if not file.exists():
            r.fail("skills", f"{skill.name}: missing SKILL.md")
            continue
        fm = frontmatter(file.read_text(encoding="utf-8-sig")) or ""
        if not re.search(rf"^name: *[\"']?{re.escape(skill.name)}[\"']? *$", fm, re.M):
            r.fail("skills", f"{skill.name}: frontmatter `name:` must match its directory")
        if not re.search(r"^description: \S", fm, re.M):
            r.fail("skills", f"{skill.name}: frontmatter needs a `description:`")
        if f"`{skill.name}`" not in text:
            r.fail("skills", f"{skill.name}: not named in backticks in CLAUDE.md or an agent "
                             "file, so nothing tells Claude to load it")


def check_rules(root: Path, r: Report) -> None:
    rules_dir = root / ".claude/rules"
    if not rules_dir.is_dir():
        return
    for rule in sorted(rules_dir.rglob("*.md")):
        name = rule.relative_to(rules_dir)
        fm = frontmatter(rule.read_text(encoding="utf-8-sig"))
        if fm is None:
            continue  # no header: loads in every session, which is allowed
        first, *items = fm.split("\n")
        if first != "paths:" or not items or not all(
            re.match(r'^ {2}- "[^"]+"$', line) for line in items
        ):
            r.fail("rules", f"{name}: header must be `paths:` then lines of `  - \"glob\"`")
            continue
        for line in items:
            glob = re.match(r'^ {2}- "([^"]+)"$', line).group(1)
            prefix = re.split(r"[*?[{]", glob)[0].rstrip("/")
            if prefix and not (root / prefix).exists():
                r.fail("rules", f"{name}: `{glob}` names `{prefix}`, which does not exist")


def check_decisions(root: Path, cfg: dict, r: Report) -> None:
    index_path = root / "docs/decisions.md"
    if not index_path.exists():
        return
    index = index_path.read_text(encoding="utf-8-sig")
    size = len(index.encode())
    if size > cfg["decisions_index_max_bytes"]:
        r.fail("decisions", f"index is {size} bytes, over {cfg['decisions_index_max_bytes']}: "
                            "prune lines (set their entries to `live, not indexed`)")
    named: set[str] = set()
    for line in index.split("\n"):
        if re.match(r"^- \d{4}-", line) and not INDEX_LINE.match(line):
            r.fail("decisions", f"index line not in `- YYYY-MM-DD — text` form: {line[:70]}")
        if INDEX_LINE.match(line) and len(line) > cfg["decisions_line_max_chars"]:
            r.fail("decisions", f"line over {cfg['decisions_line_max_chars']} chars: {line[:70]}…")
    for section in re.split(r"^(?=## )", index, flags=re.M):
        m = re.match(r"^## .*`docs/decisions/([\w-]+\.md)`", section)
        if not m:
            stray = [l for l in section.split("\n") if INDEX_LINE.match(l)]
            if stray:
                r.fail("decisions", f"index line under a heading that names no theme file: "
                                    f"{stray[0][:60]}")
            continue
        file = m.group(1)
        named.add(file)
        dates = [d.group(1) for d in (INDEX_LINE.match(l) for l in section.split("\n")) if d]
        detail = root / "docs/decisions" / file
        if not detail.exists():
            r.fail("decisions", f"theme heading names {file}, which does not exist")
            continue
        lines = detail.read_text(encoding="utf-8-sig").split("\n")
        claims: list[tuple[int, str]] = []
        for i, line in enumerate(lines):
            h = HEADING.match(line)
            if not h:
                continue
            nxt = next((l for l in lines[i + 1:] if l.strip()), "")
            s = STATUS.match(nxt)
            if not s:
                r.fail("decisions", f"{file}:{i + 1}: entry must open with a valid `**Status:**` "
                                    f"line, found: {nxt[:60]}")
                continue
            if s.group(1) != "live, indexed":
                continue
            note = s.group(2) or ""
            pointed = re.findall(r"under the (\d{4}-\d{2}-\d{2}) line", note)
            own = [] if note.startswith("under ") else [h.group(1)]
            claims += [(i + 1, d) for d in own + pointed]
        claimed = {d for _, d in claims}
        for d in dates:
            if d not in claimed:
                r.fail("decisions", f"{file}: index line {d} has no entry marked `live, indexed`")
        for line_no, d in claims:
            if d not in dates:
                r.fail("decisions", f"{file}:{line_no}: marked `live, indexed` for {d}, but the "
                                    "index has no line of that date")
    theme_dir = root / "docs/decisions"
    if theme_dir.is_dir():
        for p in sorted(theme_dir.glob("*.md")):
            if p.name not in named:
                r.fail("decisions", f"{p.name} is not named by any heading in docs/decisions.md")


def check_plans(root: Path, r: Report) -> None:
    plans = root / "docs/plans"
    readme = plans / "README.md"
    if not readme.exists():
        return
    text = readme.read_text(encoding="utf-8-sig")
    parts = re.split(r"^## Active plans$", text, flags=re.M)
    if len(parts) < 2:
        r.fail("plans", "README.md has no `## Active plans` section")
        return
    active = parts[1].split("\n## ")[0]
    listed = re.findall(r"^- `([^`]+\.md)`", active, re.M)
    status = {}
    for p in sorted(plans.glob("*.md")):
        if not re.match(r"^\d{4}-\d{2}-\d{2}-.+\.md$", p.name) or p.name.endswith("-log.md"):
            continue
        head = "\n".join(p.read_text(encoding="utf-8-sig").split("\n")[:12])
        m = re.search(r"^- \*\*Status:\*\*\s*(.*)$", head, re.M)
        if not m:
            r.fail("plans", f"{p.name}: no `- **Status:**` line in its header")
            continue
        status[p.name] = m.group(1).replace("*", "").strip()
    for name in listed:
        if name not in status:
            r.fail("plans", f"active list names {name}, which is not a dated plan here")
        elif TERMINAL.match(status[name]):
            r.fail("plans", f"{name} is listed as active but its status is: {status[name][:40]}")
    for name, s in status.items():
        if not TERMINAL.match(s) and name not in listed:
            r.fail("plans", f"{name} is open ({s[:30]}) but missing from the active list")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    root = Path(parser.parse_args().root).resolve()
    cfg_path = root / ".claude/workflow.json"
    cfg = {**DEFAULTS, **(json.loads(cfg_path.read_text(encoding="utf-8-sig")) if cfg_path.exists() else {})}
    r = Report()
    check_claude_md(root, cfg, r)
    check_skills(root, r)
    check_rules(root, r)
    check_decisions(root, cfg, r)
    check_plans(root, r)
    for failure in r.failures:
        print(f"FAIL {failure}")
    if not r.failures:
        print("workflow check: all passed")
    return 1 if r.failures else 0


if __name__ == "__main__":
    sys.exit(main())
