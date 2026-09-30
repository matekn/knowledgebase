# Evals

Test prompts for the `knowledgebase` skill. No machine-specific paths are stored here.

## Portable placeholder

Prompts in `evals.json` reference fixtures as `<skill-dir>/evals/files/...`. Before
running, substitute `<skill-dir>` with the absolute path of this skill directory.

For a fair comparison, copy the fixture folders to a **neutral** temp location first
and point the prompts there. Otherwise a baseline agent can discover the skill by
walking the input path (the fixtures live inside the skill tree), which invalidates
the `without_skill` control.

## Fixtures

| Folder | Files |
| --- | --- |
| `files/docs` | runbook.md, auth.py, sourdough.md, budget.csv |
| `files/work` | sprint_plan.md, deploy_notes.md |
| `files/personal` | sourdough_journal.md, garden.md |
| `files/cleanup` | runbook.md, meeting.md |

## Running

Use the skill-creator harness with `KB_HOME` set to an isolated store per run, e.g.:

```bash
KB_HOME=/tmp/kb-eval python <skill-dir>/scripts/run.py add <fixtures>/docs --collection docs
```

See the parent `SKILL.md` for the full command reference.
