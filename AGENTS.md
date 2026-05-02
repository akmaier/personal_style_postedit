# AGENTS.md

## Cursor Cloud specific instructions

This is a **data-only research repository** (ACL 2026 paper: "Can You Make It Sound Like You?"). It contains no source code, no applications, no services, no build system, and no automated tests.

### Repository contents

- `README.md` — Dataset schema documentation
- `logs/` — 81 JSON files, one per study participant, containing writing logs with keystroke-level edits

### Working with the data

- All data files are valid JSON, parseable with Python's built-in `json` module (Python 3 is pre-installed).
- No dependencies need to be installed.
- There are no lint, test, or build commands.
- To validate/explore the dataset, use `python3` with the `json` and `os` standard library modules.
