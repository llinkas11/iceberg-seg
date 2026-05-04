# Overleaf session snapshots

Versioned snapshots of the Overleaf LaTeX project, one folder per session. Workaround for the lack of Premium Git integration on this Overleaf account.

## Naming convention

`YYYY-MM-DD_short-kebab-summary/`

Examples: `2026-04-17_initial-import`, `2026-04-20_methods-draft`, `2026-04-22_trim-abstract`.

## Sessions (newest first)

| Date | Folder | Summary |
|------|--------|---------|
| 2026-04-17 | [2026-04-17_template-plan-md](./2026-04-17_template-plan-md) | Fresh placeholder-only IGS-class template keyed to `plan.md` + `reference/*.md`. No prose lifted from prior drafts. Primary structure: IGS class demo. Secondary section organisation: Fisser (2024). |

Earlier sessions from 2026-04-17 (initial-import and restart-with-igs-class) were archived under [`../_archive/overleaf-sessions/`](../_archive/overleaf-sessions/) during the 2026-04-17 cleanup; both carried old 3-class / 323-chip / 240-scene numbers that predate `plan.md`. See [`../_archive/README.md`](../_archive/README.md) for per-file provenance.

## Workflow

### Start a new session (from prior snapshot)

```bash
cd "/Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/paper-writing/overleaf"
cp -r "<most-recent-folder>" "YYYY-MM-DD_short-summary"
```

Update the new folder's `README.md` with planned changes. Edit files. At session end, finalize the README's "Changes" and "Files modified" sections. Update the table above in this file.

### Push edits back to Overleaf

In the Overleaf web editor, replace the corresponding files via drag-drop onto the file tree. Compile. Confirm the PDF builds. Note any new compilation errors in the session README.

### Pull Overleaf web edits into local

Overleaf Menu → Source (Download). Save the zip, unzip into a new dated folder named like `YYYY-MM-DD_sync-from-overleaf/`. Diff against the previous session folder with:

```bash
diff -r "<previous>" "<new>"
```

Log the incoming changes in the new session's README.

## Style rules

All edits must follow `../CLAUDE.md` (paper-writing scope) and the root `CLAUDE.md`. Key targets:

- Journal: Journal of Glaciology (IGS)
- Paper type: Article
- Abstract limit: 200 words
- Citations: `(Author and others, YYYY)` never `et al.`
- Units: SI, superscript notation
- References: surname + initials no periods, ISO 4 journal abbreviations, DOI when available

## Compile check (before declaring a session done)

If the session edits `.tex` or `.bib`, verify it compiles in Overleaf (drag-drop, click Recompile, confirm no new errors). Record compile status in the session README.
