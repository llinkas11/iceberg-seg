# Overleaf paper

Live LaTeX source for the IDS2026 paper. The Overleaf project at `https://git@git.overleaf.com/6990cbcf83db0311e228f7a3` is the authoritative source; we work against a local clone synced over `git pull` / `git push`.

Older manual zip-snapshot folders are archived under [`../_archive/overleaf-sessions/`](../_archive/overleaf-sessions/) (see "Historical sessions" below).

## Git workflow

The Overleaf project is mirrored at `paper-writing/overleaf/git-mirror/` (gitignored in iceberg-seg; that directory is its own git repo whose remote is Overleaf).

### One-time setup

```bash
cd /Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/paper-writing/overleaf
git clone https://git@git.overleaf.com/6990cbcf83db0311e228f7a3 git-mirror
```

When prompted for a password, paste the Overleaf personal git token (Account Settings -> Git Integration on overleaf.com). osxkeychain caches it after the first successful auth.

### Pull web edits into local

```bash
cd paper-writing/overleaf/git-mirror
git pull
```

### Push local edits to Overleaf

```bash
cd paper-writing/overleaf/git-mirror
git add -A
git commit -m "<short summary>"
git push
```

### Compile check (before declaring a session done)

If the change touches `.tex` or `.bib`, open the Overleaf web editor, click Recompile, confirm no new errors. The Overleaf side compiles automatically on each push.

## Style rules

All edits must follow [`../CLAUDE.md`](../CLAUDE.md) (paper-writing scope) and the root [`CLAUDE.md`](../../CLAUDE.md). Key targets:

- Journal: Journal of Glaciology (IGS)
- Paper type: Article
- Abstract limit: 200 words
- Citations: `(Author and others, YYYY)` never `et al.`
- Units: SI, superscript notation
- References: surname + initials no periods, ISO 4 journal abbreviations, DOI when available

## Historical sessions

Manual zip-snapshot folders from before git integration. All archived under [`../_archive/overleaf-sessions/`](../_archive/overleaf-sessions/):

| Date | Folder | Summary |
|------|--------|---------|
| 2026-04-17 | [`2026-04-17_template-plan-md`](../_archive/overleaf-sessions/2026-04-17_template-plan-md) | Placeholder-only IGS-class template keyed to `plan.md` + `reference/*.md`. Primary structure: IGS class demo. Secondary section organisation: Fisser (2024). |
| 2026-04-17 | [`2026-04-17_restart-with-igs-class`](../_archive/overleaf-sessions/2026-04-17_restart-with-igs-class) | Earlier IGS-class restart. Carried old 3-class / 323-chip / 240-scene numbers that predate `plan.md`. |
| 2026-04-17 | [`2026-04-17_initial-import`](../_archive/overleaf-sessions/2026-04-17_initial-import) | Initial Overleaf import. Same outdated numbers as above. |

See [`../_archive/README.md`](../_archive/README.md) for per-file provenance.
