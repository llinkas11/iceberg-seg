"""
scripts/assign_chips.py
=======================
Command-line tool to assign chips to labelers from the terminal.

This is an alternative to using the admin web dashboard.
Useful for batch operations or scripted setup.

Usage
-----
# List all labelers
python scripts/assign_chips.py list-labelers

# List all chips (with assignment status)
python scripts/assign_chips.py list-chips

# Assign N chips to a specific labeler
python scripts/assign_chips.py assign --labeler "Alex Smith" --n 100

# Assign N chips filtered by region / SZA bin
python scripts/assign_chips.py assign --labeler "Alex Smith" --n 50 \\
    --region kq --sza-bin sza_65_70

# Distribute all chips evenly among all labelers
python scripts/assign_chips.py distribute

# Show assignment summary
python scripts/assign_chips.py summary
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import SessionLocal, init_db
from models import Chip, Labeler, Assignment


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_db():
    init_db()
    return SessionLocal()


def find_labeler(db, name: str) -> Labeler:
    labeler = db.query(Labeler).filter(Labeler.name == name).first()
    if not labeler:
        print(f"ERROR: No labeler named '{name}'. Registered labelers:")
        for l in db.query(Labeler).filter(Labeler.is_admin == False).all():
            print(f"  - {l.name}")
        sys.exit(1)
    return labeler


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_list_labelers(args):
    db = get_db()
    labelers = db.query(Labeler).filter(Labeler.is_admin == False).all()
    if not labelers:
        print("No labelers registered yet. Labelers register themselves via the web UI.")
        return

    print(f"\n{'Name':<30} {'Assigned':>8} {'Complete':>8} {'Pending':>8}")
    print("─" * 60)
    for l in labelers:
        asgns    = db.query(Assignment).filter(Assignment.labeler_id == l.id).all()
        total    = len(asgns)
        complete = sum(1 for a in asgns if a.status == "complete")
        pending  = total - complete
        print(f"{l.name:<30} {total:>8} {complete:>8} {pending:>8}")
    db.close()


def cmd_list_chips(args):
    db = get_db()
    q  = db.query(Chip)
    if args.region:
        q = q.filter(Chip.region == args.region.lower())
    if args.sza_bin:
        q = q.filter(Chip.sza_bin == args.sza_bin)

    chips = q.all()
    print(f"\n{'ID':>6} {'Region':>6} {'SZA Bin':<12} {'Preds':>5} {'Assigned':>8} {'Status'}")
    print("─" * 70)
    for chip in chips[:200]:
        asgns  = db.query(Assignment).filter(Assignment.chip_id == chip.id).all()
        status = ", ".join(set(a.status for a in asgns)) if asgns else "unassigned"
        print(f"{chip.id:>6} {chip.region or '':>6} {chip.sza_bin or '':<12} "
              f"{chip.prediction_count:>5} {len(asgns):>8} {status}")
    if len(chips) > 200:
        print(f"  … and {len(chips) - 200} more chips")
    db.close()


def cmd_assign(args):
    db      = get_db()
    labeler = find_labeler(db, args.labeler)

    already = {a.chip_id for a in
               db.query(Assignment).filter(Assignment.labeler_id == labeler.id).all()}

    q = db.query(Chip)
    if args.region:
        q = q.filter(Chip.region == args.region.lower())
    if args.sza_bin:
        q = q.filter(Chip.sza_bin == args.sza_bin)

    # Exclude chips already assigned to this labeler
    candidates = [c for c in q.all() if c.id not in already]

    # Prioritise unassigned chips
    def _priority(chip):
        return db.query(Assignment).filter(Assignment.chip_id == chip.id).count()

    candidates.sort(key=_priority)
    to_assign = candidates[:args.n]

    if not to_assign:
        print("No eligible chips found (all may already be assigned to this labeler).")
        db.close()
        return

    for chip in to_assign:
        db.add(Assignment(chip_id=chip.id, labeler_id=labeler.id))
    db.commit()
    db.close()

    print(f"Assigned {len(to_assign)} chips to '{labeler.name}'.")


def cmd_distribute(args):
    """Distribute all unassigned chips evenly among all non-admin labelers."""
    db       = get_db()
    labelers = db.query(Labeler).filter(Labeler.is_admin == False).all()
    if not labelers:
        print("No labelers registered. Cannot distribute.")
        db.close()
        return

    assigned_ids = {a.chip_id for a in db.query(Assignment).all()}
    unassigned   = db.query(Chip).filter(~Chip.id.in_(assigned_ids)).all()

    if not unassigned:
        print("All chips are already assigned.")
        db.close()
        return

    print(f"Distributing {len(unassigned)} chips among {len(labelers)} labelers…")
    for i, chip in enumerate(unassigned):
        labeler = labelers[i % len(labelers)]
        db.add(Assignment(chip_id=chip.id, labeler_id=labeler.id))
    db.commit()
    db.close()

    per = len(unassigned) / len(labelers)
    print(f"Done — ~{per:.0f} chips per labeler.")


def cmd_summary(args):
    db = get_db()
    total     = db.query(Chip).count()
    assigned  = db.query(Assignment).count()
    complete  = db.query(Assignment).filter(Assignment.status == "complete").count()
    labelers  = db.query(Labeler).filter(Labeler.is_admin == False).count()

    # Unassigned chips
    assigned_ids = {a.chip_id for a in db.query(Assignment).all()}
    unassigned   = db.query(Chip).count() - len(assigned_ids)

    print(f"""
Assignment Summary
──────────────────
Total chips      : {total}
Unassigned chips : {unassigned}
Total assignments: {assigned}
Complete         : {complete}
Labelers         : {labelers}
""")
    db.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Assign chips to labelers")
    sub    = parser.add_subparsers(dest="command", required=True)

    # list-labelers
    sub.add_parser("list-labelers", help="List all registered labelers")

    # list-chips
    lc = sub.add_parser("list-chips", help="List chips with assignment status")
    lc.add_argument("--region",  default=None)
    lc.add_argument("--sza-bin", default=None, dest="sza_bin")

    # assign
    as_ = sub.add_parser("assign", help="Assign chips to a labeler")
    as_.add_argument("--labeler", required=True, help="Labeler name (exact)")
    as_.add_argument("--n",       required=True, type=int, help="Number of chips to assign")
    as_.add_argument("--region",  default=None)
    as_.add_argument("--sza-bin", default=None, dest="sza_bin")

    # distribute
    sub.add_parser("distribute", help="Distribute unassigned chips among all labelers")

    # summary
    sub.add_parser("summary", help="Show overall assignment summary")

    args = parser.parse_args()
    {
        "list-labelers": cmd_list_labelers,
        "list-chips":    cmd_list_chips,
        "assign":        cmd_assign,
        "distribute":    cmd_distribute,
        "summary":       cmd_summary,
    }[args.command](args)


if __name__ == "__main__":
    main()
