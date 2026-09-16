"""Create accounts, from a CSV or as a numbered class.

Inside the running API container, which is where it is usually wanted:

    docker compose exec api python -m atelier.seed_users --students 24 --password lab2026
    docker compose exec -T api python -m atelier.seed_users --csv - < class.csv

Outside it, anywhere ATELIER_DATABASE_URL points at the database:

    python scripts/seed_users.py --students 24

CSV columns: username,name,role,password. Role defaults to student; a missing
password is generated. Existing usernames are skipped rather than overwritten, so
re-running after adding a few names to the CSV does the right thing.
"""
import argparse
import csv
import io
import secrets
import sys
from typing import Any, Optional

from sqlalchemy import select

from .auth import hash_password
from .db import SessionLocal, init_db
from .models import User


def rows_from_csv(source: Any) -> list[dict[str, str]]:
    out = []
    for r in csv.DictReader(source):
        if not (r.get("username") or "").strip():
            continue
        out.append({
            "username": r["username"].strip().lower(),
            "name": (r.get("name") or "").strip(),
            "role": (r.get("role") or "student").strip(),
            "password": (r.get("password") or secrets.token_urlsafe(6)).strip(),
        })
    return out


def rows_for_class(n: int, prefix: str = "student", password: Optional[str] = None) -> list[dict[str, str]]:
    width = len(str(n))
    return [{
        "username": f"{prefix}{i:0{width}d}",
        "name": f"Student {i}",
        "role": "student",
        "password": password or secrets.token_urlsafe(6),
    } for i in range(1, n + 1)]


def create(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[str]]:
    init_db()
    created, skipped = [], []
    with SessionLocal() as db:
        for r in rows:
            if db.scalar(select(User).where(User.username == r["username"])):
                skipped.append(r["username"])
                continue
            db.add(User(
                username=r["username"],
                name=r["name"],
                role=r["role"] if r["role"] in ("admin", "teacher", "student") else "student",
                password_hash=hash_password(r["password"]),
            ))
            created.append(r)
        db.commit()
    return created, skipped


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m atelier.seed_users")
    ap.add_argument("--csv", help="CSV file with username,name,role,password — or - for stdin")
    ap.add_argument("--students", type=int, help="create N numbered student accounts")
    ap.add_argument("--prefix", default="student", help="username prefix for --students")
    ap.add_argument("--password", help="shared password for --students (random per user if omitted)")
    ap.add_argument("--out", default="accounts.csv", help="where to write the credentials of accounts created")
    args = ap.parse_args(argv)

    if args.csv == "-":
        rows = rows_from_csv(io.StringIO(sys.stdin.read()))
    elif args.csv:
        with open(args.csv, encoding="utf-8-sig") as fh:
            rows = rows_from_csv(fh)
    elif args.students:
        rows = rows_for_class(args.students, args.prefix, args.password)
    else:
        ap.error("pass --csv or --students")

    created, skipped = create(rows)
    for u in skipped:
        print(f"skip {u} (exists)")
    if not created:
        print("nothing to do")
        return 0
    # print as well as write: inside a container the file is easy to lose
    print("username,name,role,password")
    for r in created:
        print(f"{r['username']},{r['name']},{r['role']},{r['password']}")
    try:
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["username", "name", "role", "password"])
            w.writeheader()
            w.writerows(created)
        print(f"\ncreated {len(created)} accounts; credentials also written to {args.out}")
    except OSError as exc:
        print(f"\ncreated {len(created)} accounts; could not write {args.out} ({exc}) — copy them from above")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
