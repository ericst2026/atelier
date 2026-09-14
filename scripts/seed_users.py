"""Create accounts from outside a container.

    python scripts/seed_users.py --students 24 --password lab2026
    python scripts/seed_users.py --csv class.csv

Inside the running API container, call the module directly instead — it is part of
the package, so no path juggling is needed:

    docker compose exec api python -m atelier.seed_users --students 24
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from atelier.seed_users import main  # noqa: E402

raise SystemExit(main())
