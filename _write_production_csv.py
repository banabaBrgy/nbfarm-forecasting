import json
from pathlib import Path

source = Path(__file__).resolve().parents[1] / "nb_system" / "database" / "seeders" / "data" / "client_egg_production.json"
target = Path(__file__).resolve().parent / "production.csv"
days = json.loads(source.read_text(encoding="utf-8"))
lines = ["date,eggs"]
for day in days:
    eggs = sum(int(batch.get("eggs_produced") or 0) for batch in day.get("batches") or [])
    lines.append(f"{day['date']},{eggs}")
target.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {target} ({len(lines) - 1} days)")
