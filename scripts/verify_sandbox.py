"""Manual sandbox verification (no LLM calls). Run: python scripts/verify_sandbox.py

Checks the operator's hard requirement: sandboxed code cannot read arbitrary host
files or reach the network, but CAN read the data and report a result.
"""

from pathlib import Path

from geryon.codeflow.runner import get_latest_etl_output
from geryon.sandbox import run_script

DATA = get_latest_etl_output(Path.home() / "data" / "geryon_data")
print(f"data dir: {DATA}\n")

# 1. Legit analysis: read parquet, compute something, report a result.
ok = run_script(
    """
from geryon_runtime import db, report
con = db()
n = con.execute("SELECT COUNT(*) FROM clinical_patient").fetchone()[0]
print("patients:", n)
# scratch mutation must NOT touch source parquet:
con.execute("CREATE TABLE scratch AS SELECT 1 AS x")
report(effect_size=1.23, effect_size_type="demo", p_value=0.04, n_a=n, summary="ok")
""",
    DATA,
)
print("[1] legit analysis")
print(f"    success={ok.success} result={ok.result}")
print(f"    stdout={ok.stdout.strip()!r} err={ok.error}\n")

# 2. Try to read an arbitrary host file — must FAIL (file not in container).
host_file = str(Path.home() / ".zshrc")
leak = run_script(
    f"""
try:
    data = open({host_file!r}).read()
    print("LEAKED", len(data))
except Exception as e:
    print("blocked:", type(e).__name__)
""",
    DATA,
)
print("[2] read arbitrary host file (~/.zshrc)")
print(f"    stdout={leak.stdout.strip()!r}  -> want 'blocked'\n")

# 3. Try network egress — must FAIL (--network none).
net = run_script(
    """
import socket
try:
    socket.create_connection(("8.8.8.8", 53), timeout=5)
    print("NETWORK REACHABLE")
except Exception as e:
    print("blocked:", type(e).__name__)
""",
    DATA,
)
print("[3] network egress")
print(f"    stdout={net.stdout.strip()!r}  -> want 'blocked'\n")

# 4. Confirm source parquet is unchanged on disk (mounted read-only).
mut = run_script(
    """
from geryon_runtime import db
con = db()
try:
    con.execute("INSERT INTO clinical_patient SELECT * FROM clinical_patient LIMIT 1")
    print("MUTATED VIEW")
except Exception as e:
    print("blocked write to source:", type(e).__name__)
""",
    DATA,
)
print("[4] attempt to mutate source data")
print(f"    stdout={mut.stdout.strip()!r}\n")

print("DONE — checks 2 and 3 should say 'blocked'; check 1 should succeed.")
