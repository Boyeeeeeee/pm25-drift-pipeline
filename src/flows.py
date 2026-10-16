import subprocess
import sys
from pathlib import Path

from prefect import flow, serve, task

ROOT = Path(__file__).resolve().parent.parent


def run_script(name: str) -> None:
    subprocess.run([sys.executable, f"src/{name}"], cwd=ROOT, check=True)


@task(retries=2, retry_delay_seconds=300)
def ingest():
    run_script("ingest.py")


@task
def monitor():
    run_script("monitor.py")


@task
def train():
    run_script("train.py")


@flow(name="daily-ingest-monitor")
def daily():
    ingest()
    monitor()


@flow(name="weekly-retrain")
def weekly():
    ingest()
    train()
    monitor()


if __name__ == "__main__":
    serve(
        daily.to_deployment(name="daily", cron="0 7 * * *", tags=["pm25"]),
        weekly.to_deployment(name="weekly", cron="0 8 * * 1", tags=["pm25"]),
    )