"""RecallIQ data pipeline for the Government of Canada Recalls and Safety Alerts open dataset.

Stages (see build.py):  load -> filter/normalize -> group -> sample -> export (JSONL + SQLite)
"""

PIPELINE_VERSION = "1.0.0"
