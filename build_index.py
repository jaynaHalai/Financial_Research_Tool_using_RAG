# Import Path so we can read the cleaned report
from pathlib import Path

# Load the cleaned annual report
report = Path("data/raw/arm_report.txt").read_text(encoding="utf-8")

# List the sections we want to locate
sections = [
    "Item 3.",
    "Item 4.",
    "Item 5.",
    "Item 7.",
    "Item 8.",
    "Item 10.",
    "Item 11.",
    "Item 15.",
    "Item 16.",
]

# Find the first occurrence of each section
for section in sections:
    position = report.find(section)
    print(f"{section:<10} character position: {position:,}")