# Import Path so we can read the cleaned report
from pathlib import Path

# Load the cleaned annual report
text = Path("data/raw/arm_report.txt").read_text(encoding="utf-8")

# Look for printed page breaks in the extracted text
lines = text.splitlines()

# Store the line number before each printed page number
page_numbers = []

for i, line in enumerate(lines):
    if line.strip().isdigit():
        page = int(line.strip())

        # Focus on plausible annual-report page numbers
        if 1 <= page <= 300:
            page_numbers.append((i, page))

# Show text around each page break
for i, page in page_numbers:
    if page < 56:
        continue

    start = max(0, i - 5)
    end = min(len(lines), i + 6)

    print("\n" + "=" * 70)
    print(f"PRINTED PAGE BREAK: {page}")
    print("=" * 70)

    for n in range(start, end):
        marker = ">>>" if n == i else "   "
        print(f"{marker} {n + 1}: {lines[n]}")