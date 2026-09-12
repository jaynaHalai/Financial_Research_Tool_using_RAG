# Import Path so we can work with files and folders
from pathlib import Path

# Import BeautifulSoup so we can parse the HTML document
from bs4 import BeautifulSoup

# Tell Python where the ARM annual report is located
html_file = Path("data/raw/SEC Filing – Arm®.html")

# Choose where to save the cleaned text
text_file = Path("data/raw/arm_report.txt")

# Read the HTML file
html = html_file.read_text(encoding="utf-8", errors="replace")

# Parse the HTML
soup = BeautifulSoup(html, "html.parser")

# Remove XBRL metadata sections but keep the filing itself
for tag_name in ["ix:header", "ix:hidden"]:
    for tag in soup.find_all(tag_name):
        tag.decompose()

# Extract readable text
text = soup.get_text(separator="\n", strip=True)

# Remove obvious website navigation from the beginning
noise = [
    "SEC Filing – Arm®",
    "Download DOC",
    "Download PDF",
    "Download XLS",
    "Download XBRL",
]

for item in noise:
    text = text.replace(item, "", 1)

# Clean up excessive blank lines
lines = [line.strip() for line in text.splitlines()]
lines = [line for line in lines if line]

# Put the cleaned lines back together
cleaned_text = "\n".join(lines)

# Save the cleaned document
text_file.write_text(cleaned_text, encoding="utf-8")

# Confirm the result
print(f"Cleaned characters: {len(cleaned_text):,}")
print(f"Saved cleaned text to: {text_file}")

