"""Clean the saved SEC filing HTML into plain text for chunking.

The saved page wraps the filing in site navigation and inline XBRL metadata.
Both are stripped here so that everything downstream, including the printed
page numbers page_map.py depends on, works from the filing's own text.
"""

from bs4 import BeautifulSoup

from config import CLEANED_TEXT, SOURCE_HTML

# Site furniture that appears once, ahead of the filing itself.
NAVIGATION = [
    "SEC Filing – Arm®",
    "Download DOC",
    "Download PDF",
    "Download XLS",
    "Download XBRL",
]

html = SOURCE_HTML.read_text(encoding="utf-8", errors="replace")
soup = BeautifulSoup(html, "html.parser")

# Inline XBRL tagging duplicates every figure as hidden metadata; keeping it
# would put each number in the document twice.
for tag_name in ["ix:header", "ix:hidden"]:
    for tag in soup.find_all(tag_name):
        tag.decompose()

text = soup.get_text(separator="\n", strip=True)

for item in NAVIGATION:
    text = text.replace(item, "", 1)

lines = [line.strip() for line in text.splitlines()]
cleaned_text = "\n".join(line for line in lines if line)

CLEANED_TEXT.write_text(cleaned_text, encoding="utf-8")

print(f"Cleaned characters: {len(cleaned_text):,}")
print(f"Words: {len(cleaned_text.split()):,}")
print(f"Saved cleaned text to: {CLEANED_TEXT}")
