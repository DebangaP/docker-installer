#!/usr/bin/env python3
"""Script to extract CSS and JavaScript from dashboard.html"""

import re

# Read the HTML file
with open('templates/dashboard.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Extract CSS (between <style> and </style>)
css_match = re.search(r'<style>(.*?)</style>', content, re.DOTALL)
if css_match:
    css_content = css_match.group(1)
    with open('static/dashboard.css', 'w', encoding='utf-8') as f:
        f.write(css_content)
    print(f"Extracted CSS: {len(css_content)} characters")

# Extract JavaScript (between <script> and </script>, but after </style>)
# Find the first <script> after </style>
style_end = content.find('</style>')
script_start = content.find('<script>', style_end)
if script_start != -1:
    # Find the last </script> before </body>
    body_end = content.rfind('</body>')
    script_end = content.rfind('</script>', script_start, body_end)
    if script_end != -1:
        js_content = content[script_start + 8:script_end]  # +8 to skip '<script>'
        with open('static/dashboard.js', 'w', encoding='utf-8') as f:
            f.write(js_content)
        print(f"Extracted JavaScript: {len(js_content)} characters")

print("Extraction complete!")

