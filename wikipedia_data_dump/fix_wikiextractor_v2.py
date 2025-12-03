import os
import wikiextractor

# Locate the file
package_dir = os.path.dirname(wikiextractor.__file__)
target_file = os.path.join(package_dir, 'extract.py')

print(f"Reading file: {target_file}")

with open(target_file, 'r', encoding='utf-8') as f:
    content = f.read()

# The specific string causing the error in Python 3.12
broken_string = "r'\[(\w+):[^\]]+\](?i)'"
fixed_string = "r'(?i)\[(\w+):[^\]]+\]'"

if broken_string in content:
    print("Found broken regex string! Patching...")
    new_content = content.replace(broken_string, fixed_string)

    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("✅ File patched successfully.")
else:
    print("⚠️ Pattern not found using exact string match.")
    print("Please perform the manual fix described above.")