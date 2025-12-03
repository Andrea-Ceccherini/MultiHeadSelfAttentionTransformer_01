import os
import sys
import wikiextractor

# Locate the library file
package_dir = os.path.dirname(wikiextractor.__file__)
target_file = os.path.join(package_dir, 'extract.py')

print(f"🔧 Patching file: {target_file}")

with open(target_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
patched = False

for line in lines:
    # This is the specific broken line in wikiextractor 3.0.6
    if 'ExtLinkBracketedRegex = re.compile' in line and '(?i)' in line:
        # We move (?i) from the end to the start
        # Original: ... r'\[(\w+):[^\]]+\](?i)')
        # Fixed:    ... r'(?i)\[(\w+):[^\]]+\]')

        if line.strip().endswith("(?i)')"):
            print("   -> Found broken regex (Type 1). Fixing...")
            new_line = line.replace("(?i)')", "')").replace("r'[", "r'(?i)[")
            new_lines.append(new_line)
            patched = True
            continue

        # Sometimes it looks slightly different depending on version
        if '(?i)' in line:
            print("   -> Found broken regex (Type 2). Moving flag to start...")
            # Naive replace for safety
            new_line = line.replace("(?i)", "")
            # Add it back after the opening quote
            if "r'" in new_line:
                new_line = new_line.replace("r'", "r'(?i)")
            elif 'r"' in new_line:
                new_line = new_line.replace('r"', 'r"(?i)')

            new_lines.append(new_line)
            patched = True
            continue

    new_lines.append(line)

if patched:
    with open(target_file, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print("✅ WikiExtractor patched successfully! You can now run your main script.")
else:
    print("⚠️  No broken regex found. It might already be patched or this is a different version.")