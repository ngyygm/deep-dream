#!/usr/bin/env python3
"""Build a minimal lucide.js containing only the icons used by DeepDream."""

import re

ICONS_KEBAB = [
    "activity", "alert-triangle", "arrow-left", "arrow-right",
    "arrow-right-left", "bar-chart-3", "box", "brain", "camera",
    "check", "check-circle", "chevron-down", "chevron-left",
    "chevron-right", "circle", "circle-dot", "clock", "copy",
    "crosshair", "database", "dice-5", "eraser", "eye", "file-text",
    "film", "git-branch", "git-commit", "git-compare", "git-merge",
    "info", "keyboard", "layers", "layout-dashboard", "layout-grid",
    "link", "list", "list-checks", "maximize-2", "menu",
    "message-circle", "moon", "network", "pause", "pencil", "play",
    "plus", "refresh-cw", "rotate-ccw", "route", "scan",
    "scroll-text", "search", "send", "share-2", "skip-back",
    "skip-forward", "sliders-horizontal", "sparkles", "sprout",
    "square", "sun", "terminal", "trash-2", "upload", "upload-cloud",
    "x", "x-circle", "zap",
]


def kebab_to_pascal(name):
    """Convert kebab-case to PascalCase, matching lucide's toPascalCase."""
    camel = re.sub(r'^([A-Z])|[\s\-_]+(\w)', lambda m: (m.group(2) or m.group(1)).upper(), name)
    return camel[0].upper() + camel[1:]


PASCAL_NAMES = set(kebab_to_pascal(n) for n in ICONS_KEBAB)

SRC = "lucide.js"
DST = "lucide-min.js"

with open(SRC, "r") as f:
    full = f.read()

# 1. Find the iconAndAliases mapping: PascalCase -> VariableName
aliases_section = re.search(
    r'var iconAndAliases = /\*#__PURE__\*/Object\.freeze\(\{(.*?)\}\);',
    full, re.DOTALL
)
if not aliases_section:
    raise RuntimeError("Could not find iconAndAliases section")

alias_text = aliases_section.group(1)

# Build map: PascalCase -> JS variable name
icon_to_var = {}
for line in alias_text.strip().split("\n"):
    line = line.strip().rstrip(",")
    if ":" not in line or line.startswith("__proto__"):
        continue
    pascal, var = line.split(":", 1)
    pascal = pascal.strip()
    var = var.strip()
    if pascal in PASCAL_NAMES:
        icon_to_var[pascal] = var

print(f"Found {len(icon_to_var)} of {len(PASCAL_NAMES)} needed icons in alias map")
missing = PASCAL_NAMES - set(icon_to_var.keys())
if missing:
    print(f"WARNING: Missing icons: {missing}")

# 2. Find all unique JS variable names we need
needed_vars = set(icon_to_var.values())
print(f"Need {len(needed_vars)} unique icon variables")

# 3. Extract icon variable definitions (using const, not var)
icon_defs = {}
for m in re.finditer(r'const\s+(\w+)\s*=\s*\[', full):
    var_name = m.group(1)
    if var_name in needed_vars:
        start = m.start()
        # Find end of array by matching brackets
        depth = 0
        i = full.index('[', start)
        while i < len(full):
            if full[i] == '[':
                depth += 1
            elif full[i] == ']':
                depth -= 1
                if depth == 0:
                    # Include the semicolon if present
                    end = i + 1
                    if end < len(full) and full[end] == ';':
                        end += 1
                    icon_defs[var_name] = full[start:end]
                    break
            i += 1

print(f"Extracted {len(icon_defs)} icon definitions")

# 4. Extract helpers section (from defaultAttributes to first icon definition)
helper_start = full.index("const defaultAttributes")
first_icon_match = re.search(r'const [A-Z]\w+ = \[', full)
helpers_end = first_icon_match.start()
helpers = full[helper_start:helpers_end]

# 5. Extract the createIcons and replaceElement functions after iconAndAliases
# Find createIcons function
createicons_match = re.search(
    r'const createIcons = \{[^}]+\} = \{\}\) => \{(.*?)\n  \};',
    full, re.DOTALL
)

# Actually, let's extract everything after iconAndAliases until exports
# That section contains createIcons and its helpers
after_aliases = aliases_section.end()
# Find where exports start
exports_start = full.index("exports.AArrowDown")
createicons_section = full[after_aliases:exports_start]

# 6. Build the output
# Icon definitions
icon_def_lines = "\n\n".join(v for k, v in sorted(icon_defs.items()))

# Alias mapping
alias_lines = []
for pascal in sorted(PASCAL_NAMES):
    if pascal in icon_to_var:
        alias_lines.append(f"    {pascal}: {icon_to_var[pascal]},")

# Export lines
export_lines = []
for pascal in sorted(PASCAL_NAMES):
    if pascal in icon_to_var:
        export_lines.append(f"  exports.{pascal} = {icon_to_var[pascal]};")

# Extract original UMD wrapper from source
umd_header_end = full.index("'use strict';") + len("'use strict';")
umd_header = full[:umd_header_end]

# Extract footer
umd_footer = "})"
footer_pos = full.rindex(umd_footer)
# Go back to find the matching closing
umd_footer = full[full.index("exports.icons = iconAndAliases;"):]

output_parts = [
    umd_header,
    "\n\n",
    helpers,
    f"\n  // -- DeepDream minimal icon set ({len(icon_defs)} icons) --\n",
    icon_def_lines,
    "\n\n  var iconAndAliases = /*#__PURE__*/Object.freeze({\n    __proto__: null,\n",
    "\n".join(alias_lines),
    "\n  });\n\n",
    createicons_section,
    "\n".join(export_lines),
    "\n  exports.createElement = createElement;\n  exports.createIcons = createIcons;\n  exports.icons = iconAndAliases;\n\n",
    "}));\n",
]
output = "".join(output_parts)

with open(DST, "w") as f:
    f.write(output)

import os
orig_size = os.path.getsize(SRC)
new_size = os.path.getsize(DST)
print(f"\nDone! {orig_size} -> {new_size} ({new_size/orig_size*100:.1f}%)")
