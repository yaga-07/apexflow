#!/usr/bin/env python3
"""
Fix patch types in constant/polyMesh/boundary after gmshToFoam.

gmshToFoam sets every patch to type 'patch'. This script corrects the types
so OpenFOAM applies the right numerical treatment during simpleFoam.
"""
import re

PATCH_TYPES = {
    "airfoil":          "wall",
    "front":            "empty",
    "back":             "empty",
    "inlet":            "patch",
    "outlet":           "patch",
    "freestream_top":   "patch",
    "freestream_bot":   "patch",
}

bpath = "constant/polyMesh/boundary"
with open(bpath) as f:
    content = f.read()

for patch, ptype in PATCH_TYPES.items():
    # Match: patchName { ... type word; ... } (no nested braces in OF boundary blocks)
    pattern = rf'(\b{re.escape(patch)}\b\s*\{{[^}}]*?\btype\s+)\w+'
    content  = re.sub(pattern, rf'\g<1>{ptype}', content, flags=re.DOTALL)

with open(bpath, "w") as f:
    f.write(content)

print("Boundary types updated:")
for name, btype in PATCH_TYPES.items():
    print(f"  {name:16s} → {btype}")
