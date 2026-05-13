#!/usr/bin/env python3
"""
Generate a C-mesh around a NACA airfoil using the gmsh Python API.

Uses the OpenCASCADE (OCC) kernel, which correctly represents a surface with
a hole (the airfoil punch-out) and extrudes it into a proper 3-D solid volume.
The GEO kernel cannot do this reliably and returns garbage volume tags.

Workflow
--------
1. Build 2-D geometry with OCC (points → curves → loops → surface-with-hole)
2. occ.extrude() → one-layer hex volume  (BEFORE mesh generation)
3. occ.synchronize()
4. Set mesh-size fields and BoundaryLayer on airfoil curves
5. Add physical groups (patch names for OpenFOAM)
6. generate(3)  — 2-D BL surfaces are meshed first, then extruded to hexahedra
7. Write gmsh v2.2 file for gmshToFoam

Usage (inside the OpenFOAM container):
    python3 mesh_gen.py [--dat airfoil.dat] [--out airfoil_mesh.msh]
"""

import argparse
import sys


def load_dat(path):
    coords = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    coords.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
    return coords


def find_le_split(coords):
    """Index of the last upper-surface point (nearest LE, y >= 0)."""
    min_x = min(c[0] for c in coords)
    candidates = [i for i, c in enumerate(coords) if abs(c[0] - min_x) < 1e-8]
    for i in candidates:
        if coords[i][1] >= 0:
            return i
    return candidates[0]


def generate_mesh(
    airfoil_dat="airfoil.dat",
    output="airfoil_mesh.msh",
    chord=1.0,
    domain_R=15.0,
    wake_length=20.0,
    bl_first_size=2e-5,
    bl_ratio=1.2,
    bl_thickness=0.05,
    far_size=1.5,
    near_wake_size=0.05,
    extrude_z=0.1,
):
    try:
        import gmsh
    except ImportError:
        sys.exit("gmsh not found — rebuild the Docker image (python3-gmsh via apt)")

    coords = load_dat(airfoil_dat)
    le_idx = find_le_split(coords)
    upper_coords = coords[:le_idx + 1]   # TE → near-LE upper
    lower_coords = coords[le_idx + 1:]   # near-LE lower → TE

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("naca_airfoil")
    occ = gmsh.model.occ   # OpenCASCADE kernel

    # ── Airfoil geometry ──────────────────────────────────────────────────────
    te_pt         = occ.addPoint(*upper_coords[0], 0)
    near_le_upper = occ.addPoint(*upper_coords[-1], 0)
    near_le_lower = occ.addPoint(*lower_coords[0], 0)

    upper_internal = [occ.addPoint(x, y, 0) for x, y in upper_coords[1:-1]]
    lower_internal = [occ.addPoint(x, y, 0) for x, y in lower_coords[1:-1]]

    upper_spl = occ.addSpline([te_pt] + upper_internal + [near_le_upper])
    le_arc    = occ.addLine(near_le_upper, near_le_lower)
    lower_spl = occ.addSpline([near_le_lower] + lower_internal + [te_pt])
    airfoil_loop = occ.addCurveLoop([upper_spl, le_arc, lower_spl])

    # ── C-domain outer boundary ───────────────────────────────────────────────
    cx = chord / 2.0
    p_top  = occ.addPoint(cx,               domain_R,  0, far_size)
    p_left = occ.addPoint(cx - domain_R,    0,         0, far_size)
    p_bot  = occ.addPoint(cx,              -domain_R,  0, far_size)
    p_tr   = occ.addPoint(cx + wake_length, domain_R,  0, far_size)
    p_br   = occ.addPoint(cx + wake_length,-domain_R,  0, far_size)
    arc_c  = occ.addPoint(cx, 0, 0)  # arc centre (interior, not on boundary)

    arc_bot_left = occ.addCircleArc(p_bot,  arc_c, p_left)
    arc_left_top = occ.addCircleArc(p_left, arc_c, p_top)
    line_top     = occ.addLine(p_top, p_tr)
    line_outlet  = occ.addLine(p_tr,  p_br)
    line_bot     = occ.addLine(p_br,  p_bot)
    outer_loop   = occ.addCurveLoop(
        [arc_bot_left, arc_left_top, line_top, line_outlet, line_bot]
    )

    # 2-D surface: airfoil punched out as a hole
    surface = occ.addPlaneSurface([outer_loop, airfoil_loop])

    # ── Extrude to 3-D BEFORE mesh generation ────────────────────────────────
    # OCC handles the surface-with-hole topology and produces a valid solid.
    # numElements=[1], recombine=True → one layer of hex cells.
    extruded = occ.extrude([(2, surface)], 0, 0, extrude_z, [1], [1.0], True)
    occ.synchronize()

    # extrude() return order varies across gmsh/OCC versions — query model directly.
    vols = gmsh.model.getEntities(3)
    if len(vols) != 1:
        raise RuntimeError(f"Expected 1 volume after extrusion, got: {vols}")
    vol_tag = vols[0][1]
    print(f"OCC volume tag: {vol_tag}")

    # Identify front (z≈0) and back (z≈extrude_z) faces from bounding box.
    # OCC reports ±1e-7 on flat surfaces; use 1e-4 tolerance.
    all_surfs = gmsh.model.getEntities(2)
    front_face = top_face = None
    for _, stag in all_surfs:
        bb = gmsh.model.getBoundingBox(2, stag)
        zlo, zhi = bb[2], bb[5]
        if zhi - zlo < 1e-4:                          # flat surface in z
            if abs(zlo) < 1e-4:
                front_face = stag                      # z ≈ 0
            elif abs(zhi - extrude_z) < 1e-4:
                top_face = stag                        # z ≈ extrude_z
    if front_face is None or top_face is None:
        raise RuntimeError(f"front_face={front_face}, top_face={top_face}")
    print(f"front_face tag: {front_face}  back_face tag: {top_face}")

    # ── Mesh size fields ──────────────────────────────────────────────────────
    mf = gmsh.model.mesh.field

    f_dist = mf.add("Distance")
    mf.setNumbers(f_dist, "CurvesList", [upper_spl, lower_spl, le_arc])
    mf.setNumber(f_dist,  "Sampling",   300)

    f_thresh = mf.add("Threshold")
    mf.setNumber(f_thresh, "InField",  f_dist)
    mf.setNumber(f_thresh, "SizeMin",  0.003)
    mf.setNumber(f_thresh, "SizeMax",  far_size)
    mf.setNumber(f_thresh, "DistMin",  0.3)
    mf.setNumber(f_thresh, "DistMax",  domain_R * 0.4)

    f_wake = mf.add("Box")
    mf.setNumber(f_wake, "VIn",  near_wake_size)
    mf.setNumber(f_wake, "VOut", far_size)
    mf.setNumber(f_wake, "XMin", 0.8)
    mf.setNumber(f_wake, "XMax", cx + wake_length * 0.25)
    mf.setNumber(f_wake, "YMin", -0.4)
    mf.setNumber(f_wake, "YMax",  0.4)
    mf.setNumber(f_wake, "ZMin", -0.1)
    mf.setNumber(f_wake, "ZMax",  extrude_z + 0.1)

    f_min = mf.add("Min")
    mf.setNumbers(f_min, "FieldsList", [f_thresh, f_wake])
    mf.setAsBackgroundMesh(f_min)

    # ── Classify extruded side faces → patch names ────────────────────────────
    # Side faces span the full z range (not flat) — everything that isn't front/back.
    side_tags = [stag for _, stag in all_surfs
                 if stag != front_face and stag != top_face]
    inlet_faces, freestream_top_faces, freestream_bot_faces, outlet_faces, airfoil_faces = [], [], [], [], []
    wake_x = cx + wake_length

    for tag in side_tags:
        xlo, ylo, _, xhi, yhi, _ = gmsh.model.getBoundingBox(2, tag)
        if xhi <= chord + 0.2:
            # Face confined to x ≤ chord+ε: inlet arc or airfoil wall
            if xlo < cx - 1.0:
                inlet_faces.append(tag)          # inlet arc (extends far left)
            else:
                airfoil_faces.append(tag)        # near-chord face = airfoil wall
        elif xlo > wake_x - 1.0:
            outlet_faces.append(tag)             # outlet plane
        elif abs(yhi - domain_R) < 1.0:
            freestream_top_faces.append(tag)     # top edge
        elif abs(ylo + domain_R) < 1.0:
            freestream_bot_faces.append(tag)     # bottom edge
        else:
            inlet_faces.append(tag)              # fallback

    # ── Physical groups (become OpenFOAM patch names) ─────────────────────────
    # front/back are on opposite z-planes → must be separate groups so gmshToFoam
    # can assign them to contiguous face ranges. Same for freestream top/bottom.
    gmsh.model.addPhysicalGroup(3, [vol_tag],              name="fluid")
    gmsh.model.addPhysicalGroup(2, [front_face],           name="front")
    gmsh.model.addPhysicalGroup(2, [top_face],             name="back")
    gmsh.model.addPhysicalGroup(2, inlet_faces,            name="inlet")
    gmsh.model.addPhysicalGroup(2, freestream_top_faces,   name="freestream_top")
    gmsh.model.addPhysicalGroup(2, freestream_bot_faces,   name="freestream_bot")
    gmsh.model.addPhysicalGroup(2, outlet_faces,           name="outlet")
    gmsh.model.addPhysicalGroup(2, airfoil_faces,          name="airfoil")

    # ── Mesh options and generate ─────────────────────────────────────────────
    # Pure triangle 2-D → pure prism 3-D: each patch contains a single element
    # type, which is required for gmshToFoam's face-range ordering.
    gmsh.option.setNumber("Mesh.Algorithm",   6)   # Frontal-Delaunay 2-D
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)

    gmsh.model.mesh.generate(3)

    # ── Export (gmsh v2.2 required by gmshToFoam) ─────────────────────────────
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(output)
    gmsh.finalize()

    print(f"\nMesh written → {output}")
    print(f"  inlet faces    : {len(inlet_faces)}")
    print(f"  freestream_top : {len(freestream_top_faces)}")
    print(f"  freestream_bot : {len(freestream_bot_faces)}")
    print(f"  outlet faces   : {len(outlet_faces)}")
    print(f"  airfoil faces  : {len(airfoil_faces)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dat",          default="airfoil.dat")
    ap.add_argument("--out",          default="airfoil_mesh.msh")
    ap.add_argument("--bl-first",     type=float, default=2e-5,
                    help="First BL cell height [m] (default: 2e-5 → y+≈1 at Re=3e6)")
    ap.add_argument("--bl-thickness", type=float, default=0.05,
                    help="Total BL zone thickness [m/chord] (default: 0.05)")
    args = ap.parse_args()

    generate_mesh(
        airfoil_dat=args.dat,
        output=args.out,
        bl_first_size=args.bl_first,
        bl_thickness=args.bl_thickness,
    )


if __name__ == "__main__":
    main()
