# Digital-twin CAD assets

Drop CAD meshes here to skin the digital-twin scene (`digital_twin.py`) with the real
table and crate instead of the box placeholders.

## Files

| File        | What it is | **Required mesh origin** |
|-------------|------------|--------------------------|
| `table.stl` | the table  | **AprilTag ID 0** (the table anchor tag) |
| `crate.stl` | the crate  | **AprilTag ID 10** (the crate bundle anchor tag) |

Both are auto-detected if present. Override the path with `--table-stl` / `--crate-stl`,
or skip one to keep its box placeholder.

## Why the origin matters

The whole scene lives in ONE frame: the table AprilTag (id 0) is the world origin.

- **table.stl** is welded at the world origin, so its mesh origin must be where tag 0
  sits (on the table surface, tag-frame axes). Model the table in CAD with the part
  origin at the tag-0 location and the same axis convention, export, drop here.
- **crate.stl** is placed every frame at the crate pose the tracker reports. That pose
  is the **crate bundle frame = tag 10** (the calibrated anchor). So the crate mesh
  origin must be at tag 10 (the tag on the inside-centre base of the crate). Then the
  mesh rides exactly with the tracked crate.

If a mesh looks offset or rotated in the twin, the STL origin/axes don't match the tag —
re-export from CAD with the part origin on the tag.

## Axis convention — you modelled it right (Z into the surface)

CAD the parts in the AprilTag's own frame: origin on the tag, **Z pointing INTO the
surface** (into the table for tag 0; into/through the crate's interior base for tag 10).
That matches how the tags are detected on this rig. The twin handles the rest:

- **Crate:** no adjustment needed. It's placed every frame at the crate's *measured*
  pose (tag 10 in the world), and since you authored it in tag 10's frame, it lands
  perfectly aligned automatically.
- **Table:** the twin world is **Z-UP** (the rig detects tag 0 with Z pointing DOWN into
  the table, and the pipeline flips it to Z-up so the robots stand upright — the
  `--table-z-down` default). Your table is authored in the tag-0 (Z-down) frame, so the
  twin **automatically flips it 180° about X** to sit right-side-up (tabletop at z=0,
  body hanging below). This is on by default. If you ever recorded with
  `--no-table-z-down`, or you authored the table Z-up, pass `--no-table-z-down` to
  digital_twin.py so it skips the flip. (Only the table needs this — the crate rides its
  tracked pose either way.)

## Fine-alignment nudges

If a mesh is *close* but off by a few mm / degrees (e.g. the tag isn't perfectly at the
modelled origin), nudge it without re-CADding:

- `--table-offset X Y Z` — translation (m), in the WORLD frame.
- `--table-rot R P Y` — rotation (deg roll/pitch/yaw), world frame, about the table origin.
- `--crate-offset X Y Z` — translation (m), in the CRATE's own (tag-10) frame.
- `--crate-rot R P Y` — rotation (deg), in the crate frame.

The crate nudge is applied in the crate's frame, so it tracks the crate as it's
manipulated (a constant CAD-vs-tag correction). Example:
`--crate-offset 0.004 0 0 --crate-rot 0 0 1.5`.

### Or nudge LIVE in the viewer (no relaunch)

With a mesh loaded, tune the fit interactively in the OpenCV window (keys shown in the
HUD):

- `1` / `2` — select **table** / **crate** to nudge
- `h`/`l` = X∓ · `j`/`k` = Y∓ · `n`/`m` = Z∓ (translate)
- `[` / `]` = yaw ∓
- `-` / `=` = halve / double the step (starts 1 mm / 0.5°)
- `0` = reset the selected object · `p` = print the current values as `--*-offset/--*-rot`
  flags (copy them into your launch command to make them permanent)

The table nudges in the world frame; the crate nudges in its own frame and stays glued
to the crate as it moves.

## Units

STL is unitless; MuJoCo assumes metres. If your CAD exports millimetres, pass
`--table-stl-scale 0.001` / `--crate-stl-scale 0.001`.

## Formats

Binary or ASCII `.stl`. (MuJoCo also loads `.obj`/`.msh`; if you have one of those,
point `--table-stl`/`--crate-stl` straight at it.) Keep meshes reasonably decimated —
very high-poly CAD can slow the viewer.
