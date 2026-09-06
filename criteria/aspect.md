---
name: An aspect other than west
description: A non-western aspect is worth real credit; west is a mild negative.
weight: 2
grade:
  grader: best_aspect
unknown: skip
fallback: model
ask: "Which way do the main living-room windows face?"
---

## What this measures
Which way the main living space's windows face.

## Rubric
**10** — north, north-east, east, south-east or south.
**7** — south-west or north-west.
**4** — west only.
**0** — not used. A west-facing flat is a mild negative and never a problem.

## Evidence, and what not to read into it
`window_aspects` is the answer where it exists, worked out from the compass
drawn on the floorplan. It is a list, because a corner flat faces more than one
way, and **the best wall is the one that counts**: a flat facing north and west
is not a west-facing flat.

Where the field is empty and the advert states an aspect in words — "south
facing", "the living room faces west" — use that, and say so in the evidence.
"Bright", "sunny" and "light" are not aspects. Where both exist and disagree,
grade on the field and raise the disagreement as a concern. Never infer an
aspect from the address, from which way a balcony is drawn, from a room's name,
or from the light in a photograph. Return null where nothing states it.
