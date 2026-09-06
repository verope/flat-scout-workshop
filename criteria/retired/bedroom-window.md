---
name: Every bedroom has a window
description: An internal bedroom does not qualify at any price.
weight: 4
veto_at_or_below: 0
grade:
  field: bedroom_has_window
  map: {true: 10, false: 0}
  evidence: "The floorplan shows the bedroom's external wall."
unknown: skip
ask: "Does every bedroom have a window to an external wall?"
---

## What this measures
Whether a bedroom is a box room off a hallway with no external wall.

## Rubric
**10** — the floorplan positively shows a window, or an external wall, in every
bedroom.
**0** — the floorplan positively shows an internal bedroom. This rejects the
listing outright.

## Evidence, and what not to read into it
This is a hard requirement and the first thing to check on a floorplan. It is
also the one where a wrong answer is most expensive, so the middle case matters
more than either end: **a plan that simply does not draw its windows is
unknown**, not a failure. Never assume a window that cannot be seen, and never
reject a flat over a window a plan does not happen to show. Where the plan does
not settle it, say so and ask.
