---
name: Not in fact the ground floor
description: Catches what the hard filter missed, wherever the floor is written down.
weight: 2
veto_at_or_below: 0
grade:
  grader: not_ground_floor
unknown: skip
---

## What this measures
Whether anything that knows which floor this is contradicts the listing. Ground
floor is a hard filter in `config.toml`, but that filter reads the Portal's
`floor` field, and Rightmove sends it as null on nearly every Listing.

Three sources answer, and they are not equally trusted.

## Rubric
**10** — the certificate, the floorplan or the listing names a floor above the
ground.
**2** — the *floorplan's title block* or the listing says ground, lower ground,
raised ground or basement. Low, and deliberately not zero: this also raises a
red flag, so a human sees the flat and decides.
**0** — the *EPC certificate* names a ground-floor or lower-ground flat. This
rejects the listing outright.

## Evidence, and what not to read into it
The certificate prints the property type as a fact on an official document, so
it may disqualify a flat on its own. The floorplan is a model reading a
drawing, and may not — one plan in the corpus was once read as "Ground Floor"
when the drawing contains neither word. Two independent reads must now agree
before a floor is stored at all, which killed that reading, but a flat removed
on an invented fact is removed where nobody can appeal it, and that risk is not
worth a weight-2 Criterion.

A raised ground floor counts as ground. It is a few steps up, not a storey, and
it is the phrase an advert reaches for when it would rather not say it.

Where nothing names a floor, this is unknown and skips.
