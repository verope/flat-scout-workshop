---
name: Secure bike storage
description: A locked store inside the building, for two bikes.
weight: 3
grade:
  model: true
unknown: skip
# A secure store is a marketable amenity and is usually named when it exists.
# Silence therefore leans toward none, while allowing for an unadvertised store.
silence_prior: {0: 0.7, 8: 0.2, 10: 0.1}
ask: "What is the bike storage, and does it lock? Is there room for two bikes?"
---

## What this measures
Somewhere secure for two bikes, inside the building.

## Rubric
**10** — a secure or locked bike store described as such, or a bike room in a
managed scheme.
**8** — a stated bike store with nothing said about locking.
**5** — a communal area or garage mentioned, with no bike provision named.
**2** — bikes explicitly not permitted in communal areas.
**0** — the advert states there is no bike storage.

## Evidence, and what not to read into it
A rack in a shared hallway is **not** the same thing as a locked store. Treat
vague phrasing as unknown and ask, rather than reading it generously. This one
matters more than most, so a confident wrong answer costs more here than an
admission of ignorance. Return null if the advert does not mention bikes at
all.
