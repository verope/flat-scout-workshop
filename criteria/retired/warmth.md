---
name: Warmth and bills
description: Low bills, no draughts. Modern or genuinely well-insulated construction.
weight: 4
grade:
  field: epc_band
  map: {A: 10, B: 10, C: 8, D: 5, E: 2, F: 0, G: 0}
  evidence: "EPC band {value}."
unknown: skip
fallback: model
ask: "What is the EPC rating, and how is the flat heated? Roughly what does a month's heating cost?"
---

## What this measures
Whether the building will be cheap and comfortable to heat. Nothing else.

## Rubric
**10** — band A or B, or a new-build or build-to-rent scheme the advert
describes as highly insulated.
**8** — band C, or modern construction with double or triple glazing stated.
**5** — band D, or a refurbished period building with glazing mentioned.
**2** — band E. This is the legal minimum for a letting in England rather than
a standard anyone aimed at.
**0** — band F or G, or a period conversion the advert describes as original
throughout with single glazing.

## Evidence, and what not to read into it
An EPC band is evidence about warmth and bills **only**. It says nothing about
quietness — that is a separate criterion, and a good band leaves street noise
entirely unknown. Where no band was read, judge from what the advert says about
construction, glazing and heating, and return null if it says nothing.
