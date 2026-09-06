# flat-scout

Scores rental listings against a written brief, and reports how much of the
brief each listing could answer. The exhibit for a workshop on evaluating LLM
systems.

## Language

**Listing**:
A single advertised rental property as it appears on one portal.
_Avoid_: property, ad, result

**Portal**:
A third-party site that advertises Listings - Rightmove, OpenRent.
_Avoid_: site, source, provider

**Criterion**:
One thing the tenants care about, written as one file in `criteria/` with a
weight and a rubric. The Criteria together are the brief. A Criterion is graded
on its own and knows nothing about the others.
_Avoid_: rule, factor, check, filter

**Grade**:
What one Criterion says about one Listing: a mark from 0 to 10, or **unknown**,
and one sentence of evidence naming what it was read off. Unknown is a real
answer and the right one wherever the Listing does not say - never a zero.
_Avoid_: score, rating, points

**Score**:
The weighted mean of a Listing's Grades. One number, and never the whole story
- read it beside its **Coverage**. An unknown Grade enters the Score as a draw
from what that Criterion usually is, so an advert cannot score well by saying
little.
_Avoid_: grade, rating

**Coverage**:
The share of the total Criterion weight that evidence could actually answer for
one Listing. A thin advert scores on the few Criteria it happens to address, so
a Score at 40% Coverage is a guess wearing a number's clothes.
_Avoid_: confidence, completeness, quality

**Verdict**:
The evaluator's judgement of a Listing against the brief - hopeful, borderline,
or reject. It is **derived, not opinion**: reject if a Criterion with a veto
fired, hopeful if the Score clears `hopeful_threshold` and the Coverage clears
`coverage_floor`, borderline otherwise.
_Avoid_: rating, result, decision

## Relationships

- A **Portal** advertises **Listings**
- Each **Criterion** gives each **Listing** exactly one **Grade**
- A **Listing**'s **Grades** produce its **Score** and its **Coverage**, and
  those two with the vetoes produce its **Verdict**
- A **Listing** receives exactly one **Verdict**

## Flagged ambiguities

- "score" meant both what one **Criterion** said and what the whole brief said.
  Resolved: a **Criterion** gives a **Grade**; the weighted mean of the Grades
  is the **Score**.
- "unknown" and "bad" were the same thing in the original scorer - an absent
  field pulled a flat down. Resolved: they are different answers. A **Grade** is
  unknown when the **Listing** does not say, and a flat nobody could read is
  not a bad flat. Excluding unknowns from the **Score** was the first repair and
  it went too far the other way: it rewarded an advert for saying less, so the
  flat that scored highest of the first batch was the one that answered least.
  An unknown now enters the **Score** as a draw from what that **Criterion**
  usually is - or, where the brief declares a `silence_prior` and a model read
  the advert and found nothing, from what the silence itself suggests. It
  never fires a veto.
