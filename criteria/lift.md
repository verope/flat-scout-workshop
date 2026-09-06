---
name: A lift above the second floor
description: A puppy that cannot do stairs yet, and later a wet dog carried up in the rain. Below the third floor it does not matter.
weight: 1
grade:
  model: true
unknown: skip
ask: "Which floor is the flat on, and is there a lift?"
---

## What this measures
Whether a flat above the second floor can be reached without stairs.

## Rubric
**10** — the flat is on the second floor or below, whatever the building has;
or it is higher and the advert states a lift.
**0** — the flat is above the second floor and the advert states there is no
lift, or calls the building a walk-up.

## Evidence, and what not to read into it
Two facts are needed: the floor and the lift. The floor comes from the `floor`
field, from `floor_from_the_plan`, or from the description ("set on the fourth
floor"). A lift comes only from the advert saying so — "lift access", "lift to
all floors", "porter and lift". If the floor is unknown, return null: a lift on
its own does not answer this. If the floor is above the second and the advert
says nothing about a lift, return null — silence is not "no lift".
