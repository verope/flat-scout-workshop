---
name: A tenancy that accepts a dog
description: Silence means unknown. A stated no-pets policy is a dealbreaker.
weight: 3
veto_at_or_below: 0
grade:
  model: true
unknown: skip
# Pet-friendly terms are marketable, so the adverts that mention pets are
# mostly the ones happy about them. A silent advert is more likely to refuse a
# dog than to accept one, but silence never fires the veto: only a stated
# policy can.
silence_prior: {0: 0.7, 10: 0.3}
ask: "We have a dog arriving in the spring, already agreed with a rescue. What is the building's pet policy, and could you confirm it in writing?"
---

## What this measures
Whether the tenancy will permit a dog. The dog is not hypothetical: it is
agreed with a rescue and arrives in the spring, so the tenancy has to allow one
from the start.

## Rubric
**10** — an explicit pet-friendly policy at no extra cost, or a named
pet-friendly operator.
**8** — pets stated as considered, or permitted on application.
**5** — pets mentioned with conditions or an added deposit.
**0** — a stated no-pets policy. This rejects the listing outright.

## Evidence, and what not to read into it
**Very few listings say anything, and silence means unknown, not no. Never
reject on silence** — return null, and the question goes to the agent.

Read the description, the key features and the `pet_notes` field where it is
present. "Pets considered" is an 8, not a 10: it is an invitation to ask, not a
policy. Do not infer a policy from the building type, the floor, or the
presence of a garden.
