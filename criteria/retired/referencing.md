---
name: Referencing an overseas income
description: One income is a contract with an overseas employer, so in-house referencing is a practical advantage.
weight: 1
grade:
  model: true
unknown: skip
# Silence is normal here, not a strong negative: most adverts use ordinary
# outsourced referencing without describing it. A hard barrier is uncommon.
silence_prior: {0: 0.1, 5: 0.8, 8: 0.1}
ask: "How do you reference an applicant whose income is from an overseas employer?"
---

## What this measures
Whether this landlord or scheme can actually reference the tenants. One of
them has UK renting history; the other's income comes from abroad, and a
referencing agency that cannot handle that turns a good application into a
slow one.

## Rubric
**10** — in-house referencing stated, as at Get Living.
**8** — a large build-to-rent operator with its own lettings team.
**5** — a named high-street letting agency, with referencing outsourced or
not described.
**0** — a stated requirement for UK-employed applicants or a UK guarantor with
no alternative offered.

## Evidence, and what not to read into it
This is a small weight and a genuine convenience, not a requirement — the
couple can provide a UK guarantor. Every grade above needs the advert to have
named an operator or said something about referencing or income requirements.
Return null otherwise. Silence is not a 5.
