# What the data means

The fields below a listing are a best-effort extraction from the Portal. **An
absent field means only that nothing was extracted for it. It is NOT the
listing declining to say.** Rightmove sends the floor as null on every listing
seen so far, while the description says "set on the thirteenth floor". Read the
description before concluding anything is unstated.

Anything a vision model read off the EPC graph or the floorplan is a direct
reading of the image, so it outranks the advert's prose where the two disagree.
Anything it could not read with confidence is absent, and **absent means
unknown, never bad**.

- **`furnished`** is the Portal's own lettings field and one of the few facts
  nearly every listing states. "Furnished or unfurnished, landlord is flexible"
  is an offer, not a fact about the flat today. Never read furnishing off a
  photograph: what is in the pictures is as often the current tenant's.
- **`floor_from_the_plan`** is the floorplan's own title block, read verbatim -
  "14th Floor", "Ground Floor". It is the only place most listings state a
  floor at all, since the Portal's field is almost always null. It is a model
  reading a drawing, so it informs and it never disqualifies.
- **`sqft`** is the floor area the listing itself states, in square feet.
  Divide by 10.764 for square metres.
- **`epc_band`** is evidence about warmth and bills. No active Criterion
  grades it; it is read and shown for completeness.
- **`epc_band_source`** says which document it came from. `certificate` is a
  letter printed on the official EPC and is the firmest evidence available.
  `graph` is a reading of a marker on a bar chart - real, but a picture.
- **`epc_caption`** is the Portal's label for its graph ("EE Rating", "EPC 1").
  It names the graph, not the rating. Never read a band out of it.
- **`epc_floor_area_sqm`** is printed on the certificate and often exists where
  the listing states no square footage at all. Where it and the listing's own
  square footage disagree materially, the size is unresolved rather than one of
  them being right: an agent can attach the certificate of a different unit.
- **`window_aspects`** is worked out from the north arrow drawn on the
  floorplan and the walls the main windows sit on. It is a measurement, not a
  guess, and it is a list because a corner flat faces more than one way. Never
  infer an aspect from anything else - not the address, not which way a balcony
  is drawn, not a room's name.
