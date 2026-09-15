# flat-scout

Scores London rental listings against a written brief, one Criterion at a
time, and says how much of the brief each listing could actually answer. It is
the exhibit for a workshop on evaluating LLM systems: the interesting part is
not the scores but how you find out whether they are any good.

It is a trimmed and adapted copy of flat-scout, written by
Šimon Podhajský for his own London flat search in the summer of 2026. The pipeline,
the graders, the floorplan reader and the reports are his; the brief, the criteria and
the corpus are the workshop's.

## Setup

Python 3.14 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
cp .env.example .env    # then put your OPENROUTER_API_KEY in it
```

You need your own [OpenRouter API key](https://openrouter.ai/keys); every
model call goes through OpenRouter and is billed to it. The two model slugs
in `.env`, `OPENROUTER_TEXT_MODEL` for the graders and `OPENROUTER_IMAGE_MODEL`
for the EPC and floorplan reads, default to `z-ai/glm-5.3-flash`, on which a
`check` on one listing costs well under a cent. Any slug from
[openrouter.ai/models](https://openrouter.ai/models) works; the image model
has to accept image input.

`data/` holds the database and the cached listing images and is not in the
repository. If you were handed a workshop package, copy its `data/` directory
into the repository root before running anything. Without it, `check` builds
one from scratch.

`check` downloads each listing's photo, floorplan and EPC once into
`data/images/<portal>_<id>/` and stores the path beside the URL. Everything
after that, the vision reading, `grade`, `annotate --db`, reads the local file
first, so once a listing is in `data/` nothing depends on the portal still
serving it.

## The five commands

```sh
uv run flat-scout check <listing-url>
```
Downloads one Rightmove or OpenRent listing, reads its EPC and floorplan with a
vision model, grades it against every Criterion and stores the lot in
`data/flats.db`, with its images cached under `data/images/`. Prints the row.

`--holistic` sends `criteria/criteria.md` to the model once and takes back one
score; `--weighted` grades one Criterion at a time from `criteria/*.md`. Either
overrides `weighted_criteria` in `config.toml` for this run only. A listing
already evaluated is normally left alone, so to grade the same listing both
ways run `check --holistic <url>` and then `check --weighted --force <url>`.
The printed row's `evaluation_mode` says which one ran last.

```sh
uv run flat-scout grade --backfill
```
Grades every stored listing against `criteria/`. A Criterion whose stored grade
already matches the rubric on disk is skipped, so a second run is free and a
rubric edit re-grades only that Criterion. Prints the coverage distribution.
`--limit 3` prices it first.

```sh
uv run flat-scout grade --measure
```
Grades the model-graded Criteria repeatedly over the same listings and reports
two things a rubric can be wrong about: how often it comes back unknown, and how
far it disagrees with itself. Neither says the rubric is right; both say whether
it is doing any work.

```sh
uv run flat-scout annotate
```
Opens a local page on port 8765 for writing down what is really drawn on each
floorplan in `tests/fixtures/floorplans/`: where north points, which walls the
windows are on. That is the ground truth. `--no-serve` writes a static sheet
instead.

```sh
uv run flat-scout benchmark
```
Runs the floorplan reader over the annotated plans and scores it against the
human answers, stage by stage. Needs no database.

Also there: `report` prints the score histogram; `report --weights` replays a
different weighting over the stored grades with no model call, and says which
Criterion is doing the work; `report --suppressed` lists what the hard filters
stopped and why. `approve` and `reject` record your own yes or no on a listing.
`status`, `rescore` and `fit` exist too.

## Where the brief lives

| File | What it is |
|---|---|
| `criteria/*.md` | One Criterion per file: weight, rubric, how it is graded. **This is the brief.** |
| `criteria/_brief.md`, `criteria/_fields.md` | The preamble sent with every model-graded Criterion: who the tenants are, and what the extracted fields mean |
| `criteria/criteria.md` | The same brief as one document. Read only by the naive baseline (`weighted_criteria = false` in `config.toml`), which sends it to the model once and gets back one score |
| `criteria/retired/` | Criteria that no longer count. The loader reads `criteria/*.md` and does not recurse, so moving a file here retires it without losing its rubric or its stored grades |
| `config.toml` | Hard filters, the score threshold, the coverage floor, the models, the feature flags |
| `tests/fixtures/floorplans/` | The floorplan corpus and `annotations.toml`, the human answers `benchmark` scores against |
| `docs/architecture.html` | How the pieces fit together, with diagrams. Open it in a browser |

A Criterion is graded by code wherever a field answers the question (`grade:
{field, bands}`, `grade: {field, map}` or a named `grader`), and by a model
only where the question is about prose or pictures (`grade: {model: true}`, or
`fallback: model` when the field is missing). Unknown is a real answer: a
Criterion the listing does not address comes back `null`, never `0`, and
`unknown: skip` keeps it out of the coverage.

To add a Criterion, copy the shape of an existing file and run `grade
--backfill`. To retire one, move it into `criteria/retired/`. Never set
`weight: 0`; the loader refuses it on purpose.

## Watching it run

Every long command prints a progress bar, or a line every thirty seconds when
there is no terminal. Put a `LOGFIRE_TOKEN` in `.env` and every model call,
image download and page fetch is also traced to
[logfire](https://logfire.pydantic.dev), with tokens, latency and the Criterion
that asked for it. Leave the token empty and nothing is configured or sent.

## What it will not do

Download a Zoopla page: Zoopla challenges every automated client, and the
challenge is not worked around. Contact anyone: nothing here sends anything.
