# Demo runbook

Two acts. The first shows the tool working; the second asks whether it is any
good. Every command below runs from the repository root with a graded
`data/` in place. Three listings were held back from the pre-grading so the
live `check` is a real first evaluation: 83 (pet friendly, with floorplan
and EPC, evaluated both ways), 8 (no pets) and 24 (spare, same shape as 83).

If the network dies, every command's output from the rehearsal is in
`docs/demo-outputs/`.

Two things a sharp audience will ask about:

- **`vision_model` says Claude Sonnet.** The image readings were made in
  August by Sonnet, when the corpus was built, and every run since has
  reused them rather than paid to redo them. The column records who read
  the images, and that is the honest answer. The graders are GLM Flash.
- **Why the cheap model returned unknown on everything, once.** The first
  corpus run on GLM Flash graded pets unknown on 162 of 165 listings. The
  model reasoned "Grade 10" and then left the `grade` field out of its tool
  call, because the field had a default and the model omits everything it
  is not required to send. Making the field required-but-nullable fixed it.
  That is a whole evals lesson in one bug: the schema was the rubric, and
  nobody had measured the unknown rate until `grade --measure` did.

## Before the room fills

```sh
uv sync
uv run flat-scout status          # expect: new 3, evaluated 154, approved 5, rejected 5
cat .env                          # OPENROUTER_API_KEY set, models as shipped
```

## Act one: the tool (about 10 minutes)

1. **What is stored.**
   ```sh
   uv run flat-scout status
   ```
2. **The naive evaluator.** Pet friendly, with a floorplan and an EPC. The
   whole brief goes to the model once and one score comes back. The vision
   reading is already stored, so this is one text call. It takes about a
   minute and a half: one large prompt the model reasons over at length.
   Narrate the brief while it runs.
   ```sh
   uv run flat-scout check --holistic https://www.rightmove.co.uk/properties/91601766
   ```
   Point at `score`, `verdict`, `reasons`, and `evaluation_mode: holistic`.
   Note there is no `coverage`: one number, and nothing to say how much of
   the brief it rests on.
3. **The same flat, one Criterion at a time.** `--force` because the listing
   now has a Verdict. About half a minute: the calls run in parallel.
   ```sh
   uv run flat-scout check --weighted --force https://www.rightmove.co.uk/properties/91601766
   ```
   Point at `score` beside the holistic one, `coverage`, `agent_questions`,
   and `evaluation_mode: weighted`. The JSON now carries a `grades` list:
   one entry per Criterion with its grade, its weight and what the grade
   was read off. A `grade` of null is unknown, and unknown is a real answer.
4. **A veto.** The advert says "Pets Not Allowed". The Criterion fires its
   veto and the Verdict is reject whatever the Score. About half a minute.
   ```sh
   uv run flat-scout check https://www.rightmove.co.uk/properties/92077104
   ```
   Point at `verdict`, `reasons`, and the `pets` grade.
5. **The corpus.** The histogram with the threshold drawn in.
   ```sh
   uv run flat-scout report
   ```

## Act two: is it any good (about 25 minutes)

6. **Coverage.** A re-run is free and prints the coverage distribution. The
   point: a Score at 40% Coverage is a guess wearing a number's clothes.
   ```sh
   uv run flat-scout grade --backfill
   ```
7. **Judge noise.** Each model-graded Criterion asked three times over five
   listings. Unknown rate and self-disagreement per Criterion. A few
   minutes live; the saved output is in `docs/demo-outputs/measure.txt`.
   ```sh
   uv run flat-scout grade --measure --sample 5 --runs 3
   ```
8. **Score against Decision.** Ten decisions are already on record: five
   approvals and five rejections, made on the pre-graded corpus. One
   approved flat scored 3.6 and one rejected flat scored 7.1, so no
   threshold separates them, and the report says so. No model call.
   ```sh
   uv run flat-scout report --decisions
   ```
   To make one live: `uv run flat-scout approve <id>` on something from the
   histogram's top, then re-run the report.
9. **Which Criterion does the work.** The same decisions replayed per
   Criterion: mean grade among approved against rejected, and the gap. A
   Criterion with no gap is not earning its weight. No model call.
   ```sh
   uv run flat-scout report --weights
   ```
10. **Edit a rubric, re-grade only that.** Change the weight or a band in
   `criteria/lift.md`, then:
   ```sh
   uv run flat-scout grade --backfill        # only lift is stale
   uv run flat-scout report --weights
   ```
   Revert the edit afterwards (`git checkout criteria/lift.md`).
11. **Where ground truth comes from.** Nothing in act two so far has said
    whether a grade is *right*, only whether it is consistent and whether it
    separates the decisions. For the floorplan reader there is a human
    answer, and this is where it was written. Opens a page on port 8765 over
    the 24 plans in `tests/fixtures/floorplans/`. The first three are blank
    on purpose: their answers are commented out in the TOML so they can be
    annotated live with no preconception on screen. Annotate one or two of
    them in front of the room: where the north arrow points, which walls the
    living-room windows are on, bearings on the image with 0 up the page,
    then tick done. Answers save as they are made.
    ```sh
    uv run flat-scout annotate
    ```
    Then the file it writes, `tests/fixtures/floorplans/annotations.toml`.
    Point at a `# north_clock` line commented out: no indicator drawn is a
    real answer, and the benchmark scores "correctly silent" against it. The
    stubs are left blank on purpose, so nobody is asked to agree with the
    answer under test. Saving from the page rewrites the whole file and
    drops every comment, including the three original answers, so restore
    them from git afterwards rather than from the file:
    ```sh
    git checkout 472096b -- tests/fixtures/floorplans/annotations.toml
    ``` `--no-serve` writes a static sheet to
    `output/annotation-sheet.html` if the port is a problem; `--db data/flats.db`
    would add the corpus's own floorplans, which is how the set grows.
12. **The reader against that truth.** Stage by stage: north, window walls,
    aspects. Two vision calls per plan, no database needed, about six
    minutes; start it in a second terminal at the top of act two.
    ```sh
    uv run flat-scout benchmark
    ```
    Run it after the live annotation and the plans just annotated are in
    the score. Rehearsal figures over the 21 pre-annotated plans are in
    `docs/demo-outputs/benchmark.txt`. The "where it went wrong" rows
    name the plan, the answer, the truth and the method, so a wrong answer
    is a plan you can open in the annotator and look at.

If short of time, drop step 10 first, then step 8. Keep 11 and 12 together.

## Attendees afterwards

The graded `data/` is `data.zip` on the GitHub release
<https://github.com/verope/flat-scout-workshop/releases/tag/workshop-data>.
Unzip it into the repository root, put an OpenRouter key in `.env`, and every
command above works. `report` and `report --weights` cost nothing; `check` on a new
URL costs well under a cent on the default model.
