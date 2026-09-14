from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import typer

from flat_scout.config import load_settings
from flat_scout.db import Database
from flat_scout.fetch import build_async_client
from flat_scout.pipeline import process_url

app = typer.Typer(help="flat-scout - London flat triage", no_args_is_help=True)

DB_PATH = Path("data/flats.db")


@app.callback()
def main() -> None:
    """Triage for a London flat search: listings in, decisions out.

    Logging is configured here rather than in each command, so that every
    command's ERROR lines - one per Listing the pipeline drops - reach a root
    logger with something attached, rather than going nowhere at all.
    """
    from flat_scout.observe import configure_logging

    configure_logging()


@app.command()
def check(
    url: str,
    config: Path = Path("config.toml"),
    holistic: bool = typer.Option(
        False, "--holistic", help="One model call over criteria/criteria.md, one score."
    ),
    weighted: bool = typer.Option(
        False, "--weighted", help="One grade per Criterion in criteria/*.md."
    ),
    force: bool = typer.Option(
        False, "--force", help="Evaluate again even if this Listing already has a Verdict."
    ),
) -> None:
    """Download, pre-filter and evaluate a single Listing URL.

    Which evaluator runs is `features.weighted_criteria` in config.toml.
    --holistic and --weighted override it for this run only; the file is not
    touched. An evaluated Listing is otherwise never evaluated again, so
    comparing the two evaluators on one Listing is `check --holistic` followed
    by `check --weighted --force`. The stored row records which one ran last
    in `evaluation_mode`.
    """
    if holistic and weighted:
        typer.echo("--holistic and --weighted are separate evaluators. Pass one.")
        raise typer.Exit(2)
    from flat_scout.observe import start

    settings = load_settings(config)
    if holistic or weighted:
        settings.features.weighted_criteria = weighted
    start(settings, service="check")
    db = Database(DB_PATH)

    async def run() -> None:
        async with build_async_client(settings) as client:
            listing_id = await process_url(
                url, db, settings, "manual", client, force=force
            )
        if listing_id is None:
            typer.echo("Not a recognised Portal listing URL.")
            raise typer.Exit(1)
        row = db.get(listing_id)
        typer.echo(json.dumps({key: row[key] for key in row.keys()}, indent=2, default=str))

    asyncio.run(run())


@app.command()
def status(config: Path = Path("config.toml")) -> None:
    """Count Listings by status."""
    from flat_scout.report import status_counts

    counts = status_counts(Database(DB_PATH))
    if not counts:
        typer.echo("No listings yet.")
        return
    for name, count in counts:
        typer.echo(f"{name:>16}  {count}")
    typer.echo(f"{'total':>16}  {sum(count for _, count in counts)}")


@app.command()
def report(
    config: Path = Path("config.toml"),
    decisions: bool = False,
    suppressed: bool = False,
    weights: bool = False,
    shift: bool = False,
    posterior: bool = False,
) -> None:
    """Show the score distribution, to set hopeful_threshold from evidence.

    --decisions holds the holistic score against what the couple actually
    decided. --suppressed lists what was never surfaced. --weights replays a
    candidate weighting over the stored grades: it is the only view on the
    new scale, and the only one that can say which Criterion is doing the
    work. --shift asks the deployment question instead: over the Listings the
    couple already decided on, would a flat they approved still have
    reached them, and would one they passed still be filtered out. --posterior
    replays candidate (hopeful_threshold, hopeful_confidence) pairs against the
    same Decisions, and is where both of those constants come from. Every view
    is read-only, and none of --weights, --shift and --posterior makes a model
    call at all, so a weight can be edited and the report re-read as often as
    it takes.
    """
    from flat_scout.criteria import load_criteria
    from flat_scout.graders import GRADERS
    from flat_scout.report import (
        decisions_report, histogram, posterior_report, shift_report, suppressed_report,
        weights_report,
    )

    chosen = [name for name, on in (("--decisions", decisions), ("--suppressed", suppressed), ("--weights", weights), ("--shift", shift), ("--posterior", posterior)) if on]
    if len(chosen) > 1:
        typer.echo(f"{', '.join(chosen)} are separate views. Pass one.")
        raise typer.Exit(2)
    settings = load_settings(config)
    db = Database(DB_PATH)
    threshold = settings.evaluation.hopeful_threshold
    criteria = load_criteria(
        Path(settings.evaluation.criteria_dir), known_graders=set(GRADERS)
    )
    if posterior:
        typer.echo(posterior_report(db, settings, criteria))
    elif shift:
        typer.echo(shift_report(db, criteria, settings))
    elif weights:
        typer.echo(weights_report(db, criteria, settings))
    elif decisions:
        typer.echo(decisions_report(db, threshold, criteria))
    elif suppressed:
        typer.echo(suppressed_report(db, threshold))
    else:
        typer.echo(histogram(db, threshold))


@app.command()
def rescore(config: Path = Path("config.toml"), limit: int = 0) -> None:
    """Replay criteria.md over the Listings on hand. Writes nothing.

    An evaluated Listing is never evaluated twice, so an edit to the brief acts
    only on Listings that have not arrived yet. This says what the edit would
    have done to the ones that have. One model call per Listing, no vision
    spend: use --limit to try a handful first.
    """
    from flat_scout.observe import start
    from flat_scout.pipeline import rescore as run_rescore
    from flat_scout.report import rescore_report

    settings = load_settings(config)
    start(settings, service="rescore")
    db = Database(DB_PATH)
    changes = asyncio.run(run_rescore(db, settings, limit=limit))
    typer.echo(rescore_report(changes, settings.evaluation.hopeful_threshold))


@app.command()
def grade(
    config: Path = Path("config.toml"),
    limit: int = 0,
    backfill: bool = False,
    measure: bool = False,
    sample: int = 10,
    runs: int = 3,
    force: bool = False,
    criterion: list[str] = typer.Option(
        [], "--criterion", help="Only this Criterion slug. Repeatable."
    ),
) -> None:
    """Grade the stored Listings against criteria/, and report the coverage.

    Writes only `criterion_grades`. The score and the Verdict on each Listing
    are what the evaluator said at the time, and stay as they are.

    --backfill skips any Criterion whose stored grade already matches the
    rubric hash on disk, so a second run is free. That hash covers a Criterion
    file's `grade` block, its `fallback` and its body, and NOTHING ELSE:
    editing `graders.py` or the shared preamble (`_brief.md`, `_fields.md`)
    changes what a grade would be and moves no
    hash at all. After one of those edits the backfill will report nothing
    stale while newly graded Listings quietly use the new values. Pass --force
    to re-grade regardless of the stored hashes, and --criterion to scope that
    to the slugs the edit could have touched rather than paying for every
    Criterion. --criterion scopes --measure the same way.

    --measure grades the Criteria that can reach a model - a primary `model:
    true` grade or a `fallback: model` - repeatedly over a sample of Listings,
    and reports how often each returns unknown and how much it disagrees with
    itself - the two things a rubric can be measured on, since a rubric cannot
    be graded right or wrong by a machine. It writes nothing either.
    """
    if backfill and measure:
        typer.echo("--backfill and --measure are separate runs. Pass one.")
        raise typer.Exit(2)
    from flat_scout.observe import start

    if measure:
        from flat_scout.pipeline import aspect_against_annotations, measure_criteria
        from flat_scout.report import measure_report

        settings = load_settings(config)
        start(settings, service="grade --measure")
        db = Database(DB_PATH)
        spreads = asyncio.run(
            measure_criteria(db, settings, sample=sample, runs=runs, only=list(criterion))
        )
        aspect_agreement = aspect_against_annotations(db, settings)
        typer.echo(measure_report(spreads, aspect_agreement))
        return
    from flat_scout.pipeline import backfill_grades
    from flat_scout.report import backfill_report

    if not backfill:
        typer.echo("Pass --backfill. This costs one model call per Criterion per Listing.")
        raise typer.Exit(1)
    settings = load_settings(config)
    start(settings, service="grade --backfill")
    db = Database(DB_PATH)
    done = asyncio.run(
        backfill_grades(db, settings, limit=limit, force=force, only=list(criterion))
    )
    typer.echo(backfill_report(done))


@app.command()
def fit(config: Path = Path("config.toml")) -> None:
    """Fit judge noise and priors off the corpus, and print the table."""
    from flat_scout.criteria import load_criteria
    from flat_scout.fit import run_fit, serialize
    from flat_scout.graders import GRADERS
    from flat_scout.observe import start
    from flat_scout.report import fit_report

    settings = load_settings(config)
    start(settings, service="fit")
    db = Database(DB_PATH)
    criteria = load_criteria(
        Path(settings.evaluation.criteria_dir), known_graders=set(GRADERS)
    )
    fitted = run_fit(db, criteria)
    db.set_fit_cache(fitted.input_hash, serialize(fitted))
    typer.echo(fit_report(fitted, criteria))


@app.command()
def annotate(
    config: Path = Path("config.toml"),
    db: Path | None = None,
    sheet: Path = Path("output/annotation-sheet.html"),
    serve: bool = True,
    port: int = 8765,
) -> None:
    """Write down what is actually drawn on each floorplan.

    Needs nothing but the corpus in tests/fixtures/floorplans, which is in the
    repository: the images, and the answers beside them. Pass --db pointing at a
    copy of the live database to take in the floorplans of Listings that have
    arrived since, which is the only thing a database is wanted for here.

    Answers are saved as they are made. Existing ones are never overwritten -
    only plans new to the corpus get a stub, and the stubs stay blank on
    purpose, because pre-filling them with what the pipeline currently says
    would be asking someone to agree with the answer under test.
    """
    from flat_scout.annotate import (
        ANNOTATIONS,
        CORPUS,
        Annotation,
        download_plan,
        load_annotations,
        plans_in,
        render_annotations,
        render_sheet,
    )

    settings = load_settings(config)
    annotations = load_annotations()
    added = 0
    if db is not None:
        if not db.exists():
            typer.echo(f"No database at {db}.")
            raise typer.Exit(1)
        with httpx.Client(timeout=settings.fetch.timeout_seconds) as client:
            for portal_id, address, url, path in plans_in(db):
                name = download_plan(
                    portal_id, url, client, settings.fetch.user_agent, CORPUS, local=path
                )
                if name is None:
                    typer.echo(f"  could not fetch the plan for {portal_id}")
                    continue
                if portal_id not in annotations:
                    annotations[portal_id] = Annotation(
                        portal_id=portal_id, file=name, address=address
                    )
                    added += 1
        ANNOTATIONS.parent.mkdir(parents=True, exist_ok=True)
        ANNOTATIONS.write_text(render_annotations(annotations))

    if not annotations:
        typer.echo(f"No plans in {CORPUS}. Pass --db to collect some.")
        raise typer.Exit(1)

    waiting = sum(1 for entry in annotations.values() if not entry.done)
    typer.echo(f"{len(annotations)} plans in the corpus, {added} new, {waiting} unannotated.")
    typer.echo(f"Answers are saved to {ANNOTATIONS}")
    if not serve:
        media = {".jpg": "image/jpeg", ".png": "image/png", ".gif": "image/gif"}
        entries = [
            (entry, (CORPUS / entry.file).read_bytes(), media[(CORPUS / entry.file).suffix])
            for _, entry in sorted(annotations.items())
            if (CORPUS / entry.file).exists()
        ]
        sheet.parent.mkdir(parents=True, exist_ok=True)
        sheet.write_text(render_sheet(entries))
        typer.echo(f"Open {sheet} to read the bearings off, and edit the TOML by hand.")
        return

    from flat_scout.annotate import serve as serve_editor

    server = serve_editor(ANNOTATIONS, CORPUS, port=port)
    typer.echo(f"\nEditing at http://127.0.0.1:{port}  -  every change is saved as you make it.")
    typer.echo("Ctrl-C when you have had enough; it can be picked up again later.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        typer.echo("\nStopped. " + str(sum(1 for e in load_annotations().values() if e.done))
                   + " plans marked done.")
    finally:
        server.server_close()


@app.command()
def benchmark(config: Path = Path("config.toml"), limit: int = 0) -> None:
    """Score the floorplan pipeline against the annotations. Writes nothing.

    Two vision calls per plan, so the whole corpus costs a few pence. Every
    number is against what a human wrote down while looking at the drawing, and
    never against a second run of the pipeline.
    """
    from flat_scout.annotate import CORPUS
    from flat_scout.benchmark import render, run
    from flat_scout.observe import start

    settings = load_settings(config)
    start(settings, service="benchmark")
    stages, readings, troubled = asyncio.run(run(settings, CORPUS, limit=limit))
    if not stages[0].outcomes:
        typer.echo("No annotated plans yet. Run `flat-scout annotate` first.")
        raise typer.Exit(1)
    typer.echo(render(stages, readings, troubled))


@app.command()
def approve(listing_id: int, config: Path = Path("config.toml")) -> None:
    """Record a yes on a Listing."""
    db = Database(DB_PATH)
    if db.transition(listing_id, "approved"):
        typer.echo(f"#{listing_id} approved")
    else:
        typer.echo(f"#{listing_id} already decided ({db.get(listing_id)['status']})")


@app.command()
def reject(listing_id: int, config: Path = Path("config.toml")) -> None:
    """Record a no on a Listing."""
    db = Database(DB_PATH)
    if db.transition(listing_id, "rejected"):
        typer.echo(f"#{listing_id} rejected")
    else:
        typer.echo(f"#{listing_id} already decided ({db.get(listing_id)['status']})")


if __name__ == "__main__":
    app()
