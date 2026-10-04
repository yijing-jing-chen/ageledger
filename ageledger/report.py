"""Assemble the ledger JSON (CONTRACT.md) from a Dataset. Pure glue: no science here."""
from __future__ import annotations
from dataclasses import asdict
from . import __version__
from .schema import Dataset, validate

def build(ds: Dataset, rules=None, ics_text: str | None = None, experiment: dict | None = None) -> dict:
    """`ics_text`: an .ics export → ledger["week_plan"] (else None).
    `experiment`: {"lever": key, "start_date": "YYYY-MM-DD"} → ledger["experiments"]["receipt"] (else None)."""
    from .levers.engine import evaluate, load_rules
    from . import calendar as cal_mod, experiment as exp_mod
    from .cosinor import estimate as cosinor_estimate, withheld_reason
    from . import composite

    problems = validate(ds)
    if problems:
        raise ValueError("dataset violates schema: " + "; ".join(problems))

    rules = rules or load_rules()
    lev = evaluate(ds, rules)
    cos = cosinor_estimate(ds)
    withheld = list(lev.get("withheld", []))
    if cos is None:
        reason = withheld_reason(ds)
        withheld.append({"key": "cosinor_age", "reason": reason})
    comp = composite.effective_age(ds, lev, rules)
    if comp is None:
        withheld.append({"key": "composite_age", "reason": composite.withheld_reason(ds, lev, rules)})
    bio_age = {
        "cosinor": cos,
        "composite": comp,
        "fitness_age": composite.fitness_age(ds, lev, rules),
        "hrv_age_equivalent": composite.hrv_age_equivalent(lev, rules),
    }

    prof = asdict(ds.profile)
    assumed = [k for k in ("age_years", "sex") if prof.get(k) is None]
    prof["assumed"] = assumed
    prof.pop("notes", None)

    levers = lev.get("levers", [])
    week_plan = cal_mod.week_plan(ds, ics_text, rules, levers=levers) if ics_text else None
    experiments = {
        "proposals": exp_mod.propose(levers, rules),
        "receipt": exp_mod.receipt(ds, experiment["lever"], experiment["start_date"], rules) if experiment else None,
    }

    return {
        "version": __version__,
        "source": ds.source,
        "coverage": ds.coverage,
        "profile": prof,
        "bio_age": bio_age,
        "levers": levers,
        "withheld": withheld,
        "context": lev.get("context", []),
        "constants": lev.get("constants", {}),
        "ranking_note": "levers sorted by years_recoverable desc; ties by confidence",
        "notes": list(ds.profile.notes),
        "week_plan": week_plan,
        "experiments": experiments,
    }
