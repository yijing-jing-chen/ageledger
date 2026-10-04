"""ageledger — an open, cited pace-of-aging ledger from wearable exports.

Pipeline:  export file → ingest (canonical tables) → levers (cited rules) + cosinor (bio age) → report JSON.
Rule 1: never fabricate. Absent input ⇒ the lever is `withheld` with a reason, never defaulted.
Rule 2: every number that becomes "years" carries a citation (DOI) and the exact rule applied.
"""
__version__ = "0.0.1"
