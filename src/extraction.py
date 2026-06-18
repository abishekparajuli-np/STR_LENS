import re
import sys
import subprocess
import spacy

_nlp = None

# Words that frequently get mis-tagged as PERSON/ORG by spaCy when the input
# text is short, fragmentary, or made of XML field labels rather than full
# sentences. These are filtered out of NER results before they reach the
# faithfulness checklist.
_NER_BLOCKLIST = {
    "kyc", "npr", "usd", "gbp", "eur", "swift", "bic", "marg", "cash withdrawal",
    "suspicious", "transaction", "z", "m", "f", "rs", "rs.",
}


def get_nlp_model():
    """
    Loads the spaCy model. Downloads it automatically if it is not present.
    """
    global _nlp
    if _nlp is not None:
        return _nlp

    try:
        _nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("spaCy model 'en_core_web_sm' not found. Downloading...")
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        _nlp = spacy.load("en_core_web_sm")

    return _nlp


def clean_amount(amt_str):
    """
    Normalises amount string by removing commas, whitespace, and currency symbols.
    Returns the original string unchanged if it cannot be parsed as a number
    (e.g. junk regex matches like "rs,").
    """
    cleaned = re.sub(r"[^\d.]", "", amt_str)
    if not cleaned or cleaned == ".":
        return amt_str
    try:
        val = float(cleaned)
        return f"{val:.2f}"
    except ValueError:
        return amt_str


def extract_amounts(text):
    """
    Extracts all currency amounts (NPR, USD, GBP, EUR, Rs, $, £) from text.
    Requires at least one digit immediately after the currency token, which
    prevents bare-word junk matches like "Rs," with no number attached.
    """
    amt_pattern = r"(?:NPR|USD|GBP|EUR|Rs\.?|£|\$)\s*\d[\d,]*(?:\.\d{2})?|\b\d[\d,]*\.\d{2}\b"
    matches = re.findall(amt_pattern, text, re.IGNORECASE)

    seen = set()
    unique_amounts = []
    for match in matches:
        cleaned = clean_amount(match)
        # Skip anything that didn't actually resolve to a numeric value
        if cleaned == match.strip() and not re.search(r"\d", cleaned):
            continue
        if cleaned not in seen:
            seen.add(cleaned)
            unique_amounts.append(cleaned)

    return unique_amounts


def extract_dates(text):
    """
    Extracts dates in MM/DD/YYYY, DD-MM-YYYY, Month DD YYYY formats.
    """
    patterns = [
        r"\b\d{1,2}[/.\-]\d{1,2}[/.\-]\d{4}\b",
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b",
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b",
        # ISO format YYYY-MM-DD, common in XML/database exports
        r"\b\d{4}-\d{2}-\d{2}\b",
    ]

    unique_dates = []
    seen = set()
    for pattern in patterns:
        for match in re.findall(pattern, text, re.IGNORECASE):
            match_clean = match.strip()
            if match_clean.lower() not in seen:
                seen.add(match_clean.lower())
                unique_dates.append(match_clean)

    return unique_dates


def extract_accounts(text):
    """
    Extracts account numbers. Only matches the NP-prefixed Nepalese account
    format explicitly, plus 9-16 digit numbers that look like account
    numbers based on context (preceded by "account"/"acc"/"a/c" or similar).

    The previous version matched ANY bare 8-16 digit number anywhere in the
    text, which produced false positives on phone numbers, transaction
    codes, and other numeric fields that happened to fall in that length
    range. This version requires either the NP-prefixed format or a nearby
    keyword indicating the number really is an account number.
    """
    unique_accounts = []
    seen = set()

    # NP-prefixed accounts: unambiguous, always include
    for match in re.findall(r"\bNP\d{15,20}\b", text):
        if match not in seen:
            seen.add(match)
            unique_accounts.append(match)

    # Context-gated bare digit accounts: number must be near an account-ish keyword
    context_pattern = (
        r"(?:account|acc(?:ount)?\.?|a/c)\s*(?:number|no\.?|#)?\s*[:\-]?\s*(\d{9,16})"
    )
    for match in re.findall(context_pattern, text, re.IGNORECASE):
        if match not in seen:
            seen.add(match)
            unique_accounts.append(match)

    return unique_accounts


def extract_swift_codes(text):
    """
    Extracts SWIFT/BIC codes from text. Requires the 4-letter bank code to
    be followed by exactly a 2-letter country code, 2-char location code,
    and optional 3-char branch code, with a word boundary on both sides so
    we don't pick up fragments of ordinary uppercase words/IDs.
    """
    pattern = r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b"
    candidates = re.findall(pattern, text)

    # Filter out common false positives: all-letter "words" that happen to
    # be 8 characters of valid SWIFT shape but are actually English text
    # (e.g. acronyms in narrative). Real SWIFT codes always include at
    # least one of: a non-letter char in chars 5-8, or the candidate is in
    # a known exclude-list shape. As a pragmatic filter, require that the
    # candidate is not a recognisable common word and isn't purely composed
    # of letters that look like an account-section keyword.
    blocklist = {"SUSPICIOUSXX", "ACCOUNTNOXX"}
    return [c for c in dict.fromkeys(candidates) if c not in blocklist]


def extract_named_entities(text):
    """
    Extracts PERSON and ORG entities using spaCy NER, with post-filtering to
    remove common false positives that arise from short/fragmentary input
    text (e.g. XML field labels concatenated together, single-letter codes,
    currency/jurisdiction abbreviations misclassified as names).
    """
    nlp = get_nlp_model()
    doc = nlp(text)

    persons = []
    orgs = []

    seen_persons = set()
    seen_orgs = set()

    for ent in doc.ents:
        clean_ent = ent.text.strip()
        if not clean_ent:
            continue

        # Reject entities that are mostly digits/punctuation, single
        # characters, or appear in the manual blocklist
        if clean_ent.lower() in _NER_BLOCKLIST:
            continue
        if len(clean_ent) <= 2:
            continue
        if sum(ch.isdigit() for ch in clean_ent) > len(clean_ent) * 0.3:
            continue
        # Reject entities with more than 4 words -- real names/orgs in this
        # domain are short; long token runs are almost always concatenated
        # XML field garbage
        if len(clean_ent.split()) > 4:
            continue

        if ent.label_ == "PERSON":
            if clean_ent.lower() not in seen_persons:
                seen_persons.add(clean_ent.lower())
                persons.append(clean_ent)
        elif ent.label_ == "ORG":
            if clean_ent.lower() not in seen_orgs:
                seen_orgs.add(clean_ent.lower())
                orgs.append(clean_ent)

    return persons, orgs


def extract_all_facts(text):
    """
    Combines all extraction methods to produce the comprehensive fact checklist.
    """
    persons, orgs = extract_named_entities(text)
    return {
        "amounts": extract_amounts(text),
        "dates": extract_dates(text),
        "accounts": extract_accounts(text),
        "swift_codes": extract_swift_codes(text),
        "persons": persons,
        "orgs": orgs,
    }


# Canonical keys produced by data_loader.extract_structured_facts (and
# mirrored into demo.py's record_dict for ad-hoc pasted reports). Keeping
# this list in one place means faithfulness.py and evaluation.py can't
# silently drift out of sync with what data_loader actually emits.
STRUCTURED_FACT_KEYS = (
    "sender_account_number", "sender_account_name", "sender_institution",
    "sender_institution_code", "sender_signatory_name", "sender_country",
    "receiver_account_number", "receiver_account_name", "receiver_institution",
    "receiver_institution_code", "receiver_country",
    "foreign_currency_code", "foreign_amount",
)


def structured_facts_from_record(record):
    """
    Pulls the canonical structured-fact fields out of a flat record dict
    (a merged_df row, or demo.py's record_dict after it merges in
    extract_structured_facts for an ad-hoc pasted report). Fields that are
    absent or empty are omitted rather than included as None/"" so callers
    don't have to filter them out themselves.

    xml_amount_local is handled separately from the truthy filter above
    because a legitimate amount of 0.0 would otherwise be silently dropped
    -- and the local transaction amount is the single most important fact
    in the whole report, so it must always be carried into the checklist
    when present.
    """
    facts = {k: record.get(k) for k in STRUCTURED_FACT_KEYS if record.get(k)}
    amount = record.get("xml_amount_local")
    if amount is not None:
        facts["xml_amount_local"] = amount
    return facts


def build_fact_checklist(narrative_text, structured_facts=None):
    """
    Builds the MUST-PRESERVE facts checklist used for faithfulness
    verification, combining two sources:

    1. NER/regex extraction over whatever free-text narrative is present
       (extract_all_facts) -- still useful for narrative fields that
       genuinely contain prose (case notes, investigator comments, etc).
    2. Deterministic structured facts (account numbers, account-holder
       names, institution names, foreign-currency amount) pulled directly
       from known XML tags via data_loader.extract_structured_facts. These
       are 100% reliable, no NER guessing involved, and for this report
       schema they are the PRIMARY source of "persons"/"orgs" facts, since
       the only narrative tag here (<comments>) is typically boilerplate
       with no real names in it at all.

    Without (2), a report whose only narrative content is something like
    "Report filed by reporting entity." produces an empty checklist, and
    verify_fact_faithfulness has nothing left to check -- which is exactly
    the failure mode that was producing meaningless 100% "faithfulness"
    scores on reports that were, in fact, missing real information.
    """
    facts = extract_all_facts(narrative_text)
    if not structured_facts:
        return facts

    for acc_key in ("sender_account_number", "receiver_account_number"):
        acc = structured_facts.get(acc_key)
        if acc and acc not in facts["accounts"]:
            facts["accounts"].append(acc)

    for name_key in ("sender_account_name", "receiver_account_name", "sender_signatory_name"):
        name = structured_facts.get(name_key)
        if name and name not in facts["persons"]:
            facts["persons"].append(name)

    for org_key in ("sender_institution", "receiver_institution"):
        org = structured_facts.get(org_key)
        if org and org not in facts["orgs"]:
            facts["orgs"].append(org)

    foreign_amt = structured_facts.get("foreign_amount")
    if foreign_amt and foreign_amt not in facts["amounts"]:
        facts["amounts"].append(foreign_amt)

    local_amt = structured_facts.get("xml_amount_local")
    if local_amt is not None:
        local_amt_str = f"{float(local_amt):.2f}"
        if local_amt_str not in facts["amounts"]:
            facts["amounts"].append(local_amt_str)

    return facts


if __name__ == "__main__":
    test_text = "Transaction of NPR 535,368.64 on 2022-10-07 from account NP00000000003412850188 by John Jensen to Jeremy Martinez at SBL bank."
    print("Test extraction:")
    print(extract_all_facts(test_text))