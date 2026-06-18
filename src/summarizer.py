import os
import time
import asyncio
from groq import AsyncGroq

# Groq decommissioned llama3-70b-8192 on 2025-05-31. The supported
# replacement is llama-3.3-70b-versatile. Using the old name causes every
# single API call to fail with a 400 invalid_request_error, which is what
# was silently sending 100% of records through the simulated fallback path
# while still being labeled as if the real model had produced them.
GROQ_MODEL = "llama-3.3-70b-versatile"

# System Prompt detailing constraints and structure
SYSTEM_PROMPT = """You are an expert AML (Anti-Money Laundering) compliance analyst at a global financial institution. Your task is to generate a concise, structured intelligence card summarizing a suspicious transaction narrative.

You must follow these strict rules:
1. Output format MUST follow this exact structure, with sections separated by vertical pipes:
   [Suspicion Type] | [Parties] | [Transaction Summary] | [Key Red Flags]
   Do not add any conversational text or other headers.
2. Length must be strictly between 100 and 200 words.
3. You must preserve all facts listed in the 'MUST-PRESERVE FACTS CHECKLIST' provided in the user prompt (amounts, dates, account numbers, names, SWIFT codes). Do not omit, alter, or hallucinate any of these facts.
4. Keep the tone professional, objective, and analytical, as this will be read by compliance officers and law enforcement.
5. Amounts must always be written with their actual currency code (e.g. "NPR 535,368.64" or "GBP 2,603.30") exactly as given in the source data. Never substitute a "$" sign or assume USD unless the source data itself specifies USD.
"""


def _format_known_parties(record):
    """
    Renders the sender/receiver party fields -- populated from the
    structured, non-narrative parts of the source XML, see
    data_loader.extract_structured_facts -- as plain text for the LLM
    prompt.

    Without this, the model only ever saw the boilerplate <comments> field
    and the bare transaction amount, which is exactly why summaries were
    coming back with "Unknown Parties" even when the source XML clearly
    named both account holders and their institutions: the model wasn't
    hallucinating, it genuinely was never told who the parties were.
    """
    lines = []
    sender_name = record.get("sender_account_name") or record.get("sender_signatory_name")
    if sender_name or record.get("sender_institution"):
        lines.append(
            f"Sender: {sender_name or 'Unknown'} - account {record.get('sender_account_number', 'Unknown')} "
            f"at {record.get('sender_institution', 'Unknown institution')}"
        )
    receiver_name = record.get("receiver_account_name")
    if receiver_name or record.get("receiver_institution"):
        lines.append(
            f"Receiver: {receiver_name or 'Unknown'} - account {record.get('receiver_account_number', 'Unknown')} "
            f"at {record.get('receiver_institution', 'Unknown institution')}"
        )
    if record.get("foreign_currency_code") and record.get("foreign_amount"):
        lines.append(f"Foreign currency leg: {record['foreign_currency_code']} {record['foreign_amount']}")

    return "\n".join(lines) if lines else "Not available in source data."


def generate_simulated_summary(record):
    """
    Generates a realistic, highly faithful AML summary from structured record
    fields when no API key is available. This is a rule-based fallback, not
    an LLM call, and is always labeled as such by the caller.

    Field names here must match the canonical structured-fact keys now
    produced by data_loader.extract_structured_facts (sender_account_number,
    sender_account_name, sender_institution, receiver_account_number,
    receiver_account_name, receiver_institution). The previous version
    guessed at column names that didn't exist anywhere in the schema, so it
    always fell through to "[unknown ...]" placeholders regardless of the
    actual record content; this also meant the simulated fallback path
    (which fires whenever the Groq API errors or no key is set) produced
    "Unknown Parties"-style output even when the source XML had everything
    needed to populate it correctly.
    """
    xml_amt = record.get("xml_amount_local", 0.0) or 0.0
    currency = record.get("currency_code_local") or "NPR"
    amt_formatted = f"{currency} {xml_amt:,.2f}"

    def first_present(rec, candidates, default):
        for c in candidates:
            val = rec.get(c)
            if val is not None and str(val).strip() and str(val) != "nan":
                return val
        return default

    s_acc = first_present(record, ["sender_account_number"], "[unknown sender account]")
    r_acc = first_present(record, ["receiver_account_number"], "[unknown receiver account]")
    s_inst = first_present(record, ["sender_institution"], "[unknown sender institution]")
    r_inst = first_present(record, ["receiver_institution"], "[unknown receiver institution]")
    s_name = first_present(record, ["sender_account_name", "sender_signatory_name"], None)
    r_name = first_present(record, ["receiver_account_name"], None)
    date_tx = first_present(record, ["date_transaction", "Date", "date", "transaction_date"], "[unknown date]")
    time_tx = first_present(record, ["Time", "time", "transaction_time"], "[unknown time]")
    transmode = first_present(record, ["transmode_comment", "transmode_code", "transfer_mode"], "[unknown]")

    susp_type = "Potential structuring and layering activity"
    if record.get("cross_border_flag") == 1:
        susp_type = "Cross-border fund routing / potential money laundering"
    elif record.get("above_10M_NPR") == 1:
        susp_type = "High-value suspicious transaction"

    sender_label = f"{s_name} ({s_inst}, acct {s_acc})" if s_name else f"{s_inst} account {s_acc}"
    receiver_label = f"{r_name} ({r_inst}, acct {r_acc})" if r_name else f"{r_inst} account {r_acc}"
    parties = f"Sender: {sender_label}; Receiver: {receiver_label}"

    summary = (
        f"On {date_tx} at {time_tx}, a transaction of {amt_formatted} was initiated "
        f"from account {s_acc} to account {r_acc}. The transfer mode code was {transmode}."
    )

    red_flags = []
    if record.get("cross_border_flag") == 1:
        red_flags.append("Cross-border transaction")
    if record.get("currency_mismatch") == 1:
        red_flags.append("Currency mismatch between sender and receiver")
    if record.get("sender_pep") == 1 or record.get("receiver_pep") == 1:
        red_flags.append("PEP involvement flagged")
    if record.get("above_1M_NPR") == 1:
        red_flags.append("Large cash transaction exceeding 1M NPR threshold")

    red_flags_str = ", ".join(red_flags) if red_flags else "Standard high-velocity transactions"

    simulated_output = f"{susp_type} | {parties} | {summary} | Red Flags: {red_flags_str}."
    return simulated_output


async def summarize_report(record, facts_checklist, api_key=None):
    """
    Summarizes a single report. Uses the Groq API if a valid key is present;
    otherwise falls back to the rule-based simulation. The returned "model"
    field always accurately reflects which path actually produced the text,
    so downstream faithfulness/evaluation code can tell real LLM output from
    simulated output instead of mistakenly comparing simulated text against
    "model": "llama-3.3-70b-versatile" as if a real call had succeeded.
    """
    t0 = time.time()

    if api_key is None:
        api_key = os.environ.get("GROQ_API_KEY")

    if not api_key or api_key == "your_key_here":
        summary_text = generate_simulated_summary(record)
        processing_time_ms = int((time.time() - t0) * 1000)
        return {
            "summary_text": summary_text,
            "model": f"{GROQ_MODEL} (Simulated - no API key)",
            "processing_time_ms": processing_time_ms,
            "reprompted": False,
        }

    client = AsyncGroq(api_key=api_key)

    user_content = f"""Report Narrative / Details:
Reason: {record.get('xml_reason', '')}
Comments: {record.get('xml_comments', '')}
Transaction Amount: {record.get('xml_amount_local', 0.0)} {record.get('currency_code_local', 'NPR')}
Details: {record.get('narrative_text', '')}

Known Parties (from structured transaction fields):
{_format_known_parties(record)}

MUST-PRESERVE FACTS CHECKLIST:
Amounts: {', '.join(facts_checklist.get('amounts', []))}
Dates: {', '.join(facts_checklist.get('dates', []))}
Accounts: {', '.join(facts_checklist.get('accounts', []))}
Persons: {', '.join(facts_checklist.get('persons', []))}
Orgs: {', '.join(facts_checklist.get('orgs', []))}
SWIFT/BIC Codes: {', '.join(facts_checklist.get('swift_codes', []))}
"""

    try:
        response = await client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=400,
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        summary_text = response.choices[0].message.content.strip()
        model_used = GROQ_MODEL
    except Exception as e:
        # IMPORTANT: this is the exact path that was silently swallowing
        # every API error in the original code. We now print the full
        # exception (not just str(e), which can hide HTTP status details on
        # some SDK versions) and accurately label the result as simulated
        # rather than claiming the real model produced it.
        print(f"Error calling Groq API ({type(e).__name__}): {e}. Falling back to simulation...")
        summary_text = generate_simulated_summary(record)
        model_used = f"{GROQ_MODEL} (Simulated - API error: {type(e).__name__})"

    processing_time_ms = int((time.time() - t0) * 1000)

    return {
        "summary_text": summary_text,
        "model": model_used,
        "processing_time_ms": processing_time_ms,
        "reprompted": False,
    }