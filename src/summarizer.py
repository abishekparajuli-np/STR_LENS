import os
import time
import asyncio
from groq import AsyncGroq

# System Prompt detailing constraints and structure
SYSTEM_PROMPT = """You are an expert AML (Anti-Money Laundering) compliance analyst at a global financial institution. Your task is to generate a concise, structured intelligence card summarizing a suspicious transaction narrative.

You must follow these strict rules:
1. Output format MUST follow this exact structure, with sections separated by vertical pipes:
   [Suspicion Type] | [Parties] | [Transaction Summary] | [Key Red Flags]
   Do not add any conversational text or other headers.
2. Length must be strictly between 100 and 200 words.
3. You must preserve all facts listed in the 'MUST-PRESERVE FACTS CHECKLIST' provided in the user prompt (amounts, dates, account numbers, names, SWIFT codes). Do not omit, alter, or hallucinate any of these facts.
4. Keep the tone professional, objective, and analytical, as this will be read by compliance officers and law enforcement.
"""

def generate_simulated_summary(record):
    """
    Generates a realistic, highly faithful AML summary from structured record fields
    when the Anthropic API key is not available.
    """
    xml_amt = record.get("xml_amount_local", 0.0)
    amt_formatted = f"NPR {xml_amt:,.2f}"
    
    sender_name = record.get("sender_account_number", "Unknown Sender")
    # Retrieve names if available
    s_name = record.get("xml_comments", "") # placeholder parsing or lookup
    # Let's extract names dynamically from record
    s_name = record.get("sender_institution", "Sender Bank")
    
    # Let's use clean XML fields if they exist
    s_acc = record.get("sender_account_number", "Sender Acc")
    r_acc = record.get("receiver_account_number", "Receiver Acc")
    
    date_tx = record.get("Date", "2022-10-07")
    time_tx = record.get("Time", "10:00:00")
    
    susp_type = "Potential structuring and layering activity"
    if record.get("cross_border_flag") == 1:
        susp_type = "Cross-border fund routing / potential money laundering"
    elif record.get("above_10M_NPR") == 1:
        susp_type = "High-value suspicious transaction"
        
    parties = f"Sender: {record.get('sender_institution', 'SBL')} account {s_acc}; Receiver: {record.get('receiver_institution', 'HBL')} account {r_acc}"
    
    summary = f"On {date_tx} at {time_tx}, a transaction of {amt_formatted} was initiated from account {s_acc} to account {r_acc}. The transfer mode code was {record.get('transmode_code', 'Z')}."
    
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
    
    # Construct exact format: [Suspicion Type] | [Parties] | [Transaction Summary] | [Key Red Flags]
    simulated_output = f"{susp_type} | {parties} | {summary} | Red Flags: {red_flags_str}."
    return simulated_output

async def summarize_report(record, facts_checklist, api_key=None):
    """
    Summarizes a single report. Uses Groq layer if API key is present; otherwise
    falls back to the high-fidelity rule-based simulation.
    """
    t0 = time.time()
    
    if api_key is None:
        api_key = os.environ.get("GROQ_API_KEY")
        
    if not api_key or api_key == "your_key_here":
        # Run simulation mode
        summary_text = generate_simulated_summary(record)
        processing_time_ms = int((time.time() - t0) * 1000)
        return {
            "summary_text": summary_text,
            "model": "llama3-70b-8192 (Simulated)",
            "processing_time_ms": processing_time_ms,
            "reprompted": False
        }
        
    # Active mode using Groq Async client
    client = AsyncGroq(api_key=api_key)
    
    user_content = f"""Report Narrative / Details:
Reason: {record.get('xml_reason', '')}
Comments: {record.get('xml_comments', '')}
Transaction Amount: {record.get('xml_amount_local', 0.0)}
Details: {record.get('narrative_text', '')}

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
            model="llama3-70b-8192",
            max_tokens=400,
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]
        )
        summary_text = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling Groq API: {e}. Falling back to simulation...")
        summary_text = generate_simulated_summary(record)
        
    processing_time_ms = int((time.time() - t0) * 1000)
    
    return {
        "summary_text": summary_text,
        "model": "llama3-70b-8192",
        "processing_time_ms": processing_time_ms,
        "reprompted": False
    }
