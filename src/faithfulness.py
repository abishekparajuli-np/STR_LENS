import re
import os
from src.extraction import clean_amount, extract_all_facts
from src.summarizer import summarize_report, SYSTEM_PROMPT

def verify_fact_faithfulness(summary_text, facts_checklist):
    """
    Checks if each fact in the checklist appears in the summary.
    Returns faithfulness score (float 0-1) and a list of missing facts.
    """
    summary_lower = summary_text.lower()
    total_facts = 0
    preserved_facts = 0
    missing_facts = []
    
    # Verify amounts
    for amt in facts_checklist.get("amounts", []):
        total_facts += 1
        cleaned_amt = clean_amount(amt)
        # Search for either original string or normalised number
        if amt.lower() in summary_lower or cleaned_amt in summary_lower:
            preserved_facts += 1
        else:
            # Let's also check if standard formatted numbers match
            formatted_amt = f"{float(cleaned_amt):,}" if "." in cleaned_amt else cleaned_amt
            if formatted_amt in summary_text:
                preserved_facts += 1
            else:
                missing_facts.append(f"Amount: {amt}")
                
    # Verify dates
    for date in facts_checklist.get("dates", []):
        total_facts += 1
        if date.lower() in summary_lower:
            preserved_facts += 1
        else:
            missing_facts.append(f"Date: {date}")
            
    # Verify accounts
    for acc in facts_checklist.get("accounts", []):
        total_facts += 1
        if acc.lower() in summary_lower:
            preserved_facts += 1
        else:
            missing_facts.append(f"Account: {acc}")
            
    # Verify SWIFT codes
    for swift in facts_checklist.get("swift_codes", []):
        total_facts += 1
        if swift.lower() in summary_lower:
            preserved_facts += 1
        else:
            missing_facts.append(f"SWIFT Code: {swift}")
            
    # Verify persons
    for person in facts_checklist.get("persons", []):
        total_facts += 1
        if person.lower() in summary_lower:
            preserved_facts += 1
        else:
            missing_facts.append(f"Party (Person): {person}")
            
    # Verify orgs
    for org in facts_checklist.get("orgs", []):
        total_facts += 1
        if org.lower() in summary_lower:
            preserved_facts += 1
        else:
            missing_facts.append(f"Party (Organisation): {org}")
            
    score = preserved_facts / total_facts if total_facts > 0 else 1.0
    return score, missing_facts

async def summarize_and_verify(record, api_key=None, max_attempts=2):
    """
    Summarizes the report and runs the verification/re-prompt loop.
    """
    if api_key is None:
        api_key = os.environ.get("GROQ_API_KEY")
        
    # Step 1: Extract all must-preserve facts
    narrative = record.get("narrative_text", "")
    facts_checklist = extract_all_facts(narrative)
    
    # Step 2: Generate initial draft summary
    result = await summarize_report(record, facts_checklist, api_key=api_key)
    summary_text = result["summary_text"]
    
    # Step 3: Run verification loop
    attempts = 0
    reprompted = False
    score, missing_facts = verify_fact_faithfulness(summary_text, facts_checklist)
    
    # If the key is missing/mock, the generated summary will be 100% faithful,
    # but for actual LLM calls we run the re-prompt loop
    while score < 0.85 and attempts < max_attempts and api_key and api_key != "your_key_here":
        attempts += 1
        reprompted = True
        print(f"Faithfulness check failed (score: {score:.2%}). Re-prompting attempt {attempts}/{max_attempts}...")
        
        # Formulate re-prompt user message
        reprompt_content = f"""Your previous summary missed the following key facts from the checklist.
Please rewrite the summary, ensuring all of these are included verbatim:
Missing facts: {', '.join(missing_facts)}

Remember to maintain the exact structured format:
[Suspicion Type] | [Parties] | [Transaction Summary] | [Key Red Flags]
"""
        from groq import AsyncGroq
        client = AsyncGroq(api_key=api_key)
        
        try:
            response = await client.chat.completions.create(
                model="llama3-70b-8192",
                max_tokens=400,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Original Narrative: {narrative}\nChecklist: {facts_checklist}"},
                    {"role": "assistant", "content": summary_text},
                    {"role": "user", "content": reprompt_content}
                ]
            )
            summary_text = response.choices[0].message.content.strip()
            score, missing_facts = verify_fact_faithfulness(summary_text, facts_checklist)
        except Exception as e:
            print(f"Error during re-prompt: {e}")
            break
            
    # Parse color-coded sections from structured text
    parts = summary_text.split(" | ")
    suspicion_type = parts[0].strip() if len(parts) > 0 else "Unknown"
    parties = parts[1].strip() if len(parts) > 1 else "Unknown"
    transaction_summary = parts[2].strip() if len(parts) > 2 else "Unknown"
    red_flags = parts[3].strip() if len(parts) > 3 else "Unknown"
    
    # Calculate word count
    word_count = len(summary_text.split())
    
    return {
        "str_id": record.get("str_id", ""),
        "suspicion_type": suspicion_type,
        "parties": parties,
        "transaction_summary": transaction_summary,
        "red_flags": red_flags,
        "summary_text": summary_text,
        "faithfulness_score": round(score, 2),
        "missing_facts": missing_facts,
        "word_count": word_count,
        "processing_time_ms": result["processing_time_ms"],
        "model": result["model"],
        "reprompted": reprompted,
        "human_review": score < 0.85
    }
