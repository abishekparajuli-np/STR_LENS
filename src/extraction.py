import re
import sys
import subprocess
import spacy

_nlp = None

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
    """
    # Remove currency symbols and whitespace
    cleaned = re.sub(r"[^\d\.]", "", amt_str)
    try:
        # Standardise to float representation with 2 decimal places if valid
        val = float(cleaned)
        return f"{val:.2f}"
    except ValueError:
        return amt_str

def extract_amounts(text):
    """
    Extracts all currency amounts (NPR, USD, GBP, EUR, Rs, $, £) from text.
    """
    # Match currencies: $, £, Rs, Rs., NPR, USD, GBP, EUR followed by numbers
    amt_pattern = r"(?:NPR|USD|GBP|EUR|Rs\.?|Rs|£|\$)\s*[\d,]+(?:\.\d{2})?|\b[\d,]+\.\d{2}\b"
    matches = re.findall(amt_pattern, text, re.IGNORECASE)
    
    # Clean and deduplicate while maintaining order
    seen = set()
    unique_amounts = []
    for match in matches:
        cleaned = clean_amount(match)
        if cleaned not in seen:
            seen.add(cleaned)
            unique_amounts.append(cleaned)
            
    return unique_amounts

def extract_dates(text):
    """
    Extracts dates in MM/DD/YYYY, DD-MM-YYYY, Month DD YYYY formats.
    """
    patterns = [
        # MM/DD/YYYY or DD-MM-YYYY
        r"\b\d{1,2}[/\.-]\d{1,2}[/\.-]\d{4}\b",
        # Month DD, YYYY (e.g., January 14, 2023 or Jan 14 2023)
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b",
        # Month YYYY (e.g., Jan 2023)
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b",
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
    Extracts account numbers (NP-prefixed Nepalese accounts and standard 8-16 digits).
    """
    # NP prefixed format or 8-16 digit numbers surrounded by boundaries
    patterns = [
        r"\bNP\d{20}\b",
        r"\b\d{8,16}\b"
    ]
    
    unique_accounts = []
    seen = set()
    for pattern in patterns:
        for match in re.findall(pattern, text):
            match_clean = match.strip()
            if match_clean not in seen:
                seen.add(match_clean)
                unique_accounts.append(match_clean)
                
    return unique_accounts

def extract_swift_codes(text):
    """
    Extracts SWIFT/BIC codes from text.
    """
    pattern = r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b"
    matches = re.findall(pattern, text)
    return list(dict.fromkeys(matches)) # deduplicate keeping order

def extract_named_entities(text):
    """
    Extracts PERSON and ORG entities using spaCy NER.
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
        "orgs": orgs
    }

if __name__ == "__main__":
    test_text = "Transaction of NPR 535,368.64 on 2022-10-07 from account NP00000000003412850188 by John Jensen to Jeremy Martinez at SBL bank."
    print("Test extraction:")
    print(extract_all_facts(test_text))
