import os
import glob
import json
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path

# Fields that hold genuine free-text narrative prose, as opposed to
# structured/categorical fields (IDs, codes, enums). Only these are
# concatenated to build the narrative used for entity extraction and
# summarisation. This is what fixes the NER garbage problem: feeding spaCy
# "RPT-2026-000001 Structuring TXN-2026-000001 535368.64 Customer John..."
# produces nonsense entities, whereas feeding it just the comments field
# produces clean PERSON/ORG extractions.
NARRATIVE_FIELD_NAMES = {"comments", "narrative", "description", "details", "notes"}


def _local_tag(tag):
    """
    Returns an XML tag's local name, stripping any namespace prefix.
    ElementTree represents namespaced tags as '{uri}localname', which never
    matches a plain string like 'comments'. Without stripping the namespace,
    every comparison against NARRATIVE_FIELD_NAMES silently fails on
    namespaced documents and the function falls through to dumping the
    entire tree's text -- which is the unreadable wall of text users see in
    the UI. This helper makes the comparison namespace-agnostic.
    """
    if not isinstance(tag, str):
        return ""
    if "}" in tag:
        tag = tag.split("}", 1)[1]
    return tag.lower()


def extract_narrative_text(elem):
    """
    Walks the XML tree and concatenates text only from nodes whose local tag
    name indicates genuine free-text narrative content (see
    NARRATIVE_FIELD_NAMES), rather than every text node in the document.
    Structured fields like report_id, transactionnumber, and amount_local
    are excluded since they are not prose and only pollute downstream
    NER/regex extraction.

    Unlike the previous implementation, this does NOT silently fall back to
    dumping every text node in the document when no narrative field is
    found. That fallback was the source of the unreadable, unlabeled wall of
    concatenated field values shown in the UI -- it looks like a working
    narrative but is actually a symptom of the parser failing to find any
    real narrative field (often due to an XML namespace it didn't strip, or
    a schema that genuinely doesn't have a free-text field). Returning an
    empty string instead makes that failure visible immediately rather than
    disguising it as a (garbled) "narrative".
    """
    texts = []
    for node in elem.iter():
        tag = _local_tag(node.tag)
        if tag in NARRATIVE_FIELD_NAMES and node.text and node.text.strip():
            texts.append(node.text.strip())

    return " ".join(texts)


def extract_structured_facts(root):
    """
    Pulls the ground-truth party/account/institution facts directly from the
    structured (non-narrative) parts of a goAML-style <transaction> element:
    t_from_my_client/from_account and t_to/to_account.

    This is the fix for the root cause behind misleadingly perfect
    "faithfulness" scores: the account holder names, institution names,
    account numbers, and foreign-currency amount all live in these
    structured elements, NOT in any field that NARRATIVE_FIELD_NAMES
    recognises as prose. The only narrative-shaped tag in this schema is
    <comments>, which is typically boilerplate ("Report filed by reporting
    entity.") and contains none of this. extract_all_facts() run over that
    boilerplate alone finds nothing -- not because the report lacks facts,
    but because the facts were never narrative text in the first place.
    Pulling them here, directly from their known tags, is deterministic and
    trustworthy by construction (no NER guessing involved), and feeds both
    the faithfulness checklist and the LLM prompt with the data they were
    missing.

    Returns a flat dict. Any field not present in the document is simply
    omitted (never set to a placeholder string), so callers decide their
    own defaults/fallback display text.
    """
    facts = {}
    tx = root.find("transaction")
    if tx is None:
        return facts

    def text_of(elem, tag):
        if elem is None:
            return None
        node = elem.find(tag)
        return node.text.strip() if node is not None and node.text and node.text.strip() else None

    # --- Sender side: t_from_my_client / from_account ---
    from_party = tx.find("t_from_my_client")
    if from_party is not None:
        from_account = from_party.find("from_account")
        if from_account is not None:
            acc_num = text_of(from_account, "account")
            if acc_num:
                facts["sender_account_number"] = acc_num
            acc_name = text_of(from_account, "account_name")
            if acc_name:
                facts["sender_account_name"] = acc_name
            inst_name = text_of(from_account, "institution_name")
            if inst_name:
                facts["sender_institution"] = inst_name
            inst_code = text_of(from_account, "institution_code")
            if inst_code:
                facts["sender_institution_code"] = inst_code

            # Primary signatory's name, as a fallback/extra for the account
            # holder name (some schemas populate one but not the other).
            signatory = from_account.find("signatory")
            if signatory is not None:
                person = signatory.find("t_person")
                if person is not None:
                    first = text_of(person, "first_name")
                    last = text_of(person, "last_name")
                    full = " ".join(p for p in (first, last) if p)
                    if full:
                        facts["sender_signatory_name"] = full

        country = text_of(from_party, "from_country")
        if country:
            facts["sender_country"] = country

        fx = from_party.find("from_foreign_currency")
        if fx is not None:
            code = text_of(fx, "foreign_currency_code")
            amt = text_of(fx, "foreign_amount")
            if code:
                facts["foreign_currency_code"] = code
            if amt:
                facts["foreign_amount"] = amt

    # --- Receiver side: t_to / to_account ---
    to_party = tx.find("t_to")
    if to_party is not None:
        to_account = to_party.find("to_account")
        if to_account is not None:
            acc_num = text_of(to_account, "account")
            if acc_num:
                facts["receiver_account_number"] = acc_num
            acc_name = text_of(to_account, "account_name")
            if acc_name:
                facts["receiver_account_name"] = acc_name
            inst_name = text_of(to_account, "institution_name")
            if inst_name:
                facts["receiver_institution"] = inst_name
            inst_code = text_of(to_account, "institution_code")
            if inst_code:
                facts["receiver_institution_code"] = inst_code

        country = text_of(to_party, "to_country")
        if country:
            facts["receiver_country"] = country

    return facts


def parse_xml_reports(reports_dir):
    """
    Parses all XML files in the reports directory and extracts transaction details.
    """
    xml_files = sorted(glob.glob(str(Path(reports_dir) / "*.xml")))
    records = []

    for fp in xml_files:
        try:
            tree = ET.parse(fp)
            root = tree.getroot()

            report_id = root.find("report_id").text if root.find("report_id") is not None else ""
            reason = root.find("reason").text if root.find("reason") is not None else ""

            tx_elem = root.find("transaction")
            if tx_elem is not None:
                tx_num_elem = tx_elem.find("transactionnumber")
                tx_num = tx_num_elem.text if tx_num_elem is not None and tx_num_elem.text else ""
                comments_elem = tx_elem.find("comments")
                comments = comments_elem.text if comments_elem is not None and comments_elem.text else ""
                amt_elem = tx_elem.find("amount_local")
                try:
                    amount_local = float(amt_elem.text) if amt_elem is not None and amt_elem.text else 0.0
                except (ValueError, TypeError):
                    amount_local = 0.0

                # The row index in transactions.csv is the last part of transactionnumber.
                # Guard against non-numeric trailing segments instead of letting a bare
                # ValueError abort parsing of the whole file.
                parts = tx_num.split("-") if tx_num else []
                row_index = -1
                if parts and parts[-1].isdigit():
                    row_index = int(parts[-1])
                elif parts:
                    print(f"Warning: could not parse row_index from transactionnumber '{tx_num}' in {fp}")
            else:
                row_index = -1
                comments = ""
                amount_local = 0.0

            raw_text = extract_narrative_text(root)
            if not raw_text:
                # Surface the failure instead of hiding it -- comments is the
                # documented narrative field for this schema, so fall back to
                # it directly when the generic narrative-field walk finds
                # nothing (e.g. due to namespaces or unexpected nesting).
                raw_text = comments
                if not raw_text:
                    print(f"Warning: no narrative text found in {fp}")

            # Scalar transaction-level fields and the structured party/
            # account/institution facts (see extract_structured_facts) --
            # these live outside any narrative field, so they have to be
            # pulled directly from their known tags rather than relying on
            # the narrative-text walk above to ever surface them.
            date_elem = tx_elem.find("date_transaction") if tx_elem is not None else None
            date_transaction = date_elem.text.strip() if date_elem is not None and date_elem.text else None

            transmode_elem = tx_elem.find("transmode_comment") if tx_elem is not None else None
            transmode_comment = transmode_elem.text.strip() if transmode_elem is not None and transmode_elem.text else None

            currency_elem = root.find("currency_code_local")
            currency_code_local = currency_elem.text.strip() if currency_elem is not None and currency_elem.text else None

            structured_facts = extract_structured_facts(root)

            records.append({
                "str_id": report_id,
                "xml_file": os.path.basename(fp),
                "xml_amount_local": amount_local,
                "xml_reason": reason,
                "xml_comments": comments,
                "row_index": row_index,
                "narrative_text": raw_text,
                "word_count": len(raw_text.split()),
                "char_count": len(raw_text),
                "date_transaction": date_transaction,
                "transmode_comment": transmode_comment,
                "currency_code_local": currency_code_local,
                **structured_facts,
            })
        except (ET.ParseError, ValueError, AttributeError) as e:
            print(f"Error parsing {fp}: {e}")

    return pd.DataFrame(records)


def load_data(reports_dir="reports", data_dir="data"):
    """
    Main ingestion function. Loads XML reports and CSVs, merges them, and checks quality.

    `data_dir` and `reports_dir` are resolved relative to the project root
    (one level up from this file, i.e. the parent of `src/`) when given as
    plain relative paths, rather than relative to the current working
    directory. Without this, `pd.read_csv("data/transactions.csv")` only
    works if Streamlit happens to be launched from the project root; launch
    it from inside `app/` (a common setup when demo.py lives there) and the
    same relative path silently points at a nonexistent `app/data/` instead,
    raising FileNotFoundError. Passing an absolute path still works as-is.
    """
    project_root = Path(__file__).resolve().parent.parent

    def _resolve(path_str):
        p = Path(path_str)
        return p if p.is_absolute() else project_root / p

    data_dir_path = _resolve(data_dir)
    reports_dir_path = _resolve(reports_dir)

    print("Parsing XML reports...")
    xml_df = parse_xml_reports(reports_dir_path)
    print(f"Parsed {len(xml_df)} XML reports.")

    print("Loading CSV datasets...")
    transactions = pd.read_csv(data_dir_path / "transactions.csv")
    ml_features = pd.read_csv(data_dir_path / "ml_features.csv")
    accounts = pd.read_csv(data_dir_path / "accounts.csv")

    # ml_features has row_index in transactions order, align row_index
    if "row_index" not in ml_features.columns:
        ml_features["row_index"] = transactions["row_index"].values

    print("Merging datasets...")
    # Drop rows where row_index could not be determined (-1 sentinel) before
    # merging, so they don't accidentally collide with a real row_index of -1
    # in the transactions data and silently corrupt the join
    valid_xml_df = xml_df[xml_df["row_index"] >= 0].copy()
    dropped = len(xml_df) - len(valid_xml_df)
    if dropped > 0:
        print(f"Warning: dropping {dropped} XML reports with unparseable row_index before merge.")

    merged_df = pd.merge(valid_xml_df, transactions, on="row_index", how="inner")

    # Merge with ml_features to obtain the is_suspicious_tx label
    label_df = ml_features[["row_index", "is_suspicious_tx"]]
    merged_df = pd.merge(merged_df, label_df, on="row_index", how="inner")

    print(f"Merged dataset shape: {merged_df.shape}")
    join_rate = len(merged_df) / len(xml_df) if len(xml_df) > 0 else 0
    print(f"Join rate: {join_rate:.2%}")

    return merged_df, transactions, accounts, xml_df


def save_clean_data(df, output_dir="data"):
    """
    Saves clean merged data as Parquet.
    """
    output_path = Path(output_dir) / "clean_data.parquet"
    df.to_parquet(output_path, index=False)
    print(f"Saved clean merged data to {output_path}")


def generate_quality_report(merged_df, xml_df, output_path="data_quality_report.md"):
    """
    Generates a markdown data quality report using real computed values
    rather than hardcoded placeholder numbers.
    """
    avg_words = xml_df["word_count"].mean() if len(xml_df) > 0 else 0
    max_words = xml_df["word_count"].max() if len(xml_df) > 0 else 0
    join_rate = len(merged_df) / len(xml_df) if len(xml_df) > 0 else 0
    null_count = int(merged_df.isnull().sum().sum()) if len(merged_df) > 0 else 0
    total_tx = len(merged_df)
    suspicious_count = int(merged_df["is_suspicious_tx"].sum()) if "is_suspicious_tx" in merged_df.columns else 0
    suspicious_rate = (suspicious_count / total_tx * 100) if total_tx > 0 else 0
    pass_fail = "PASS" if join_rate >= 0.90 else "FAIL"

    report_content = f"""# STR-Lens — Data Quality Report

## Ingestion Summary
* **Total XML Reports Found**: {len(xml_df)}
* **Successfully Parsed**: {len(xml_df)}
* **Join Rate with CSV (on transaction row_index)**: {join_rate:.2%} (Target: >90%)
* **Average Narrative Word Count**: {avg_words:.1f}
* **Max Narrative Word Count**: {max_words}

## Profiling & Quality Indicators
* **Null Values**: {null_count} nulls detected in the merged dataset.
* **Target Imbalance**: Out of {total_tx:,} transactions, {suspicious_count} are marked as suspicious ({suspicious_rate:.2f}% positive rate).

## Verification Status
* **Phase 1 Ingestion Check**: {pass_fail} (Join rate is {join_rate:.1%}, threshold is 90.0%).
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Generated quality report at {output_path}")


def generate_sample_record(merged_df, output_path="sample_joined_record.json"):
    """
    Saves the first merged record to a JSON file.
    """
    if not merged_df.empty:
        record = merged_df.iloc[0].to_dict()
        for k, v in record.items():
            if hasattr(v, "item"):
                record[k] = v.item()
            elif pd.isna(v):
                record[k] = None

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        print(f"Saved sample record to {output_path}")


if __name__ == "__main__":
    merged, tx, acc, xml = load_data()
    save_clean_data(merged)
    generate_quality_report(merged, xml)
    generate_sample_record(merged)