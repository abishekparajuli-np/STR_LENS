import sys
sys.path.insert(0, '.')
from src.data_loader import load_data
import pandas as pd

merged_df, transactions, accounts, xml_df = load_data()

# Use the transactions.csv as a stand-in for the user-uploaded batch CSV
batch_df = pd.read_csv('data/transactions.csv')

print('merged_df columns (what load_data() produces):')
print(list(merged_df.columns))
print()

to_process = pd.merge(batch_df[["row_index"]], merged_df, on='row_index', how='inner')
print('Columns AFTER the batch tab merge:')
print(list(to_process.columns))
print()

key_fields = [
    'xml_reason', 'xml_comments', 'narrative_text',
    'sender_account_name', 'sender_institution', 'sender_account_number',
    'receiver_account_name', 'receiver_institution', 'receiver_account_number',
    'xml_amount_local',
]
print('Field-by-field check (does the unsuffixed name survive?):')
for col in key_fields:
    matches = [c for c in to_process.columns if c == col or c.startswith(col + '_')]
    exists_unsuffixed = col in to_process.columns
    print(f'  {col:28s} unsuffixed_present={exists_unsuffixed!s:5s} all_matches={matches}')
print()
if 'xml_reason' in to_process.columns:
    print("Sample value of record.get('xml_reason') for row 0:")
    print(repr(to_process.iloc[0].get('xml_reason')))
else:
    print(">>> 'xml_reason' is NOT a column after merge -- record.get('xml_reason','') will always return '' <<<")
