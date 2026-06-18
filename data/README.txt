FINANCIAL TRANSACTION INTELLIGENCE - HACKATHON DATASET
=======================================================
Synthetic banking transactions from Nepali banks (amounts in NPR). Fully
synthetic - no real customers or accounts. Build models for any of the tracks.

FILES
  transactions.csv  - transactions with engineered features (no answer column)
  accounts.csv      - customer/account KYC records
  graph_edges.csv   - sender -> receiver money-flow edge list
  ml_features.csv   - model-ready numeric features
  reports/          - sample transaction-report files (XML)

TRACKS
  1. Report quality scoring        4. Risk scoring / prioritisation
  2. Entity resolution (dedupe)    5. Behaviour clustering
  3. Network / graph detection     6. Report narrative summarisation
