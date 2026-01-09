# CUAD - Contract Understanding Atticus Dataset

## About
CUAD is a legal contract QA dataset with 510 contracts and 13,000+ expert-annotated clauses across 41 legal clause categories.

## Download
Manual download from: https://www.atticusprojectai.org/cuad

Or via GitHub:
```bash
git clone https://github.com/TheAtticusProject/cuad.git
```

## Format (SQuAD-style)
```json
{
  "data": [
    {
      "title": "contract_name",
      "paragraphs": [
        {
          "context": "Full contract text...",
          "qas": [
            {
              "id": "unique_id",
              "question": "What is the termination clause?",
              "answers": [
                {"text": "Either party may terminate...", "answer_start": 1234}
              ],
              "is_impossible": false
            }
          ]
        }
      ]
    }
  ]
}
```

## Question Types (41 categories)
- Document Name
- Parties
- Agreement Date
- Effective Date
- Expiration Date
- Renewal Term
- Notice Period To Terminate Renewal
- Governing Law
- Most Favored Nation
- Non-Compete
- Exclusivity
- No-Solicit Of Customers
- No-Solicit Of Employees
- Non-Disparagement
- Termination For Convenience
- Rofr/Rofo/Rofn
- Change Of Control
- Anti-Assignment
- Revenue/Profit Sharing
- Price Restrictions
- Minimum Commitment
- Volume Restriction
- Ip Ownership Assignment
- Joint Ip Ownership
- License Grant
- Non-Transferable License
- Affiliate License-Licensor
- Affiliate License-Licensee
- Unlimited/All-You-Can-Eat-License
- Irrevocable Or Perpetual License
- Source Code Escrow
- Post-Termination Services
- Audit Rights
- Uncapped Liability
- Cap On Liability
- Liquidated Damages
- Warranty Duration
- Insurance
- Covenant Not To Sue
- Third Party Beneficiary

## Key Differences from FinanceBench
| Aspect | FinanceBench | CUAD |
|--------|--------------|------|
| Domain | Financial filings | Legal contracts |
| Answer type | Numbers, facts | Clause extractions |
| Context | Tables + prose | Dense legal text |
| Retrieval challenge | Metadata filtering | Clause identification |
