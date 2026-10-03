# Malaysia XX Bank: Card & Payments Fraud Operations SOP

Malaysia XX Bank, Fraud Operations. Version 1.0. Internal use only.

## 3.1 Card-not-present thresholds
Card-not-present (CARD_ECOM) transactions above RM 1,500, or above 5 times the customer's 30-day average amount, must be held for analyst review before settlement. Three or more card-not-present attempts within one hour on the same card trigger an automatic temporary block and an alert in the queue.

## 3.2 New device and SIM-swap indicators
A transaction from a device not previously linked to the customer is a high-risk indicator. When a new device is combined with a recent SIM change, a password reset, or a transaction between 1am and 5am, treat the alert as priority 1. Do not approve a priority 1 alert on the customer's verbal confirmation alone; use an out-of-band callback to the registered phone number.

## 3.3 High-risk merchant categories
Merchant categories CRYPTO and GAMING are high risk. Foreign-currency transactions in these categories, or any such transaction above RM 1,000, require analyst review. ELECTRONICS and LUXURY merchants are resale-risk categories and require review when combined with a new device or a new account (under 60 days old).

## 4.1 Alert triage SLA
Priority 1 alerts (probability 0.90 or above) must be triaged within 15 minutes. Priority 2 alerts (probability 0.80 to 0.89) must be triaged within 2 hours. Every alert needs a recorded disposition: confirmed fraud, false positive, or escalated to a fraud supervisor.

## 4.2 Customer contact and card block
For confirmed or strongly suspected fraud, block the card immediately, then contact the customer through the registered phone number only. Never contact the customer through a number or link supplied in the transaction or by a caller. Offer a replacement card and log the call in the case record.

## 5.1 Suspicious transaction reporting to the regulator
Confirmed fraud involving more than RM 10,000, any mule-account pattern, or any suspected money laundering must be reported as a suspicious transaction report (STR) to the Financial Intelligence and Enforcement Department of Bank Negara Malaysia. The STR must be filed within 3 working days of the fraud supervisor's decision. Do not tip off the customer that an STR has been filed.

## 6.1 Data handling and PII
Analysts and tools must not copy customer names, NRIC numbers, full card numbers, or phone numbers out of the case system. AI assistants may only receive transaction facts, model scores, reason codes, and retrieved policy text. Customer identifiers must never be included in prompts or in answers.
