"""SEC filing-code glossary — same dict the fine-tune training set uses.

Kept here as a hardcoded copy so the UI can render hover-tooltips without a model call.
For exotic codes not in this list, the agent (Chat tab) is the fallback.
"""

from __future__ import annotations

SEC_FORMS: dict[str, str] = {
    "10-K": "Annual report filed by U.S. public companies under section 13 of the Exchange Act. Contains audited financial statements, MD&A, risk factors, and a description of the business. Filed within 60–90 days of fiscal year-end.",
    "10-Q": "Quarterly report filed by U.S. public companies for the first three fiscal quarters. Contains unaudited financial statements and updates to risk factors and MD&A. Filed within 40–45 days of quarter-end.",
    "8-K": "Current report disclosing material events between periodic filings: acquisitions, executive changes, bankruptcy, earnings releases, departures of directors. Generally filed within 4 business days of the event.",
    "S-1": "Initial registration statement for new securities, used by companies going public via IPO. Discloses business model, financials, risk factors, and use of proceeds.",
    "S-3": "Simplified registration statement for follow-on offerings by companies already reporting under the Exchange Act. Used for shelf registrations and seasoned offerings.",
    "DEF 14A": "Definitive proxy statement sent to shareholders before the annual meeting. Discloses executive compensation, board nominees, and proposals being voted on.",
    "DEFA14A": "Additional proxy soliciting material filed after the DEF 14A. Used for supplementary statements during a proxy contest or to correct errors.",
    "3": "Initial statement of beneficial ownership filed within 10 days of becoming an officer, director, or 10%+ shareholder.",
    "4": "Statement of changes in beneficial ownership. Insiders report any purchase or sale of company stock within 2 business days.",
    "5": "Annual statement of changes in beneficial ownership for transactions not reported on Form 4 during the year.",
    "SC 13D": "Filed by anyone acquiring more than 5% of a company's stock with intent to influence control. Filed within 10 days of crossing the threshold.",
    "SC 13G": "Filed by passive investors (no intent to influence control) holding more than 5%. Less detailed than 13D.",
    "144": "Notice of intent to sell restricted or control securities, filed by affiliates before selling. Limits how much insiders can sell in a 90-day window.",
    "ARS": "Annual Report to Shareholders. The narrative version of the 10-K mailed to shareholders. Required to accompany the proxy statement.",
    "6-K": "Report filed by foreign private issuers to disclose information that is material in their home country. Functional equivalent of an 8-K for non-U.S. companies.",
    "20-F": "Annual report filed by foreign private issuers. Equivalent of the 10-K for non-U.S. companies. Reconciles foreign GAAP / IFRS to U.S. accounting rules where required.",
    "11-K": "Annual report of an employee stock purchase, savings, or similar plan.",
    "SC TO-T": "Tender offer statement filed by a third-party bidder making a public offer to acquire shares of a target.",
    "SC 14D9": "Solicitation/recommendation statement — the target's official response to a tender offer.",
    "N-CSR": "Certified annual or semi-annual shareholder report filed by registered investment companies (mutual funds, ETFs).",
}


def lookup(code: str) -> str | None:
    """Case-insensitive lookup, ignoring whitespace differences."""
    if not code:
        return None
    norm = code.strip().upper().replace("FORM ", "")
    for k, v in SEC_FORMS.items():
        if k.upper() == norm:
            return v
    return None
