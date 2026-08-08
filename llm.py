from google import genai
import streamlit as st


MODEL = "gemini-flash-lite-latest"


@st.cache_resource
def get_gemini_client():
    """Create one Gemini client per app process from Streamlit secrets."""
    return genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )


@st.cache_data(
    ttl="24h",
    max_entries=50,
    show_spinner=False
)
def _generate_cached(llm_prompt: str) -> str:
    """Cache one management brief per anomaly."""

    prompt = f"""
You are a senior retail analytics consultant.

Below is an anomaly detected by an AI forecasting system.

{llm_prompt}

Write a professional executive report in Markdown using exactly these sections:

## Executive summary

Summarise the event and its likely commercial meaning in 2-3 sentences.

## Business interpretation

Explain what likely happened using only the supplied evidence.

## Business risks

Cover relevant inventory, customer, revenue, and operational risks.

## Recommended actions

Provide 3-5 specific, prioritised recommendations.

## Management takeaway

Finish with one concise sentence for senior leadership.

Keep the response under 300 words.

Do not invent unsupported figures.
Clearly distinguish observed facts from hypotheses.
"""


    response = get_gemini_client().models.generate_content(
        model=MODEL,
        contents=prompt,
    )

    return response.text or "No analysis was returned for this event."


def generate_insights(row):
    """
    Build the LLM context directly from the selected anomaly row.

    This avoids depending on an 'LLM Prompt' dataframe column,
    which is not present in dynamically generated anomaly rows.
    """

    def value(column, default="Not available"):
        if column in row.index:
            return row[column]
        return default

    llm_prompt = f"""
ANOMALY EVENT
-------------
Week: {value("Week")}

Anomaly Type: {value("Type")}

Actual Sales: £{value("Actual Sales", value("Actual", "Not available"))}

Forecast Sales: £{value("Forecast Sales", value("Forecast", "Not available"))}

Deviation: {value("Deviation %", "Not available")}%

ROOT-CAUSE CONTEXT
------------------
Top Product: {value("Top Product")}

Top Customer: {value("Top Customer")}

Top Country: {value("Top Country")}

Primary Driver: {value("Root Cause Driver")}

Driver Increase / Contribution:
£{value("Driver Increase", "Not available")}

ADDITIONAL RCA EVIDENCE
-----------------------
Product Contribution: {value("Product Contribution %", "Not available")}%

Customer Contribution: {value("Customer Contribution %", "Not available")}%

Country Contribution: {value("Country Contribution %", "Not available")}%

Product WoW Change: {value("Product WoW %", "Not available")}%

Customer WoW Change: {value("Customer WoW %", "Not available")}%

Country WoW Change: {value("Country WoW %", "Not available")}%

IMPORTANT:
Use only the evidence supplied above.
Do not fabricate product behaviour, customer behaviour,
market conditions, or financial figures.
Where the evidence is insufficient to determine a cause,
state that it is a hypothesis requiring validation.
"""

    return _generate_cached(llm_prompt)
