from google import genai
import streamlit as st

client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)

MODEL = "gemini-flash-lite-latest"


def generate_insights(row):

    # prompt = row["LLM Prompt"]
    prompt = f"""
        You are a Senior Retail Analytics Consultant at Deloitte.

        Below is an anomaly detected by an AI forecasting system.

        {row["LLM Prompt"]}

        Instructions:

        Write a professional executive report in Markdown.

        Use the following sections:

        ## 📌 Executive Summary
        (2-3 sentences)

        ## 📊 Business Interpretation
        Explain what likely happened.

        ## ⚠️ Business Risks
        Mention inventory, customer satisfaction, revenue or operational risks.

        ## ✅ Recommended Actions
        Provide 3-5 actionable recommendations.

        ## 🎯 Management Takeaway
        Conclude with one concise sentence for senior leadership.

        Keep the response under 300 words.
    """

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    return response.text