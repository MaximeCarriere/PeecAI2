import os
from dotenv import load_dotenv
from flask import Flask, render_template, request
import openai
import pandas as pd
from io import StringIO

# Load environment variables from .env
load_dotenv()

# Configure OpenAI key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set your OPENAI_API_KEY in the .env file or environment variables.")
openai.api_key = OPENAI_API_KEY

# Function to fetch and parse product models via ChatGPT

def fetch_product_models(product: str,
                         functionality_list: list,
                         country: str,
                         model: str = "gpt-4",
                         temperature: float = 0.2) -> pd.DataFrame:
    """
    Ask ChatGPT for real, currently available product models and parse into a DataFrame.
    """
    # Build the prompt
    prompt = (
        f"Please list **real**, currently available {product} models in {country} "
        f"that are known for being {', '.join(functionality_list)}.\n"
        "For each, give:\n"
        "1. Model number/name\n"
        "2. Brand\n"
        "3. A short 'Specificity' (key features or capacity)\n"
        "4. Retail price in local currency\n"
        "5. A link to an official manufacturer or major retailer page\n\n"
        "Format **exactly** as a Markdown table with columns:\n"
        "| Model | Brand | Specificity | Price | Link |\n"
        "| --- | --- | --- | --- | --- |\n"
        "…and one row per product."
    )

    # Call the ChatGPT API
    response = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    content = response.choices[0].message.content

    # Extract and clean the Markdown table lines
    lines = [ln.strip().strip("|") for ln in content.splitlines() if ln.strip().startswith("|")]
    if len(lines) >= 2:
        del lines[1]  # remove the separator row
    csv_str = "\n".join(lines)

    # Parse into DataFrame
    df = pd.read_csv(StringIO(csv_str), sep="|")
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()

    # Convert Markdown links to HTML anchors
    df['Link'] = df['Link'].str.replace(
        r'\[([^\]]+)\]\(([^)]+)\)',
        r'<a href="\2" target="_blank" class="text-blue-300 hover:underline">\1</a>',
        regex=True
    )
    return df

# Initialize Flask
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    df_html = None
    error = None
    if request.method == "POST":
        product = request.form.get("product")
        features = [f.strip() for f in request.form.get("features", "").split(",") if f.strip()]
        country = request.form.get("country")
        try:
            df = fetch_product_models(product, features, country)
            df_html = df.to_html(
                classes="table-auto text-sm text-left overflow-x-auto w-full",
                index=False,
                escape=False
            )
        except Exception as e:
            error = str(e)
    return render_template("index.html", table=df_html, error=error)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
