import os
import csv
import json
import time
import argparse
from typing import List, Dict, Any, Tuple

from anthropic import Anthropic, APIStatusError

ALLOWED_CATEGORIES = [
    "Cheating Concerns",
    "Positive Learning Use",
    "Negative Experiences",
    "Overreliance",
    "Trust Issues",
    "Policy/School Rules",
    "Mixed Views",
    "Ethical/Privacy Concerns",
    "No Use",
    "Other",
]

SYSTEM_MSG = (
    "You are a strict data labeling assistant. Your task is to classify a single free-text response "
    "about AI in education into exactly ONE category from a fixed label set. Return only valid JSON."
)

USER_PROMPT_TEMPLATE = """Classify the following free-text response into exactly ONE of the categories below.

Categories (use exact strings):
- Cheating Concerns: Mentions of academic dishonesty, misuse
- Positive Learning Use: Says AI helped them understand/study
- Negative Experiences: Confusing, inaccurate, unhelpful
- Overreliance: Worries about becoming lazy or dependent
- Trust Issues: Doesn’t trust responses; always double-checks
- Policy/School Rules: Mentions bans, restrictions, teacher feedback
- Mixed Views: Likes AI but has concerns
- Ethical/Privacy Concerns: Worries about AI’s effect on society/privacy
- No Use: “I don’t use AI” or “never used it”
- Other: Doesn’t fit above or is off-topic

Instructions:
- Choose exactly one category that best fits overall.
- If none clearly fit, use "Other".
- Output JSON only in this schema:
  {{
    "category": "<one of the allowed categories>",
    "reason": "<short rationale>"
  }}

Response:
\"\"\"{text}\"\"\""""


def load_records(input_path: str, input_column: str) -> List[Dict[str, Any]]:
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".csv":
        with open(input_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                return []
            if input_column not in rows[0]:
                raise ValueError(
                    f"Column '{input_column}' not found in CSV headers: {list(rows[0].keys())}"
                )
            return rows
    elif ext == ".json":
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON must be a list of objects.")
            if not data:
                return []
            if input_column not in data[0]:
                raise ValueError(
                    f"Key '{input_column}' not found in JSON objects: {list(data[0].keys())}"
                )
            return data
    else:
        raise ValueError("Input file must be .csv or .json")


def normalize_category(cat: str) -> str:
    if not isinstance(cat, str):
        return "Other"
    cat_norm = cat.strip()
    for allowed in ALLOWED_CATEGORIES:
        if cat_norm.lower() == allowed.lower():
            return allowed
    return "Other"


def classify_text(client: Anthropic, model: str, text: str, max_retries: int = 5) -> Tuple[str, str]:
    user_prompt = USER_PROMPT_TEMPLATE.format(text=text)
    delay = 1.0
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=200,
                temperature=0,
                system=SYSTEM_MSG,
                messages=[{"role": "user", "content": user_prompt}],
            )
            content_blocks = resp.content or []
            combined = "".join([getattr(b, "text", "") for b in content_blocks if getattr(b, "text", "")])
            combined = combined.strip()
            start = combined.find("{")
            end = combined.rfind("}")
            if start != -1 and end != -1 and end > start:
                combined = combined[start : end + 1]
            data = json.loads(combined)
            category = normalize_category(data.get("category", "Other"))
            reason = (data.get("reason") or "").strip()
            return category, reason
        except (json.JSONDecodeError, KeyError):
            if attempt == max_retries - 1:
                return "Other", "Could not parse model output"
        except APIStatusError as e:
            if getattr(e, "status_code", None) in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 16)
                continue
            if attempt == max_retries - 1:
                return "Other", f"API error: {e}"
        except Exception as e:
            if attempt == max_retries - 1:
                return "Other", f"Unexpected error: {e}"
            time.sleep(delay)
            delay = min(delay * 2, 16)
    return "Other", "Unknown error"


def main():
    parser = argparse.ArgumentParser(
        description="Classify AI-in-education free responses into categories with Claude 3.5 Sonnet."
    )
    parser.add_argument("--in", dest="input_path", required=True, help="Absolute path to input .csv or .json")
    parser.add_argument(
        "--input-column",
        dest="input_column",
        default="Response",
        help="Column/key containing the free text",
    )
    parser.add_argument("--out", dest="output_path", required=True, help="Absolute path to output .json or .csv")
    parser.add_argument(
        "--model",
        dest="model",
        default="claude-3-5-sonnet-20240620",
        help="Anthropic model name",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY in your environment.")

    client = Anthropic(api_key=api_key)
    rows = load_records(args.input_path, args.input_column)

    results: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        text = (row.get(args.input_column) or "").strip()
        category, reason = classify_text(client, args.model, text)
        out_row = dict(row)
        out_row["category"] = category
        out_row["reason"] = reason
        results.append(out_row)
        time.sleep(0.1)

    out_ext = os.path.splitext(args.output_path)[1].lower()
    if out_ext == ".json":
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    elif out_ext == ".csv":
        if results:
            fieldnames = list(results[0].keys())
        else:
            fieldnames = [args.input_column, "category", "reason"]
        with open(args.output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    else:
        raise ValueError("Output file must be .json or .csv")


if __name__ == "__main__":
    main()


