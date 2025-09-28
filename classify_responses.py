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


def read_csv_answers(input_path: str) -> List[str]:
    ext = os.path.splitext(input_path)[1].lower()
    if ext != ".csv":
        raise ValueError("Input file must be .csv")
    answers: List[str] = []
    with open(input_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            text = (row[0] if len(row) > 0 else "").strip()
            if text:
                answers.append(text)
    return answers


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
            full_output = "".join([getattr(b, "text", "") for b in content_blocks if getattr(b, "text", "")]).strip()
            # Parse category from JSON region within the output (be lenient)
            json_region = full_output
            start = full_output.find("{")
            end = full_output.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_region = full_output[start : end + 1]
            data = json.loads(json_region)
            category = normalize_category(data.get("category", "Other"))
            return category, full_output
        except (json.JSONDecodeError, KeyError):
            if attempt == max_retries - 1:
                return "Other", full_output if 'full_output' in locals() else ""
        except APIStatusError as e:
            if getattr(e, "status_code", None) in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 16)
                continue
            if attempt == max_retries - 1:
                return "Other", full_output if 'full_output' in locals() else ""
        except Exception:
            if attempt == max_retries - 1:
                return "Other", full_output if 'full_output' in locals() else ""
            time.sleep(delay)
            delay = min(delay * 2, 16)
    return "Other", ""


def build_summary(categories: List[str]) -> Dict[str, Any]:
    total = len(categories)
    counts: Dict[str, int] = {label: 0 for label in ALLOWED_CATEGORIES}
    for c in categories:
        counts[c] = counts.get(c, 0) + 1
    percents: Dict[str, float] = {}
    for label in ALLOWED_CATEGORIES:
        if total == 0:
            percents[label] = 0.0
        else:
            percents[label] = round(counts[label] * 100.0 / total, 2)
    return {
        "total_inputs": total,
        "category_counts": counts,
        "category_percents": percents,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Classify free-response CSV answers into one category each and output summary JSON."
    )
    parser.add_argument("--in", dest="input_path", required=True, help="Absolute path to input .csv (single column)")
    parser.add_argument("--out", dest="output_path", required=True, help="Absolute path to output .json")
    parser.add_argument(
        "--model",
        dest="model",
        default="claude-3-5-sonnet-20240620",
        help="Anthropic model name",
    )
    args = parser.parse_args()

    if not args.output_path.lower().endswith(".json"):
        raise ValueError("Output file must be .json")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY in your environment.")

    client = Anthropic(api_key=api_key)
    answers = read_csv_answers(args.input_path)

    records: List[Dict[str, Any]] = []
    categories: List[str] = []
    for text in answers:
        category, full_output = classify_text(client, args.model, text)
        records.append({
            "input": text,
            "output": full_output,
            "category": category,
        })
        categories.append(category)
        time.sleep(0.05)

    summary = build_summary(categories)
    final_output = {
        "summary": summary,
        "data": records,
    }

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()


