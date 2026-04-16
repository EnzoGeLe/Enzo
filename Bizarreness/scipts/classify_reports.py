"""
classify_reports.py
-------------------
Reads ClasseurGPT.csv (semicolon-separated, columns: txt_cleaned;CodeGpt),
sends each non-empty text to an OpenAI chat completion using the 15-dimension
dream/wakefulness classification prompt, parses the structured output, and
writes all scores, NDS, classification, confidence and reasoning to an output CSV.

Usage
-----
    export OPENAI_API_KEY="sk-..."
    python classify_reports.py [--input PATH] [--output PATH] [--model MODEL]
                               [--batch-size N] [--max-workers N] [--resume]

Defaults
--------
    --input   ../ClasseurGPT.csv
    --output  ../ClasseurGPT_classified.csv
    --model   gpt-4o-mini
    --batch-size  50   (rows committed to disk at a time)
    --max-workers 4    (parallel API requests)

Checkpoint / resume
-------------------
If --resume is given (default: True) and the output file already exists, rows
whose CodeGpt is already present are skipped and the file is appended to.

Output columns
--------------
txt_cleaned, CodeGpt,
Bizarreness, Visual, Space, Social, Settings, Movements, Limitations,
Tactile, Auditory, Arousal, Thought, Agentivity, Time, Body, Valence,
Dimensions_evidenced, RAW, DENOM, NDS,
Dream_favoring_signals, Wakefulness_favoring_signals,
Classification, Confidence, Reasoning,
Parse_error
"""

import argparse
import csv
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("openai package not found. Install with: pip install openai", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Classification prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert classifier trained to distinguish between DREAM REPORTS and WAKEFULNESS REPORTS based on their semantic and linguistic content.
A dream report is a verbal account of a subjective conscious experience occurring during sleep, recorded immediately upon awakening. A wakefulness report is a verbal account of thoughts, experiences, or mental content occurring during normal waking life.
Analyze the following text and classify it as either DREAM or WAKEFULNESS. Then provide a confidence score from 0 to 100.
________________________________________
SCORING PRINCIPLE
For each dimension, ask: does this text provide clear evidence about this feature, either confirming its presence or its absence?
• If YES: assign a score from 1 to 9 as instructed below
• If NO evidence in either direction: leave the dimension unscored
This rule applies identically to all reports. A single sentence containing clear evidence of bizarreness should score BIZARRENESS=9. A long report that never mentions spatial detail should score SPACE=1. What matters is whether the text provides genuine evidence, not how many words it contains. Do not assign a default score of 1 simply because a feature is not mentioned — unmentioned features are left unscored.
________________________________________
STEP 1 — SCORE THE DIMENSIONS
DREAM-FAVORING DIMENSIONS:
1. BIZARRENESS Score 1–9: How strange, illogical, or impossible are the events described?
• 1 = Completely realistic, coherent, plausible everyday scenario
• 5 = Partially strange; some unusual elements but generally coherent
• 9 = Extremely bizarre; impossible, incoherent, or surreal events
2. VISUAL Score 1–9: How much does the text explicitly reference visual details, appearances, colors, shapes, or actions requiring sight?
• 1 = No visual references whatsoever
• 5 = Moderate visual content
• 9 = Richly visual throughout
3. SPACE Score 1–9: How much does the text describe surrounding environments, physical spaces, rooms, landscapes, or spatial layout?
• 1 = No spatial description
• 5 = Some mention of location or environment
• 9 = Rich, detailed spatial descriptions throughout
4. SOCIAL Score 1–9: How much does the text describe interactions between people?
• 1 = No social interaction; narrator alone
• 5 = Some social interaction present
• 9 = Predominantly social; events driven by interpersonal dynamics
5. SETTINGS Score 1–9: How many abrupt or unexplained changes of scene, location, or environment occur?
• 1 = Single stable setting throughout
• 5 = A few transitions between settings
• 9 = Frequent, abrupt, or inexplicable changes of setting
6. MOVEMENTS Score 1–9: How much physical movement by the narrator or other characters is described?
• 1 = No physical movement
• 5 = Some movement described
• 9 = Constant or extreme physical movement throughout
7. LIMITATIONS Score 1–9: How much are characters' actions constrained — physically or socially/morally?
• 1 = No limitations; characters act freely
• 5 = Some constraints present
• 9 = Severe or pervasive constraints on action
8. TACTILE Score 1–9: How much does the text reference touch, physical contact, or textures?
• 1 = No tactile content
• 5 = Some tactile references
• 9 = Rich tactile description throughout
9. AUDITORY Score 1–9: How much does the text reference sounds, voices, music, or hearing?
• 1 = No auditory content
• 5 = Some auditory references
• 9 = Rich auditory detail throughout
10. AROUSAL Score 1–9: How emotionally intense is the narration, regardless of whether positive or negative?
• 1 = Completely flat, emotionally inert
• 5 = Moderately intense
• 9 = Extremely intense emotional content
________________________________________
WAKEFULNESS-FAVORING DIMENSIONS:
11. THOUGHT Score 1–9: How much does the narrator describe abstract thoughts, reflections, reasoning, planning, opinions, or metacognitive processes?
• 1 = No abstract thought; purely event-based narration
• 5 = Some reflective content mixed with events
• 9 = Predominantly abstract thoughts, reasoning, or inner commentary
12. AGENTIVITY Score 1–9: How much is the narrator the active agent of events, in control of their actions, making deliberate choices?
• 1 = Completely passive; narrator is acted upon or observes events
• 5 = Balanced; sometimes active, sometimes passive
• 9 = Fully active agent; narrator drives all described actions
13. TIME Score 1–9: How much does the text reference chronological order, duration, timestamps, schedules, or sequential time?
• 1 = No temporal references
• 5 = Some time references
• 9 = Strong temporal awareness and sequencing throughout
14. BODY Score 1–9: How much does the text reference bodily needs, functions, or instincts?
• 1 = No body references
• 5 = Some body awareness
• 9 = Prominent bodily themes throughout
15. VALENCE Score 1–9: What is the overall emotional tone?
• 1 = Extremely negative (fear, despair, anger, sadness)
• 5 = Neutral
• 9 = Extremely positive (joy, happiness, excitement)
________________________________________
STEP 2 — COMPUTE THE NORMALIZED DISCRIMINANT SCORE (NDS)
Using only the dimensions you scored, compute:
RAW = sum of each (score × weight) using the table below
DENOM = sum of the absolute weight values for scored dimensions only
NDS = RAW / DENOM
Dimension	Weight
Bizarreness	+2.85
Visual	+1.52
Space	+1.40
Social	+1.40
Settings	+1.29
Movements	+0.83
Limitations	+0.82
Tactile	+0.49
Auditory	+0.42
Arousal	+0.21
Thought	−1.77
Agentivity	−1.58
Time	−1.20
Body	−0.73
Valence	−0.49
The NDS is normalized: dividing by DENOM ensures that a report where only one dimension was evidenced is directly comparable to a report where all fifteen were evidenced.
Interpret the NDS:
NDS < 0 = WAKEFULNESS ; NDS > 0 = DREAM
OUTPUT FORMAT — respond with EXACTLY this block, filling in every field:
DIMENSION SCORES: Bizarreness: [1–9 or —] Visual: [1–9 or —] Space: [1–9 or —] Social: [1–9 or —] Settings: [1–9 or —] Movements: [1–9 or —] Limitations: [1–9 or —] Tactile: [1–9 or —] Auditory: [1–9 or —] Arousal: [1–9 or —] Thought: [1–9 or —] Agentivity: [1–9 or —] Time: [1–9 or —] Body: [1–9 or —] Valence: [1–9 or —] Dimensions evidenced: [N of 15]
RAW: [weighted sum] DENOM: [sum of absolute weights of scored dimensions] NDS: [RAW / DENOM]
LEXICAL SIGNALS DETECTED: Dream-favoring: [list] Wakefulness-favoring: [list]
CLASSIFICATION: [DREAM / WAKEFULNESS / UNCERTAIN] CONFIDENCE: [0–100] REASONING: [2–4 sentences]
________________________________________
On language: All dimensions and structural rules are language-agnostic. Reports in any language should be scored identically."""

USER_TEMPLATE = "TEXT:\n{text}"

# ---------------------------------------------------------------------------
# Weights table
# ---------------------------------------------------------------------------

WEIGHTS = {
    "Bizarreness": 2.85,
    "Visual": 1.52,
    "Space": 1.40,
    "Social": 1.40,
    "Settings": 1.29,
    "Movements": 0.83,
    "Limitations": 0.82,
    "Tactile": 0.49,
    "Auditory": 0.42,
    "Arousal": 0.21,
    "Thought": -1.77,
    "Agentivity": -1.58,
    "Time": -1.20,
    "Body": -0.73,
    "Valence": -0.49,
}

DIMENSION_NAMES = list(WEIGHTS.keys())

OUTPUT_COLUMNS = (
    ["txt_cleaned", "CodeGpt"]
    + DIMENSION_NAMES
    + [
        "Dimensions_evidenced",
        "RAW",
        "DENOM",
        "NDS",
        "Dream_favoring_signals",
        "Wakefulness_favoring_signals",
        "Classification",
        "Confidence",
        "Reasoning",
        "Parse_error",
    ]
)

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_SCORE_RE = re.compile(r"(\w+):\s*([1-9]|—|-)")
_DIMENSIONS_EVIDENCED_RE = re.compile(r"Dimensions evidenced:\s*(\d+)", re.IGNORECASE)
_RAW_RE = re.compile(r"\bRAW:\s*([-\d.]+)", re.IGNORECASE)
_DENOM_RE = re.compile(r"\bDENOM:\s*([-\d.]+)", re.IGNORECASE)
_NDS_RE = re.compile(r"\bNDS:\s*([-\d.]+)", re.IGNORECASE)
_DREAM_FAV_RE = re.compile(r"Dream-favoring:\s*(.+?)(?=\s*Wakefulness-favoring:|$)", re.IGNORECASE)
_WAKE_FAV_RE = re.compile(r"Wakefulness-favoring:\s*(.+?)(?=\s*CLASSIFICATION:|$)", re.IGNORECASE)
_CLASS_RE = re.compile(r"CLASSIFICATION:\s*(DREAM|WAKEFULNESS|UNCERTAIN)", re.IGNORECASE)
_CONF_RE = re.compile(r"CONFIDENCE:\s*(\d+)", re.IGNORECASE)
_REASON_RE = re.compile(r"REASONING:\s*(.+)", re.IGNORECASE | re.DOTALL)

SCORE_ORDER = {name.upper(): name for name in DIMENSION_NAMES}


def _maybe_float(m):
    return float(m.group(1)) if m else None


def parse_response(text: str) -> dict:
    """Parse the LLM output into a flat dict of result fields."""
    result = {name: None for name in DIMENSION_NAMES}
    result["Dimensions_evidenced"] = None
    result["RAW"] = None
    result["DENOM"] = None
    result["NDS"] = None
    result["Dream_favoring_signals"] = None
    result["Wakefulness_favoring_signals"] = None
    result["Classification"] = None
    result["Confidence"] = None
    result["Reasoning"] = None
    result["Parse_error"] = ""

    # --- dimension scores ---
    # Find the DIMENSION SCORES block
    dim_block_match = re.search(
        r"DIMENSION SCORES:(.*?)(?:RAW:|$)", text, re.IGNORECASE | re.DOTALL
    )
    if dim_block_match:
        block = dim_block_match.group(1)
        for m in _SCORE_RE.finditer(block):
            key_raw = m.group(1).strip().upper()
            if key_raw in SCORE_ORDER:
                val = m.group(2).strip()
                if val not in ("—", "-"):
                    result[SCORE_ORDER[key_raw]] = int(val)

        de_m = _DIMENSIONS_EVIDENCED_RE.search(block)
        if de_m:
            result["Dimensions_evidenced"] = int(de_m.group(1))

    # --- RAW / DENOM / NDS ---
    result["RAW"] = _maybe_float(_RAW_RE.search(text))
    result["DENOM"] = _maybe_float(_DENOM_RE.search(text))
    result["NDS"] = _maybe_float(_NDS_RE.search(text))

    # --- lexical signals ---
    df_m = _DREAM_FAV_RE.search(text)
    if df_m:
        result["Dream_favoring_signals"] = df_m.group(1).strip()
    wf_m = _WAKE_FAV_RE.search(text)
    if wf_m:
        result["Wakefulness_favoring_signals"] = wf_m.group(1).strip()

    # --- classification / confidence / reasoning ---
    cl_m = _CLASS_RE.search(text)
    result["Classification"] = cl_m.group(1).upper() if cl_m else None

    cf_m = _CONF_RE.search(text)
    result["Confidence"] = int(cf_m.group(1)) if cf_m else None

    # Reasoning: everything after REASONING: until the end
    re_m = _REASON_RE.search(text)
    if re_m:
        reasoning = re_m.group(1).strip()
        # Remove trailing separator line if present
        reasoning = re.sub(r"\s*_{10,}\s*$", "", reasoning).strip()
        result["Reasoning"] = reasoning

    # flag missing critical fields
    missing = [
        f for f in ("Classification", "Confidence", "NDS")
        if result[f] is None
    ]
    if missing:
        result["Parse_error"] = f"Missing: {', '.join(missing)}"

    return result


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------


def _empty_result(parse_error: str = "") -> dict:
    """Return a result dict with all fields set to None and an optional parse_error."""
    return {
        **{name: None for name in DIMENSION_NAMES},
        "Dimensions_evidenced": None,
        "RAW": None,
        "DENOM": None,
        "NDS": None,
        "Dream_favoring_signals": None,
        "Wakefulness_favoring_signals": None,
        "Classification": None,
        "Confidence": None,
        "Reasoning": None,
        "Parse_error": parse_error,
    }


def classify_text(client: OpenAI, text: str, model: str, max_tokens: int = 2048,
                  max_retries: int = 5) -> dict:
    """Send text to the LLM and return parsed result dict."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(text=text)},
    ]
    delay = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=max_tokens,
            )
            raw_output = response.choices[0].message.content
            result = parse_response(raw_output)
            return result
        except Exception as exc:
            if attempt == max_retries:
                return _empty_result(f"API error after {max_retries} attempts: {exc}")
            time.sleep(delay)
            delay *= 2


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def load_already_done(output_path: Path) -> set:
    """Return set of CodeGpt values already present in the output file."""
    done = set()
    if output_path.exists():
        with output_path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("CodeGpt"):
                    done.add(row["CodeGpt"])
    return done


def write_rows(writer, rows):
    for row in rows:
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Classify dream/wakefulness reports using an LLM.")
    parser.add_argument(
        "--input",
        default=str(Path(__file__).parent.parent / "ClasseurGPT.csv"),
        help="Path to input CSV (default: ../ClasseurGPT.csv relative to this script)",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent.parent / "ClasseurGPT_classified.csv"),
        help="Path to output CSV",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument(
        "--max-tokens", type=int, default=2048,
        help="Maximum tokens for LLM response (default: 2048)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=50, help="Rows to accumulate before flushing to disk"
    )
    parser.add_argument(
        "--max-workers", type=int, default=4, help="Parallel API request threads"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Overwrite output file instead of resuming",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Read input — use 'replace' for decode errors but warn so data quality issues are visible
    with input_path.open(encoding="utf-8-sig", errors="replace", newline="") as f:
        content = f.read()
    if "\ufffd" in content:
        print(
            "Warning: replacement characters (U+FFFD) found in input — some bytes could not be "
            f"decoded as UTF-8 in {input_path.name}. Verify encoding if results look garbled.",
            file=sys.stderr,
        )
    import io
    reader = csv.DictReader(io.StringIO(content), delimiter=";")
    all_rows = [r for r in reader if r.get("txt_cleaned", "").strip()]

    print(f"Loaded {len(all_rows)} non-empty rows from {input_path.name}")

    # Determine already-done rows
    resume = not args.no_resume
    done_codes = load_already_done(output_path) if resume else set()
    pending = [r for r in all_rows if r["CodeGpt"] not in done_codes]
    print(f"Already done: {len(done_codes)}  |  Pending: {len(pending)}")

    if not pending:
        print("Nothing to do.")
        return

    # Open output file (append or write)
    mode = "a" if resume and output_path.exists() else "w"
    out_f = output_path.open(mode, encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(out_f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
    if mode == "w":
        writer.writeheader()

    batch = []
    completed = 0
    errors = 0

    def process_row(row):
        result = classify_text(client, row["txt_cleaned"], args.model, args.max_tokens)
        out_row = {"txt_cleaned": row["txt_cleaned"], "CodeGpt": row["CodeGpt"], **result}
        return out_row

    print(f"Classifying {len(pending)} texts with {args.max_workers} workers…")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_row, row): row for row in pending}
        for future in as_completed(futures):
            out_row = future.result()
            batch.append(out_row)
            completed += 1
            if out_row.get("Parse_error"):
                errors += 1

            if len(batch) >= args.batch_size:
                write_rows(writer, batch)
                out_f.flush()
                batch.clear()
                print(f"  {completed}/{len(pending)} done ({errors} errors)…")

    # Flush remaining
    if batch:
        write_rows(writer, batch)
        out_f.flush()

    out_f.close()

    print(f"\nDone. {completed} rows classified, {errors} errors.")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
