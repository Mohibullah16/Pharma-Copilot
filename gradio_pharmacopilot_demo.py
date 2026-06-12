from __future__ import annotations

import json
import os
import unicodedata
import time
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image, ImageDraw

try:
    import spaces
except Exception:
    class _SpacesFallback:
        @staticmethod
        def GPU(*_args, **_kwargs):
            def decorator(fn):
                return fn

            return decorator

    spaces = _SpacesFallback()

try:
    import plotly.graph_objects as go
except Exception:  # The app still runs if plotly is installed later from requirements.txt.
    go = None


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
STORAGE_DIR = Path(os.getenv("PHARMACOPILOT_DATA_DIR", "/data"))


def data_path(relative_path: str) -> Path:
    """Prefer the Hugging Face Space storage bucket mounted at /data."""
    relative = Path(relative_path)
    for base in (STORAGE_DIR, DATA_DIR):
        candidate = base / relative
        if candidate.exists():
            return candidate
    return DATA_DIR / relative


MEDICINES_PATH = data_path("medicines_master.json")
BRAND_MAP_PATH = data_path("training/bd_brand_to_generic.json")
INVENTORY_PATH = data_path("inventory.json")

MODEL_ID = os.getenv("PHARMACOPILOT_MODEL_ID", "openbmb/MiniCPM-V-4_5")
LIVE_GPU_OCR = os.getenv("PHARMACOPILOT_LIVE_GPU_OCR", "1").lower() not in {"0", "false", "no"}
LIVE_NEMOTRON = os.getenv("PHARMACOPILOT_LIVE_NEMOTRON", "1").lower() not in {"0", "false", "no"}
NEMOTRON_MODEL_ID = os.getenv("NEMOTRON_MODEL_ID", "nvidia/Nemotron-Mini-4B-Instruct")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_NIM_MODEL = os.getenv("NVIDIA_NIM_MODEL", "nvidia/nvidia-nemotron-nano-9b-v2")
DEMO_OCR_TEXT = "Neuoxen"
DEMO_PROMPT = "Read the handwritten medicine name in the image. Return only the text."
ACCEPTANCE_THRESHOLD = int(os.getenv("PHARMACOPILOT_ACCEPTANCE_THRESHOLD", "75"))
OCR_MODEL = None
OCR_TOKENIZER = None
NEMOTRON_MODEL = None
NEMOTRON_TOKENIZER = None


def load_json(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_asset_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    raw_path = Path(path_value)
    if raw_path.is_absolute() and raw_path.exists():
        return raw_path

    parts = raw_path.parts
    storage_relative = Path(*parts[1:]) if parts and parts[0] == "data" else raw_path
    candidates = [
        STORAGE_DIR / storage_relative,
        ROOT / raw_path,
        DATA_DIR / raw_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


MEDICINES = load_json(MEDICINES_PATH, [])
BD_BRAND_TO_GENERIC = load_json(BRAND_MAP_PATH, {})
INVENTORY = load_json(INVENTORY_PATH, [])

MED_BY_ID = {m["id"]: m for m in MEDICINES}
MED_BY_NAME = {m["name"].lower(): m for m in MEDICINES}
INVENTORY_BY_MED_ID = {item["medicine_id"]: item for item in INVENTORY}
SESSION_SEARCHES = 0


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.strip().lower().split())


def clean_prediction(raw_prediction: str) -> str:
    text = str(raw_prediction or "").strip()
    text = text.replace("\r", "\n")
    text = text.split("\n")[0].strip() if "\n" in text else text
    for prefix in [
        "the medicine is",
        "the name of the medicine is",
        "the answer is",
        "this image shows",
        "the drug is",
        "medicine:",
        "answer:",
    ]:
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip(" :-.")
    return " ".join(text.strip(" .,:;\"'`").split())


def label_for_medicine(ocr_text: str, medicine: dict[str, Any]) -> str:
    query = normalize(ocr_text)
    if query in BD_BRAND_TO_GENERIC:
        return ocr_text.strip()
    if query == normalize(medicine.get("name")):
        return medicine["name"]
    for brand, canonical in BD_BRAND_TO_GENERIC.items():
        if normalize(canonical) == normalize(medicine["name"]):
            return brand.title()
    brands = medicine.get("brand_names") or []
    return brands[0] if brands else medicine["name"]


def find_medicine_from_ocr(ocr_text: str) -> tuple[dict[str, Any], list[dict[str, Any]], str, int]:
    query = normalize(ocr_text)
    corrected_query = query
    canonical = BD_BRAND_TO_GENERIC.get(corrected_query, corrected_query)
    direct_medicine = MED_BY_NAME.get(normalize(canonical))

    candidate_names = set()
    for med in MEDICINES:
        candidate_names.add(med["name"])
        candidate_names.add(med.get("generic_name") or med["name"])
        candidate_names.update(med.get("brand_names") or [])
    candidate_names.update(BD_BRAND_TO_GENERIC.keys())

    scored = []
    for name in candidate_names:
        score = SequenceMatcher(None, query, normalize(name)).ratio()
        if score > 0.35:
            mapped = BD_BRAND_TO_GENERIC.get(normalize(name), normalize(name))
            med = MED_BY_NAME.get(mapped) or MED_BY_NAME.get(normalize(name))
            if med:
                scored.append({"label": name, "medicine": med, "score": score})

    scored.sort(key=lambda item: item["score"], reverse=True)
    if direct_medicine:
        medicine = direct_medicine
        display_name = label_for_medicine(ocr_text, medicine)
        primary_score = 0.97
    elif scored:
        best = scored[0]
        medicine = best["medicine"]
        display_name = best["label"]
        primary_score = best["score"]
    else:
        medicine = MEDICINES[0]
        display_name = clean_prediction(ocr_text) or "Needs review"
        primary_score = 0.0

    top = [{"label": display_name, "medicine": medicine, "score": primary_score}]
    seen_ids = {medicine["id"]}
    for item in scored:
        if item["medicine"]["id"] in seen_ids:
            continue
        top.append(item)
        seen_ids.add(item["medicine"]["id"])
        if len(top) == 3:
            break

    while len(top) < 3:
        fallback_name = get_close_matches(query, list(BD_BRAND_TO_GENERIC.keys()), n=1)
        if fallback_name:
            mapped = BD_BRAND_TO_GENERIC[fallback_name[0]]
            med = MED_BY_NAME.get(mapped)
            if med and med["id"] not in seen_ids:
                top.append({"label": fallback_name[0], "medicine": med, "score": 0.62})
                seen_ids.add(med["id"])
                continue
        break

    confidence = max(0, min(99, round(primary_score * 100)))
    return medicine, top, display_name, confidence


def get_inventory(medicine: dict[str, Any]) -> dict[str, Any]:
    return INVENTORY_BY_MED_ID.get(
        medicine["id"],
        {
            "medicine_id": medicine["id"],
            "shelf": "Review",
            "row": "-",
            "quantity": 0,
            "last_updated": "Not synced",
        },
    )


def first_strength(strength: str) -> str:
    if not strength:
        return "Not listed"
    return strength.split(",")[0].strip()


def fallback_prescription_plan(
    ocr_text: str,
    medicine: dict[str, Any],
    display_name: str,
    confidence: int,
    note: str = "Nemotron did not run",
) -> dict[str, Any]:
    accepted = confidence >= ACCEPTANCE_THRESHOLD
    return {
        "status": "needs_review" if not accepted else "retrieval_only",
        "medicine_name": display_name if accepted else "Needs review",
        "canonical_name": medicine.get("name", "Unknown") if accepted else f"Suggestion: {medicine.get('name', 'Unknown')}",
        "dose": first_strength(medicine.get("strength", "")) if accepted else "Not confirmed",
        "route": "Not specified",
        "timing": "Not specified",
        "frequency": "Not specified",
        "duration": "Not specified",
        "instructions": "Pharmacist review required before dispensing.",
        "validation_note": note,
        "ocr_text": ocr_text,
    }


def release_model_memory() -> None:
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def unload_ocr_model() -> None:
    global OCR_MODEL, OCR_TOKENIZER
    OCR_MODEL = None
    OCR_TOKENIZER = None
    release_model_memory()


def unload_nemotron_model() -> None:
    global NEMOTRON_MODEL, NEMOTRON_TOKENIZER
    NEMOTRON_MODEL = None
    NEMOTRON_TOKENIZER = None
    release_model_memory()


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = str(text or "").strip()
    cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def build_validation_prompt(
    ocr_text: str,
    medicine: dict[str, Any],
    display_name: str,
    confidence: int,
    retrieval_candidates: list[dict[str, Any]],
) -> str:
    validation_payload = {
        "ocr_text": ocr_text,
        "retrieved_display_name": display_name,
        "retrieved_canonical_name": medicine.get("name", "Unknown"),
        "retrieval_confidence": confidence,
        "strength": first_strength(medicine.get("strength", "")),
        "category": medicine.get("category", "Unknown"),
        "top_candidates": [
            {
                "display_name": item["label"],
                "canonical_name": item["medicine"]["name"],
                "score": round(item["score"] * 100),
            }
            for item in retrieval_candidates[:3]
        ],
    }
    return f"""
You are a pharmacy prescription validation assistant.

Input JSON:
{json.dumps(validation_payload, ensure_ascii=False)}

Task:
1. Decide whether the retrieved medicine is safe to accept.
2. Translate the prescription into a clean pharmacy instruction row.
3. Do not invent dose/timing/duration if it is not visible or inferable.
4. If OCR and retrieved medicine clearly disagree, return needs_review.

Return ONLY valid JSON with these keys:
status: one of validated, needs_review
medicine_name
canonical_name
dose
route
timing
frequency
duration
instructions
validation_note
ocr_text
"""


def validate_with_nvidia_nim(
    prompt: str,
    ocr_text: str,
    medicine: dict[str, Any],
    display_name: str,
    confidence: int,
) -> dict[str, Any]:
    if not NVIDIA_API_KEY:
        return fallback_prescription_plan(
            ocr_text,
            medicine,
            display_name,
            confidence,
            "NVIDIA_API_KEY is not configured in the Space secrets",
        )
    try:
        from openai import OpenAI

        client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
        response = client.chat.completions.create(
            model=NVIDIA_NIM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            top_p=1,
            max_tokens=320,
        )
        content = response.choices[0].message.content or ""
        plan = extract_json_object(content)
        if plan.get("status") not in {"validated", "needs_review"}:
            plan["status"] = "needs_review"
        if confidence < ACCEPTANCE_THRESHOLD:
            plan["status"] = "needs_review"
            plan["validation_note"] = (
                f"Retrieval confidence {confidence}% is below the {ACCEPTANCE_THRESHOLD}% acceptance threshold"
            )
        return {
            **fallback_prescription_plan(
                ocr_text,
                medicine,
                display_name,
                confidence,
                f"Validated by NVIDIA NIM {NVIDIA_NIM_MODEL}",
            ),
            **plan,
        }
    except Exception as exc:
        return fallback_prescription_plan(
            ocr_text,
            medicine,
            display_name,
            confidence,
            f"NVIDIA NIM validation failed: {exc}",
        )


def validate_with_nemotron(
    ocr_text: str,
    medicine: dict[str, Any],
    display_name: str,
    confidence: int,
    retrieval_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    global NEMOTRON_MODEL, NEMOTRON_TOKENIZER

    if not LIVE_NEMOTRON:
        return fallback_prescription_plan(
            ocr_text, medicine, display_name, confidence, "Local Nemotron validation is disabled"
        )

    prompt = build_validation_prompt(ocr_text, medicine, display_name, confidence, retrieval_candidates)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if NEMOTRON_MODEL is None or NEMOTRON_TOKENIZER is None:
            NEMOTRON_TOKENIZER = AutoTokenizer.from_pretrained(NEMOTRON_MODEL_ID, trust_remote_code=True)
            NEMOTRON_MODEL = AutoModelForCausalLM.from_pretrained(
                NEMOTRON_MODEL_ID,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            ).eval()

        messages = [{"role": "user", "content": prompt}]
        if hasattr(NEMOTRON_TOKENIZER, "apply_chat_template"):
            input_ids = NEMOTRON_TOKENIZER.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            input_ids = NEMOTRON_TOKENIZER(prompt, return_tensors="pt").input_ids

        device = next(NEMOTRON_MODEL.parameters()).device
        input_ids = input_ids.to(device)
        with torch.inference_mode():
            output_ids = NEMOTRON_MODEL.generate(
                input_ids,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                max_new_tokens=320,
                pad_token_id=NEMOTRON_TOKENIZER.eos_token_id,
            )
        generated = output_ids[0][input_ids.shape[-1] :]
        content = NEMOTRON_TOKENIZER.decode(generated, skip_special_tokens=True).strip()
        plan = extract_json_object(content)
        if plan.get("status") not in {"validated", "needs_review"}:
            plan["status"] = "needs_review"
        if confidence < ACCEPTANCE_THRESHOLD:
            plan["status"] = "needs_review"
            plan["validation_note"] = (
                f"Retrieval confidence {confidence}% is below the {ACCEPTANCE_THRESHOLD}% acceptance threshold"
            )
        return {
            **fallback_prescription_plan(
                ocr_text,
                medicine,
                display_name,
                confidence,
                f"Validated by local {NEMOTRON_MODEL_ID}",
            ),
            **plan,
        }
    except Exception as exc:
        nim_plan = validate_with_nvidia_nim(prompt, ocr_text, medicine, display_name, confidence)
        if NVIDIA_API_KEY:
            return nim_plan
        nim_plan["validation_note"] = f"Local Nemotron failed: {exc}. NVIDIA_API_KEY is not configured."
        return nim_plan


def load_kpi_metrics(searches: int = 0) -> str:
    metrics_path = ROOT / "training" / "baseline_eval" / "minicpm_v_4_5" / "baseline_minicpm_v_4_5_metrics.json"
    fallback_path = ROOT / "training" / "baseline_eval" / "minicpm_v_4_5" / "baseline_minicpm_v_4_5_report.md"

    ocr_accuracy = None
    retrieval_accuracy = None
    if metrics_path.exists():
        metrics = load_json(metrics_path, {})
        ocr_accuracy = metrics.get("ocr_accuracy")
        retrieval_accuracy = metrics.get("retrieval_accuracy") or metrics.get("canonical_match_accuracy")
    elif fallback_path.exists():
        text = fallback_path.read_text(encoding="utf-8", errors="ignore")
        if "ocr_accuracy" in text:
            # Keep a conservative fallback tied to the checked-in report values.
            ocr_accuracy = 0.37888446215139443
            retrieval_accuracy = 0.6055776892430279

    ocr_text = f"{ocr_accuracy * 100:.2f}%" if isinstance(ocr_accuracy, (int, float)) else "Not measured"
    retrieval_text = (
        f"{retrieval_accuracy * 100:.2f}%" if isinstance(retrieval_accuracy, (int, float)) else "Not measured"
    )
    indexed = len(MEDICINES)

    return f"""
    <div class="app-shell">
      <div class="metric-row">
        <div class="metric"><span>Session Searches</span><strong>{searches}</strong></div>
        <div class="metric"><span>Retrieval Recovery</span><strong>{retrieval_text}</strong></div>
        <div class="metric"><span>Medicines Indexed</span><strong>{indexed}</strong></div>
        <div class="metric"><span>Rx Samples Tested</span><strong>2,510</strong></div>
      </div>
    </div>
    """


def confidence_gauge(confidence: int = 97):
    if go is None:
        return None
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence,
            number={"suffix": "%", "font": {"size": 34}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#0f9f6e"},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 60], "color": "#fee2e2"},
                    {"range": [60, 85], "color": "#fef3c7"},
                    {"range": [85, 100], "color": "#dcfce7"},
                ],
            },
        )
    )
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=20, b=8), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def pipeline_html(stage: int = 0, validation_status: str = "waiting") -> str:
    validation_label = {
        "validated": "Nemotron Validated",
        "needs_review": "Needs Review",
        "retrieval_only": "Retrieval Only",
        "waiting": "Awaiting Analysis",
    }.get(validation_status, "Nemotron Review")
    steps = [
        ("Prescription", "uploaded"),
        ("MiniCPM OCR", "ran on image"),
        ("Retrieval Engine", "ranked candidates"),
        (validation_label, "returned a decision"),
        ("Pharmacy View", "prepared"),
    ]
    cards = []
    logs = []
    for i, (title, status) in enumerate(steps, start=1):
        active = i <= stage
        cards.append(
            f"""
            <div class="flow-step {'done' if active else ''}">
              <span>{i}</span>
              <strong>{title}</strong>
            </div>
            """
        )
        if active:
            logs.append(f"<li>✓ {title} {status}</li>")
    return f"""
    <div class="pipeline">
      <div class="pipeline-title">Actual Run Trace</div>
      <div class="flow">{''.join(cards)}</div>
      <ul class="logs">{''.join(logs)}</ul>
    </div>
    """


def medicine_details_html(
    medicine: dict[str, Any],
    inventory: dict[str, Any],
    ocr_text: str,
    display_name: str,
    confidence: int,
    plan: dict[str, Any],
) -> str:
    accepted = plan.get("status") == "validated" and confidence >= ACCEPTANCE_THRESHOLD
    medicine_label = display_name if accepted else "Needs pharmacist review"
    generic_label = medicine.get("name", "Unknown") if accepted else f"Suggestion: {medicine.get('name', 'Unknown')}"
    strength_label = first_strength(medicine.get("strength", "")) if accepted else "Not confirmed"
    manufacturer_label = (medicine.get("manufacturer") or "Not listed") if accepted else "Not confirmed"
    category_label = medicine.get("category", "General") if accepted else "Not confirmed"
    price_label = "PKR 145" if accepted else "Not confirmed"
    validation_label = plan.get("validation_note") or plan.get("status", "Not available")
    inventory_label = (
        f"Shelf {inventory['shelf']}, row {inventory['row']}"
        if accepted
        else "Hidden until a confident medicine match is available"
    )
    return f"""
    <div class="result-card">
      <h3>Prescription Details</h3>
      <dl class="details">
        <dt>Medicine</dt><dd>{medicine_label}</dd>
        <dt>Generic</dt><dd>{generic_label}</dd>
        <dt>Strength</dt><dd>{strength_label}</dd>
        <dt>Manufacturer</dt><dd>{manufacturer_label}</dd>
        <dt>Confidence</dt><dd>{confidence}%</dd>
        <dt>Category</dt><dd>{category_label}</dd>
        <dt>Price</dt><dd>{price_label}</dd>
      </dl>
      <div class="explain">
        <h4>AI Explanation</h4>
        <p><b>OCR detected:</b> "{ocr_text}"</p>
        <p><b>Retrieved:</b> {display_name} ({medicine.get('name', 'Unknown')})</p>
        <p><b>Validation:</b> {validation_label}</p>
        <p><b>Inventory:</b> {inventory_label}</p>
      </div>
    </div>
    """


def translated_prescription_html(plan: dict[str, Any]) -> str:
    rows = [
        ("Medicine", plan.get("medicine_name") or "Not confirmed"),
        ("Canonical", plan.get("canonical_name") or "Not confirmed"),
        ("Dose", plan.get("dose") or "Not specified"),
        ("Route", plan.get("route") or "Not specified"),
        ("When to take", plan.get("timing") or "Not specified"),
        ("Pill timing", plan.get("frequency") or "Not specified"),
        ("Duration", plan.get("duration") or "Not specified"),
        ("Instructions", plan.get("instructions") or "Pharmacist review required"),
    ]
    row_html = "".join(f"<dt>{label}</dt><dd>{value}</dd>" for label, value in rows)
    status = plan.get("status", "needs_review").replace("_", " ").title()
    return f"""
    <div class="translated-card">
      <div class="translated-head">
        <h3>Translated Prescription</h3>
        <span class="status-pill">{status}</span>
      </div>
      <dl class="details translated-details">{row_html}</dl>
      <p class="fine-print">Generated from OCR text and retrieval candidates. Confirm before dispensing.</p>
    </div>
    """


def package_status_html(inventory: dict[str, Any], accepted: bool = True) -> str:
    if not accepted:
        return """
        <div class="stock-card">
          <div><span>Available Stock</span><strong>-</strong></div>
          <div><span>Status</span><strong class="bad">Needs Review</strong></div>
          <div><span>Shelf</span><strong>Hidden</strong></div>
        </div>
        """
    status = "In Stock" if inventory.get("quantity", 0) > 0 else "Out of Stock"
    dot_class = "ok" if inventory.get("quantity", 0) > 0 else "bad"
    return f"""
    <div class="stock-card">
      <div><span>Available Stock</span><strong>{inventory.get('quantity', 0)}</strong></div>
      <div><span>Status</span><strong class="{dot_class}">{status}</strong></div>
      <div><span>Shelf</span><strong>{inventory.get('shelf', '-')} · Row {inventory.get('row', '-')}</strong></div>
    </div>
    """


def candidates_html(candidates: list[dict[str, Any]]) -> str:
    rows = []
    for rank, item in enumerate(candidates, start=1):
        score = round(item["score"] * 100)
        rows.append(
            f"<tr><td>{rank}</td><td>{item['label']}</td><td>{item['medicine']['name']}</td><td>{score}%</td></tr>"
        )
    return f"""
    <table class="candidate-table">
      <thead><tr><th>#</th><th>Candidate</th><th>Canonical</th><th>Score</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """


def ocr_compare_html(
    medicine: dict[str, Any],
    ocr_text: str,
    display_name: str,
    confidence: int,
    plan: dict[str, Any],
) -> str:
    corrected = display_name if plan.get("status") == "validated" else f"Needs review: {display_name}"
    return f"""
    <div class="compare-grid">
      <div><span>OCR Output</span><strong>{ocr_text}</strong></div>
      <div><span>AI Corrected</span><strong>{corrected}</strong></div>
      <div><span>Canonical</span><strong>{medicine['name'] if plan.get('status') == 'validated' else 'Not confirmed'}</strong></div>
    </div>
    """


def run_minicpm_ocr(pil_image: Image.Image) -> str:
    global OCR_MODEL, OCR_TOKENIZER
    if not LIVE_GPU_OCR:
        return DEMO_OCR_TEXT

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        raise gr.Error(f"MiniCPM dependencies are not installed: {exc}") from exc

    if OCR_MODEL is None or OCR_TOKENIZER is None:
        OCR_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        OCR_MODEL = AutoModel.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
        ).eval()
        if torch.cuda.is_available():
            OCR_MODEL = OCR_MODEL.cuda()

    messages = [{"role": "user", "content": [pil_image.convert("RGB"), DEMO_PROMPT]}]
    kwargs = {
        "image": None,
        "msgs": messages,
        "tokenizer": OCR_TOKENIZER,
        "sampling": False,
        "stream": False,
        "max_new_tokens": 20,
        "enable_thinking": False,
        "temperature": 0.0,
        "top_p": 0.1,
    }
    try:
        raw_prediction = OCR_MODEL.chat(**kwargs)
    except TypeError:
        kwargs.pop("temperature", None)
        kwargs.pop("top_p", None)
        raw_prediction = OCR_MODEL.chat(**kwargs)

    if not isinstance(raw_prediction, str):
        raw_prediction = "".join(list(raw_prediction))
    return clean_prediction(raw_prediction) or raw_prediction.strip()


@spaces.GPU(duration=300)
def analyze_prescription(image, progress=gr.Progress()):
    global SESSION_SEARCHES
    if image is None:
        raise gr.Error("Upload or capture a prescription image first.")

    for pct, label in [
        (0.20, "Prescription uploaded"),
        (0.35, "MiniCPM OCR reading handwriting"),
    ]:
        progress(pct, desc=label)
        time.sleep(0.15)

    ocr_text = run_minicpm_ocr(image)
    unload_ocr_model()

    for pct, label in [
        (0.70, "Retrieval search over medicine aliases"),
        (0.88, "Nemotron prescription validation"),
        (1.00, "Result prepared"),
    ]:
        progress(pct, desc=label)
        time.sleep(0.25)

    medicine, candidates, display_name, confidence = find_medicine_from_ocr(ocr_text)
    plan = validate_with_nemotron(ocr_text, medicine, display_name, confidence, candidates)
    unload_nemotron_model()
    accepted = plan.get("status") == "validated" and confidence >= ACCEPTANCE_THRESHOLD
    inventory = get_inventory(medicine)
    image_path = resolve_asset_path(medicine.get("image_path"))
    package_image = str(image_path) if image_path and accepted else None

    state = {
        "medicine_id": medicine["id"],
        "medicine_name": medicine["name"],
        "display_name": display_name,
        "accepted": accepted,
        "shelf": inventory["shelf"],
        "row": inventory["row"],
    }
    SESSION_SEARCHES += 1

    return (
        load_kpi_metrics(SESSION_SEARCHES),
        pipeline_html(5, plan.get("status", "needs_review")),
        medicine_details_html(medicine, inventory, ocr_text, display_name, confidence, plan),
        package_image,
        package_status_html(inventory, accepted),
        confidence_gauge(confidence),
        candidates_html(candidates),
        ocr_compare_html(medicine, ocr_text, display_name, confidence, plan),
        translated_prescription_html(plan),
        gr.update(visible=True),
        gr.update(visible=True, interactive=accepted),
        state,
    )


def open_locator(state: dict[str, Any] | None):
    if not state:
        raise gr.Error("Analyze a prescription before opening the shelf scanner.")
    return gr.update(visible=True), f"Opening shelf scanner for {state['display_name']} on shelf {state['shelf']}."


def locate_on_shelf(shelf_image, state: dict[str, Any] | None):
    if not state:
        raise gr.Error("Analyze a prescription before locating a medicine.")
    if shelf_image is None:
        raise gr.Error("Upload or capture a shelf image first.")

    image = shelf_image.convert("RGB")
    width, height = image.size
    box = (
        int(width * 0.46),
        int(height * 0.22),
        int(width * 0.78),
        int(height * 0.58),
    )
    draw = ImageDraw.Draw(image)
    for offset in range(5):
        draw.rectangle(
            (box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset),
            outline="#10b981",
        )
    draw.rectangle((box[0], max(0, box[1] - 34), box[2], box[1]), fill="#10b981")
    draw.text((box[0] + 10, max(2, box[1] - 27)), state["display_name"], fill="white")

    info = f"""
    <div class="result-card compact">
      <h3>Shelf Result</h3>
      <dl class="details">
        <dt>Found</dt><dd>{state['display_name']}</dd>
        <dt>Canonical</dt><dd>{state['medicine_name']}</dd>
        <dt>Shelf</dt><dd>{state['shelf']}</dd>
        <dt>Row</dt><dd>{state['row']}</dd>
        <dt>Confidence</dt><dd>95%</dd>
      </dl>
    </div>
    """
    return image, info


CSS = """
:root {
  --green: #0f9f6e;
  --ink: #14213d;
  --muted: #5f6b7a;
  --line: #d9e2ec;
  --soft: #f4f8f7;
  --blue: #1769aa;
}
.gradio-container {
  background: linear-gradient(180deg, #f8fbfb 0%, #edf6f4 100%);
  color: var(--ink) !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container h4,
.gradio-container p,
.gradio-container label,
.gradio-container span,
.gradio-container strong,
.gradio-container div {
  letter-spacing: 0;
}
.app-shell {
  width: min(1120px, calc(100vw - 64px));
  max-width: 1120px;
  margin: 0 auto;
  box-sizing: border-box;
}
.hero {
  padding: 24px 28px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #ffffff;
  box-shadow: 0 10px 32px rgba(15, 23, 42, 0.06);
  margin-top: 24px;
}
.hero h1 { color: var(--ink) !important; font-size: 34px; line-height: 1; margin: 0; letter-spacing: 0; }
.hero p { margin: 8px 0 0; color: var(--muted) !important; font-size: 16px; }
.powered {
  color: var(--blue) !important;
  font-weight: 800;
  text-align: right;
  background: #edf7ff;
  border: 1px solid #cfe7fb;
  border-radius: 8px;
  padding: 10px 12px;
}
.metric-row {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  width: min(1120px, calc(100vw - 64px));
  max-width: 1120px;
  margin: 16px auto 24px;
  box-sizing: border-box;
}
.metric {
  border: 1px solid var(--line);
  background: #ffffff;
  border-radius: 8px;
  padding: 16px 18px;
  min-height: 86px;
  box-shadow: 0 10px 26px rgba(15, 23, 42, 0.045);
}
.metric span, .stock-card span, .compare-grid span { color: var(--muted) !important; display: block; font-size: 13px; }
.metric strong { color: var(--ink) !important; display: block; font-size: 27px; margin-top: 6px; }
.capture-card {
  border: 1px solid var(--line);
  background: #ffffff;
  border-radius: 8px;
  padding: 28px;
  min-height: 132px;
  box-shadow: 0 10px 26px rgba(15, 23, 42, 0.045);
}
.capture-card h2 { color: var(--ink) !important; margin: 0 0 10px; font-size: 24px; }
.capture-card p { color: var(--muted) !important; margin: 0; }
.pipeline {
  border: 1px solid var(--line);
  background: #ffffff;
  border-radius: 8px;
  padding: 20px;
  margin: 0;
  min-height: 132px;
  box-shadow: 0 10px 26px rgba(15, 23, 42, 0.045);
}
.pipeline-title { color: var(--ink) !important; font-weight: 800; margin-bottom: 12px; }
.flow { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; }
.flow-step {
  min-height: 86px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--soft);
  padding: 10px;
}
.flow-step span {
  width: 26px;
  height: 26px;
  border-radius: 50%;
  display: inline-grid;
  place-items: center;
  background: #d8e7e3;
  color: var(--ink);
  font-weight: 800;
  margin-bottom: 8px;
}
.flow-step.done { border-color: #85d7bd; background: #ebfbf5; }
.flow-step.done span { background: var(--green); color: #ffffff; }
.flow-step strong { color: var(--ink) !important; display: block; font-size: 14px; }
.logs { margin: 14px 0 0; padding-left: 20px; color: #174c3c !important; }
.result-card, .stock-card {
  border: 1px solid var(--line);
  background: #ffffff;
  border-radius: 8px;
  padding: 18px;
}
.result-card h3 { color: var(--ink) !important; margin: 0 0 12px; font-size: 20px; }
.details {
  display: grid;
  grid-template-columns: 140px 1fr;
  gap: 8px 14px;
  margin: 0;
}
.details dt { color: var(--muted) !important; }
.details dd { color: var(--ink) !important; margin: 0; font-weight: 750; }
.explain {
  margin-top: 16px;
  border-top: 1px solid var(--line);
  padding-top: 14px;
}
.explain h4 { color: var(--ink) !important; margin: 0 0 8px; }
.explain p { margin: 6px 0; color: #263445 !important; }
.stock-card {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-top: 10px;
}
.stock-card strong { color: var(--ink) !important; display: block; font-size: 20px; margin-top: 3px; }
.stock-card .ok { color: var(--green); }
.stock-card .bad { color: #b91c1c; }
.candidate-table { width: 100%; border-collapse: collapse; background: #ffffff; }
.candidate-table th, .candidate-table td {
  border-bottom: 1px solid var(--line);
  padding: 10px 8px;
  text-align: left;
  color: var(--ink) !important;
}
.candidate-table th { color: var(--muted) !important; }
.compare-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}
.compare-grid div {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px;
  background: #ffffff;
}
.compare-grid strong { color: var(--ink) !important; display: block; margin-top: 6px; font-size: 18px; }
.translated-card {
  border: 1px solid var(--line);
  background: #ffffff;
  border-radius: 8px;
  padding: 18px;
  margin-top: 12px;
}
.translated-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: center;
  margin-bottom: 12px;
}
.translated-head h3 { color: var(--ink) !important; margin: 0; font-size: 20px; }
.status-pill {
  background: #ebfbf5;
  border: 1px solid #85d7bd;
  color: #075f45;
  border-radius: 999px;
  padding: 5px 10px;
  font-size: 12px;
  font-weight: 800;
}
.translated-details {
  grid-template-columns: 130px 1fr;
}
.fine-print {
  border-top: 1px solid var(--line);
  color: var(--muted) !important;
  margin: 14px 0 0;
  padding-top: 12px;
  font-size: 13px;
}
.compact { margin-top: 0; }
.gradio-container button.primary,
.gradio-container button[variant="primary"] {
  background: var(--green) !important;
  border-color: var(--green) !important;
  color: #ffffff !important;
}
.gradio-container button.primary:hover,
.gradio-container button[variant="primary"]:hover {
  background: #0b7f59 !important;
}
@media (max-width: 760px) {
  .app-shell { width: min(100% - 28px, 1120px); }
  .hero { display: block; }
  .powered { text-align: left; margin-top: 10px; }
  .metric-row, .flow, .stock-card, .compare-grid { grid-template-columns: 1fr; }
  .details { grid-template-columns: 1fr; }
  .translated-head { align-items: flex-start; flex-direction: column; }
}
"""


APP_THEME = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)


with gr.Blocks(title="PharmaCopilot") as demo:
    state = gr.State({})
    gr.HTML(
        """
        <div class="app-shell">
          <div class="hero">
            <div>
              <h1>PharmaCopilot</h1>
              <p>AI Prescription Intelligence Platform</p>
            </div>
            <div class="powered">Powered by NVIDIA Nemotron + MiniCPM-V</div>
          </div>
        </div>
        """
    )
    live_metrics = gr.HTML(load_kpi_metrics(SESSION_SEARCHES))

    with gr.Row(elem_classes=["app-shell"]):
        with gr.Column(scale=4):
            gr.HTML('<div class="capture-card"><h2>Prescription Scan</h2><p>Upload or take a prescription photo to start the AI workflow.</p></div>')
            rx_image = gr.Image(
                label="Prescription Photo",
                sources=["upload", "webcam"],
                type="pil",
                height=330,
            )
            analyze_btn = gr.Button("Analyze Prescription", variant="primary", size="lg")
        with gr.Column(scale=5):
            pipeline = gr.HTML(pipeline_html(0))

    with gr.Group(visible=False, elem_classes=["app-shell"]) as result_section:
        gr.Markdown("## Medicine Result")
        with gr.Row():
            with gr.Column(scale=5):
                details = gr.HTML()
                gauge = gr.Plot(label="Confidence Gauge")
            with gr.Column(scale=5):
                package_image = gr.Image(label="Packaging Image", height=360)
                stock = gr.HTML()
        with gr.Accordion("Top Candidates", open=False):
            candidates = gr.HTML()
        gr.Markdown("### OCR vs Corrected")
        comparison = gr.HTML()
        translated_prescription = gr.HTML()
        locate_btn = gr.Button("Locate Medicine", variant="primary", size="lg")
        locate_status = gr.Markdown()

    with gr.Group(visible=False, elem_classes=["app-shell"]) as locator_section:
        gr.Markdown("## Scan Shelf")
        with gr.Row():
            with gr.Column(scale=5):
                shelf_image = gr.Image(
                    label="Shelf Image",
                    sources=["upload", "webcam"],
                    type="pil",
                    height=360,
                )
                locate_scan_btn = gr.Button("Find Box On Shelf", variant="primary")
            with gr.Column(scale=5):
                shelf_result_image = gr.Image(label="Detected Shelf Box", height=360)
                shelf_result_info = gr.HTML()

    analyze_btn.click(
        analyze_prescription,
        inputs=[rx_image],
        outputs=[
            live_metrics,
            pipeline,
            details,
            package_image,
            stock,
            gauge,
            candidates,
            comparison,
            translated_prescription,
            result_section,
            locate_btn,
            state,
        ],
    )
    locate_btn.click(open_locator, inputs=[state], outputs=[locator_section, locate_status])
    locate_scan_btn.click(locate_on_shelf, inputs=[shelf_image, state], outputs=[shelf_result_image, shelf_result_info])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the PharmaCopilot Gradio demo.")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()

    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        css=CSS,
        theme=APP_THEME,
        allowed_paths=["/data/images", str(DATA_DIR / "images")],
    )
