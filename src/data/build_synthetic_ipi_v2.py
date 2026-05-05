# 构造“半合成间接提示词注入样本”
# src/data/build_synthetic_ipi_v2.py
import csv
import json
import random
import uuid
from typing import Dict, List, Tuple

from src.utils.paths import DATA_DIR, IPI_DATA_DIR, ensure_dir


class DPIToIPIConverter:
    """
    Convert a direct prompt injection dataset (DPI) into a semi-synthetic
    indirect prompt injection dataset (IPI).

    Input CSV format:
        text,label

    Output CSV fields:
        id
        source_id
        scenario_type
        label
        user_prompt
        retrieved_materials
        conversation_history
        flat_text
        original_text
    """

    def __init__(self, random_seed: int = 42):
        self.rng = random.Random(random_seed)

        self.user_prompts = [
            "Please summarize the retrieved document.",
            "Please summarize the webpage.",
            "Read the following materials and extract the key points.",
            "Analyze the retrieved content and provide a concise summary.",
            "Review the following materials and answer the question.",
            "Continue processing the retrieved materials.",
            "Please extract the important information from the following content.",
        ]

        self.benign_wrappers = [
            "This document describes the system architecture and deployment workflow.",
            "The following section outlines the experimental setup and evaluation metrics.",
            "This webpage provides developer-facing documentation and configuration notes.",
            "The API returns structured data in JSON format and supports multiple endpoints.",
            "This section summarizes key design considerations and implementation details.",
            "The document discusses usage constraints, parameters, and default behavior.",
            "This page explains deployment steps, requirements, and environment settings.",
        ]

        self.html_templates = [
            "<html><body><h1>Documentation</h1><p>{prefix}</p><p>{payload}</p><p>{suffix}</p></body></html>",
            "<html><body><div class='content'>{prefix}</div><!-- {payload} --><footer>{suffix}</footer></body></html>",
            "<html><body><section>{prefix}</section><aside>{payload}</aside><section>{suffix}</section></body></html>",
        ]

        self.markdown_templates = [
            "# Overview\n\n{prefix}\n\n> {payload}\n\n{suffix}",
            "## Documentation\n\n{prefix}\n\n---\n\n{payload}\n\n---\n\n{suffix}",
            "| Section | Content |\n|---|---|\n| Intro | {prefix} |\n| Note | {payload} |\n| End | {suffix} |",
        ]

        self.json_templates = [
            '{{"description": "{prefix}", "debug_note": "{payload}", "status": "{suffix}"}}',
            '{{"meta": {{"overview": "{prefix}", "note": "{payload}"}}, "result": "{suffix}"}}',
            '{{"content": "{prefix}", "internal_message": "{payload}", "summary": "{suffix}"}}',
        ]

        self.ocr_templates = [
            "[OCR]\nFigure 2 caption: {prefix}\n\nEmbedded text: {payload}\n\nFooter: {suffix}",
            "[OCR]\nScanned note:\n{prefix}\n\n{payload}\n\n{suffix}",
        ]

    # ------------------------------------------------
    # Public API
    # ------------------------------------------------

    def convert_row(self, row_id: int, text: str, label: str) -> List[Dict]:
        """
        Convert one DPI row into multiple IPI scenarios.
        """
        text = self._clean_text(text)
        label = self._clean_label(label)

        return [
            self._build_rag_basic(row_id, text, label),
            self._build_rag_wrapped(row_id, text, label),
            self._build_structured_source(row_id, text, label),
            self._build_conversation(row_id, text, label),
        ]

    # ------------------------------------------------
    # Scenario builders
    # ------------------------------------------------

    def _build_rag_basic(self, row_id: int, text: str, label: str) -> Dict:
        user_prompt = self.rng.choice(self.user_prompts)

        retrieved_materials = [
            text
        ]

        flat_text = self._compose_flat_text(
            user_prompt=user_prompt,
            retrieved_materials=retrieved_materials,
            conversation_history=[],
        )

        return self._build_record(
            row_id=row_id,
            scenario_type="rag_basic",
            label=label,
            user_prompt=user_prompt,
            retrieved_materials=retrieved_materials,
            conversation_history=[],
            flat_text=flat_text,
            original_text=text,
        )

    def _build_rag_wrapped(self, row_id: int, text: str, label: str) -> Dict:
        user_prompt = self.rng.choice(self.user_prompts)

        prefix = self.rng.choice(self.benign_wrappers)
        suffix = self.rng.choice(self.benign_wrappers)

        injected_doc = f"{prefix}\n\n{text}\n\n{suffix}"

        retrieved_materials = [
            self.rng.choice(self.benign_wrappers),
            injected_doc,
        ]

        flat_text = self._compose_flat_text(
            user_prompt=user_prompt,
            retrieved_materials=retrieved_materials,
            conversation_history=[],
        )

        return self._build_record(
            row_id=row_id,
            scenario_type="rag_wrapped",
            label=label,
            user_prompt=user_prompt,
            retrieved_materials=retrieved_materials,
            conversation_history=[],
            flat_text=flat_text,
            original_text=text,
        )

    def _build_structured_source(self, row_id: int, text: str, label: str) -> Dict:
        user_prompt = self.rng.choice(self.user_prompts)

        prefix = self.rng.choice(self.benign_wrappers)
        suffix = self.rng.choice(self.benign_wrappers)

        source_family = self.rng.choice(["html", "markdown", "json", "ocr"])

        if source_family == "html":
            template = self.rng.choice(self.html_templates)
            payload_doc = template.format(
                prefix=self._escape_html(prefix),
                payload=self._escape_html(text),
                suffix=self._escape_html(suffix),
            )
        elif source_family == "markdown":
            template = self.rng.choice(self.markdown_templates)
            payload_doc = template.format(
                prefix=prefix,
                payload=text,
                suffix=suffix,
            )
        elif source_family == "json":
            template = self.rng.choice(self.json_templates)
            payload_doc = template.format(
                prefix=self._escape_json_string(prefix),
                payload=self._escape_json_string(text),
                suffix=self._escape_json_string(suffix),
            )
        else:
            template = self.rng.choice(self.ocr_templates)
            payload_doc = template.format(
                prefix=prefix,
                payload=text,
                suffix=suffix,
            )

        retrieved_materials = [
            payload_doc
        ]

        flat_text = self._compose_flat_text(
            user_prompt=user_prompt,
            retrieved_materials=retrieved_materials,
            conversation_history=[],
        )

        return self._build_record(
            row_id=row_id,
            scenario_type=f"structured_source_{source_family}",
            label=label,
            user_prompt=user_prompt,
            retrieved_materials=retrieved_materials,
            conversation_history=[],
            flat_text=flat_text,
            original_text=text,
        )

    def _build_conversation(self, row_id: int, text: str, label: str) -> Dict:
        user_prompt = self.rng.choice([
            "Continue processing the materials.",
            "Please continue the analysis.",
            "Proceed with the next step.",
            "Review the previously provided content.",
        ])

        conversation_history = [
            {"role": "user", "content": "I will paste a webpage snippet."},
            {"role": "assistant", "content": "Okay, please provide it."},
            {"role": "user", "content": text},
        ]

        flat_text = self._compose_flat_text(
            user_prompt=user_prompt,
            retrieved_materials=[],
            conversation_history=conversation_history,
        )

        return self._build_record(
            row_id=row_id,
            scenario_type="conversation",
            label=label,
            user_prompt=user_prompt,
            retrieved_materials=[],
            conversation_history=conversation_history,
            flat_text=flat_text,
            original_text=text,
        )

    # ------------------------------------------------
    # Record helpers
    # ------------------------------------------------

    def _build_record(
        self,
        row_id: int,
        scenario_type: str,
        label: str,
        user_prompt: str,
        retrieved_materials: List[str],
        conversation_history: List[Dict[str, str]],
        flat_text: str,
        original_text: str,
    ) -> Dict:
        return {
            "id": f"ipi_{uuid.uuid4().hex[:10]}",
            "source_id": row_id,
            "scenario_type": scenario_type,
            "label": label,
            "user_prompt": user_prompt,
            "retrieved_materials": json.dumps(retrieved_materials, ensure_ascii=False),
            "conversation_history": json.dumps(conversation_history, ensure_ascii=False),
            "flat_text": flat_text,
            "original_text": original_text,
        }

    def _compose_flat_text(
        self,
        user_prompt: str,
        retrieved_materials: List[str],
        conversation_history: List[Dict[str, str]],
    ) -> str:
        parts: List[str] = []

        parts.append("### CURRENT USER PROMPT")
        parts.append(user_prompt.strip())

        if conversation_history:
            parts.append("\n### CONVERSATION HISTORY")
            for idx, turn in enumerate(conversation_history, start=1):
                role = turn.get("role", "unknown").upper()
                content = turn.get("content", "").strip()
                parts.append(f"[TURN {idx}] {role}: {content}")

        if retrieved_materials:
            parts.append("\n### RETRIEVED MATERIALS")
            for idx, doc in enumerate(retrieved_materials, start=1):
                parts.append(f"[DOC {idx}] {doc}")

        return "\n".join(parts).strip()

    # ------------------------------------------------
    # Small utilities
    # ------------------------------------------------

    def _clean_text(self, text: str) -> str:
        return str(text).replace("\r", " ").strip()

    def _clean_label(self, label: str) -> str:
        label = str(label).strip().lower()
        if label not in {"benign", "jailbreak"}:
            raise ValueError(f"Unsupported label: {label}")
        return label

    def _escape_html(self, text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    def _escape_json_string(self, text: str) -> str:
        return (
            text.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", " ")
        )


def load_dataset(csv_path) -> List[Dict]:
    rows: List[Dict] = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if "text" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("Input CSV must contain 'text' and 'label' columns.")

        for idx, row in enumerate(reader):
            rows.append({
                "row_id": idx,
                "text": row["text"],
                "label": row["label"],
            })

    return rows


def save_dataset(rows: List[Dict], output_path) -> None:
    if not rows:
        raise ValueError("No rows to save.")

    fieldnames = list(rows[0].keys())

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    input_path = DATA_DIR / "dataset.csv"
    output_dir = ensure_dir(IPI_DATA_DIR)
    output_path = output_dir / "dataset_ipi.csv"

    print(f"Loading DPI dataset from: {input_path}")
    raw_rows = load_dataset(input_path)

    converter = DPIToIPIConverter(random_seed=42)

    converted_rows: List[Dict] = []

    for row in raw_rows:
        converted_rows.extend(
            converter.convert_row(
                row_id=row["row_id"],
                text=row["text"],
                label=row["label"],
            )
        )

    save_dataset(converted_rows, output_path)

    print(f"Input rows: {len(raw_rows)}")
    print(f"Output rows: {len(converted_rows)}")
    print(f"Saved IPI dataset to: {output_path}")


if __name__ == "__main__":
    main()
