import re
from typing import Dict, List, Any


def parse_structured_context(context: str) -> Dict[str, Any]:
    """
    Parse the synthetic/converted IPI context format into structured fields.

    Expected sections:
    - ### CURRENT USER PROMPT
    - ### CONVERSATION HISTORY
    - ### RETRIEVED MATERIALS
    """
    text = (context or "").strip()
    if not text:
        return {
            "user_prompt": "",
            "conversation_history": [],
            "retrieved_materials": [],
        }

    lines = [ln.rstrip() for ln in text.splitlines()]
    section = None
    prompt_lines: List[str] = []
    history: List[Dict[str, str]] = []
    docs: List[str] = []

    for line in lines:
        stripped = line.strip()

        if stripped == "### CURRENT USER PROMPT":
            section = "prompt"
            continue
        if stripped == "### CONVERSATION HISTORY":
            section = "history"
            continue
        if stripped == "### RETRIEVED MATERIALS":
            section = "docs"
            continue
        if not stripped:
            continue

        if section == "prompt":
            prompt_lines.append(stripped)
            continue

        if section == "history":
            # Format example: [TURN 1] USER: content
            m = re.match(r"^\[TURN\s+\d+\]\s*([A-Z_]+)\s*:\s*(.*)$", stripped)
            if m:
                history.append(
                    {
                        "role": m.group(1).lower(),
                        "content": m.group(2).strip(),
                    }
                )
            else:
                history.append({"role": "unknown", "content": stripped})
            continue

        if section == "docs":
            # Format example: [DOC 1] text...
            m = re.match(r"^\[DOC\s+\d+\]\s*(.*)$", stripped)
            docs.append(m.group(1).strip() if m else stripped)
            continue

    user_prompt = " ".join(prompt_lines).strip()

    # Fallback for non-structured text:
    # keep entire text in user_prompt so detectors always receive valid input.
    if not user_prompt and not history and not docs:
        user_prompt = text

    return {
        "user_prompt": user_prompt,
        "conversation_history": history,
        "retrieved_materials": docs,
    }
