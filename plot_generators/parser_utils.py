import re
from typing import Optional
import pandas as pd

# Language tokens aligned with parse_visual semantics
YES_WORDS = {
    'en': ['yes'], 'ro': ['da'], 'hu': ['igen'], 'ru': ['да']
}
NO_WORDS = {
    'en': ['no'], 'ro': ['nu'], 'hu': ['nem'], 'ru': ['нет']
}


def clean_response_parse_visual(response_text: Optional[str]) -> Optional[str]:
    if not isinstance(response_text, str):
        return response_text
    s = response_text
    # Remove fenced code blocks
    s = re.sub(r"```[\s\S]*?```", " ", s)
    # Remove <think> blocks, even if unclosed
    s = re.sub(r"(?is)<think>.*?(?:</think>|$)", " ", s)
    # Remove [think] blocks, even if unclosed
    s = re.sub(r"(?is)\[think\].*?(?:\[/think\]|$)", " ", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def infer_prompt_type(sys_id: Optional[str]) -> Optional[str]:
    if not isinstance(sys_id, str):
        return None
    s = sys_id.lower()
    if 'yesno' in s:
        return 'yesno'
    if 'scale' in s:
        return 'scale'
    return None


def parse_yesno_pv(resp: Optional[str], lang: Optional[str]) -> Optional[bool]:
    if not isinstance(resp, str) or lang not in YES_WORDS:
        return None
    cleaned = ''.join(ch for ch in resp if ch not in '.,!?').lower().strip()
    if cleaned in YES_WORDS[lang]:
        return True
    if cleaned in NO_WORDS.get(lang, []):
        return False
    return None


def parse_scale_pv(resp: Optional[str]) -> Optional[float]:
    if not isinstance(resp, str) or not resp:
        return None
    try:
        val = pd.to_numeric(resp, errors='coerce')
    except Exception:
        return None
    if val is None or pd.isna(val):
        return None
    try:
        fv = float(val)
    except Exception:
        return None
    return fv if 1 <= fv <= 10 else None

