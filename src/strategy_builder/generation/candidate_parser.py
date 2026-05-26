import json
import re
import logging
from typing import List, Dict, Any, Optional, Union

class CandidateParser:
    @staticmethod
    def parse(raw_output: str) -> List[Dict[str, Any]]:
        cleaned = CandidateParser.strip_markdown_fences(raw_output.strip())
        parsed = CandidateParser.parse_json_loose(cleaned)
        if parsed is None:
            return []

        candidates = parsed if isinstance(parsed, list) else [parsed]
        results = []
        for c in candidates:
            if isinstance(c, dict):
                results.append(c)
        return results

    @staticmethod
    def parse_json_loose(text: str) -> Any:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            fragment = CandidateParser.extract_balanced_json_fragment(text)
            if fragment:
                try:
                    return json.loads(fragment)
                except json.JSONDecodeError:
                    return None
            return None

    @staticmethod
    def strip_markdown_fences(text: str) -> str:
        text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
        return text.strip()

    @staticmethod
    def extract_balanced_json_fragment(text: str) -> Optional[str]:
        if not text:
            return None

        match = re.search(r'[\[{]', text)
        if not match:
            return None
        
        start_idx = match.start()
        stack = []
        in_string = False
        escape = False

        for i in range(start_idx, len(text)):
            ch = text[i]

            if in_string:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == '{':
                    stack.append('}')
                elif ch == '[':
                    stack.append(']')
                elif ch in ['}', ']']:
                    if not stack:
                        return None
                    expected = stack.pop()
                    if expected != ch:
                        return None
                    if not stack:
                        return text[start_idx : i + 1]
        return None
