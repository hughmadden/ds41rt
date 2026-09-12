"""Original GLMRT semantic quality contract, preserved for DS41RT release inputs.

Copyright (c) 2026 T.J. Purtell. MIT License (see LICENSE).
Source revision and file hashes: fixtures/release-semantic-corpus.json.
These checks do not execute generated code or constitute comprehensive quality evaluation.
"""
import ast
import json
import re
from typing import Any

QUALITY_CONTRACT_VERSION = 'glmrt-semantic-decode-contract-v3'


def _python_block(content: str) -> tuple[str | None, list[str]]:
    match = re.fullmatch(
        r"\s*```(?:python|py)?\s*\n(?P<code>.*)\n```\s*",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        return None, ["response is not exactly one Python code block"]
    return match.group("code"), []


def _structured_json_content(content: str, *, allow_fence: bool) -> str:
    stripped = content.strip()
    if not allow_fence:
        return stripped
    match = re.fullmatch(
        r"```(?:json)?\s*\n(?P<json>.*)\n```",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return match.group("json").strip() if match is not None else stripped


def validate_case_content(case_id: str, content: str) -> dict[str, Any]:
    """Check prompt-visible contracts without executing generated content."""

    issues: list[str] = []
    stripped = content.strip()
    if not stripped:
        issues.append("response is empty")
    elif case_id == "count":
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if lines != [str(value) for value in range(1, 65)]:
            issues.append("response is not exactly the integers 1 through 64")
    elif case_id == "repeat":
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if lines != ["red green blue"] * 24:
            issues.append("response is not exactly 24 requested repetition lines")
    elif case_id == "code":
        code, block_issues = _python_block(content)
        issues.extend(block_issues)
        if code is not None:
            try:
                tree = ast.parse(code)
            except SyntaxError:
                issues.append("Python code does not parse")
            else:
                functions = [
                    node
                    for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == "merge_intervals"
                ]
                if len(functions) != 1:
                    issues.append("merge_intervals function is missing or duplicated")
                else:
                    function = functions[0]
                    if (
                        not function.args.args
                        or function.args.args[0].annotation is None
                        or function.returns is None
                    ):
                        issues.append("merge_intervals lacks requested type hints")
                    if ast.get_docstring(function) is None:
                        issues.append("merge_intervals lacks a docstring")
                if sum(isinstance(node, ast.Assert) for node in ast.walk(tree)) < 3:
                    issues.append("fewer than three assert examples were provided")
    elif case_id == "math":
        normalized = stripped.replace(",", "")
        if re.search(r"(?<![0-9])(?:\$\s*)?194\.4(?:0)?(?![0-9])", normalized) is None:
            issues.append("response does not contain the correct final price 194.40")
        if (
            "240" not in normalized
            or not any(token in normalized for token in ("25", "75", "0.75", ".75"))
            or not any(token in normalized for token in ("8", "1.08"))
        ):
            issues.append("response does not show the requested calculation inputs")
    elif case_id == "fable":
        words = re.findall(r"\b[\w'-]+\b", stripped, flags=re.UNICODE)
        if not 140 <= len(words) <= 170:
            issues.append(f"fable has {len(words)} words, outside 140..170")
        sentence_matches = list(
            re.finditer(r"(?:^|(?<=[.!?]))\s*([^.!?]+[.!?])", stripped)
        )
        final_sentence = (
            sentence_matches[-1].group(1).strip() if sentence_matches else ""
        )
        moral_words = re.findall(r"\b[\w'-]+\b", final_sentence, flags=re.UNICODE)
        moral_terms = (
            "credit",
            "share",
            "sharing",
            "together",
            "cooperat",
            "team",
            "recognition",
            "praise",
            "glory",
            "harmony",
            "humility",
            "fair",
            "both",
        )
        if not 3 <= len(moral_words) <= 32 or not any(
            term in final_sentence.casefold() for term in moral_terms
        ):
            issues.append(
                "response does not end with a concise moral about sharing credit"
            )
    elif case_id == "hello":
        if len(stripped) > 512:
            issues.append("short greeting response is unexpectedly long")
    elif case_id == "topic":
        bullets = [
            line
            for line in stripped.splitlines()
            if re.match(r"^\s*(?:[-*•]|[1-5][.)])\s+", line)
        ]
        if len(bullets) != 5:
            issues.append(f"response has {len(bullets)} bullets, expected five")
        lowered = stripped.casefold()
        for term in ("paging", "page fault", "tlb"):
            if term not in lowered:
                issues.append(f"response omits {term}")
    elif case_id in {"structured-json", "structured-json-schema"}:
        encoded = _structured_json_content(
            content,
            allow_fence=case_id == "structured-json",
        )
        try:
            value = json.loads(encoded)
        except json.JSONDecodeError:
            issues.append(
                "response is not valid bare-or-fenced JSON"
                if case_id == "structured-json"
                else "constrained response is not bare valid JSON"
            )
        else:
            expected_keys = {"path", "operation", "line_start", "line_end", "rationale"}
            if not isinstance(value, dict) or set(value) != expected_keys:
                issues.append("JSON object has the wrong key set")
            elif (
                value.get("path") != "src/cache.rs"
                or value.get("operation") != "replace"
                or value.get("line_start") != 41
                or value.get("line_end") != 47
                or not isinstance(value.get("rationale"), str)
                or not value["rationale"].strip()
            ):
                issues.append("JSON object does not preserve the requested edit")
    elif case_id == "multilingual":
        bullets = [
            line
            for line in stripped.splitlines()
            if re.match(r"^\s*(?:[-*•]|[1-4][.)、])\s*", line)
        ]
        if len(bullets) != 4:
            issues.append(f"response has {len(bullets)} bullets, expected four")
        lowered = stripped.casefold()
        if not ("寫入時複製" in stripped or "copy-on-write" in lowered):
            issues.append("response omits copy-on-write")
        if "fork" not in lowered or "頁" not in stripped:
            issues.append("response omits the requested fork/page example")
    elif case_id == "syntax-rust":
        variants = [
            int(match.group(1))
            for line in stripped.splitlines()
            if (match := re.match(r"^\s*Op([0-9]{3}),?\s*$", line))
        ]
        if variants != list(range(128)):
            issues.append("Rust enum does not contain exactly Op000 through Op127")
    elif case_id == "syntax-python":
        code, block_issues = _python_block(content)
        issues.extend(block_issues)
        if code is not None:
            try:
                tree = ast.parse(code)
            except SyntaxError:
                issues.append("Python code does not parse")
            else:
                assignments = [
                    node
                    for node in tree.body
                    if isinstance(node, ast.Assign)
                    and any(
                        isinstance(target, ast.Name) and target.id == "POWERS_OF_TWO"
                        for target in node.targets
                    )
                ]
                if len(assignments) != 1 or not isinstance(
                    assignments[0].value, ast.Tuple
                ):
                    issues.append("POWERS_OF_TWO tuple assignment is missing")
                else:
                    exponents = []
                    for element in assignments[0].value.elts:
                        if (
                            not isinstance(element, ast.BinOp)
                            or not isinstance(element.op, ast.Pow)
                            or not isinstance(element.left, ast.Constant)
                            or element.left.value != 2
                            or not isinstance(element.right, ast.Constant)
                            or not isinstance(element.right.value, int)
                        ):
                            break
                        exponents.append(element.right.value)
                    if exponents != list(range(128)):
                        issues.append("tuple is not exactly 2**0 through 2**127")
    else:
        issues.append(f"no quality validator exists for {case_id}")
    return {
        "quality_contract_version": QUALITY_CONTRACT_VERSION,
        "quality_contract_passed": not issues,
        "quality_contract_issues": issues,
    }
