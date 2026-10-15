"""Normalize equivalent judge JSON envelopes without changing any judgment."""
import json


def normalize_judge_json(text):
    value = text.strip()
    if value.startswith("```") and value.endswith("```"):
        value = value.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    data = json.loads(value)
    if isinstance(data, list):
        data = {"results": data}
    elif isinstance(data, dict) and {"expectedOutcome", "reasoning", "metExpectation"} <= data.keys():
        data = {"results": [data]}
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ValueError("Judge must return a results array")
    for row in data["results"]:
        if not isinstance(row, dict) or not {"expectedOutcome", "reasoning", "metExpectation"} <= row.keys():
            raise ValueError("Incomplete judge result")
        if not isinstance(row["metExpectation"], bool):
            raise ValueError("Judge decisions must be JSON booleans")
    return data


def configure_judge(module, expected, records):
    from tau2.utils.llm_utils import generate
    from tau2.data_model.message import SystemMessage

    def structured_generate(**kwargs):
        messages = list(kwargs["messages"])
        messages.insert(1, SystemMessage(role="system", content=(
            'Return ONLY {"results": [{"expectedOutcome": "verbatim assertion", '
            '"reasoning": "brief explanation", "metExpectation": true_or_false}]}. '
            'The results array must contain exactly one entry for EACH assertion below, including unmet assertions. '
            'Copy every expectedOutcome verbatim, with no additional assertions. Assertions: '
            + json.dumps(expected))))
        response = generate(**{**kwargs, "messages": messages})
        records.append({"raw_content": response.content})
        parsed = normalize_judge_json(response.content or "")
        response.content = json.dumps(parsed)
        return response

    module.generate = structured_generate
