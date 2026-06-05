import json
import time
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SYSTEM_PROMPT = """You are a senior hardware failure analyst with 20 years of experience in semiconductor manufacturing. You analyze test data from hardware validation runs to identify root causes of failures.

You will receive a cluster summary containing: dominant failure code, mean test duration, anomaly rate, and sample error messages observed in that failure cluster.

Your task is to return exactly 3 ranked root-cause hypotheses, ordered by likelihood. For each hypothesis, provide:
- root_cause: A concise, specific technical explanation
- confidence: "high", "medium", or "low"
- rationale: Why this hypothesis fits the provided data (cite specific evidence from the summary)
- triage_action: A single actionable next step for the manufacturing engineering team

Return your response as a JSON object with this exact structure:
{
  "hypotheses": [
    {
      "rank": 1,
      "root_cause": "...",
      "confidence": "high",
      "rationale": "...",
      "triage_action": "..."
    },
    {
      "rank": 2,
      "root_cause": "...",
      "confidence": "medium",
      "rationale": "...",
      "triage_action": "..."
    },
    {
      "rank": 3,
      "root_cause": "...",
      "confidence": "low",
      "rationale": "...",
      "triage_action": "..."
    }
  ]
}

Respond ONLY with the JSON object. No preamble, no explanation, no markdown fences."""


class RootCauseAnalyzer:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        max_retries: int = 3,
        base_delay: float = 2.0,
    ):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. Provide it or set it in your .env file."
            )
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay

    def analyze(self, cluster_summary: dict) -> dict:
        user_message = json.dumps(cluster_summary, indent=2)

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=0.3,
                )
                raw = response.choices[0].message.content or ""
                return self._parse_response(raw)

            except Exception as exc:
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2**attempt)
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"OpenAI API call failed after {self.max_retries} attempts: {exc}"
                ) from exc

        raise RuntimeError("OpenAI API call failed — unexpected retry loop exit")

    def _parse_response(self, raw: str) -> dict:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        parsed = json.loads(raw)
        if not isinstance(parsed, dict) or "hypotheses" not in parsed:
            raise ValueError("Response missing 'hypotheses' key")

        hypotheses = parsed["hypotheses"]
        if not isinstance(hypotheses, list) or len(hypotheses) == 0:
            raise ValueError("'hypotheses' must be a non-empty list")

        required_keys = {"rank", "root_cause", "confidence", "rationale", "triage_action"}
        valid_confidences = {"high", "medium", "low"}

        for i, hyp in enumerate(hypotheses):
            missing = required_keys - set(hyp.keys())
            if missing:
                raise ValueError(f"Hypothesis {i} missing keys: {missing}")
            if hyp["confidence"] not in valid_confidences:
                raise ValueError(
                    f"Hypothesis {i} has invalid confidence '{hyp['confidence']}'"
                )

        return parsed
