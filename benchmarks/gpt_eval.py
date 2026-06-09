import base64
import os
import re
from openai import OpenAI

_VL_CLIENT: OpenAI | None = None
_JUDGE_VERSION = "finger_count_v1"


def get_judge_version() -> str:
    return _JUDGE_VERSION


def _get_vl_client(api_key: str) -> OpenAI:
    global _VL_CLIENT
    base_url = os.getenv("ROUTERAI_BASE_URL", "https://routerai.ru/api/v1")
    if _VL_CLIENT is None:
        _VL_CLIENT = OpenAI(api_key=api_key, base_url=base_url)
    return _VL_CLIENT


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def _requires_finger_count(prompt: str) -> bool:
    return "6 finger" in prompt.lower()


def _finger_count_rubric() -> str:
    return """
**FINGER COUNT RULE (mandatory for this prompt):**
Before scoring alignment, count every extended digit on the hand/glove, including the thumb.
Report the count in ### FINGER COUNT: N (integer).

Alignment scoring for finger-count prompts:
- FINGER COUNT = 6 → alignment may be 5 only if the glove is white and clearly visible
- FINGER COUNT = 5 (standard hand) → alignment MUST be 2 or lower
- FINGER COUNT any other value → alignment MUST be 2 or lower
- If digits are unclear or merged → FINGER COUNT: -1, alignment MUST be 1

Do NOT give alignment 4 or 5 unless you have explicitly counted exactly 6 digits.
"""


def _parse_finger_count(text: str) -> int | None:
    if "### FINGER COUNT:" not in text:
        return None
    raw = text.split("### FINGER COUNT:")[1].split("\n")[0].strip()
    match = re.search(r"-?\d+", raw)
    if not match:
        return None
    return int(match.group())


def _apply_finger_count_cap(alignment_score: int, finger_count: int | None) -> int:
    if finger_count is None:
        return min(alignment_score, 2)
    if finger_count != 6:
        return min(alignment_score, 2)
    return alignment_score


def evaluate_image_with_gpt(image_path, prompt, key):
    api_key = key or os.getenv("ROUTERAI_API_KEY", "")
    if not api_key:
        raise ValueError("Missing API key. Set ROUTERAI_API_KEY environment variable.")
    client = _get_vl_client(api_key)
    max_tokens = int(os.getenv("SAP_VL_MAX_TOKENS", "1024"))

    finger_rubric = _finger_count_rubric() if _requires_finger_count(prompt) else ""
    response_format = (
        "### FINGER COUNT: N\n"
        "### ALIGNMENT SCORE: score\n"
        "### ALIGNMENT EXPLANATION: explanation\n"
        "### QUALITY SCORE: score\n"
        "### QUALITY EXPLANATION: explanation"
        if finger_rubric
        else "### ALIGNMENT SCORE: score\n"
        "### ALIGNMENT EXPLANATION: explanation\n"
        "### QUALITY SCORE: score\n"
        "### QUALITY EXPLANATION: explanation"
    )

    eval_prompt = f"""You are an assistant evaluating an image on two **independent** aspects: \
(1) how well it aligns with the meaning of a given text prompt, and \
(2) its visual quality.

The text prompt is: \"{prompt}\"
{finger_rubric}
---

**PART 1: PROMPT ALIGNMENT (Semantic Fidelity)**  
Evaluate only the *meaning* conveyed by the image — ignore visual artifacts.  
Focus on:
- Are the correct objects present and depicted in a way that clearly demonstrates their intended roles and actions from the prompt?
- Does the scene illustrate the intended situation or use-case in a concrete and functional way, rather than through symbolic, metaphorical, or hybrid representation?
- If the described usage or interaction is missing or unclear, alignment should be penalized.
- Focus strictly on the presence, roles, and relationships of the described elements — not on rendering quality.


Score from 1 to 5:
5: Fully conveys the prompt's meaning with correct elements
4: Mostly accurate — main elements are correct, with minor conceptual or contextual issues
3: Main subjects are present but important attributes or actions are missing or wrong
2: Some relevant components are present, but key elements or intent are significantly misrepresented
1: Does not reflect the prompt at all

---

**PART 2: VISUAL QUALITY (Rendering Fidelity)**  
Now focus only on how the image looks visually — ignore whether it matches the prompt.  
Focus on:
- Are there rendering artifacts, distortions, or broken elements?

- Are complex areas like faces, hands, and shaped objects well-formed and visually coherent?
- Are complex areas like faces, hands, limbs, and object grips well-formed and anatomically correct?

- Is lighting, texture, and perspective consistent across the scene?
- Do elements appear physically coherent — i.e., do objects connect naturally (no floating tools, clipped limbs, or merged shapes)?
- Distortion, warping, or implausible blending of objects (e.g. melted features, fused geometry) should reduce the score.
- Unusual or surreal objects are acceptable **if** they are clearly rendered and visually deliberate.

Score from 1 to 5:
5: Clean, realistic, and fully coherent — no visible flaws
4: Mostly clean with minor visual issues or stiffness  
3: Noticeable visual flaws (e.g. broken grips, distorted anatomy), but the image is still readable  
2: Major visual issues — warped or broken key elements disrupt coherence  
1: Severe rendering failure — image appears nonsensical or corrupted

---

Respond using this format:
{response_format}"""

    base64_image = encode_image(image_path)

    print("waiting for qwen/qwen3-vl-235b-a22b-thinking response")
    response = client.chat.completions.create(
        model="qwen/qwen3-vl-235b-a22b-thinking",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": eval_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            }
        ],
        max_tokens=max_tokens,
    )
    text = response.choices[0].message.content
    print(text)

    alignment_score = int(text.split("### ALIGNMENT SCORE:")[1].split("\n")[0].strip())
    alignment_explanation = text.split("### ALIGNMENT EXPLANATION:")[1].split("### QUALITY SCORE:")[0].strip()
    quality_score = int(text.split("### QUALITY SCORE:")[1].split("\n")[0].strip())
    quality_explanation = text.split("### QUALITY EXPLANATION:")[1].strip()

    finger_count = None
    if _requires_finger_count(prompt):
        finger_count = _parse_finger_count(text)
        alignment_score = _apply_finger_count_cap(alignment_score, finger_count)
        if finger_count is not None:
            alignment_explanation = (
                f"[finger_count={finger_count}] {alignment_explanation}"
            )

    output_dict = {
        "alignment score": alignment_score,
        "alignment explanation": alignment_explanation,
        "quality score": quality_score,
        "quality explanation": quality_explanation,
    }
    if finger_count is not None:
        output_dict["finger count"] = finger_count
    return output_dict
