"""Prompt construction for zero-shot MedGemma report generation.

LO-VLM (BLIP) generates text in the format

    "Diagnosed disease: <disease>, Layer Information: <report>"

and its metrics.py regex-parses that format. To keep the comparison fair,
the MedGemma prompts below steer the model toward the same output format.
"""

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert ophthalmologist writing reports for retinal "
    "optical coherence tomography (OCT) B-scans."
)

DEFAULT_USER_PROMPT = (
    "You are given a retinal OCT B-scan image. Analyse it and respond in "
    "exactly the following format, with no other text:\n"
    "Diagnosed disease: <disease name>, Layer Information: <layer-by-layer "
    "description of the retinal layers and pathologies visible in the scan>"
)


def build_messages(
    user_prompt: str = DEFAULT_USER_PROMPT,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    image=None,
) -> list:
    """Build the chat messages list in the format MedGemma's processor expects."""
    content: list = []
    if user_prompt:
        content.append({"type": "text", "text": user_prompt})
    if image is not None:
        content.append({"type": "image", "image": image})
    messages = []
    if system_prompt:
        messages.append(
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        )
    messages.append({"role": "user", "content": content})
    return messages
