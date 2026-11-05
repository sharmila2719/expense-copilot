"""Amazon Bedrock LLM provider.

Uses Anthropic Claude (vision + reasoning) and Amazon Titan (embeddings)
through the Bedrock Runtime API. Credentials resolve via the standard AWS
chain (env vars, ~/.aws/credentials, IAM role).
"""
from __future__ import annotations

import base64
import json
from typing import List

from .base import LLMProvider


class BedrockProvider(LLMProvider):
    name = "bedrock"

    def __init__(self, region: str, text_model: str, embed_model: str) -> None:
        import boto3  # imported lazily so local mode needs no boto3 runtime

        self._text_model = text_model
        self._embed_model = embed_model
        self._client = boto3.client("bedrock-runtime", region_name=region)

    # -- generation -------------------------------------------------------
    def generate(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ],
        }
        resp = self._client.invoke_model(
            modelId=self._text_model, body=json.dumps(body)
        )
        payload = json.loads(resp["body"].read())
        return payload["content"][0]["text"]

    # -- vision -----------------------------------------------------------
    def extract_receipt_text(self, image_bytes: bytes, media_type: str) -> str:
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1500,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": encoded,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Transcribe every line of text visible on this "
                                "receipt exactly as printed. Preserve amounts, "
                                "dates, and the vendor name."
                            ),
                        },
                    ],
                }
            ],
        }
        resp = self._client.invoke_model(
            modelId=self._text_model, body=json.dumps(body)
        )
        payload = json.loads(resp["body"].read())
        return payload["content"][0]["text"]

    # -- embeddings -------------------------------------------------------
    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            body = {"inputText": text}
            resp = self._client.invoke_model(
                modelId=self._embed_model, body=json.dumps(body)
            )
            payload = json.loads(resp["body"].read())
            vectors.append(payload["embedding"])
        return vectors
