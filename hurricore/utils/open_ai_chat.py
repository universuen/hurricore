from __future__ import annotations

import os
from io import BytesIO
from openai import OpenAI

from PIL import Image
import base64


class OpenAIChat:
    def __init__(
        self,
        model_name: str = "gpt-5",
        system_prompt: str | Image.Image | list[str | Image.Image] | None = None,
        multi_turn: bool = False,
        api_key: str | None = os.getenv("OPENAI_API_KEY"),
        base_url: str | None = os.getenv("OPENAI_API_BASE_URL"),
    ) -> None:
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.reset()
        self.multi_turn = multi_turn
    

    def reset(self) -> None:
        self.messgages = [
            {
                "role": "system",
                "content": self._build_content(self.system_prompt),
            }
        ] if self.system_prompt is not None else []


    def _build_content(self, content_sequence: str | Image.Image | list[str | Image.Image]) -> list[dict]:
        content = []
        if not isinstance(content_sequence, list):
            content_sequence = [content_sequence]
            
        for item in content_sequence:
            if isinstance(item, dict):
                content.append(item)
            elif isinstance(item, str):
                content.append(
                    {
                        "type": "text",
                        "text": item,
                    }
                )
            elif isinstance(item, Image.Image):
                with BytesIO() as buffer:
                    item.save(buffer, format="PNG")
                    base64_image = base64.b64encode(buffer.getvalue()).decode()
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    )
            else:
                raise ValueError(f"Invalid content type: {type(item)}")
        return content


    def ask(
            self, 
            content_sequence: str | Image.Image | list[str | Image.Image],
            **kwargs,
        ) -> str:
        content = self._build_content(content_sequence)
        self.messgages.append(
            {
                "role": "user",
                "content": content,
            }
        )
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messgages,
            **kwargs,
        )
        assistant_response_content = completion.choices[0].message.content
        self.messgages.append(
            {
                "role": "assistant",
                "content": assistant_response_content,
            }
        )
        if self.multi_turn is False:
            self.reset()
        return assistant_response_content
