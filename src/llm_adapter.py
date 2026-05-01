"""
LangChain-compatible adapter for local LLaMA/OpenAI-compatible servers.
Supports streaming and reasoning_content extraction via AsyncOpenAI.

/src/llm_adapter.py
Version:     0.17.6
Author:      Soror L.'.L.'.
Updated:     2026-05-01

Patch Notes v0.17.6 (by pytraveler):
  [+] Extracted from main.py: LlamaChatModel (BaseChatModel subclass).
  [*] No functional changes from original main.py code.
"""

import asyncio
from typing import Any, List, Optional, Dict

from openai import AsyncOpenAI
from pydantic import Field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
    AIMessageChunk,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.callbacks.manager import AsyncCallbackManagerForLLMRun


class LlamaChatModel(BaseChatModel):
    base_url: str = Field(default="http://127.0.0.1:8088/v1")
    model: str = Field(default="Falcon-H1R-7B-Q8_0.gguf")
    api_key: str = Field(default="not-needed")
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 180

    _client: Any = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout
        )

    @property
    def _llm_type(self) -> str:
        return "llama-chat"

    def _convert_messages(self, messages: List[BaseMessage]) -> List[Dict]:
        result = []
        for m in messages:
            if isinstance(m, SystemMessage):
                result.append({"role": "system", "content": m.content})
            elif isinstance(m, HumanMessage):
                result.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                result.append({"role": "assistant", "content": m.content})
            else:
                result.append({"role": "user", "content": m.content})
        return result

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return asyncio.run(self._agenerate(messages, stop, run_manager, **kwargs))

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):
        openai_messages = self._convert_messages(messages)

        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
            **kwargs
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta
            content = delta.content or ""
            reasoning = getattr(delta, 'reasoning_content', "") or ""

            lc_chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content=content,
                    additional_kwargs={"reasoning_content": reasoning}
                )
            )
            yield lc_chunk

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        result_content = ""
        result_reasoning = ""
        async for chunk in self._astream(messages, stop, run_manager, **kwargs):
            result_content += chunk.message.content
            if "reasoning_content" in chunk.message.additional_kwargs:
                result_reasoning += chunk.message.additional_kwargs["reasoning_content"]

        ai_message = AIMessage(
            content=result_content,
            additional_kwargs={"reasoning_content": result_reasoning}
        )
        return ChatResult(generations=[ChatGeneration(message=ai_message)])
