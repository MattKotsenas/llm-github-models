import llm
from llm.models import Attachment, Conversation, Prompt, Response
from typing import Optional, Iterator, List

from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import (
    ChatRequestMessage,
    AssistantMessage,
    AudioContentItem,
    TextContentItem,
    ImageContentItem,
    ContentItem,
    InputAudio,
    AudioContentFormat,
    ImageDetailLevel,
    ImageUrl,
    SystemMessage,
    UserMessage,
    JsonSchemaFormat,
)

INFERENCE_ENDPOINT = "https://models.inference.ai.azure.com"

CHAT_MODELS = [
    ("AI21-Jamba-1.5-Large", True, False, ["text"], ["text"]),
    ("AI21-Jamba-1.5-Mini", True, False, ["text"], ["text"]),
    ("Codestral-2501", True, False, ["text"], ["text"]),
    ("Cohere-command-r", True, False, ["text"], ["text"]),
    ("Cohere-command-r-08-2024", True, False, ["text"], ["text"]),
    ("Cohere-command-r-plus", True, False, ["text"], ["text"]),
    ("Cohere-command-r-plus-08-2024", True, False, ["text"], ["text"]),
    ("DeepSeek-R1", True, False, ["text"], ["text"]),
    ("DeepSeek-V3", True, False, ["text"], ["text"]),
    ("DeepSeek-V3-0324", True, False, ["text"], ["text"]),
    ("Llama-3.2-11B-Vision-Instruct", True, False, ["text", "image", "audio"], ["text"]),
    ("Llama-3.2-90B-Vision-Instruct", True, False, ["text", "image", "audio"], ["text"]),
    ("Llama-3.3-70B-Instruct", True, False, ["text"], ["text"]),
    ("Meta-Llama-3-70B-Instruct", True, False, ["text"], ["text"]),
    ("Meta-Llama-3-8B-Instruct", True, False, ["text"], ["text"]),
    ("Meta-Llama-3.1-405B-Instruct", True, False, ["text"], ["text"]),
    ("Meta-Llama-3.1-70B-Instruct", True, False, ["text"], ["text"]),
    ("Meta-Llama-3.1-8B-Instruct", True, False, ["text"], ["text"]),
    ("Ministral-3B", True, False, ["text"], ["text"]),
    ("Mistral-Large-2411", True, False, ["text"], ["text"]),
    ("Mistral-Nemo", True, False, ["text"], ["text"]),
    ("Mistral-large", True, False, ["text"], ["text"]),
    ("Mistral-large-2407", True, False, ["text"], ["text"]),
    ("Mistral-small", True, False, ["text"], ["text"]),
    ("Phi-3-medium-128k-instruct", True, False, ["text"], ["text"]),
    ("Phi-3-medium-4k-instruct", True, False, ["text"], ["text"]),
    ("Phi-3-mini-128k-instruct", True, False, ["text"], ["text"]),
    ("Phi-3-mini-4k-instruct", True, False, ["text"], ["text"]),
    ("Phi-3-small-128k-instruct", True, False, ["text"], ["text"]),
    ("Phi-3-small-8k-instruct", True, False, ["text"], ["text"]),
    ("Phi-3.5-MoE-instruct", True, False, ["text"], ["text"]),
    ("Phi-3.5-mini-instruct", True, False, ["text"], ["text"]),
    ("Phi-3.5-vision-instruct", True, False, ["text", "image"], []),
    ("Phi-4", True, False, ["text"], ["text"]),
    ("Phi-4-mini-instruct", True, False, ["text"], ["text"]),
    ("Phi-4-multimodal-instruct", True, False, ["audio", "image", "text"], ["text"]),
    ("gpt-4o", True, True, ["text", "image", "audio"], ["text"]),
    ("gpt-4o-mini", True, True, ["text", "image", "audio"], ["text"]),
    ("jais-30b-chat", True, False, ["text"], ["text"]),
    ("o1", False, True, ["text", "image"], ["text"]),
    ("o1-mini", False, False, ["text"], ["text"]),
    ("o1-preview", False, False, ["text"], ["text"]),
    ("o3-mini", False, True, ["text"], ["text"]),
]


EMBEDDING_MODELS = [
    "Cohere-embed-v3-english",
    "Cohere-embed-v3-multilingual",
    "text-embedding-3-large",
    "text-embedding-3-small",
]


@llm.hookimpl
def register_models(register):
    # Register both sync and async versions of each model
    # TODO: Dynamically fetch this list
    for model_id, can_stream, supports_schema, input_modalities, output_modalities in CHAT_MODELS:
        register(
            GitHubModels(
                model_id,
                can_stream=can_stream,
                supports_schema=supports_schema,
                input_modalities=input_modalities,
                output_modalities=output_modalities,
            )
        )


IMAGE_ATTACHMENTS = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
}

AUDIO_ATTACHMENTS = {
    "audio/wav",
    "audio/mpeg",
}


def attachment_as_content_item(attachment: Attachment) -> ContentItem:
    if attachment.resolve_type().startswith("audio/"):
        audio_format = (
            AudioContentFormat.WAV
            if attachment.resolve_type() == "audio/wav"
            else AudioContentFormat.MP3
        )
        return AudioContentItem(
            input_audio=InputAudio.load(
                audio_file=attachment.path, audio_format=audio_format
            )
        )
    if attachment.resolve_type().startswith("image/"):
        if attachment.url:
            return ImageContentItem(
                image_url=ImageUrl(
                    url=attachment.url,
                    detail=ImageDetailLevel.AUTO,
                ),
            )
        if attachment.path:
            return ImageContentItem(
                image_url=ImageUrl.load(
                    image_file=attachment.path,
                    image_format=attachment.resolve_type().split("/")[1],
                    detail=ImageDetailLevel.AUTO,
                ),
            )

    raise ValueError(f"Unsupported attachment type: {attachment.resolve_type()}")


def build_messages(
    prompt: Prompt, conversation: Optional[Conversation]
) -> List[ChatRequestMessage]:
    messages: List[ChatRequestMessage] = []
    current_system = None
    if conversation is not None:
        for prev_response in conversation.responses:
            if (
                prev_response.prompt.system
                and prev_response.prompt.system != current_system
            ):
                messages.append(SystemMessage(prev_response.prompt.system))
                current_system = prev_response.prompt.system
            if prev_response.attachments:
                attachment_message: list[ContentItem] = []
                if prev_response.prompt.prompt:
                    attachment_message.append(
                        TextContentItem(text=prev_response.prompt.prompt)
                    )
                for attachment in prev_response.attachments:
                    attachment_message.append(attachment_as_content_item(attachment))
                messages.append(UserMessage(attachment_message))
            else:
                messages.append(UserMessage(prev_response.prompt.prompt))
            messages.append(AssistantMessage(prev_response.text_or_raise()))
    if prompt.system and prompt.system != current_system:
        messages.append(SystemMessage(prompt.system))
    if not prompt.attachments:
        messages.append(UserMessage(content=prompt.prompt))
    else:
        attachment_message = []
        if prompt.prompt:
            attachment_message.append(TextContentItem(text=prompt.prompt))
        for attachment in prompt.attachments:
            attachment_message.append(attachment_as_content_item(attachment))
        messages.append(UserMessage(attachment_message))
    return messages


class GitHubModels(llm.Model):
    needs_key = "github"
    key_env_var = "GITHUB_MODELS_KEY"

    def __init__(
        self,
        model_id: str,
        can_stream: bool,
        supports_schema: bool,
        input_modalities: Optional[List[str]] = None,
        output_modalities: Optional[List[str]] = None,
    ):
        self.model_id = f"github/{model_id}"
        self.model_name = model_id
        self.can_stream = can_stream
        self.supports_schema = supports_schema
        self.attachment_types = set()
        if input_modalities and "image" in input_modalities:
            self.attachment_types.update(IMAGE_ATTACHMENTS)
        if input_modalities and "audio" in input_modalities:
            self.attachment_types.update(AUDIO_ATTACHMENTS)

        self.input_modalities = input_modalities
        self.output_modalities = output_modalities


    def execute(
        self,
        prompt: Prompt,
        stream: bool,
        response: Response,
        conversation: Optional[Conversation],
    ) -> Iterator[str]:
        key = self.get_key()

        extra = {}
        extra["api_version"] = "2025-03-01-preview" # Use latest version

        client = ChatCompletionsClient(
            endpoint=INFERENCE_ENDPOINT,
            credential=AzureKeyCredential(key),
            model=self.model_name,
            **extra,
        )

        if prompt.schema:
            response_format = JsonSchemaFormat(
                name="output",
                schema=prompt.schema
            )
        else:
            response_format = "text"

        messages = build_messages(prompt, conversation)
        if stream:
            completion = client.complete(
                messages=messages,
                stream=True,
                response_format=response_format,
            )
            chunks = []
            for chunk in completion:
                chunks.append(chunk)
                try:
                    content = chunk.choices[0].delta.content
                except IndexError:
                    content = None
                if content is not None:
                    yield content
            response.response_json = None  # TODO
        else:
            completion = client.complete(
                messages=messages,
                stream=False,
                response_format=response_format,
            )
            response.response_json = None  # TODO
            yield completion.choices[0].message.content
