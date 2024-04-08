import sys
from typing import List

import click
import llm
from llm.default_plugins.openai_models import remove_dict_none_values
from zhipuai import ZhipuAI


def combine_chunks(chunks: List) -> dict:
    content = ""
    role = None
    finish_reason = None
    # If any of them have log probability, we're going to persist
    # those later on
    logprobs = []

    for item in chunks:
        for choice in item.choices:
            if hasattr(choice, "logrobs") and choice.logprobs and hasattr(choice.logprobs, "top_logprobs"):
                logprobs.append(
                    {
                        "text": choice.text if hasattr(choice, "text") else None,
                        "top_logprobs": choice.logprobs.top_logprobs,
                    }
                )

            if not hasattr(choice, "delta"):
                content += choice.text
                continue
            role = choice.delta.role
            if choice.delta.content is not None:
                content += choice.delta.content
            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason

    # Imitations of the OpenAI API may be missing some of these fields
    combined = {
        "content": content,
        "role": role,
        "finish_reason": finish_reason,
    }
    if logprobs:
        combined["logprobs"] = logprobs
    for key in ("id", "object", "model", "created", "index"):
        value = getattr(chunks[0], key, None)
        if value is not None:
            combined[key] = value

    return combined

@llm.hookimpl
def register_commands(cli):
    @cli.command()
    @click.option("-s", "--system", help="System prompt to use")
    @click.option("-l", "--url", help="Image URL to use")
    def image_identify_chatglm(system, url):
        "Use chatglm to identify image"

        model_id = 'glm-4v'
        model = ChatGLMMessages(model_id)
        prompt = [
      {
        "type": "text",
        "text": "解释一下图中的现象"
      },
      {
        "type": "image_url",
        "image_url": {
          "url" : url
        }
      }
      ]
        response = model.prompt(prompt,system)
        for chunk in response:
            print(chunk, end="")
            sys.stdout.flush()

@llm.hookimpl
def register_models(register):
    #https://open.bigmodel.cn/dev/api
    register(ChatGLMMessages("glm-3-turbo"), aliases=("glm3",))
    register(ChatGLMMessages("glm-4"), aliases=("glm4",))
    register(ChatGLMMessages("glm-4v"), aliases=("glm4v",))


class ChatGLMMessages(llm.Model):
    needs_key = "chatglm"
    key_env_var = "CHATGLM_API_KEY"
    can_stream = True

    def __init__(self, model_id):
        self.model_id = model_id

    def execute(self, prompt, stream, response, conversation):
        client = ZhipuAI(api_key=self.get_key())
        messages = []
        current_system = None
        if conversation is not None:
            for prev_response in conversation.responses:
                if (
                    prev_response.prompt.system
                    and prev_response.prompt.system != current_system
                ):
                    messages.append(
                        {"role": "system", "content": prev_response.prompt.system}
                    )
                    current_system = prev_response.prompt.system
                messages.append(
                    {"role": "user", "content": prev_response.prompt.prompt}
                )
                messages.append({"role": "assistant", "content": prev_response.text()})
        if prompt.system and prompt.system != current_system:
            messages.append({"role": "system", "content": prompt.system})
        messages.append({"role": "user", "content": prompt.prompt})
        response._prompt_json = {"messages": messages}
        kwargs = {}
        if stream:
            completion = client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                stream=True,
                **kwargs,
            )
            chunks = []
            for chunk in completion:
                chunks.append(chunk)
                content = chunk.choices[0].delta.content
                if content is not None:
                    yield content
            response.response_json = remove_dict_none_values(combine_chunks(chunks))
        else:
            completion = client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                stream=False,
                **kwargs,
            )
            response.response_json = remove_dict_none_values(completion.dict())
            yield completion.choices[0].message.content




    def __str__(self):
        return "ChatGLM Messages: {}".format(self.model_id)
