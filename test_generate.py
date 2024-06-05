# %%
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image

import requests
from io import BytesIO
import re


# %%
def infer_conv_mode(model_name, input_conv_mode):
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if input_conv_mode is not None and conv_mode != input_conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, input_conv_mode, input_conv_mode))
    else:
        input_conv_mode = conv_mode

    return input_conv_mode

# %%
def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def load_and_process_image(image_file, image_processor, model):
    image = load_image(image_file)
    image_size = image.size
    # Similar operation in model_worker.py
    image_tensor = process_images([image], image_processor, model.config)
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    return image_tensor, image_size

# %%
def create_prompt(query, model, model_name, conv_mode):

    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in query:
        if model.config.mm_use_im_start_end:
            query = re.sub(IMAGE_PLACEHOLDER, image_token_se, query)
        else:
            query = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, query)
    else:
        if model.config.mm_use_im_start_end:
            query = image_token_se + "\n" + query
        else:
            query = DEFAULT_IMAGE_TOKEN + "\n" + query

    conv_mode = infer_conv_mode(model_name, input_conv_mode=conv_mode)
    conv = conv_templates[conv_mode].copy()

    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)

    return conv.get_prompt()


# %%
def load_model(
    model_path,
    model_base,
    load_8bit,
    load_4bit,
    device,
):
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, load_8bit, load_4bit, device)

    return model_name, tokenizer, model, image_processor

# %%
def run_generate(
    model,
    model_name,
    image_processor,
    tokenizer,
    image_file,
    query,
    conv_mode,
    temperature,
    top_p,
    num_beams,
    max_new_tokens,
    debug,
):

    prompt = create_prompt(query, model, model_name, conv_mode)
    image_tensor, image_size = load_and_process_image(image_file, image_processor, model)

    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors='pt',
        ).unsqueeze(0).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.unk_token_id,
            )

    outputs = tokenizer.decode(output_ids[0]).strip()
    print(outputs)

    if debug:
        print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

# %%
model_path = "./checkpoints/merged_llava-v1.5-Mistral-7B-Instruct-v0.2-finetune_lora_mistral_generated_data"
image_file = "./llava/serve/examples/pokemon_card.jpeg"


model_name, tokenizer, model, image_processor = load_model(
    model_path,
    model_base=None,
    load_8bit=False,
    load_4bit=False,
    device="cuda",
    )

run_generate(
    model,
    model_name,
    image_processor,
    tokenizer,
    image_file,
    "List five keywords that describe the item for selling on an e-commerce website",
    conv_mode="mistral_instruct",
    temperature=0.2,
    top_p=1,
    num_beams=1,
    max_new_tokens=512,
    debug=True,
    )


# %%
def time_it(reps, func):
    from time import time

    inference_time = []

    for _ in range(reps):
        start = time()
        func(
             model,
             model_name,
             image_processor,
             tokenizer,
             image_file,
            #  "List five keywords that could be used to describe the item for selling on an online e-commerce website",
             "List five keywords in numbered bullet point format that could be used to describe the item for selling online",
             conv_mode="mistral_instruct",
             temperature=0.2,
             top_p=1,
             num_beams=1,
             max_new_tokens=512,
             debug=False,
        )
        end = time()
        inference_time.append((end - start))

    return inference_time

# %%
# import pandas as pd
# inference_time = time_it(1, run_generate)

# df = pd.DataFrame(inference_time)
# print(df.describe())
