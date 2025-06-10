#ISSUES
#1. what is output.sequence
#2. why i cant access language_model.lm_head


import torch
import torch.nn.functional as F
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    LogitsProcessorList,
    TopKLogitsWarper,
)
from PIL import Image
import requests
import numpy as np

def retrieve_logit_lens(
    model_id: str,
    img_path: str,
    text_prompt: str = None,
    top_k: int = 50,
    max_new_tokens: int = 300,
):
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.output_hidden_states = True
    model = model.eval()
    processor = AutoProcessor.from_pretrained(model_id)

    if img_path.startswith("http"):
        image = Image.open(
            requests.get(img_path, headers={"User-Agent": "example"}, stream=True).raw
        ).convert("RGB")
    else:
        image = Image.open(img_path).convert("RGB")

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert radiologist."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this X-ray"},
                {"type": "image", "image": image}
            ]
        }
    ]
    if text_prompt:
        messages[1]["content"].insert(0, {"type": "text", "text": text_prompt})

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
    output_ids = outputs.sequences[0][input_len:] ## doesnt work for gemma3
    caption = processor.decode(output_ids, skip_special_tokens=True).strip()
    last_hidden_states = []
    for step_hidden_states in outputs.hidden_states:
        last_layer_hidden_state = step_hidden_states[-1]
        last_token_hidden_state = last_layer_hidden_state[:, -1:, :]
        last_hidden_states.append(last_token_hidden_state)
    hidden_states = torch.cat(last_hidden_states, dim=1)
    logits_warper = TopKLogitsWarper(top_k=top_k, filter_value=float("-inf"))
    logits_processor = LogitsProcessorList([])
    generated_ids = outputs.sequences[:, input_len:]
    with torch.inference_mode():
        logits = model.language_model.lm_head(hidden_states).cpu().float() ##exists but doesnt work for gemma3
        log_probs = F.log_softmax(logits, dim=-1)
        processed_log_probs = logits_processor(generated_ids.cpu(), log_probs)
        warped_log_probs = logits_warper(generated_ids.cpu(), processed_log_probs)
        softmax_probs = F.softmax(warped_log_probs, dim=-1)
    softmax_probs = softmax_probs.detach().cpu().numpy().squeeze(0)

    return caption, softmax_probs

def example_usage():
    model_id = "google/medgemma-4b-it"
    image_url = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
    
    caption, logit_lens_data = retrieve_logit_lens(
        model_id=model_id,
        img_path=image_url,
        text_prompt="Describe this X-ray:",
        top_k=50,
        max_new_tokens=200,
    )
    print(f"Generated caption:\n{caption}\n")
    print(f"Logit-lens data shape: {logit_lens_data.shape}")
    print(f"Shape means: ({logit_lens_data.shape[0]} generated tokens, {logit_lens_data.shape[1]} vocab size)")

if __name__ == "__main__":
    example_usage()