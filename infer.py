import torch
from omegaconf import OmegaConf
import hydra
from src.models import MultiModalityCausalLM, VLChatProcessor
import numpy as np
import os
import PIL.Image
from tqdm import tqdm
from PIL import Image
import argparse
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str, default="ckpts/GoT-R1-7B")
parser.add_argument("--cfg_weight", type=float, default=5)
parser.add_argument("--text_temperature", type=float, default=1)
parser.add_argument("--image_temperature", type=float, default=1)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if "7B" in args.ckpt_path:
    processor_cfg = OmegaConf.load("configs/processor/processor_7B.yaml")
    agent_model_cfg = OmegaConf.load("configs/clm_models/GoT-R1-7B.yaml")
    save_dir = "examples/GoT-R1-7B"
elif "1B" in args.ckpt_path:
    processor_cfg = OmegaConf.load("configs/processor/processor_1B.yaml")
    agent_model_cfg = OmegaConf.load("configs/clm_models/GoT-R1-1B.yaml")
    save_dir = "examples/GoT-R1-1B"
else:
    raise ValueError(f"Invalid checkpoint path: {args.ckpt_path}, please specify the model size (7B or 1B)")

processor = hydra.utils.instantiate(processor_cfg)
tokenizer = processor.tokenizer
agent_model = hydra.utils.instantiate(agent_model_cfg).to(torch.float16).to(device).eval()

image_start_token = tokenizer.encode("<begin_of_image>")[-1]

@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    text_temperature: float = 1,
    image_temperature: float = 1,
    parallel_size: int = 16,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    seed: int = 1000,
):
    """
    parallel_size: number of images to generate from one reasoning chain
    seed: random seed for reproducibility
    """
    # Set random seed for reproducibility
    set_seed(seed)
    prompt = f"Follow the caption to generate an image through a chain of thought process: {prompt}"

    conversation = [
        {
            "role": "User",
            "content": prompt,
        },
        {"role": "Assistant", "content": ""},
    ]

    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    prompt = sft_format
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids).to(device).unsqueeze(0)

    use_cfg = cfg_weight > 1
    
    inputs_embeds = mmgpt.language_model.get_input_embeddings()(input_ids).to(device)
    generation_tokens = []
    i = 0
    next_token = torch.tensor(1)
    past_key_values = None

    while next_token.item() != image_start_token and i < 600:
        outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values)

        hidden_states = outputs.last_hidden_state
        logits = mmgpt.language_model.lm_head(hidden_states[:, -1, :])
        probs = torch.softmax(logits / text_temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        text_embeds = mmgpt.language_model.get_input_embeddings()(next_token).to(device)
        inputs_embeds = text_embeds
        generation_tokens.append(next_token)
        past_key_values = outputs.past_key_values
        i += 1
    
    if generation_tokens[-1].item() != image_start_token:
        generation_tokens.append(torch.tensor(image_start_token).to(device))

    output_ids = torch.tensor(generation_tokens).to(torch.int).to(device)
    
    input_ids = torch.cat([input_ids.squeeze(0), output_ids.to(device)], dim=-1)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(device)


    tokens = torch.zeros((parallel_size*(2 if use_cfg else 1), len(input_ids)), dtype=torch.int).to(device)
    for i in range(parallel_size*(2 if use_cfg else 1)):
        tokens[i, :] = input_ids
        if i % 2 != 0 and use_cfg:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    past_key_values = None
    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    for i in tqdm(range(image_token_num_per_image)):
        outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values)
        hidden_states = outputs.last_hidden_state
        
        logits = mmgpt.gen_head(hidden_states[:, -1, :])

        if use_cfg:
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)

        probs = torch.softmax(logits / image_temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)
        
        if use_cfg:
            next_token = torch.cat([next_token, next_token], dim=1).view(-1).unsqueeze(-1)

        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds
        past_key_values = outputs.past_key_values


    dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    return {
        "text": tokenizer.decode(output_ids),
        "images": [Image.fromarray(visual_img[i]) for i in range(parallel_size)]
    }

if __name__ == "__main__":
    prompt = "A vibrant still life painting featuring a bouquet of red tulips in a yellow vase, accompanied by a teal vase, all set on a colorful tablecloth with floral patterns"
    outputs = generate(
        mmgpt=agent_model, 
        vl_chat_processor=processor, 
        prompt=prompt,
        cfg_weight=args.cfg_weight,
        text_temperature=args.text_temperature,
        image_temperature=args.image_temperature,
        parallel_size=1,
        seed=1000,
    )
    save_dir = os.path.join(save_dir, prompt)
    os.makedirs(save_dir, exist_ok=True)
    for i, img in enumerate(outputs["images"]):
        img.save(os.path.join(save_dir, f"generated_image_{i}.jpg"))

    with open(os.path.join(save_dir, "reasoning_chain.txt"), "w") as f:
        f.write(outputs["text"])

    print(f"Generated {len(outputs['images'])} images and saved to {save_dir}")