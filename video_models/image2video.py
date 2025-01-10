from diffusers import DiffusionPipeline
import os
from transformers import AutoTokenizer

current_dir = os.path.dirname(__file__)
cache_dir = os.path.join(current_dir, "cache")
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", cache_dir=cache_dir)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
tokenizer = AutoTokenizer.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt")

# Tokenize the prompt
tokenized_prompt = tokenizer(prompt, return_tensors="pt")
image = pipe(tokenized_prompt).images[0]