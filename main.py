import torch
from fastapi import FastAPI
from transformers import AutoModelForCausalLM
from fastapi.middleware.cors import CORSMiddleware
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
from pydantic import BaseModel
import time

class UserInput(BaseModel):
    user_input: str

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Load model and tokenizer during application startup
model_path = "deepseek-ai/deepseek-vl-1.3b-chat"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer
vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).eval()

def get_response(user_input: str) -> str:
    start_time = time.time()
    conversation = [{"role": "User", "content": user_input}, {"role": "Assistant", "content": ""}]
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True
    ).to(vl_gpt.device)
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True
    )
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    end_time = time.time()
    print(end_time-start_time)
    return answer

@app.post("/ask/")
async def ask_question(user_input: UserInput):
    if not user_input.user_input:
        return {"error": "Invalid input. Please provide a non-empty string."}
    try:
        assistant_response = get_response(user_input.user_input)
        return {"User": user_input.user_input, "Assistant": assistant_response}
    except Exception as e:
        return {"error": str(e)}
