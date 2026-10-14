import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from engine.speculative_decoder import SpeculativeDecoder
from engine.block_allocator import BlockAllocator

app = FastAPI(title="ApexInfer Server")

WEB_DIR = Path(__file__).resolve().parent / "web"
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

# Load models once at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Initializing ApexInfer engine on {device}...")

target_name = "gpt2"
draft_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(target_name)
target_model = AutoModelForCausalLM.from_pretrained(target_name).to(device)
draft_model = AutoModelForCausalLM.from_pretrained(draft_name).to(device)
target_model.eval()
draft_model.eval()

decoder = SpeculativeDecoder(
    draft_model=draft_model,
    target_model=target_model,
    tokenizer=tokenizer,
    gamma=4,
    temperature=0.0,
    device=device,
)

allocator = BlockAllocator(
    num_layers=12,
    num_heads=12,
    head_dim=64,
    total_num_blocks=64,
    block_size=16,
    device=device,
)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 35
    gamma: int = 4

@app.get("/")
async def get_index():
    return FileResponse(WEB_DIR / "index.html")

@app.post("/api/generate")
async def generate(req: GenerateRequest):
    decoder.gamma = req.gamma
    res = decoder.generate(req.prompt, max_new_tokens=req.max_tokens)
    return res

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
