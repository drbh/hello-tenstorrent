from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "microsoft/phi-1"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="cpu" 
)

print(f"Model loaded: {model}")
prompt = "In a world where technology and magic coexist, "

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
