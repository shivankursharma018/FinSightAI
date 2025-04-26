
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi.responses import JSONResponse

# Initialize FastAPI app
app = FastAPI(
    title="Financial Advisor API",
    description="API for generating financial advice using a fine-tuned distilgpt2 model.",
)

# Define input model for structured JSON
class PreferredInvestments(BaseModel):
    Mutual_Funds: int
    Equity_Market: int
    Debentures: int
    Government_Bonds: int
    Fixed_Deposits: int
    PPF: int
    Gold: int

    model_config = ConfigDict(
        alias_generator=lambda x: x.replace("_", " "),
        populate_by_name=True
    )

class UserProfileInput(BaseModel):
    Gender: str
    Age: int
    Investment_Avenues_Interested: str
    Preferred_Investments: PreferredInvestments
    Investment_Objectives: str
    Investment_Purpose: str
    Investment_Duration: str
    Expected_Returns: str
    Savings_Objective: str
    Source_of_Information: str

# Load model and tokenizer globally (on startup)
model_path = "financial_advisor_pretrained"
device = torch.device("cpu")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # CPU-friendly
        low_cpu_mem_usage=True,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token
except Exception as e:
    raise RuntimeError(f"Failed to load model or tokenizer: {e}")

# List of valid investments for extraction
VALID_INVESTMENTS = [
    "Mutual Funds", "Equity Market", "Debentures", "Government Bonds",
    "Fixed Deposits", "PPF", "Gold"
]

# Function to convert structured input to string format
def format_profile_to_string(input_data: UserProfileInput) -> str:
    investments = input_data.Preferred_Investments.dict(by_alias=True)
    preferred_investments_str = "\n".join(
        f"  - {key} (Preference: {value})"
        for key, value in investments.items()
    )
    profile = (
        f"User Profile:\n"
        f"- Gender: {input_data.Gender}\n"
        f"- Age: {input_data.Age}\n"
        f"- Investment Avenues Interested: {input_data.Investment_Avenues_Interested}\n"
        f"- Preferred Investments:\n"
        f"{preferred_investments_str}\n"
        f"- Investment Objectives: {input_data.Investment_Objectives}\n"
        f"- Investment Purpose: {input_data.Investment_Purpose}\n"
        f"- Investment Duration: {input_data.Investment_Duration}\n"
        f"- Expected Returns: {input_data.Expected_Returns}\n"
        f"- Savings Objective: {input_data.Savings_Objective}\n"
        f"- Source of Information: {input_data.Source_of_Information}\n\n"
        f"Question:\n"
        f"What investment strategies should I consider?"
    )
    return profile

# Function to get the highest-rated investment
def get_highest_rated_investment(preferred_investments: Dict) -> str:
    investments = preferred_investments
    return max(
        investments.items(),
        key=lambda x: x[1]
    )[0].replace("_", " ")

# Function to validate advice and extract recommended investment
def validate_and_extract_advice(advice: str, input_data: UserProfileInput) -> tuple:
    for investment in VALID_INVESTMENTS:
        if investment in advice and input_data.Investment_Objectives.lower() in advice.lower():
            return advice, investment
    # Fallback: Generate template-based advice
    highest_rated = get_highest_rated_investment(input_data.Preferred_Investments.dict(by_alias=True))
    template_advice = (
        f"Considering your objectives of {input_data.Investment_Objectives} and "
        f"{input_data.Investment_Purpose} over {input_data.Investment_Duration}, "
        f"you might explore investment avenues like {highest_rated}. "
        f"Given your expected returns of {input_data.Expected_Returns}, these options "
        f"align with your goals. Remember to diversify your portfolio and assess the risks "
        f"involved. Consulting a financial advisor can provide personalized guidance."
    )
    return template_advice, highest_rated

# Function to format response
def format_response(advice: str, recommended_investment: str) -> str:
    return (
        f"Generated Advice:\n"
        f"{advice}\n\n"
        f"What investments should you consider?:\n"
        f"{recommended_investment}"
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "distilgpt2", "device": str(device)}

# Generate advice endpoint
@app.post("/generate_advice")
async def generate_advice(input_data: UserProfileInput):
    try:
        # Convert structured input to string format
        user_profile = format_profile_to_string(input_data)

        # Prepare prompt
        prompt = f"input: {user_profile}\noutput:"
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(device)

        # Generate advice
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            no_repeat_ngram_size=2,
            early_stopping=True,
            do_sample=True,
            temperature=0.7,
        )

        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        advice = response.split("output:")[-1].strip()

        # Validate and extract investment
        validated_advice, recommended_investment = validate_and_extract_advice(advice, input_data)

        # Format response
        formatted_response = format_response(validated_advice, recommended_investment)

        return {"response": formatted_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating advice: {str(e)}")

# Run the server (for local testing)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
