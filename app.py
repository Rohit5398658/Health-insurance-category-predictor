from fastapi import FastAPI
from pydantic import BaseModel, Field, computed_field
from typing import Literal, Annotated
import pandas as pd
import pickle
from fastapi.responses import JSONResponse

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

# City Tier Lists
tier_1 = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"]
tier_2 = ["Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi", "Visakhapatnam", "Coimbatore",
          "Bhopal", "Nagpur", "Vadodara", "Surat", "Rajkot", "Jodhpur", "Raipur", "Amritsar", "Varanasi",
          "Agra", "Dehradun", "Mysore", "Jabalpur", "Guwahati", "Thiruvananthapuram", "Ludhiana", "Nashik",
          "Allahabad", "Udaipur", "Aurangabad", "Hubli", "Belgaum", "Salem", "Vijayawada", "Tiruchirappalli",
          "Bhavnagar", "Gwalior", "Dhanbad", "Bareilly", "Aligarh", "Gaya", "Kozhikode", "Warangal",
          "Kolhapur", "Bilaspur", "Jalandhar", "Noida", "Guntur", "Asansol", "Siliguri"]

# Pydantic Input Model
class UserInput(BaseModel):
    age: Annotated[int, Field(..., ge=0)]
    weight: Annotated[float, Field(..., gt=0)]
    height: Annotated[float, Field(..., gt=0)]
    income_lpa: Annotated[float, Field(..., gt=0)]
    smoker: Annotated[Literal["yes", "no"], Field(...)]
    city: str
    occupation: Literal["retired", "freelancer", "student", "government_job", "business_owner", "unemployed", "private_job"]

    @computed_field
    @property
    def bmi(self) -> float:
        return self.weight / (self.height ** 2)

    @computed_field
    @property
    def lifestyle_risk(self) -> str:
        is_smoker = self.smoker == "yes"
        if is_smoker and self.bmi > 30:
            return "high"
        elif is_smoker or self.bmi > 27:
            return "medium"
        else:
            return "low"

    @computed_field
    @property
    def age_group(self) -> str:
        if self.age < 25:
            return "young"
        elif self.age < 45:
            return "adult"
        elif self.age < 60:
            return "middle_aged"
        return "senior"

    @computed_field
    @property
    def city_tier(self) -> int:
        if self.city in tier_1:
            return 1
        elif self.city in tier_2:
            return 2
        return 3

@app.post("/predict")
def predict_premium(data: UserInput):
    input_df = pd.DataFrame([{
        "income_lpa": data.income_lpa,
        "occupation": data.occupation,
        "bmi": data.bmi,
        "age_group": data.age_group,
        "lifestyle_risk": data.lifestyle_risk,
        "city_tier": data.city_tier
    }])
    
    prediction = model.predict(input_df)[0]
    return JSONResponse(content={"predicted_category": prediction})
