from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

# Initialize FastAPI app for internal retail tools
app = FastAPI(title="Retail Customer Segmentation API", version="1.0")

# Load the saved ML model and feature scaler
model = joblib.load('models/customer_segment_model.joblib')
scaler = joblib.load('models/rfm_scaler.joblib')

# Define input structure expected from employees/dashboards
class CustomerInput(BaseModel):
    recency: float      # Days since last purchase
    frequency: float    # Total orders placed
    monetary: float     # Total spend amount

# Dictionary mapping cluster numbers to clear business actions for staff
CLUSTER_MEANINGS = {
    0: {"segment": "Active Regulars", "action": "Maintain routine engagement; recommend cross-category add-ons."},
    1: {"segment": "VIP Champions", "action": "Provide priority delivery, exclusive early access, and loyalty perks."},
    2: {"segment": "At-Risk Churners", "action": "URGENT: Trigger automated WhatsApp retention voucher (15% off) within 24 hours."},
    3: {"segment": "Low-Spend Occasional", "action": "Target with bulk-buy bundling deals to increase basket size."}
}

@app.get("/")
def home():
    return {"message": "Retail Customer Segmentation API is live! Use the /segment endpoint to analyze buyer behavior."}

@app.post("/segment")
def predict_segment(data: CustomerInput):
    # Prepare input data frame
    input_df = pd.DataFrame([{
        'Recency': data.recency,
        'Frequency': data.frequency,
        'Monetary': data.monetary
    }])
    
    # Scale input using our saved production scaler
    scaled_input = scaler.transform(input_df)
    
    # Predict cluster
    cluster_id = int(model.predict(scaled_input)[0])
    segment_info = CLUSTER_MEANINGS.get(cluster_id, {"segment": "Unknown", "action": "Review manually."})
    
    return {
        "input_metrics": data.dict(),
        "assigned_cluster": cluster_id,
        "customer_segment": segment_info["segment"],
        "recommended_employee_action": segment_info["action"]
    }
