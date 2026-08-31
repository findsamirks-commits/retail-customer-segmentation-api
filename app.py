from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel, Field
import logging

# Configure logging for production auditing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with comprehensive metadata
app = FastAPI(
    title="Retail Customer Segmentation API",
    description="Enterprise backend service for automated RFM customer behavioral clustering and employee action routing.",
    version="2.0"
)

# Load the saved ML model and feature scaler safely
try:
    model = joblib.load('models/customer_segment_model.joblib')
    scaler = joblib.load('models/rfm_scaler.joblib')
    logger.info("Machine learning model and scaler successfully loaded into memory.")
except Exception as e:
    logger.error(f"Failed to load model artifacts: {e}")
    raise e

# Enhanced Pydantic schema with validation constraints
class CustomerInput(BaseModel):
    recency: float = Field(..., ge=0, description="Days since last purchase (must be >= 0)")
    frequency: float = Field(..., gt=0, description="Total orders placed (must be > 0)")
    monetary: float = Field(..., ge=0, description="Total spend amount (must be >= 0)")

# Dictionary mapping cluster numbers to clear business actions for staff
CLUSTER_MEANINGS = {
    0: {"segment": "Active Regulars", "action": "Maintain routine engagement; recommend cross-category add-ons."},
    1: {"segment": "VIP Champions", "action": "Provide priority delivery, exclusive early access, and loyalty perks."},
    2: {"segment": "At-Risk Churners", "action": "URGENT: Trigger automated WhatsApp retention voucher (15% off) within 24 hours."},
    3: {"segment": "Low-Spend Occasional", "action": "Target with bulk-buy bundling deals to increase basket size."}
}

@app.get("/")
def home():
    return {
        "status": "online",
        "service": "Retail Customer Segmentation API",
        "docs_url": "/docs"
    }

@app.post("/segment")
def predict_segment(data: CustomerInput):
    try:
        # Prepare input dataframe
        input_df = pd.DataFrame([{
            'Recency': data.recency,
            'Frequency': data.frequency,
            'Monetary': data.monetary
        }])
        
        # Scale input using our production scaler
        scaled_input = scaler.transform(input_df)
        
        # Predict cluster
        cluster_id = int(model.predict(scaled_input)[0])
        segment_info = CLUSTER_MEANINGS.get(cluster_id, {"segment": "Unknown", "action": "Review manually."})
        
        # Log the prediction event
        logger.info(f"Classified customer profile -> Segment: {segment_info['segment']} (Cluster ID: {cluster_id})")
        
        return {
            "status": "success",
            "input_metrics": data.dict(),
            "assigned_cluster": cluster_id,
            "customer_segment": segment_info["segment"],
            "recommended_employee_action": segment_info["action"]
        }
        
    except Exception as e:
        logger.error(f"Error during segmentation inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))
