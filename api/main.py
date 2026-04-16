from fastapi import FastAPI
from pydantic import BaseModel
from tools.acf_tool import get_acf_factor
from tools.estimate_tool import get_project_estimate


app = FastAPI(title="Construction Cost Estimation API")


# ---- Request Schema ----
class ACFRequest(BaseModel):
    city: str
    state: str
    lat: float | None = None
    lon: float | None = None


# ---- Response Endpoint ----
@app.post("/acf")
def acf_endpoint(req: ACFRequest):
    return get_acf_factor(
        city=req.city,
        state=req.state,
        lat=req.lat,
        lon=req.lon,
    )


# ---- Health Check ----
@app.get("/")
def health():
    return {"status": "ok"}


class EstimateRequest(BaseModel):
    project_description: str
    city: str
    state: str
    county_name: str = ""
    project_type: str = "unknown"
    project_category: str = "unknown"
    ciqs_complexity_category: str = "Category 1"
    official_budget_range: str = "$0-$1M"
    cnt_division: int = 0
    cnt_item_code: int = 0
    inflation_factor: float = 1.0
    area_type: str | None = None
    lat: float | None = None
    lon: float | None = None

@app.post("/estimate")
def estimate(req: EstimateRequest):
    return get_project_estimate(
        project_description=req.project_description,
        city=req.city,
        state=req.state,
        county_name=req.county_name,
        project_type=req.project_type,
        project_category=req.project_category,
        ciqs_complexity_category=req.ciqs_complexity_category,
        official_budget_range=req.official_budget_range,
        cnt_division=req.cnt_division,
        cnt_item_code=req.cnt_item_code,
        inflation_factor=req.inflation_factor,
        area_type=req.area_type,
        lat=req.lat,
        lon=req.lon,
    )
