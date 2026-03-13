# Construction Cost Estimation System - Project Plan

## Overview
Build a production-ready ML and LLM-powered construction cost estimation system with parametric modeling, location-based adjustments, document extraction, and standardized reporting.

---

## Phase 1: Data Foundation & Schema Design
**Goal**: Establish a normalized data layer that supports multi-provider integration and consistent feature representation.

### 1.1 Define Normalized Schema
- [ ] Design master schema for cost datasets with standard fields:
  - `project_id`, `provider_id` (raw tracing)
  - `project_type` (normalized taxonomy)
  - `location` (city, state, region, lat/lon)
  - `size_sf`, `size_gsf` (standardized units)
  - `complexity_score` (normalized 1-5 scale)
  - `systems` (structural, MEP, finishes, etc.)
  - `phase` (schematic, DD, CD, bid)
  - `cost_total_usd`, `cost_per_sf`
  - `year_completed`, `inflation_adjusted_cost`
- [ ] Document unit conversion rules (SF ↔ GSF, currency normalization)
- [ ] Create project type taxonomy mapping table
- [ ] Define complexity scoring rubric

### 1.2 Build ETL Pipeline
- [ ] Create ingestion adapters for each data provider format
- [ ] Implement data validation and quality checks
- [ ] Build transformation layer for feature normalization
- [ ] Store raw + normalized data with lineage tracking
- [ ] Set up automated ETL scheduling (if applicable)

### 1.3 Feature Engineering for ML
- [ ] Define feature set for parametric model:
  - Categorical: project_type, region, phase, primary_system
  - Numerical: size_sf, complexity_score, year_index
  - Derived: system_counts, regional_cost_index
- [ ] Define feature set for ACF model:
  - Geospatial: lat/lon, urban density, regional zone
  - Labor: prevailing wage index, union density, labor availability
  - Macro: material cost index, regional CPI, permit cost index

**Deliverables**:
- Schema documentation
- ETL pipeline code
- Feature engineering module
- Data quality dashboard/reports

---

## Phase 2: Parametric Cost Engine
**Goal**: Train, validate, and package a parametric cost prediction model.

### 2.1 Model Development
- [ ] Perform exploratory data analysis (EDA) on historical cost data
- [ ] Document data distributions, correlations, and assumptions
- [ ] Select model architecture (e.g., gradient boosting, ensemble, or neural network)
- [ ] Implement train/validation/test split strategy (temporal or random)
- [ ] Train baseline model with hyperparameter tuning
- [ ] Evaluate model performance (MAE, MAPE, R², coverage of prediction intervals)

### 2.2 Output Specification
- [ ] Define output schema:
  ```json
  {
    "cost_low": float,
    "cost_mid": float,
    "cost_high": float,
    "cost_per_sf_low": float,
    "cost_per_sf_mid": float,
    "cost_per_sf_high": float,
    "confidence_level": float,
    "model_version": string
  }
  ```
- [ ] Implement quantile regression or prediction intervals for low/mid/high
- [ ] Add confidence scoring based on input similarity to training data

### 2.3 Productionization
- [ ] Package model as inference function with clear API:
  ```python
  def get_parametric_estimate(
      project_type: str,
      size_sf: float,
      complexity: int,
      systems: List[str],
      phase: str,
      year: int
  ) -> ParametricEstimate
  ```
- [ ] Implement input validation and error handling
- [ ] Add logging and monitoring hooks
- [ ] Write unit and integration tests
- [ ] Document assumptions, limitations, and valid input ranges

**Deliverables**:
- Trained parametric model artifact
- Inference API/function
- Model documentation (assumptions, training data, performance metrics)
- Test suite

---

## Phase 3: Area Cost Factor (ACF) Engine
**Goal**: Build a location adjustment model for US-wide cost normalization.

### 3.1 Data Preparation
- [ ] Collect and normalize geospatial features (zip code, MSA, region)
- [ ] Integrate labor market indicators (BLS wage data, union rates)
- [ ] Integrate macroeconomic indicators (RS Means city cost indices, regional CPI)
- [ ] Handle missing data and spatial interpolation

### 3.2 Model Development
- [ ] Train ACF model to predict location multiplier
- [ ] Evaluate generalization across regions (hold-out region testing)
- [ ] Implement confidence output based on data density in region
- [ ] Validate against known indices (RS Means, ENR)

### 3.3 Output Specification
- [ ] Define output schema:
  ```json
  {
    "location_factor": float,
    "confidence": float,
    "base_location": string,
    "target_location": string,
    "components": {
      "labor": float,
      "materials": float,
      "equipment": float
    }
  }
  ```

### 3.4 Productionization
- [ ] Package as inference function:
  ```python
  def get_acf_factor(
      location: str,  # city, state or zip
      project_type: str = None,  # optional type-specific adjustment
      base_location: str = "national_average"
  ) -> ACFFactor
  ```
- [ ] Implement geocoding fallback for flexible location input
- [ ] Add caching for repeated location lookups
- [ ] Write tests and documentation

**Deliverables**:
- Trained ACF model artifact
- Inference API/function
- ACF documentation and validation report
- Test suite

---

## Phase 4: Foundry Tool Integration
**Goal**: Register ML engines as callable tools for LLM orchestration.

### 4.1 Tool Schema Definition
- [ ] Define OpenAI-compatible tool schemas:
  ```json
  {
    "name": "get_parametric_estimate",
    "description": "Predicts baseline construction cost range based on project parameters",
    "parameters": {...}
  }
  ```
- [ ] Define schema for `get_acf_factor`
- [ ] (Optional) Define schema for `get_program_estimate` (multi-project mode)

### 4.2 Tool Implementation
- [ ] Create wrapper functions that:
  - Parse LLM-provided arguments
  - Call underlying ML inference functions
  - Format responses for LLM consumption
- [ ] Handle edge cases (invalid inputs, out-of-range values)
- [ ] Implement rate limiting and usage tracking (if needed)

### 4.3 Testing
- [ ] Test tool calls with mock LLM inputs
- [ ] Validate round-trip JSON serialization
- [ ] Test error handling and graceful degradation

**Deliverables**:
- Foundry tool schemas (JSON)
- Tool wrapper implementations
- Integration test suite

---

## Phase 5: LLM Pipeline Integration
**Goal**: Orchestrate LLM workflows for Q&A, document extraction, and report generation.

### 5.1 Core LLM Orchestration
- [ ] Implement LLM pipeline framework with tool calling
- [ ] Design prompt templates for:
  - Single-project estimation workflow
  - Multi-project program mode
  - PDF specification extraction
  - Q&A about cost estimates
- [ ] Implement tool result parsing and re-injection

### 5.2 Document Extraction Pipeline
- [ ] Integrate PDF parsing (text extraction, table extraction)
- [ ] Create extraction prompts for scope identification:
  - Project type and size
  - Location
  - Systems and specifications
  - Timeline and phasing
- [ ] Implement validation and human-in-the-loop review flags

### 5.3 18-Section Report Generator
- [ ] Define report template with 18 standardized sections:
  1. Executive Summary
  2. Project Overview
  3. Scope Description
  4. Location Analysis
  5. Size and Program
  6. Complexity Assessment
  7. Systems Breakdown
  8. Parametric Cost Estimate
  9. Area Cost Factor Adjustment
  10. Adjusted Cost Range
  11. Cost per SF Analysis
  12. Confidence Assessment
  13. Assumptions and Exclusions
  14. Risk Factors
  15. Benchmark Comparison
  16. Recommendations
  17. Methodology Notes
  18. Appendix (data sources, model versions)
- [ ] Implement section generators with LLM prompts
- [ ] Create report assembly logic
- [ ] Support output formats (JSON, PDF, DOCX)

### 5.4 Program Mode (Multi-Project)
- [ ] Design program-level aggregation logic
- [ ] Implement batch estimation across projects
- [ ] Create program-level summary sections
- [ ] Handle cross-project comparisons and totals

**Deliverables**:
- LLM orchestration framework
- Prompt templates library
- Document extraction pipeline
- 18-section report generator
- Program mode implementation

---

## Phase 6: Testing, Documentation & Deployment
**Goal**: Ensure production readiness with comprehensive testing and documentation.

### 6.1 Testing
- [ ] Unit tests for all modules (data, ML inference, tools, LLM pipeline)
- [ ] Integration tests for end-to-end workflows
- [ ] Regression tests with golden datasets
- [ ] Performance/load testing for API endpoints

### 6.2 Documentation
- [ ] API documentation (endpoints, parameters, responses)
- [ ] User guide for LLM interactions
- [ ] Model cards for parametric and ACF models
- [ ] Data dictionary and schema documentation
- [ ] Runbook for operations and troubleshooting

### 6.3 Deployment
- [ ] Package for deployment (Docker, serverless, or platform-specific)
- [ ] Set up CI/CD pipeline
- [ ] Configure monitoring and alerting
- [ ] Implement model versioning and rollback capability

**Deliverables**:
- Complete test suite
- Documentation package
- Deployment artifacts
- Monitoring dashboard

---

## Dependencies & Sequencing

```
Phase 1 (Data Foundation)
    │
    ├──► Phase 2 (Parametric Engine) ──┐
    │                                   │
    └──► Phase 3 (ACF Engine) ─────────┼──► Phase 4 (Foundry Tools)
                                        │          │
                                        │          ▼
                                        └────► Phase 5 (LLM Pipeline)
                                                   │
                                                   ▼
                                            Phase 6 (Testing & Deployment)
```

- **Phase 1** is prerequisite for Phases 2 and 3
- **Phases 2 and 3** can run in parallel
- **Phase 4** requires completion of Phases 2 and 3
- **Phase 5** can start partially during Phase 4 (prompt design, framework)
- **Phase 6** runs continuously but final testing after Phase 5

---

## Suggested Timeline (Approximate)

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Data Foundation | 2-3 weeks | None |
| Phase 2: Parametric Engine | 3-4 weeks | Phase 1 |
| Phase 3: ACF Engine | 3-4 weeks | Phase 1 (parallel with Phase 2) |
| Phase 4: Foundry Tools | 1-2 weeks | Phases 2, 3 |
| Phase 5: LLM Pipeline | 3-4 weeks | Phase 4 |
| Phase 6: Testing & Deployment | 2-3 weeks | Phase 5 |

**Total: ~12-16 weeks** (assuming dedicated resources and parallel execution of Phases 2 and 3)

---

## Risk Factors & Mitigations

| Risk | Mitigation |
|------|------------|
| Insufficient historical data for training | Augment with public indices; use Bayesian priors |
| Location data gaps (rural areas) | Spatial interpolation; fallback to regional averages |
| Model drift over time | Implement monitoring; schedule retraining cadence |
| LLM hallucination in reports | Enforce structured outputs; validate against tool results |
| Schema changes from providers | Abstract provider adapters; version schema migrations |

---

## Success Metrics

- **Parametric Model**: MAPE < 15% on held-out test set; 90% of actuals within low-high range
- **ACF Model**: < 5% deviation from RS Means benchmark indices
- **LLM Pipeline**: 95% successful extraction rate from standard PDFs
- **Report Quality**: Stakeholder satisfaction score > 4/5 on sample reports
- **System Uptime**: 99.5% availability for API endpoints
