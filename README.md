# Geo AI – Soil Quality Estimator

Estimates a 0–100 soil quality score and grade (Poor / Fair / Good / Excellent)
from 11 soil measurements: pH, organic carbon, nitrogen, phosphorus, potassium,
EC, moisture, bulk density, CEC, clay %, sand %.

```bash
pip install -r requirements.txt
python soil_ai.py train
python soil_ai.py predict --ph 6.5 --organic_carbon 2.1 --nitrogen 0.15 --phosphorus 25 --potassium 180 --ec 0.8 --moisture 25 --bulk_density 1.3 --cec 18 --clay 25 --sand 40
```

## Important limitation
No real dataset was supplied, so by default the model trains on **synthetic data
labelled with a rule-based soil quality index**. It therefore reproduces that rule
set and is a prototype, not a validated scientific tool. To make it real, train on
your own lab/field data: `python soil_ai.py train --data my_soil.csv` (CSV needs the
feature columns plus `soil_quality_score`).

Status: code written but not yet run/tested.
