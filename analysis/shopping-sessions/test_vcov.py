import pandas as pd
from pyfixest.estimation import feols

panel = pd.read_parquet('data/netflix_panel.parquet')
panel = panel[panel['sample_type'] != 'double_negative'].copy()

# Quick test model
model = feols('outcome ~ adstock_click_1hr | user_id', data=panel, weights='sample_weight', vcov='hetero')
print('Model type:', type(model))
print('Coef type:', type(model.coef()))

# Try to get vcov - check attributes
vcov_attrs = [x for x in dir(model) if 'vcov' in x.lower() or 'cov' in x.lower()]
print('Vcov-related attributes:', vcov_attrs)

# Check _vcov directly
if hasattr(model, '_vcov'):
    print('_vcov type:', type(model._vcov))
    print('_vcov shape:', model._vcov.shape if hasattr(model._vcov, 'shape') else 'N/A')
    print('_vcov:\n', model._vcov)