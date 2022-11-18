# GeoDetector

This is a simple GeoDetector which can be used to the determinant
power of a covariate X of Y.

Now only interaction_detect, and q_values is available.

# Install
```bash
pip install geodetector
```
then the geodetector will be installed 
```

here is a basic example
```python
from geodetector import geodetector

df = load_disease()
gd = GeoDetector(df,["type", "region", "level"], "incidence")

# directly plot the result
gd.plot() # show the q_values bar of each covariate
gd.plot_interaction() # show the interaction bar of each covariate

# interaction detect and save the result
result = gd.interaction_detect()
result.to_excel("q.xlsx")

```
