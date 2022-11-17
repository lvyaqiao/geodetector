# GeoDetector

This is a simple GeoDetector which can be used to the determinant
power of a covariate X of Y.

here is a basic example
```python
df = load_disease()
gd = GeoDetector(["type", "region", "level"], "incidence", df)
result = gd.interaction_detect()
result.to_excel("q.xlsx")
```
