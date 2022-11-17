from src import GeoDetector
from src.dataset import load_disease

df = load_disease()
gd = GeoDetector(["type", "region", "level"], "incidence", df)
result = gd.interaction_detect()
result.to_excel("q.xlsx")
