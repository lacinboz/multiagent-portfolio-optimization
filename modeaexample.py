import pandas as pd
df = pd.read_csv("data/ablation/mode_a_ablation_results.csv")
ok = df[df["status"] == "ok"]
print(ok.groupby("alpha")["delta_return"].mean().round(4) * 100)