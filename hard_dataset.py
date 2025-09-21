import json
import pandas as pd

f = open("dataset/math500.jsonl", "r")
data = f.readlines()
f.close()

problems = []
for line in data:
    line = line.strip()
    problems.append(json.loads(line))

df = pd.DataFrame(problems)
df.to_parquet("dataset/math500.parquet", index=False)