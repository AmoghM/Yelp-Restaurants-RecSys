import pandas as pd

with open('PeshVsQuetta.json', encoding='utf-8-sig') as f_input:
    df = pd.read_json(f_input)

df.to_csv('PeshVsQuetta.csv', encoding='utf-8', index=False)