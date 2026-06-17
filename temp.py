import pandas as pd
df=pd.read_csv(r"C:\Users\hp\Documents\S6 EEBI\Model AI\combined_data.csv")

print(df[['Temp1','Temp2']].max())
