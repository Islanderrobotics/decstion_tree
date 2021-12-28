---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python pycharm={"name": "#%%\n"}
import pandas as pd
import numpy as np
import irdatacleaning as ird
import islanders as ir
olympic = pd.read_csv("/Users/williammckeon/Downloads/Winter_Olympic_Medals.csv")
```

```python pycharm={"name": "#%%\n"}
olympic.info()
```

```python pycharm={"name": "#%%\n"}
olympic.head()
```

```python pycharm={"name": "#%%\n"}
olympic
```

```python pycharm={"name": "#%%\n"}
olympic.drop(columns = ["Year","Host_country","Host_city","Country_Code"],inplace=True)
```

```python pycharm={"name": "#%%\n"}
new_data = pd.DataFrame(olympic.groupby(by = "Country_Name").sum().sort_values(by = "Gold", ascending =False))
```

```python pycharm={"name": "#%%\n"}
new_data
```

```python pycharm={"name": "#%%\n"}
gets_gold = []
for i in new_data.Gold:
    if i >0:
        gets_gold.append(1)
    else:
        gets_gold.append(0)
```

```python pycharm={"name": "#%%\n"}
gets_gold
```

```python pycharm={"name": "#%%\n"}
new_data["gets_gold"] = gets_gold
new_data.drop(columns="Gold", inplace=True)
```

```python pycharm={"name": "#%%\n"}
new_data
```

```python pycharm={"name": "#%%\n"}
new_data.columns
```

```python pycharm={"name": "#%%\n"}
X = np.array(new_data.iloc[:,:-1].values)
```

```python pycharm={"name": "#%%\n"}
y = np.array(new_data.iloc[:,-1].values)
```

```python pycharm={"name": "#%%\n"}
predict = np.array(new_data.iloc[-5:,:-1].values)
```

```python pycharm={"name": "#%%\n"}
predict
```

```python pycharm={"name": "#%%\n"}
dec = ir.DT(X,y, test_size=0.2)
```

```python pycharm={"name": "#%%\n"}
dt,X_test,y_test = dec.build(test = True)
```

```python pycharm={"name": "#%%\n"}
dt.score(X_test,y_test)
```

```python pycharm={"name": "#%%\n"}
dec.show()
```

```python pycharm={"name": "#%%\n"}
dt.predict_proba(predict)
```

```python pycharm={"name": "#%%\n"}
new_data
```

```python pycharm={"name": "#%%\n"}

```
