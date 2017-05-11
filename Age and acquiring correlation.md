

```python
import pandas as pd
df = pd.read_csv('/home/artem/Projects/wolf_of_crunchbase/data.csv')
```

    /usr/local/lib/python3.5/dist-packages/IPython/core/interactiveshell.py:2683: DtypeWarning: Columns (6,13,15,16,17,24,39,44,59,64,69,74,79,84,94,99,104,109,114) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)



```python
age = pd.DataFrame(pd.to_datetime('2014-01-01') - pd.to_datetime(df['founded_at']))
corr = pd.DataFrame()
corr['age'] = age
corr.loc[:, 'age'] = corr['age'].apply(lambda x: x.days)
corr['is_acquired'] = df['acquired_at'].notnull()
corr.corr()

```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>is_acquired</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>1.000000</td>
      <td>0.124559</td>
    </tr>
    <tr>
      <th>is_acquired</th>
      <td>0.124559</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
