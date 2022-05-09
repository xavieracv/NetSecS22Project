import numpy      as np
import pandas     as pd

IDN = pd.read_csv("datasets/IDN.csv", usecols = [0,1], header=None)

IDN[0] = IDN[0].map(lambda x: x[x.find('//') + 2:-1])
IDN[0] = IDN[0].map(lambda x: x[:x.find('/')])

print(IDN.head())

IDN.to_csv ("test.csv", header=["qname", "label"], index=False)