import re, pandas as pd, numpy as np
spec = {}
for line in open("nssrn_2022_puf_flat.do", encoding="latin-1"):
    m = re.match(r"^str\s+(\S+)\s+(\d+)-(\d+)", line.strip()) or re.match(r"^str\s+(\S+)\s+(\d+)$", line.strip())
    if m:
        g = m.groups(); a = int(g[1]); b = int(g[2]) if len(g) > 2 and g[2] else a
        spec[g[0]] = (a - 1, b)
def load(cols):
    return pd.read_fwf("nssrn_2022_puf_flat.txt", colspecs=[spec[c] for c in cols], names=cols, dtype=str)
