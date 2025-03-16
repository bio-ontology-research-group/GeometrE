import sys

filename = sys.argv[1]

with open(filename, 'r') as f:
    lines = f.readlines()[:-100]

metrics = {'1p': None, '2p': None, '3p': None, '2i': None, '3i': None, 'pi': None, 'ip': None, '2u-DNF': None, 'up-DNF': None, '2in': None, '3in': None, 'inp': None, 'pin': None, 'pni': None}
query_types = list(metrics.keys())

for line in lines:
    if "MRR" in line:
        for query_type in query_types:
            if f" {query_type} " in line:
                value = line.strip().split(": ")[-1]
                metrics[query_type] = float(value)


metrics = {k: v*100 for k, v in metrics.items()}
string = "\t&\t".join(metrics.keys()) + "\n" + "\t&\t".join([f"{v:.1f}" for v in metrics.values()]) + "\n"
print(string)
