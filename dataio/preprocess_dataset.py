import os
import collections

max_tw = -1
name = "fb"

max_ndid = -1

d_time_edges = collections.defaultdict(list)


f = open("../dataset/" + name + "/graph.tedgelist", "r")

for line in f.readlines():
    token = line.split("::")
    time = int(token[2])
    s = int(token[0])
    d = int(token[1])
    d_time_edges[time].append((s,d))
    max_ndid = max(max_ndid, s)
    max_ndid = max(max_ndid, d)

f.close()

max_tw = max(d_time_edges.keys())

for k, v in d_time_edges.items():
    f = open("../dataset/" + name + "/graph_" + str(k) + ".tedgelist", "w")
    for edge in v:
        f.write(str(edge[0]) + "::" + str(edge[1]) + "\n")
    f.close()


f = open("../dataset/" + name + "/graph.info", "w")
f.write(str(max_tw+1) + "\n")
f.write(str(max_ndid+1))
f.close()



