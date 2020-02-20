f = open('GloveDataV1/resultaat/lijn_10000_15000.txt', 'r', errors='ignore')
line = f.readline()
getal = 1
while line:

    title = "GloveDataV1/test/ok/zin" + str(getal) + ".txt"
    nf = open(title, "w+")
    nf.write(line)
    nf.close()
    line = f.readline()
    getal += 1

f.close()
