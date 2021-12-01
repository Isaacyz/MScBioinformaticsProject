import re
import os

def cut_into_chunks(l, n):  
    for i in range(0, len(l), n):  
        yield l[i:i + n] 

if not os.path.exists("./urls"):
    os.mkdir("./urls")

# Need download the file table from ENA database
with open('./ENA_FASTQ_download_table.txt', 'r') as f:
    f = f.readlines()
    f = f[1:]
    f = list(cut_into_chunks(f,10))

    for i in range(len(f)):
        wf = open("./urls/{}.txt".format(i+1), "w")
        quantiseq_input = open("./{}_quantiseq_input_file.txt".format(i+1), "w")

        for lines in f[i]:
            d = lines.split("\t")[6].split(";")
            file_name = str(re.findall("/\w+_", d[1])[0]).strip("_").strip("/")
            wf.write("ftp://"+d[0]+"\n" + "ftp://"+d[1]+"\n")
            quantiseq_input.write("{}\t{}\t{}\n".format(file_name, file_name+"_1.fastq.gz", file_name+"_2.fastq.gz"))
        wf.close()
        quantiseq_input.close()





