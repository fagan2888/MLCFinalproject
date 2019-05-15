
filepath = 'citibike_final_merge_may4.csv'  
with open(filepath) as fp: 
    with open('raw_data_small.csv','w') as outfile: 
        line = fp.readline()
        cnt = 0
        while line:
            if cnt % 400 == 0:
                outfile.write(line)
            line = fp.readline()
            cnt += 1
