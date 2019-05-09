
filepath = 'train_data.csv'  
with open(filepath) as fp: 
    with open('train_data_small.csv','w') as outfile: 
        line = fp.readline()
        cnt = 0
        while line:
            if cnt % 10 == 0:
                outfile.write(line)
            line = fp.readline()
            cnt += 1
