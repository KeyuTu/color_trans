qianzui="/ghome/lijj/DA/DA_ours/data/txt/multi/unlabeled_target_images_real"
data_file=qianzui+"_3.txt"
data_train=qianzui+"_3_train.txt"
data_test=qianzui+"_3_test.txt"
f=open(data_file,'r')
f_train=open(data_train,'w')
f_test=open(data_test,'w')
i=0
for lines in f.readlines():
    #print(lines)
    i=i+1
    if (i%3)==0:
       f_test.write(lines)
    else:
       f_train.write(lines) 
    #exit()
print(i)
f_train.close()
f_test.close()
exit()
