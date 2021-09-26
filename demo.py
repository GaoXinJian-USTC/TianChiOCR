import os

txt1 = "/home/will/ocr_data/train_list.txt"
txt2 = "/home/will/ocr_data/val_list.txt"
txt1_= "/home/will/ocr_data/train_list_.txt"
txt2_= "/home/will/ocr_data/val_list_.txt"

out = [txt1_,txt2_]

for i, l in enumerate([txt1,txt2]):
    with open(l,"r+") as f:
        with open(out[i],"a+") as o:
            for line in f:
                line = line.replace("\t"," ")
                o.writelines(line)
