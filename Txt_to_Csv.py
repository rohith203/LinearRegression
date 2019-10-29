import os.path
import csv
import pandas as pd
 
save_path = "/home/divakar/Downloads/FODS/Assignmet1/"
 
completeName_in  = os.path.join(save_path,'3D_spatial_network'+'.txt')
completeName_out = os.path.join(save_path,'Data'+'.csv')
 
  
file1=open(completeName_in,'r')
In_text = csv.reader(file1,delimiter = ',')
 
file2 =open(completeName_out,'w')
out_csv = csv.writer(file2)
 
file3 = out_csv.writerows(In_text)
Datafrmae = pd.read_csv('./Data.csv')
Datafrmae.to_pickle('./Data.pickle')
file1.close()
file2.close()