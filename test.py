file1 = open("myfile.txt","w")
L = ["This is Delhi","This is Paris","This is London"] 
  
# \n is placed to indicate EOL (End of Line)
file1.write("Hello \n")
file1.writelines(L)
file1.close() #to change file access modes