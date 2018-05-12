import urllib.request

url = 'http://pmaweb.caltech.edu/~physlab/herx1.dat' 
response = urllib.request.urlopen(url)
dat = response.read().decode('utf-8')

# write data to .txt file
with open('data/her-x-1.txt', 'w') as data:
    data.write(dat)


 


        


