import urllib.request, urllib.parse, string
import numpy as np
import matplotlib.pyplot as plt

url = 'http://nesssi.cacr.caltech.edu/cgi-bin/getcssconedbid_release2.cgi'
query = {
        'Name': 'Her X-1',
        'DB': 'photcat',
        'OUT': 'csv',
        'SHORT': 'short'
        }

query_val = urllib.parse.urlencode(query)
query_dat = query_val.encode()

with urllib.request.urlopen(url, query_dat) as response:
    split_page = response.read().decode().split('\n')[-3]
    
    # get csv url from page
    parsed = split_page[len('<font size=2> (right-mouse-click and save as to <a href='):]
    csv_url = parsed.rstrip('>download</a>)')
    
    with urllib.request.urlopen(csv_url) as outfile:
        CSV = outfile.read().decode()
        
        # write data to .html and .txt files
        with open('data/csv.html', 'w') as dat_html, open('data/csv.txt', 'w') as dat_txt:
            dat_html.write(CSV)
            dat_txt.write(CSV)

    MasterID, Mag, Magerr, RA, Dec, MJD, Blend = np.loadtxt('data/csv.txt', delimiter=',', unpack=True, skiprows=1)

    # plot magnitude vs time
    plt.scatter(MJD, Mag)
    plt.gca().invert_yaxis()
    plt.xlabel('Date (MJD)')
    plt.ylabel('Magnitude')
    plt.title(query['Name'])
    plt.show()
        


        


