import urllib.parse, urllib.request
from astropy.io.votable import parse_single_table
import matplotlib.pyplot as plt

url = 'http://nesssi.cacr.caltech.edu/cgi-bin/getcssconedbid_release2.cgi'
query = {
            'Name': 'Her X-1',
            'DB': 'photcat',
            'OUT': 'vot',
            'SHORT': 'short'
                                                }
query_val = urllib.parse.urlencode(query)
query_dat = query_val.encode()

with urllib.request.urlopen(url, query_dat) as response:
    split_page = response.read().decode().split('\n')[-7]
    
    # get votable url from page
    parsed = split_page[len('<font size=2> (right-mouse-click and save as to <a href='):]
    vot_url = parsed.rstrip('>download</a>)')
    
    with urllib.request.urlopen(vot_url) as outfile:
        vot = outfile.read().decode()
        urllib.request.urlretrieve(vot_url, 'data/vot.xml')

table = parse_single_table('data/vot.xml')
Mag = table.array['Mag']
MJD = table.array['ObsTime']

# plot magnitude vs time
plt.scatter(MJD, Mag)
plt.gca().invert_yaxis()
plt.xlabel('Date (MJD)')
plt.ylabel('Magnitude')
plt.title(query['Name'])
plt.show()




