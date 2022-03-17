import urllib2
import time
num=0
myAPI = '9TESEW9N1RD052O8'
baseURL = 'https://api.thingspeak.com/update?api_key=%s' % myAPI
while True:
    time.sleep(10.0)
    conn = urllib2.urlopen(baseURL + '&field3=%s' % (num))
    conn.close()
    
