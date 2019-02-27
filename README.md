# Desktop-Notifier
It is an desktop app which gives pop-up notification of current news and notifies when charger has to be plugged in or removed.

Web Scraping Using Python :
Web Scraping involves extracting and processing vast amount of data from a website.
We accomplish this preferably using python as it provides a lot of libraries and functions which expedites the process. 
To perform web scraping, we import the libraries as mentioned. The urllib.request module is used to open URLs. The Beautiful Soup package is used to extract data from html files. The Beautiful Soup library's name is bs4 which stands for Beautiful Soup, version 4.
Python is best suited language for this work because beautifulsoup library which we use for scraping is fast. We can write the code for extracting any html information with the help of it.
Initially to learn the basics we scraped a website coreyms.com.

Regarding this project we learnt about Beautiful soup which is a python package used to parse data from HTML and XML documents. 
We wrote the code in python so firstly, we needed to import the beautiful soup from bs4 and wrote:
->from bs4 import Beautifulsoup
After that we needed to import requests library so we could get the texts and required things from the website that we wanted to scrap
So, we wrote
-> Import requests
-> source=requests.get(‘http://coreyms.com’).text
After that we needed a parser so we wrote 
-> soup=BeautifulSoup(source,’lxml’)
Then we needed to scrap the article , it’s heading , summary and it’s youtube link so we wrote down:
-> article=soup.find(‘article’)
As by inspecting the page we found that headline was in h2 tag of the article and summary was in ‘div’ of class ‘entry content’ so we wrote
-> headline =article.h2.a.text
-> print(headline)
-> Summary = article.find(‘div’,class_=’entry content’).p.text
-> print(summary)
Also while inspecting the page we found that the youtube link was in iframe of class ‘youtube-player’ and to find the video id we needed to split the link in different strings so finally we wrote
-> vid_src=article.find(‘iframe’,class_=’youtube-player’)[‘src’]
-> vid_id=vid_src.split(‘\’)[4]
-> vid_id=vid_id.split(‘?’)[0]
-> Y_link=f’http://youtube.com/watch?v={vid_id}’
-> Print(y_link)
Lastly , we repeated the whole process for each article using for loop.

After completion of basic implementation we wrote a program to add notification tune whenever a notification pops up.

