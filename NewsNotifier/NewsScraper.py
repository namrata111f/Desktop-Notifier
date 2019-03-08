"""python program to scrap news from timesofindia website and show notification on desktop"""
import feedparser
import notify2
import time
import os
from pygame import mixer

mixer.init()
mixer.music.load('alarm.mp3')


def Parsefeed():
    f = feedparser.parse("http://timesofindia.indiatimes.com/rssfeedstopstories.cms")
    ICON_PATH = os.getcwd() + "/icon.ico"
    notify2.init('News Notify')

    for newsitem in f['items']:
        print(newsitem['title'])
        print(newsitem['summary'])
        print('\n')

        n = notify2.Notification(newsitem['title'],newsitem['summary'],icon=ICON_PATH)

        n.set_urgency(notify2.URGENCY_NORMAL)
        n.show()
        n.set_timeout(100)
        time.sleep(10)
        
if __name__ == '__main__':
    try:
        Parsefeed()
        mixer.music.play()
        time.sleep(10)
        mixer.music.pause()
    except:
        print("Error")
"""created by Vishnu Kumar and team"""
