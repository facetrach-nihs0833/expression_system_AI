#!/usr/bin/python3.9
from playsound import playsound
import threading

class playy:
    def __init__(self, URL):
        self.playsong=URL
    def start(self):
	#把程式放進子執行緒
        threading.Thread(target=self.playsound,name="playla",  args=()).start()

    def playsound(self):    
        print(f"啟動播放程序:{self.playsong}")
        
        playsound(self.playsong,'mp3')    

