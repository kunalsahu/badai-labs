import pyttsx3
engine = pyttsx3.init() # object creation

""" RATE"""
#rate = engine.getProperty('rate')   # getting details of current speaking rate
#print (rate)                        #printing current voice rate
engine.setProperty('rate', 120)     # setting up new voice rate


"""VOLUME"""
#volume = engine.getProperty('volume')   #getting to know current volume level (min=0 and max=1)
#print (volume)                          #printing current volume level
#engine.setProperty('volume',1.0)    # setting up volume level  between 0 and 1

"""VOICE"""
voices = engine.getProperty('voices')       #getting details of current voice
engine.setProperty('voice', 'english+f1')  #changing index, changes voices. o for male
#engine.setProperty('voice', voices[1].id)   #changing index, changes voices. 1 for female

#engine.say("Hello World!")
engine.say('who are you ?')
engine.say('I am kunal, I am from bihar, india.')
engine.say("can you speak in female voice ?")
engine.setProperty('gender','female')
engine.say("yes,I can speak in female voice as well")

engine.runAndWait()
engine.stop()