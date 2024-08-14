from logging import NullHandler
from random import random
from tkinter import Frame
from xmlrpc.client import boolean
import cv2
from cv2 import flip
import time
import random

import mingus
from mingus.midi import fluidsynth
import mingus.core.notes as notes
from mingus.containers import Note, Instrument
from mingus.containers import MidiInstrument
from mingus.containers import NoteContainer
from mingus.containers import Track
import mingus.core.chords as chords

def Check_note(X,Y):
    Tonal_Triadic = [[-1,11,-1,7,7,9,-1,-1,-1,-1,-1,-1],
                 [11,-1,-1,-1,8,8,10,-1,-1,-1,-1,-1],
                 [-1,-1,-1,-1,-1,9,9,11,-1,-1,-1,-1],
                 [7,-1,-1,-1,-1,-1,10,0,0,-1,-1,-1],
                 [7,8,-1,-1,-1,-1,-1,11,1,1,-1,-1],
                 [9,8,9,-1,-1,-1,-1,-1,0,2,2,-1],
                 [-1,10,9,11,10,-1,-1,-1,-1,-1,1,3,3],
                 [-1,-1,11,0,11,-1,-1,-1,-1,-1,2,4],
                 [-1,-1,-1,0,1,0,-1,-1,-1,-1,-1,3],
                 [-1,-1,-1,-1,1,2,1,-1,-1,-1,-1,-1],
                 [-1,-1,-1,-1,-1,1,2,1,-1,-1,-1,-1,-1],
                 [-1,-1,-1,-1,-1,-1,3,4,3,-1,-1,-1]]
        
    return Tonal_Triadic[X][Y]


def corres_chord(X, Y):
    note_list = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    Cor_note_one = [[-1,6,-1,1,9,5,-1,-1,-1,-1,-1,-1],
                 [6,-1,-1,-1,10,10,6,-1,-1,-1,-1,-1],
                 [-1,-1,-1,-1,-1,3,3,7,-1,-1,-1,-1],
                 [1,-1,-1,-1,-1,-1,3,0,8,-1,-1,-1],
                 [9,10,-1,-1,-1,-1,-1,4,1,9,-1,-1],
                 [5,10,3,-1,-1,-1,-1,-1,5,2,10,-1],
                 [-1,6,3,3,-1,-1,-1,-1,-1,-1,6,10,11],
                 [-1,-1,7,0,4,-1,-1,-1,-1,-1,7,4],
                 [-1,-1,-1,8,1,5,-1,-1,-1,-1,-1,8],
                 [-1,-1,-1,-1,9,2,6,-1,-1,-1,-1,-1],
                 [-1,-1,-1,-1,-1,10,10,7,-1,-1,-1,-1,-1],
                 [-1,-1,-1,-1,-1,-1,11,4,8,-1,-1,-1]]

    chord = chords.major_triad(note_list[Cor_note_one[X][Y]])

    return chord



def coordinate_to_Note(X, Y):
    note_list = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    chord_one = corres_chord(X, Y)
    n = [Note(note_list[X],velocity=100),Note(note_list[Y],velocity=100),
            Note(note_list[Check_note(X,Y)],velocity=100)]
    c = NoteContainer([Note(chord_one[0], 3), Note(chord_one[1], 3), Note(chord_one[2], 3)])
    
    random.shuffle(n)
    return n,c


 
class mpHands:
    import mediapipe as mp
    def __init__(self,maxHands=2,tol1=.5,tol2=.5):
        self.hands = self.mp.solutions.hands.Hands(maxHands,tol1,tol2)
    def Marks(self,frame):
        myHands=[]
        handsType=[]
        frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=self.hands.process(frameRGB)
        if results.multi_hand_landmarks != None:
            for hand in results.multi_handedness:
                handType=hand.classification[0].label
                handsType.append(handType)
            for handLandMarks in results.multi_hand_landmarks:
                myHand=[]
                for landMark in handLandMarks.landmark:
                    myHand.append((int(landMark.x*width),int(landMark.y*height)))
                myHands.append(myHand)
        return myHands,handsType

fluidsynth.init("test.sf2")
fluidsynth.set_instrument(1, 2)

 
width=1440
height=840
grid_w = 600
grid_h = 600
cap = cv2.VideoCapture(0)
findHands=mpHands(2)

vertical = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
horizontal = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
inds = [8]

X = -1
Y = -1

while True:
    ignore,  frame = cap.read()
    frame=cv2.resize(frame,(width,height))
    flip_img = cv2.flip(frame,1)
    handData, handsType=findHands.Marks(flip_img)
    for hand,handType in zip(handData,handsType):
        if handType=='Right':
            handColor=(255,0,0)
            for i in range(12):
                for j in range(12):
                    if grid_w//12*j+(width-grid_w)//2 < hand[8][0] < grid_w//12*(j+1)+(width-grid_w)//2 and grid_h/12*i < hand[8][1] < grid_h/12*(i+1):
                        cv2.putText(flip_img, '('+str(horizontal[j])+','+str(vertical[i])+')', (hand[8][0]+10,hand[8][1]+10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
                        
                        if Check_note(horizontal[j], horizontal[i])!= -1:
                            
                            Y = horizontal[i]

        if handType=='Left':
            handColor=(0,0,255)
            for i in range(12):
                for j in range(12):
                    if grid_w//12*j+(width-grid_w)//2 < hand[8][0] < grid_w//12*(j+1)+(width-grid_w)//2 and grid_h/12*i < hand[8][1] < grid_h/12*(i+1):
                        cv2.putText(flip_img, '('+str(horizontal[j])+','+str(vertical[i])+')', (hand[8][0]+10,hand[8][1]+10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
                        if Check_note(horizontal[j], horizontal[i])!= -1:
                            
                            X = horizontal[j]
        
        if X != -1 and Y != -1:
            print('(', X, ', ', Y, ')')
            t, c = coordinate_to_Note(X, Y)
            fluidsynth.play_NoteContainer(c)

            for i in range(3):
                fluidsynth.play_Note(t[i], 1)        
                fluidsynth.stop_Note(t[i], 1)
                #print(t[i])

            fluidsynth.stop_NoteContainer(c)
            X = -1
            Y = -1
            #print(c)
        

        for ind in inds:
            #cv2.putText(flip_img, str(ind/4), (hand[ind][0]-15,hand[ind][1]-15) , cv2.FONT_HERSHEY_PLAIN, 2, (225, 225, 255), 2)
            cv2.circle(flip_img,hand[ind],5,handColor,cv2.FILLED)
        

    # draw the grid
    h1 = cv2.line(flip_img, ((width-grid_w)//2, grid_h//12*0), ((width-grid_w)//2+grid_w, grid_h//12*0), (0, 255, 0), 2)
    h2 = cv2.line(h1, ((width-grid_w)//2, grid_h//12*1), ((width-grid_w)//2+grid_w, grid_h//12*1), (0, 255, 0), 2)
    h3 = cv2.line(h2, ((width-grid_w)//2, grid_h//12*2), ((width-grid_w)//2+grid_w, grid_h//12*2), (0, 255, 0), 2)
    h4 = cv2.line(h3, ((width-grid_w)//2, grid_h//12*3), ((width-grid_w)//2+grid_w, grid_h//12*3), (0, 255, 0), 2)
    h5 = cv2.line(h4, ((width-grid_w)//2, grid_h//12*4), ((width-grid_w)//2+grid_w, grid_h//12*4), (0, 255, 0), 2)
    h6 = cv2.line(h5, ((width-grid_w)//2, grid_h//12*5), ((width-grid_w)//2+grid_w, grid_h//12*5), (0, 255, 0), 2)
    h7 = cv2.line(h6, ((width-grid_w)//2, grid_h//12*6), ((width-grid_w)//2+grid_w, grid_h//12*6), (0, 255, 0), 2)
    h8 = cv2.line(h7, ((width-grid_w)//2, grid_h//12*7), ((width-grid_w)//2+grid_w, grid_h//12*7), (0, 255, 0), 2)
    h9 = cv2.line(h8, ((width-grid_w)//2, grid_h//12*8), ((width-grid_w)//2+grid_w, grid_h//12*8), (0, 255, 0), 2)
    h10 = cv2.line(h9, ((width-grid_w)//2, grid_h//12*9), ((width-grid_w)//2+grid_w, grid_h//12*9), (0, 255, 0), 2)
    h11 = cv2.line(h10, ((width-grid_w)//2, grid_h//12*10), ((width-grid_w)//2+grid_w, grid_h//12*10), (0, 255, 0), 2)
    h12 = cv2.line(h11, ((width-grid_w)//2, grid_h//12*11), ((width-grid_w)//2+grid_w, grid_h//12*11), (0, 255, 0), 2)
    h13 = cv2.line(h12, ((width-grid_w)//2, grid_h//12*12), ((width-grid_w)//2+grid_w, grid_h//12*12), (0, 255, 0), 2)

    v1 = cv2.line(h12, ((width-grid_w)//2,0), ((width-grid_w)//2, grid_h), (0, 255, 0), 2)
    v2 = cv2.line(v1, ((width-grid_w)//2+grid_w//12, 0), ((width-grid_w)//2+grid_w//12, grid_h), (0, 255, 0), 2)
    v3 = cv2.line(v2, ((width-grid_w)//2+grid_w//12*2, 0), ((width-grid_w)//2+grid_w//12*2, grid_h), (0, 255, 0), 2)
    v4 = cv2.line(v3, ((width-grid_w)//2+grid_w//12*3, 0), ((width-grid_w)//2+grid_w//12*3, grid_h), (0, 255, 0), 2)
    v5 = cv2.line(v4, ((width-grid_w)//2+grid_w//12*4, 0), ((width-grid_w)//2+grid_w//12*4, grid_h), (0, 255, 0), 2)
    v6 = cv2.line(v5, ((width-grid_w)//2+grid_w//12*5, 0), ((width-grid_w)//2+grid_w//12*5, grid_h), (0, 255, 0), 2)
    v7 = cv2.line(v6, ((width-grid_w)//2+grid_w//12*6, 0), ((width-grid_w)//2+grid_w//12*6, grid_h), (0, 255, 0), 2)
    v8 = cv2.line(v7, ((width-grid_w)//2+grid_w//12*7, 0), ((width-grid_w)//2+grid_w//12*7, grid_h), (0, 255, 0), 2)
    v9 = cv2.line(v8, ((width-grid_w)//2+grid_w//12*8, 0), ((width-grid_w)//2+grid_w//12*8, grid_h), (0, 255, 0), 2)
    v10 = cv2.line(v9, ((width-grid_w)//2+grid_w//12*9, 0), ((width-grid_w)//2+grid_w//12*9, grid_h), (0, 255, 0), 2)
    v11 = cv2.line(v10, ((width-grid_w)//2+grid_w//12*10, 0), ((width-grid_w)//2+grid_w//12*10, grid_h), (0, 255, 0), 2)
    v12 = cv2.line(v11, ((width-grid_w)//2+grid_w//12*11, 0), ((width-grid_w)//2+grid_w//12*11, grid_h), (0, 255, 0), 2)
    v13 = cv2.line(v12, ((width-grid_w)//2+grid_w//12*12, 0), ((width-grid_w)//2+grid_w//12*12, grid_h), (0, 255, 0), 2)

    cv2.imshow('my WEBcam', v11)
    cv2.moveWindow('my WEBcam',0,0)
    if cv2.waitKey(1) & 0xff ==ord('q'):
        break
cap.release()




