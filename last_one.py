from logging import NullHandler
from random import random
from tkinter import E, Frame
from weakref import ref
from xmlrpc.client import boolean
import cv2
from cv2 import flip
import time
import random,ast
from threading import Thread, Event
from queue import Queue

from mingus.midi import fluidsynth
from mingus.containers import Note, Instrument
from mingus.containers import NoteContainer
import mingus.core.chords as chords
from mingus.containers import Bar
import mediapipe as mp
import multiprocessing

def Check_note(X,Y,pos):
    Tonal_Triadic =[[-1,  11,   -1,     7,     [7,9],9,    -1,   -1,  -1,     -1,   -1,   -1],
                    [11,  -1,   -1,    -1,     8,   [8,10],10,   -1,  -1,     -1,   -1,   -1],
                    [-1,  -1,   -1,    -1,    -1,    9,   [9,11],11,  -1,     -1,   -1,   -1],
                    [7,   -1,   -1,    -1,    -1,   -1,    10,  [10,0],0,     -1,   -1,   -1],
                    [[7,9],8,   -1,    -1,    -1,   -1,    -1,   11,   [11,1], 1,   -1,   -1],
                    [9,   [8,10],9,    -1,    -1,   -1,    -1,   -1,   0,     [0,2], 2,   -1],
                    [-1,  10,   [9,11],11,    10,   -1,    -1,   -1,  -1,      1,   [1,3], 3],
                    [-1,  -1,   11,    [10,0],11,   -1,    -1,   -1,  -1,     -1,    2,   [2,4]],
                    [-1,  -1,   -1,     0,    [11,1],0,    -1,   -1,  -1,     -1,   -1,    3],
                    [-1,  -1,   -1,    -1,     1,    [0,2], 1,   -1,  -1,     -1,   -1,   -1],
                    [-1,  -1,   -1,    -1,    -1,    2,    [1,3], 2,  -1,     -1,   -1,   -1],
                    [-1,  -1,   -1,    -1,    -1,   -1,     3,  [2,4], 3,     -1,   -1,   -1]]

    Tonal_Triadic_corre = Tonal_Triadic[X][Y]
    if pos == -1:
        if isinstance(Tonal_Triadic_corre,list):
            return Tonal_Triadic_corre[0]
        else:
            return Tonal_Triadic_corre
    else:
        return Tonal_Triadic_corre[pos]

def Note_refer(c_n):
    Tonal_Tri_init=[[-1,  11,   -1,     7,     [7,9],9,    -1,   -1,  -1,     -1,   -1,   -1],
                    [11,  -1,   -1,    -1,     8,   [8,10],10,   -1,  -1,     -1,   -1,   -1],
                    [-1,  -1,   -1,    -1,    -1,    9,   [9,11],11,  -1,     -1,   -1,   -1],
                    [7,   -1,   -1,    -1,    -1,   -1,    10,  [10,0],0,     -1,   -1,   -1],
                    [[7,9],8,   -1,    -1,    -1,   -1,    -1,   11,   [11,1], 1,   -1,   -1],
                    [9,   [8,10],9,    -1,    -1,   -1,    -1,   -1,   0,     [0,2], 2,   -1],
                    [-1,  10,   [9,11],11,    10,   -1,    -1,   -1,  -1,      1,   [1,3], 3],
                    [-1,  -1,   11,    [10,0],11,   -1,    -1,   -1,  -1,     -1,    2,   [2,4]],
                    [-1,  -1,   -1,     0,    [11,1],0,    -1,   -1,  -1,     -1,   -1,    3],
                    [-1,  -1,   -1,    -1,     1,    [0,2], 1,   -1,  -1,     -1,   -1,   -1],
                    [-1,  -1,   -1,    -1,    -1,    2,    [1,3], 2,  -1,     -1,   -1,   -1],
                    [-1,  -1,   -1,    -1,    -1,   -1,     3,  [2,4], 3,     -1,   -1,   -1]]
    
    Tonal_Tri_zero=[[-1,     -1,   -1,     -1,     [-1,-1],-1,    -1,   -1,  -1,     -1,   -1,   -1],
                    [-1,     -1,   -1,    -1,     -1,   [-1,-1],-1,   -1,  -1,     -1,   -1,   -1],
                    [-1,     -1,   -1,    -1,    -1,    -1,   [-1,-1],-1,  -1,     -1,   -1,   -1],
                    [-1,     -1,   -1,    -1,    -1,   -1,    -1,  [-1,-1],-1,     -1,   -1,   -1],
                    [[-1,-1],-1,   -1,    -1,    -1,   -1,    -1,   -1,   [-1,-1], -1,   -1,   -1],
                    [-1,     [-1,-1],-1,    -1,    -1,   -1,    -1,   -1,   -1,     [-1,-1], -1,   -1],
                    [-1,     -1,   [-1,-1],-1,    -1,   -1,    -1,   -1,  -1,      -1,   [-1,-1], -1],
                    [-1,     -1,   -1,    [-1,-1],-1,   -1,    -1,   -1,  -1,     -1,    -1,   [-1,-1]],
                    [-1,     -1,   -1,     -1,    [-1,-1],-1,    -1,   -1,  -1,     -1,   -1,    -1],
                    [-1,     -1,   -1,    -1,     -1,    [-1,-1], 1,   -1,  -1,     -1,   -1,   -1],
                    [-1,     -1,   -1,    -1,    -1,    -1,    [-1,-1], -1,  -1,     -1,   -1,   -1],
                    [-1,     -1,   -1,    -1,    -1,   -1,     -1,  [-1,-1], -1,     -1,   -1,   -1]]
    
    Chord_with_pitch = [[[0,4,0]],  #C
                        [[1,5,0]],  #Db
                        [[2,6,0]],  #D
                        [[3,7,0]],  #Eb
                        [[4,8,0]],  #E
                        [[0,5],[5,9,0]], #F
                        [[1,6],[6,10,0]], #F*
                        [[1,6]],    #Gb
                        [[2,7],[7,11,0]], #G
                        [[3,8]],    #Ab
                        [[4,9]],    #A
                        [[5,10]],   #Bb
                        [[6,11]],   #B
                        [[0,3],[3,7,1]],  #c
                        [[4,8,1]],  #c*
                        [[1,4]],    #db
                        [[2,5],[5,9,1]], #d
                        [[3,6],[6,10,1]], #eb
                        [[4,7],[7,11,1]], #e
                        [[1,5,1],[5,8]],  #f
                        [[6,9]],    #f*
                        [[7,10]],   #g
                        [[8,11]],   #g*
                        [[0,4,1]],  #a
                        [[2,6,1]]   #b
                        ]
    Follow_chord = {0: [2,5,8,16,18,23],
                    1: [0],
                    2: [0,5,8,18,23],
                    3: [-1], 4: [-1],
                    5: [0,8,16,23,24],
                    6: [-1], 7: [-1],
                    8: [0,5,11,16,18,23],
                    9: [-1],
                    10:[0,5,16],
                    11:[-1],
                    12:[0,5,11],
                    13:[-1],14:[-1],15:[-1],
                    16:[0,5,8,18,23],
                    17:[-1],
                    18:[0,5,8,11,16,23],
                    19:[0],20:[-1],21:[5],22:[-1],
                    23:[0,5,8,16,18],
                    24:[-1]}

    if c_n != -1:
        fol_chords = Follow_chord[c_n]
        for pitchs_chords in fol_chords:
            if pitchs_chords != -1:
                pitchs_cor = Chord_with_pitch[pitchs_chords]
                for pitch in pitchs_cor:
                    pitch_x = -1
                    pitch_y = -1
                    if pitch[0]<pitch[1]:
                        pitch_x = pitch[0]
                        pitch_y = pitch[1]
                    else:
                        pitch_x = pitch[1]
                        pitch_y = pitch[0]
                    if abs(pitch_x-pitch_y)==4:
                        Tonal_Tri_zero[pitch_x][pitch_y][pitch[2]] = Tonal_Tri_init[pitch_x][pitch_y][pitch[2]]
                    else:
                        Tonal_Tri_zero[pitch_x][pitch_y] = Tonal_Tri_init[pitch_x][pitch_y]
        return Tonal_Tri_zero
    else:
        return Tonal_Tri_init

def Search_refer(X,Y,refer_tonal):
    note_num = refer_tonal[X][Y]
    check_result = False
    check_pos = -1

    if isinstance(note_num,list):
        for i in range(0,2):
            if note_num[i] != -1:
                check_result = True
                check_pos = i
    else:
        if note_num != -1: 
            check_result = True

    return check_result, check_pos  
        

def corres_chord(X, Y, pos):
    Chord_Major_list = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'] #12
    Chord_Minor_list = ['C', 'C#', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A','B'] #24    
    Cor_Chord_one =[[-1,     11,     -1,     13,     [ 0, 23],23,     -1,   -1,       -1,     -1,     -1,     -1],
                    [ 7,     -1,     -1,     -1,     15,      [ 1,19], 6,     -1,     -1,     -1,     -1,     -1],
                    [-1,     -1,     -1,     -1,     -1,      16,     [ 2,24], 8,     -1,     -1,     -1,     -1],
                    [13,     -1,     -1,     -1,     -1,      -1,     17,     [ 3,13], 9,     -1,     -1,     -1],
                    [[ 0,23],15,     -1,     -1,     -1,      -1,     -1,     18,     [ 4,14],10,     -1,     -1],
                    [23,     [ 1,19],16,     -1,     -1,      -1,     -1,     -1,     19,     [ 5,16],12,     -1],
                    [-1,      6,     [ 2,24],17,     10,      -1,     -1,     -1,     -1,     20,     [ 6,17],12],
                    [-1,     -1,      8,     [ 3,13],18,      -1,     -1,     -1,     -1,     -1,     21,     [ 7,18]],
                    [-1,     -1,     -1,      9,     [ 4,14], 19,     -1,     -1,     -1,     -1,     -1,     22],
                    [-1,     -1,     -1,     -1,     10,      [ 5,16],20,     -1,     -1,     -1,     -1,     -1],
                    [-1,     -1,     -1,     -1,     -1,      12,     [ 6,17],21,     -1,     -1,     -1,     -1],
                    [-1,     -1,     -1,     -1,     -1,      -1,     12,     [ 7,18],22,     -1,     -1,     -1]]

    chord_corr = Cor_Chord_one[X][Y]
    chord_num = -1
    if pos == -1:
        if(isinstance(chord_corr,list)):
            pos = random.randint(0,1)
            chord_num = chord_corr[pos]
        else:
            chord_num = chord_corr
    else:
        chord_num = chord_corr[pos]

    if chord_num < 13:
        chord = chords.major_triad(Chord_Major_list[chord_num])
    else:
        chord = chords.minor_triad(Chord_Minor_list[chord_num-13])

    return chord, pos, chord_num


def get_bar(X,Y,pos_chord):
    b = Bar()
    b.set_meter([4,1])
    note_list = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    choice_list = [note_list[X],note_list[Y],note_list[Check_note(X,Y,pos_chord)]]
    b.place_notes("{}-4".format(choice_list[random.randint(0,2)]),100)
    return b
 
    
def coordinate_to_Note(X, Y, pos):
    chord_one, pos_chord, c_n, = corres_chord(X, Y, pos)
    b = get_bar(X, Y, pos_chord)
    c = NoteContainer([Note(chord_one[0], 3), Note(chord_one[1], 3), Note(chord_one[2], 3)])
    return b,c,c_n,pos_chord


fluidsynth.init("test_bian.sf2")
fluidsynth.set_instrument(1, 2)

width=1440
height=840

# following the order above
# record_x, record_y, record_c, record_b, record_c_n, record_pos, record_refer
record_list = [-1,-1,0,0,-1,-1,-1]

# init X Y
record_xy = [-1,-1]

def position2coordinate(pos_x, pos_y):
    #print('X: ',pos)
    if pos_x > 420 and pos_x < 1020 and pos_y > 0 and pos_y < 600:
        return (pos_x-420)//50, pos_y//50
    else:
        return -1, -1


class mpHands:
    def __init__(self,maxHands=2,tol1=.5,tol2=.5):
      #  print([maxHands,tol1,tol2])
        self.hands=mp.solutions.hands.Hands(static_image_mode=False,max_num_hands=maxHands,min_detection_confidence=tol1)
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

def detect():
    width=1440
    height=840
    grid_w = 600
    grid_h = 600
    cap = cv2.VideoCapture(0)
    findHands=mpHands(2)

    # vertical = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # horizontal = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    inds = [8]

    X = record_xy[0]
    Y = record_xy[1]
    #X = -1
    #Y = -1

    while True:
        ignore, frame = cap.read()
        frame=cv2.resize(frame,(width,height))
        flip_img = cv2.flip(frame,1)
        handData, handsType=findHands.Marks(flip_img)
        for hand,handType in zip(handData,handsType):
            if handType=='Right':
                handColor=(255,0,0)
                NA, Y = position2coordinate(hand[8][0], hand[8][1])
                cv2.putText(flip_img, '('+str(Y)+')', (hand[8][0]+10,hand[8][1]+10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

            if handType=='Left':
                handColor=(0,0,255)
                X, NA = position2coordinate(hand[8][0], hand[8][1])
                cv2.putText(flip_img, '('+str(X)+')', (hand[8][0]+10,hand[8][1]+10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
            
            with open('pos.txt','w') as ff:
                ff.write("{},{}".format(X,Y))
                # print("detect part {},{}".format(X,Y))
            # out_q.put(data)                    

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

        v1 = cv2.line(h13, ((width-grid_w)//2,0), ((width-grid_w)//2, grid_h), (0, 255, 0), 2)
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

        cv2.imshow('my WEBcam', v13)
        cv2.moveWindow('my WEBcam',0,0)
        if cv2.waitKey(1) & 0xff ==ord('q'):
            break
    cap.release()

def play_sound():
    while True:
    #[[0.0, 0.25, ['G-4']]]
        with open('bar.txt','r') as f:
            finfo= f.readlines()
        if len(finfo)>0:           
            x = ast.literal_eval(finfo[0])
            fluidsynth.play_Bar(x)
          #  time.sleep(1)
            
def music():
    # following the order
    # record_x, record_y, record_c, record_b, record_c_n, record_pos, record_refer
    
    record_list = [-1,-1,0,0,-1,-1,-1]
    X = -1
    Y = -1
    
    while True:
      #  print("STATUS:{},{}".format(current_status,kk))
        with open('pos.txt','r') as ff:
            posInfo = ff.readlines()
            if len(posInfo)>0:
                posInfo = posInfo[0].split(',')
                X = int(posInfo[0])
                Y = int(posInfo[1])
                time.sleep(0.01)
              
        first_play = 0
        if record_list[4] == -1:
            # record_c = chords.major_triad('C')
            # record_c_n = 0
            record_list[3], self_c, self_c_n,record_list[5]= coordinate_to_Note(X, Y,-1)
            if self_c_n != -1:
                major_c = chords.major_triad('C')
                record_list[2] = NoteContainer([Note(major_c[0], 3), Note(major_c[1], 3), Note(major_c[2], 3)])
                record_list[4] = 0
                fluidsynth.play_NoteContainer(record_list[2])
                fluidsynth.play_Bar(record_list[3])
             
                record_list[0] = X
                record_list[1] = Y
                first_play = 1
            record_list[6] = Note_refer(record_list[4])
        
        check_refer, refer_pos = Search_refer(X,Y,record_list[6])
     #   print("refer",check_refer)
        if check_refer != False:
            fluidsynth.stop_NoteContainer(record_list[2])
            record_list[3], record_list[2], record_list[4], record_list[5] = coordinate_to_Note(X, Y,refer_pos)
            record_list[0] = X
            record_list[1] = Y
            record_list[6] = Note_refer(record_list[4])
            fluidsynth.play_NoteContainer(record_list[2])
            fluidsynth.play_Bar(record_list[3])
        else:
            print("c_n:{} f_p:{}".format(record_list[4],first_play))
            if record_list[4] != -1 and first_play != 1 :
                # fluidsynth.stop_NoteContainer(record_list[2])
                # print("play")
                fluidsynth.play_NoteContainer(record_list[2])
                play_b = get_bar(record_list[0],record_list[1],record_list[5])
                fluidsynth.play_Bar(play_b)
       

def call_func(opt):
    if opt == 1:
        music()
    else:
        detect()

if __name__ == "__main__":
    t2 = Thread(target=music)
    t2.start()
    detect()
    t2.join()