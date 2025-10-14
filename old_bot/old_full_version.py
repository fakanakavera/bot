#import socket
import cv2
import numpy as np
from PIL import ImageGrab, Image
import time
import glob
import os
#import string
from ctypes import windll, Structure, c_ulong, byref
#import  itertools
#from math import sqrt
import random
import copy
#import pyautogui
#import pytesseract
#import keyboard
#import pyautogui
#import pytesseract
#import keyboard

# from posfunc import *
#from dir import *
#from c import FindTemplate

#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'


# *****************************************************************************************************************************
# *****************************************************************************************************************************
#                                                       SETTING UP VARIABLES

# DONE DIR = 'C:/Users/RoKeR/Dropbox/poker/2019/'
# DONE TOPLEFT = 'template/roomlogo/topleft/*.png'
POT = {'PLAYMONEY':'template/pot/playmoney.png', 'REALMONEY':'template/pot/realmoney.png', 'all':'template/pot/*.png'}
# DONE DEALER = 'template/dealer/*.png'
#HANDNUMBER = 'template/handnumber/0.png'    #NAO TEM
#HANDNUMBER_PAR = 'template/handnumber/'     #NAO TEM
HOLECARDS = 'template/read_hole/cards/*.png'
# DONE CHECKCARDS = 'template/check_cards/*.png'
WAITFASTFOLD = 'template/fastfold/*.png'
ALLIN = 'template/allin/allin.png'
PLTHINKING = 'template/plthinking/0.png'
# DONE PLACTION = 'template/plaction/*.png'
DECISIONREAD = ['template/decisionread/fold.png', 'template/decisionread/*.png', 'template/decisionread/raise.png']
WAITOWNACTION = 'template/waitownaction/0.png'
POT_NUMBERS = 'template/pot/numbers/*.png'
RAISE_NUMBERS = 'template/raise/numbers/*.png'
HAND_NUMBERS = 'template/handnumber/numbers/*.png'
WAITING_PL_THINK = 'template/waiting/0.png'

    
# SETTINGS
AUTOPLAYING = False
SELF_PLAYING = False
ZOOM = False
PLAYING_MONEY = True
LOGGING = False
DECISION_LOG = False
NPLAYERS = 6

if PLAYING_MONEY:
    POTMONEY = 'PLAYMONEY'
else:
    POTMONEY = 'REALMONEY'

if NPLAYERS == 9:
    SELF_SEAT = 5
elif NPLAYERS == 6:
    SELF_SEAT = 3


TABLE_X_SIZE = 1320 # TABLE SIZE X
TABLE_Y_SIZE = 940  # TABLE SIZE Y

GPIO_DICT = {'fold':11, '':11, 'call':12, 'check':12, 'bet':13, '3x':40, '40':16, '45':18, '55':38, '65':38, 'None':11}

CARDS_DIC = {'2':1,'3':2,'4':3,'5':4,'6':5,'7':6,'8':7,'9':8,'T':9,'J':10,'Q':11,'K':12,'A':13}
SUIT_DIC = {'c':0, 'd':1, 's':2, 'h':3}
DICT_CARDS = {0:'a', 1:'2', 2:'3', 3:'4', 4:'5', 5:'6', 6:'7', 7:'8', 8:'9', 9:'T', 10:'J', 11:'Q', 12:'K', 13:'A'}
DICT_SUIT = {0:'c', 1:'d', 2:'s', 3:'h'}

STREET = ['pre-flop', 'flop', 'turn', 'river',' HAND IS OVER ']

DICT_STR = {0:'NOTHING', 1:'BACKDOOR', 2:'GUTSHOT',         3:'1 SIDE STR DRAW', 4:'UPDOWN STR DRAW', 5:'STRAIGHT MADE', 6:'NOTHING'} #CARD_VAL
DICT_FSH = {0:'RAINBOW', 1:'BACKDOOR', 2:'DOUBLE BACKDOOR', 3:'FSH DRAW 2 NEEDED', 4:'FSH DRAW 1 NEEDED', 5:'FLUSH MADE', 6:'NOTHING'}
DICT_PAIR = {0:'NOTHING', 1:'PAIR', 2:'2 PAIRS', 3:'TRIPLE', 4:'FULLHOUSE', 5:'QUAD'}
DICT_FINAL = {0:'NOTHING', 1:'PAIR', 2:'2 PAIRS', 3:'TRIPLE', 4:'STRAIGHT', 5:'FLUSH', 6:'FULLHOUSE', 7:'QUAD', 8:'STRAIGHT FLUSH', 9:'ROYAL FLUSH'}

COD_ACTION = {'fold':0, 'check':1, 'call':2, 'bet':3, 'raise':3, 'allin':30}
            

HOLE_RANK = ['AAo','KKo','QQo','JJo','AKs','AQs','TTo','AKo','AJs','KQs','99o','ATs','AQo','KJs','88o','QJs','KTs','A9s','AJo','QTs','KQo','77o','JTs','A8s','K9s','ATo','A5s','A7s','KJo','66o','T9s','A4s','Q9s','J9s','QJo','A6s','55o','A3s','K8s','KTo','98s','T8s','K7s','A2s','87s','QTo','Q8s','44o','A9o','J8s','76s','JTo','97s','K6s','K5s','K4s','T7s','Q7s','K9o','65s','T9o','86s','A8o','J7s','33o','54s','Q6s','K3s','Q9o','75s','22o','J9o','64s','Q5s','K2s','96s','Q3s','J8o','98o','T8o','97o','A7o','T7o','Q4s','Q8o','J5s','T6o','75o','J4s','74s','K8o','86o','53s','K7o','63s','J6s','85o','T6s','76o','A6o','T2o','95s','84o','62o','T5s','95o','A5o','Q7o','T5o','87o','83o','65o','Q2s','94o','74o','54o','A4o','T4o','82o','64o','42o','J7o','93o','85s','73o','53o','T3o','63o','K6o','J6o','96o','92o','72o','52o','Q4o','K5o','J5o','43s','Q3o','43o','K4o','J4o','T4s','Q6o','Q2o','J3s','J3o','T3s','A3o','Q5o','J2o','84s','82s','42s','93s','73s','K3o','J2s','92s','52s','K2o','T2s','62s','32o','A2o','83s','94s','72s','32s']
HOLE_RANK = np.array(HOLE_RANK)

CARDS_LEFT = [0, 47.000, 46.000]

if NPLAYERS == 9: 
    POSITION_DICT =         {       0:'BB',     1:'SB',     2:'BU',     3:'CO',     4:'HJ',     5:'MP+1',   6:'MP',     7:'UTG+1',  8:'UTG'     }
    SETOR_DICT = { 0:'BLINDS',     1:'BLINDS',     2:'LATE',     3:'LATE',     4:'MID',     5:'MID',   6:'EARLY',     7:'EARLY',  8:'EARLY'     }
if NPLAYERS == 6:
    POSITION_DICT =         {       0:'BB',     1:'SB',     2:'BU',     3:'CO',     4:'MP',     5:'UTG'     }
    SETOR_DICT =            {0:'BLINDS',     1:'BLINDS',     2:'LATE',     3:'MID',     4:'MID',     5:'EARLY'}


# *****************************************************************************************************************************
# *****************************************************************************************************************************
#                                                           SETTING UP PRE-FLOP RANGE

ratio_a = {'passive':1.10, 'normal':1.00, 'agressive':0.85}

D = ['passive', 'normal', 'agressive']
tipo = {'passive':0, 'normal':1, 'agressive':2}

if NPLAYERS == 9:
    #                              |   BLINDS    |     LATE      |      MID      |        EARLY          |
    HAND_RANGE = {'open':{       #   BB     SB      BU      CO      HJ      MP1     MP      UTG1    UTG    
                        'passive':  [], 
                        'normal':   [95,    95,     59,     59,     69,     69,     84,     84,     84],  #******
                        'agressive':[]}, 
                  'call':{      #    BB     SB      BU      CO      HJ      MP1     MP      UTG1    UTG
                        'passive':  [], 
                        'normal':   [100,   100,    79,     80,     84,     89,     89,     93,     94],  #******
                        'agressive':[]},
                  'reraise':{   #    BB     SB      BU      CO      HJ      MP1     MP      UTG1    UTG  
                        'passive':  [], 
                        'normal':   [98,    98,     89,     91,     94,     95,     96,     96,     97],  #****** 
                        'agressive':[]},}

    for k, v in HAND_RANGE.items():  
        for p in range(0, NPLAYERS): 
            HAND_RANGE[k]['passive'].append(HAND_RANGE[k]['normal'][p]*ratio_a['passive'])
            HAND_RANGE[k]['agressive'].append(HAND_RANGE[k]['normal'][p]*ratio_a['agressive'])
       
    MIN_RANGE =  min(HAND_RANGE['open']['agressive'])

if NPLAYERS == 6:
    #                              |   BLINDS   |  LATE  |     MID      |  EARLY   |
    HAND_RANGE = {'open':{       #   BB     SB      BU      CO      MP      UTG    
                        'passive':  [], 
                        'normal':   [95,    95,     59,     59,     84,     84],  #******
                        'agressive':[]}, 
                  'call':{      #   BB     SB      BU      CO      MP      UTG  
                        'passive':  [], 
                        'normal':   [100,   100,    79,     80,     89,     94],  #******
                        'agressive':[]},
                  'reraise':{   #   BB     SB      BU      CO      MP      UTG  
                        'passive':  [], 
                        'normal':   [98,    98,     89,     91,     96,     97],  #****** 
                        'agressive':[]},}

    for k, v in HAND_RANGE.items():  
        for p in range(0, NPLAYERS): 
            HAND_RANGE[k]['passive'].append(HAND_RANGE[k]['normal'][p]*ratio_a['passive'])
            HAND_RANGE[k]['agressive'].append(HAND_RANGE[k]['normal'][p]*ratio_a['agressive'])
       
    MIN_RANGE =  min(HAND_RANGE['open']['agressive'])




# *****************************************************************************************************************************
# *****************************************************************************************************************************
#                                                           POSITIONS ON SCREEN

def action_pos(p):  #GOOD ENOUGH
    if NPLAYERS == 9:
        r = [(0, 0, 0, 0), (799, 56, 108, 78),(1040, 151, 108, 78), (1139, 359, 108, 78), (970, 564, 108, 78), (610, 611, 108, 78), (234, 564, 108, 78), (72, 359, 108, 78),(172, 151, 108, 78), (425, 56, 108, 78)]
    if NPLAYERS == 6:
        r = [(0, 0, 0, 0), (930, 88, 102, 78), (1137, 334, 102, 78), (930, 589, 102, 78), (283, 589, 102, 78), (77, 334, 102, 78), (283, 88, 102, 78)]
    return r[p]

def cards_pos(p):   #OPTIMAL +-5px    
    if NPLAYERS == 9:
        r = [(0, 0, 0, 0), (795, 142, 111, 99),(960, 215, 111, 99), (1024, 343, 99, 99), (905, 470, 111, 99), (607, 495, 111, 99), (300, 468, 111, 99), (189, 342, 111, 99), (260, 212, 111, 99), (442, 142, 111, 99)]
    if NPLAYERS == 6:
        r = [(0, 0, 0, 0), (898, 198, 55, 70), (1022, 362, 55, 70), (898, 487, 55, 70), (367, 487, 55, 70), (227, 362, 55, 70), (367, 198, 55, 70)]
    return r[p]

def dealer_pos():   #OPTIMAL +-5px 
    if NPLAYERS == 9:
        return [(0,0,0,0),(785, 166, 40, 37),(961, 209, 40, 37),(1068, 329, 40, 37),(1006, 484, 40, 37),(720, 539, 40, 37),(396, 539, 40, 37),(225, 439, 40, 37),(233, 284, 40, 37),(388, 184, 40, 37)]
    if NPLAYERS == 6:
        return [(0, 0, 0, 0), (825, 167, 40, 37), (1051, 311, 40, 37), (975, 504, 40, 37), (430, 539, 40, 37), (216, 422, 40, 37),(315, 214, 40, 37)]
    
def hud_pos(p):
    x = 290
    y = 110
    if NPLAYERS == 9:
        r = [(0, 0, 0, 0), (882,33,x,y), (1100,230,x,y), (1080,440,x,y), (1040,575,x,y), (737,620,x,y), (235,660,x,y), (1,470,x,y), (1,765,x,y), (140,40,x,y)]
    if NPLAYERS == 6:
        r = [(0, 0, 0, 0), (882,33,x,y), (1100,230,x,y), (1080,440,x,y), (1040,575,x,y), (737,620,x,y), (235,660,x,y), (1,470,x,y), (1,765,x,y), (140,40,x,y)]
    return r[p]

def hole_pos():
    x = 35
    y = 65
    if NPLAYERS == 9:
        r = [(565,590,x,y), (651,590,x,y)]
    if NPLAYERS == 6:
        r = [(914, 555, 79, 109), (998, 554, 79, 110)]
    return r

def board_pos(t=False):
    x = 40
    y = 60
    if not t:
        yy = 285
        xx = 90
        start = 350
        r = [(0,0,0,0), (start+xx,yy,x,y), (start+(xx*2),yy,x,y), (start+(xx*3),yy,x,y), (start+(xx*4),yy,x,y), (start+(xx*5),yy,x,y)]
    
    if t:
        yy = 220
        y2 = 350
        xx = 90
        start = 350
        r = [(0,0,0,0), (start+xx,yy,x,y), (start+(xx*2),yy,x,y), (start+(xx*3),yy,x,y), (start+(xx*4),yy,x,y), (start+(xx*5),yy,x,y)],[(0,0,0,0), (start+xx,y2,x,y), (start+(xx*2),y2,x,y), (start+(xx*3),y2,x,y), (start+(xx*4),y2,x,y), (start+(xx*5),y2,x,y)]
    return r

def pot_pos():
    if NPLAYERS == 9:
        r = (600,50,125,40)
    if NPLAYERS == 6:
        r = (600,50,125,40)
    return r

def decision_pos(): #GOOD ENOUGH
    x = 210
    y = 95
    if NPLAYERS == 9:
        r = [(653,843,x,y), (871,842,x,y), (1092,843,x,y)]
    if NPLAYERS == 6:
        r = [(653,843,x,y), (871,842,x,y), (1092,843,x,y)]
    return r

def allin_pos(p):
    x = 250
    y = 60
    if NPLAYERS == 9:
        r = [(0,0,x,y), (922,75,x,y), (1135,264,x,y), (1113,494,x,y), (1098,621,x,y), (738,645,x,y), (361,619,x,y), (46,493,x,y), (22,263,x,y), (248,76,x,y)]
    if NPLAYERS == 6:
        r = [(0,0,x,y), (760, 44, 155, 66), (1119, 442, 163, 59), (755, 609, 163, 69), (394, 609, 162, 68), (31, 441, 163, 64), (388, 42, 168, 69)]
    return r[p]

def wait_pos():
    if NPLAYERS == 9:
        return [(0,0,0,0),(868, 42, 62, 60),(1109, 137, 62, 60),(1208, 345, 62, 60),(1039, 550, 62, 60),(679, 597, 62, 60),(303, 550, 62, 60),(141, 345, 62, 60),(241, 137, 62, 60),(494, 42, 62, 60)]
    if NPLAYERS == 6:
        return [(0, 0, 0, 0), (998, 21, 62, 60), (1205, 267, 62, 60), (998, 522, 62, 60), (351, 522, 62, 60), (145, 267, 62, 60), (351, 21, 62, 60)]

def run_twice():
    if NPLAYERS == 9:
        return (545, 158, 231, 60)
    
def raise_size_pos():
    if NPLAYERS == 9:
        return [1120, 890, 180, 44]
    if NPLAYERS == 6:
        return [1120, 890, 180, 44]

def to_call_pos():
    if NPLAYERS == 9:
        return [874, 887, 205, 43]
    if NPLAYERS == 6:
        return [874, 887, 205, 43]
    
def to_allin_call_pos():
    if NPLAYERS == 9:
        return [1096, 886, 201, 42]
    if NPLAYERS == 6:
        return [1096, 886, 201, 42]
    
def hand_num_pos():
    return (78, 33, 149, 28)





# *****************************************************************************************************************************
# *****************************************************************************************************************************
#                                                   RASPBERRY PI CONNECTION

# if AUTOPLAYING:
#     #GPIO_DICT = {'fold':11, '':11, 'call':12, 'check':12, 'bet':13, '3x':15, '40':16, '45':18, '55':38, '65':40}
#     GPIO_DICT = {'fold':11, '':11, 'call':12, 'check':12, 'bet':13, '3x':40, '40':16, '45':18, '55':38, '65':38}

#     def sendAcao(a):
#         sock.send(str.encode(str(a)))
        
#     def SendFastFold():
#         sock.send(str.encode(str('99')))

#     def setSocket():
#         host = '192.168.0.200'
#         port = 5560

#         sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         sock.connect((host, port))
#         return sock
        
#     sock = setSocket()    


# *****************************************************************************************************************************
# *****************************************************************************************************************************
#                                                   FUNCTIONS

def show(img): # DONE
    #SHOW IMG, 'Q' TO DESTROY
    while(True):
        cv2.imshow('output',img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

def GetFrame(x1=0, y1=0, x2=1920, y2=1080, coor=False):
    if not coor:
        coordenadas = bbox=(x1, y1, (x1+x2), (y1+y2))
    else:
        coordenadas = bbox=(coor[0], coor[1], (coor[0]+coor[2]), (coor[1]+coor[3]))

    img = np.array(ImageGrab.grab(coordenadas)) 
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #ajusta RGB

def FindTemplate(template_input, wanted='bool', frame=False, threshold=0.9, crop_pos=False, tp=(0, 0)):
    result = False
    if wanted == 'until_not_in_same_position':
        old_pt = -1

    while not result:
        if isinstance(frame, list):
            try:
                if TP > -1:
                    img = np.array(ImageGrab.grab(bbox=(TABLECOOR)))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # ajusta RGB
            except:
                img = np.array(ImageGrab.grab(bbox=(0, 0, 1920, 1080)))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # ajusta RGB

        else:
            if isinstance(frame, str):
                img = cv2.imread(frame)

            elif isinstance(frame, np.ndarray):
                img = frame

            else:
                if tp == (0,0):
                    img = GetFrame()
                else:
                    img = GetFrame(coor=tp)

        if isinstance(template_input, str):
            template_input = glob.glob(template_input)


        if isinstance(template_input, list):
            for template_temp in template_input:
                template = cv2.imread(template_temp)
                h, w = template.shape[:-1]
                res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res >= threshold)
                for pt in zip(*loc[::-1]):
                    if wanted is 'show':
                        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)
                        print(os.path.basename(template_temp)[:-4])
                        print("---")
                        if len(template_input) > 1:
                            try:
                                template_input.remove(template_temp)
                            except:
                                pass
                            print(len(template_input))
                            print("---")
                            print(template_temp)
                            print("--------------")
                            
                        else:
                            return True

                        #show(img)
                        #return True

                    if wanted == 'position':
                        return pt

                    if wanted == 'pthw':
                        return (pt[0], pt[1], h, w)

                    elif wanted == 'center':
                        return ((pt[0] + (w / 2)), (pt[1] + (h / 2)))

                    elif wanted == 'until_not_in_same_position':
                        if old_pt == -1:
                            old_pt = pt

                        elif (pt[0] - old_pt[0]) > 10 or (pt[0] - old_pt[0]) < -10:
                            return (int((pt[0] + (w / 2))), int((pt[1] + (h / 2))))


                    elif wanted == 'frame':
                        # cv2.rectangle(img, (pt[0] - 2, pt[1] - 2), ((pt[0] + w + 2), (pt[1] + 2 + h)), (0, 0, 255), 1)
                        # ((pt[0] + w), (pt[1]-2), (w*4), (h+4))
                        return img

                    elif wanted == 'filename' or wanted == 'filename_or_nada':
                        return os.path.basename(template_temp)[:-4]

                    elif wanted == 'bool':
                        return True

                    elif wanted == 'read_num':
                        #print(template_temp)
                        return os.path.basename(template_temp)[:-4], (pt[0], pt[1], w, h)

                    elif wanted == 'cropped_from_full':  # crop_pos = [W, H, +W, +H]
                        return img[crop_pos[1]:(crop_pos[1] + crop_pos[3]), crop_pos[0]:(crop_pos[0] + crop_pos[2])]

                    elif wanted == 'cropped_from_position':
                        # cv2.rectangle(img, (
                        #     ((pt[0] * crop_pos['start_ptx_ratio']) + (w * crop_pos['start_w_ratio']) + crop_pos['start_fix_x']),
                        #     ((pt[1] * crop_pos['start_pty_ratio']) + (h * crop_pos['start_h_ratio']) + crop_pos['start_fix_y'])),
                        #     (((pt[0] * crop_pos['end_ptx_ratio']) + (w * crop_pos['end_w_ratio']) + crop_pos['end_fix_x']),
                        #      ((pt[1] * crop_pos['end_pty_ratio']) + (h * crop_pos['end_h_ratio']) + crop_pos['end_fix_y'])), (0, 255, 0), 1)
                        # return img
                        return img[((pt[1] * crop_pos['start_pty_ratio']) + (h * crop_pos['start_h_ratio']) + crop_pos['start_fix_y']):
                                   ((pt[1] * crop_pos['end_pty_ratio']) + (h * crop_pos['end_h_ratio']) + crop_pos['end_fix_y']),

                                   ((pt[0] * crop_pos['start_ptx_ratio']) + (w * crop_pos['start_w_ratio']) + crop_pos['start_fix_x']):
                                   ((pt[0] * crop_pos['end_ptx_ratio']) + (w * crop_pos['end_w_ratio']) + crop_pos['end_fix_x'])]

            else:
                if wanted == 'bool' or wanted == 'position':
                    return False

                if wanted == 'filename_or_nada':
                    return 'nada'

                if wanted == 'filename':
                    return False

                if wanted == 'read_num':
                    return -1, (-1, -1, -1, -1)

                if wanted == 'cropped_from_position':
                    frame = GetFrame(coor=tp)    

                if wanted == 'pthw':
                    return False

def inittopleft(x1=0, y1=0, x2=1920, y2=1080):
    temp_list = glob.glob(DIR+TOPLEFT) #dir pra templates
    result = FindTemplate(temp_list, wanted='position')
  
    tp = result
    tablecoor = (tp[0], tp[1], (tp[0]+TABLE_X_SIZE), (tp[1]+TABLE_Y_SIZE))
    crop = (tablecoor[0], tablecoor[1], (tablecoor[2]-tablecoor[0]), (tablecoor[3]-tablecoor[1]))
    img = FindTemplate(temp_list, wanted='cropped_from_full', crop_pos=crop)
    #cv2.imwrite('logs/t.png', img)
    return tp, tablecoor


TP, TABLECOOR = inittopleft() # CANT MOVE
print('TP: '+str(TP)+' | TABLECOOR: '+str(TABLECOOR))


def crop_img(img, cpc):  # USING IN THE NEW VERSION - DONE
    # cpc[0] = x inicial, [1] = y inicial, [2] = x added, [3] = y added
    return img[cpc[1]:(cpc[1] + cpc[3]), cpc[0]:(cpc[0] + cpc[2])]

def ReadWithTemplate(template_input, frame_input, crop_pos=False):
    x_sum = 0
    if crop_pos:
        frame_x = crop_img(frame_input, crop_pos)
    else:
        frame_x = frame_input #frame_x nunca sera modificado.

    result = ''
    out = ''
    x_start = 0

    temp = glob.glob(template_input)
    max_x = 0

    for x in temp:
        if max_x < cv2.imread(x).shape[1]:
            max_x = cv2.imread(x).shape[1]

    frame = frame_x[0:frame_x.shape[0], 0:int(max_x * 1.3)]

    loopcount = 1
    while True:
        try:
            result, pt = FindTemplate(template_input, frame=frame, wanted='read_num', threshold=0.90)
            #print(template_input)

        except:
            break
        # print(str(result) + ' ' + str(pt))
        if result and pt:
            if result == -1 and out == '':
                frame = frame_x[0:frame_x.shape[0], 0:int(max_x * 1.3 * loopcount)]
                loopcount += 0.5
                # show(frame)


            if pt[0] > -1:
                frame = frame_x[0:frame_x.shape[0], (pt[0] + pt[2] + x_sum):(pt[0] + pt[2] + x_sum + int(max_x * 1.5))]
                x_start = (pt[0] + pt[2] + x_sum)
                x_sum += (pt[0] + pt[2])

            elif out != '':
                break

            if result != 'c' and result != -1:
                out += result

        elif (frame.shape[1] + 2) < frame_x.shape[1]:
            frame = frame_x[0:frame_x.shape[0], x_start:frame.shape[1] + int(max_x * 0.5)]

        else:
            break

    return out

def ReadPot(template_input=False, frame_input=False):
    #DIR = 'C:/Users/RoKeR/Dropbox/poker/2019/'

    if isinstance(frame_input, np.ndarray):
        frame_s = frame_input
    else:
        frame_s = GetFrame()

    crop = {'start_ptx_ratio': 1, 'start_w_ratio': 1, 'start_fix_x': 0, 'end_ptx_ratio': 1, 'end_w_ratio': 1, 'end_fix_x': 100,
            'start_pty_ratio': 1, 'start_h_ratio': 0, 'start_fix_y': 0, 'end_pty_ratio': 1, 'end_h_ratio': 1, 'end_fix_y': 0}

    frame_s = FindTemplate(DIR+POT[POTMONEY], frame=frame_s, wanted='cropped_from_position', threshold=0.9, crop_pos=crop)
    #cv2.imwrite("C:/Users/RoKeR/Dropbox/poker/2019/logs/"+str(int(time.time()))+".png", frame_s)
    #template_list = 'C:/Users/RoKeR/Dropbox/poker/2019/template/pot/numbers/*.png'

    return ReadWithTemplate(template_input=DIR+POT_NUMBERS, frame_input=frame_s)

def ReadRaise(template_input=False, frame_input=False):
    # DIR = 'C:/Users/RoKeR/Dropbox/poker/2019/logs/'
    # FILEDIR = '0.png'

    if isinstance(frame_input, np.ndarray):
        frame_s = frame_input
    else:
        frame_s = GetFrame(coor=TABLECOOR)

    frame_s = crop_img(frame_s, (1097, 890, 250, 25))

    return ReadWithTemplate(template_input=DIR+RAISE_NUMBERS, frame_input=frame_s)

def ReadHandNumber(template_input=False, frame_input=False):
    if isinstance(frame_input, np.ndarray):
        frame_s = frame_input
    else:
        frame_s = GetFrame(coor=frame_input)

    crop = {'start_ptx_ratio': 1, 'start_w_ratio': 1, 'start_fix_x': 0, 'end_ptx_ratio': 1, 'end_w_ratio': 1, 'end_fix_x': 200,
            'start_pty_ratio': 1, 'start_h_ratio': 0, 'start_fix_y': 0, 'end_pty_ratio': 1, 'end_h_ratio': 1, 'end_fix_y': 0}

    frame_s = FindTemplate(DIR + HANDNUMBER_PAR + 'hand.png', frame=frame_s, wanted='cropped_from_position', threshold=0.9, crop_pos=crop, tp=frame_input)
    #cv2.imwrite("C:/Users/RoKeR/Dropbox/poker/2019/logs/"+str(int(time.time()))+".png", frame_s)
    frame = FindTemplate(DIR + HANDNUMBER_PAR + 'hash.png', frame=frame_s, wanted='cropped_from_position', threshold=0.9, crop_pos=crop, tp=frame_input)
    # saved = frame
    #show(frame)
    # template_list = 'C:/Users/RoKeR/Dropbox/poker/2019/template/handnumber/numbers/*.png'

    return ReadWithTemplate(template_input=DIR+HAND_NUMBERS, frame_input=frame)

def fnk(file, string):
    with open(DIR+file, 'a') as fl:
        fl.write(string+'\n')
        fl.flush()

def pl_action(img): #retorna string da acao. [ bet call check fold ]   ESPERAR 3 SECS PRA LER DENOVO PRA NAO LER ACAO DA RODADA ANTERIOR
    result = FindTemplate(template_input=DIR+PLACTION, wanted='filename_or_nada', frame=img, threshold=0.95)
    r = ['sitout', 'empty', 'take', 'r']

    if result in r:
        return 'fold'
    else:
        return result

def read_hole(frame):
    r = FindTemplate(template_input=DIR+HOLECARDS, wanted='filename_or_nada', threshold=0.95, frame=frame)

    if r == 'nada':
        r = False
        
    return r




# MUDAR PARA NEW TEMPLATE READ
# 0 = LEFT. RETORNA FOLD OU 'NADA'
# 1 = MID. RETORNA 'CHECK' OU CALL AMMOUNT(CROP BOT PART AND READ WITH TEMPLATE READ, WILL NEED TEMPLATE OF ALL NUMBERS)
# 2 = RIGHT. RETORNA MIN RAISE AMMOUNT(CROP BOT PART AND READ WITH TEMPALTE READ, PROB CAN USE THE SAME TEMPLATES FROM CALL AMMOUNT)
def DecisionRead(d): # d = [ 0 = left, 1 = mid, 2 = right ] !!!!!!!!!! MELHORAR !!!!!!!!!!
    temp_list = glob.glob(DIR+DECISIONREAD[d])
    
    frame = GetFrame(coor=TABLECOOR)

    if d == 0:
        return FindTemplate(template_input=temp_list, wanted='filename', frame=frame, threshold=0.9)

    elif d == 1:
        if FindTemplate(template_input=temp_list, wanted='filename_or_nada', frame=frame, threshold=0.9) == 'check':
            return 'check'
        else:
            # template_list = 'C:/Users/RoKeR/Dropbox/poker/2019/template/raise/numbers/*.png'
            pt = FindTemplate(template_input=temp_list, frame=frame, wanted='pthw', threshold=0.9)
            return ReadWithTemplate(template_input=DIR+RAISE_NUMBERS, frame_input=crop_img(frame, (pt[0], (pt[1]+pt[2]), pt[3], pt[2])))

    elif d == 2: 
            # template_list = 'C:/Users/RoKeR/Dropbox/poker/2019/template/raise/numbers/*.png'
            pt = FindTemplate(template_input=temp_list, frame=frame, wanted='pthw', threshold=0.9)
            if pt:
                return ReadWithTemplate(template_input=DIR+RAISE_NUMBERS, frame_input=crop_img(frame, (pt[0], (pt[1]+pt[2]), pt[3], pt[2])))
            else:
                return False

def hand_order(d, in_hand):
    b = []
    c = []
    if d > 0 and d < (NPLAYERS+1) and len(in_hand) > 1:
        b = list(in_hand[int(((in_hand.index(d)+3)-(int(((in_hand.index(d)+2)/len(in_hand)))*len(in_hand)))):]) + list(in_hand[:int(((in_hand.index(d)+3)-(int(((in_hand.index(d)+2)/len(in_hand)))*len(in_hand))))])
        c = list(in_hand[int(((in_hand.index(d)+1)-(int(((in_hand.index(d)+1)/len(in_hand)))*len(in_hand)))):]) + list(in_hand[:int(((in_hand.index(d)+1)-(int(((in_hand.index(d)+1)/len(in_hand)))*len(in_hand))))])  
        return [b, c]

    else:
        return False

def arruma_list(d, lista):
    #print 'LISTA ',lista
    #print 'ARRUMADA ', list(lista[lista.index(d):]) + list(lista[:lista.index(d)])
    return list(lista[lista.index(d):]) + list(lista[:lista.index(d)])

def pl_thinking():
    result = FindTemplate(template_input=DIR+PLTHINKING, wanted='position', threshold=0.9, frame=GetFrame(coor=TABLECOOR))

    if result != False:
        return find_player_num(result, wait_pos())
    else:
        return 'pass'

def check_coor(p, r):       # USING IN THE NEW VERSION - DONE
    if p[0] > r[0] and p[0] < (r[0]+r[2]) and p[1] > r[1] and p[1] < (r[1]+r[3]):
        return True
    else:
        return False

def find_player_num(p, r):    
    for x in range(1,(NPLAYERS+1)):
        if check_coor(p, r[x]):
            r = x
            break
    return r

def wait_pl_think(p):
    temp_list = glob.glob(DIR+WAITING_PL_THINK) #dir pra templates
    wait = wait_pos()
    t = 0
    #TP = top_left()

    while True:
        frame = GetFrame(coor=TABLECOOR)
        c = FindTemplate(wanted='position', template_input=temp_list, threshold=0.95, frame=frame)
        if c:
            if not check_coor(c, wait[p]):
                return True
        else:
            t += 1
            if t > 3:
                return True

def break_cards(cards):
    hole_c = []
    hole_s = []
    
    for x in range(len(cards)):
        hole_c.append(cards[x][0:1])
        hole_s.append(cards[x][1:2])

    return hole_c, hole_s
    
def break_hole(hole): # USING IN THE NEW VERSION - DONE
    temp_card = ['','']
    temp_suit = ['','']
    
    for x in range(len(hole)):
        temp_card[x] = hole[x][0:1]
        temp_suit[x] = hole[x][1:2]

    return temp_card, temp_suit

def hand_outs(hand_in):  # return hand now / outs 
    hand_cards, hand_suit = break_cards(hand_in)
    return_val = ''
    outs = 0
    pair_check = 0
    card_val = 0
    suit_check = 0
    hand_val = 0
    # out vals
    pair_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    suit_count = [0, 0, 0, 0]
    cards_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    board_str_counter = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    board_cards_val = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    gutshot = [False, False, False, False, False, False, False, False, False, False, False, False, False, False]

    if True:                            # BOARD SUIT VALUES.    suit_check
        for x in hand_suit:            
            suit_count[SUIT_DIC[x]] += 1
        
        
        if len(hand_suit) == 5:                # FLOP
            if suit_count.count(5) == 1:
                suit_check = 5
            elif suit_count.count(4) == 1:
                suit_check = 4
                outs += 9
            elif suit_count.count(3) == 1:
                suit_check = 3
            elif suit_count.count(2) == 2:
                suit_check = 2
            elif suit_count.count(2) == 1:
                suit_check = 1
            else:
                suit_check = 0
                
        
        if len(hand_suit) == 6:                # TURN
            if suit_count.count(5) == 1 or suit_count.count(6) == 1:
                suit_check = 5
            elif suit_count.count(4) == 1:
                suit_check = 4
                outs += 9
            else:
                suit_check = 0
                
        if len(hand_suit) == 7:                # RIVER
            if suit_count.count(5) == 1 or suit_count.count(6) == 1 or suit_count.count(7) == 1:
                suit_check = 5
            else:
                suit_check = 0
    # BOARD SUIT VALUES END.
        flush_naipe = fmax_index(suit_count)

    if True:                            # BOARD CARDS VALUES.   card_val
        for x in hand_cards:
            cards_count[CARDS_DIC[x]] += 1
            if x == 'A':
                cards_count[0] += 1
        
        for x in range(len(cards_count)):
            count = 0
            if cards_count[x] > 0:
                board_str_counter[x] += 1 
                while True:
                    count += 1
                    if (x+count) <= 13:
                        if cards_count[x+count] > 0:
                            board_str_counter[x] += 1  
                        elif x+count+1 <= 13:
                            if cards_count[x+count+1] > 0 and not gutshot[x]:
                                board_str_counter[x] += 2
                                count += 1
                                gutshot[x] = True
                            else:
                                break
                        else:
                            break
                    else:
                        break
                       
        straight_min_card = fmax_index(board_str_counter)

        for x in range(len(board_str_counter)):
            if board_str_counter[x] >= 5:
                if not gutshot[x]:
                    board_cards_val[x] = 5
                if gutshot[x]:
                    board_cards_val[x] = 2
 
            elif board_str_counter[x] == 4:
                if not gutshot[x]: 
                    if x < 10 and x > 0:
                        board_cards_val[x] = 4                      
                    else:
                        board_cards_val[x] = 3
                if gutshot[x]:
                    board_cards_val[x] = 1
                    
            elif board_str_counter[x] == 3 and not gutshot[x]: # > 0 and < 11 == espaco pra up down    
                    board_cards_val[x] = 1  
            else:
                    board_cards_val[x] = 0
        
        
        card_val = fmax(board_cards_val)
        if card_val == 2 or card_val == 3:
            outs += 4
        elif card_val == 4:
            outs += 8
        n_str = fmax_index(board_cards_val) # 9 == max straight

    # BOARD CARDS VALUES END.

    if True:                            # BOARD PAIRS VALUES.   pair_check
        for x in hand_cards:
            pair_count[CARDS_DIC[x]] += 1


        if fmax(pair_count) == 4:
            pair_check = 5
            pair_card = fmax_index(pair_count)
        elif fmax(pair_count) == 3 and fmax(pair_count, 2) == 2:
            pair_check = 4
            pair_card = fmax_index(pair_count)
        elif fmax(pair_count) == 3:
            pair_check = 3
            pair_card = fmax_index(pair_count)
        elif fmax(pair_count) == 2 and pair_count.count(2) == 2:
            pair_check = 2
            pair_card = fmax_index(pair_count)
        elif fmax(pair_count) == 2 and pair_count.count(2) == 1:
            pair_check = 1
            pair_card = fmax_index(pair_count)
        else:
            pair_check = 0
            pair_card = fmax_index(pair_count)
    # BOARD PAIRS VALUES END.
    
    if suit_check == 5 and card_val == 5:
        if n_str == 9:
            hand_val = 9
        elif n_str < 9:
            hand_val = 8
    elif pair_check == 5:
        hand_val = 7
    elif pair_check == 4:
        hand_val = 6
    elif suit_check == 5:
        hand_val = 5
    elif card_val == 5:
        hand_val = 4
    elif pair_check == 3:
        hand_val = 3
    elif pair_check == 2:
        hand_val = 2
    elif pair_check == 1:
        hand_val = 1
    else:
        hand_val = 0
    

    if pair_card != '':
        return_val = pair_card
        
    if outs == 17:
        outs -= 2
    elif outs == 13:
        outs -= 1

    return hand_val, outs, flush_naipe, straight_min_card, card_val

def hole_percent(hole):     # return ??? % hand in the rank. !!!!!!!!!! MELHORAR !!!!!!!!!!   USAR PRA RANGE.
    s = list()
    c = list()
    if CARDS_DIC[hole[0][0]] > CARDS_DIC[hole[1][0]]:
        c.append(hole[0][0])
        c.append(hole[1][0])
    else:
        c.append(hole[1][0])
        c.append(hole[0][0])

    s.append(hole[0][1])
    s.append(hole[1][1])

    f_hand = ''
    f_hand += c[0]
    f_hand += c[1]
    if s[0] == s[1]:
        f_hand += 's'
    else:
        f_hand += 'o'
    
    f_hand = np.where(HOLE_RANK == f_hand)  
    
    return int(100 - (f_hand[0][0] * 0.5952))

def split(a):
    return a.split(',')

def fmax(n, d=1):
    n = np.array(n) 
    n.sort()
    return n[d*-1]

def fmin(n):
    n = np.array(n)    
    return min(n)
    
def fmax_index(n, d=0):
    n = np.array(n)
    n = np.where(n == max(n))[0]    
    return n[0] 
    
def fmin_index(n):
    n = np.array(n)
    n = np.where(n == min(n))[0]    
    return n[0]

def fcount(n, d):
    n = np.array(n)
    return n

def board_read(board): # return type of boards. use to alter actions
    # RETURN  [ SUIT_VAL, SUIT_INDEX ],[ STR_VAL, STR_INDEX ]

    count = 0
    pair_check = 0
    pair_index = 0
    board_cards, board_suit = break_cards(board)
    suit_count = [0, 0, 0, 0]
    pair_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cards_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    board_str_counter = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    board_cards_val = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    gutshot = [False, False, False, False, False, False, False, False, False, False, False, False, False, False]
    
    if True:                            # BOARD SUIT VALUES.
        for x in board_suit:
            if x != '':
                suit_count[SUIT_DIC[x]] += 1
                count += 1
        
        if count == 3:                # FLOP
            if suit_count.count(1) == 3:
                suit_check = 0
            elif suit_count.count(2) == 1:
                suit_check = 1
            elif suit_count.count(3) == 1:
                suit_check = 3
        
        if count == 4:                # TURN
            if suit_count.count(2) == 2:
                suit_check = 2
            elif suit_count.count(3) == 1:
                suit_check = 3
            elif suit_count.count(4) == 1:
                suit_check = 4
            elif suit_count.count(2) == 1:
                suit_check = 1
            else:
                suit_check = 0
                
        if count == 5:                # RIVER
            if suit_count.count(5) == 1:
                suit_check = 5
            elif suit_count.count(4) == 1:
                suit_check = 4
            elif suit_count.count(3) == 1:
                suit_check = 3
            else:
                suit_check = 0
    # BOARD SUIT VALUES END.
    
    if True:                            # BOARD CARDS VALUES.
        for x in board_cards:
            if x != '':
                cards_count[CARDS_DIC[x]] += 1
                if x == 'A':
                    cards_count[0] += 1
                
        for x in range(len(cards_count)):
            count = 0
            if cards_count[x] > 0:
                board_str_counter[x] += 1 
                while True:
                    count += 1
                    if (x+count) < 13:
                        if cards_count[x+count] > 0:
                            board_str_counter[x] += 1   
                        elif cards_count[x+count+1] > 0 and not gutshot[x]:
                            board_str_counter[x] += 2 
                            count += 1
                            gutshot[x] = True
                        else:
                            break
                    elif (x+count) == 13:
                        if cards_count[x+count] > 0:
                            board_str_counter[x] += 1 
                    else:
                        break
                        
        max = fmax(board_str_counter)

        for x in range(len(board_str_counter)):
            if board_str_counter[x] >= 5:
                if not gutshot[x]:
                    board_cards_val[x] = 5
                if gutshot[x]:
                    board_cards_val[x] = 2
 
            elif board_str_counter[x] == 4:
                if not gutshot[x]: 
                    if x < 11 and x > 0:
                        board_cards_val[x] = 4   
                    else:
                        board_cards_val[x] = 3
                elif len(board) < 4:
                    board_cards_val[x] = 1
                else:
                    board_cards_val[x] = 0
                    
                    
            elif board_str_counter[x] == 3 and not gutshot[x]: # > 0 and < 11 == espaco pra up down    
                board_cards_val[x] = 1   
    # BOARD CARDS VALUES END.
    
    if True:                            # BOARD PAIRS VALUES.   pair_check
        for x in board_cards:
            if x != '':
                pair_count[CARDS_DIC[x]] += 1
                if x == 'A':
                    pair_count[0] += 1
                    
        if fmax(pair_count) == 4:
            pair_check = 7            # quad
            pair_index = fmax_index(pair_count)
        
        elif fmax(pair_count) == 3 and fmax(pair_count, 2) == 2:
            pair_check = 6            # FULLHOUSE
            pair_index = fmax_index(pair_count)
        elif fmax(pair_count) == 3:
            pair_check = 3            # TRIPLE
            pair_index = fmax_index(pair_count)
        
        elif fmax(pair_count) == 2 and pair_count.count(2) == 2:
            pair_check = 2            # 2 PAIRS
            pair_index = fmax_index(pair_count)
        
        elif fmax(pair_count) == 2 and pair_count.count(2) == 1:
            pair_check = 1               # 1 PAIR
            pair_index = fmax_index(pair_count)
        
        else:
            pair_check = 0            # NOTHING
            pair_index = False

            
    #                   MAX INDEX 13
    #       STRAIGHT                    SUITS    
    # 0 = RAINBOW                   RAINBOW
    # 1 = BACKDOOR                  BACKDOOR
    # 2 = GUTSHOT                   DOUBLE BACKDOOR
    # 3 = STRAIGHT DRAW 1 SIDE      FLUSHDRAW 2 NEEDED
    # 4 = UPDOWN STRAIGHT DRAW      FLUSHDRAW 1 NEEDED
    # 5 = STRAIGHT MADE             FLUSH MADE

    if fmax(board_cards_val) == -1:
        board_cards_val[fmax_index(board_str_counter)] = 0

    return [suit_check, fmax_index(suit_count)], [board_cards_val[fmax_index(board_str_counter)], fmax_index(board_str_counter)], [pair_check, pair_index]

def AddHandList(l, x):
    if len(l) > 5:
        l.pop(0)
    l.append(x)
    
    
def Action():
    while not FindTemplate(template_input=DIR+WAITOWNACTION, wanted='bool', threshold=0.95, tp=TABLECOOR):
        pass

    return True




# *********************************************************************************************************************************************
# *********************************************************************************************************************************************
#                                                               PRE FLOP RANGE
# *********************************************************************************************************************************************
# *********************************************************************************************************************************************

OPENRANGE = {"9":{"open":{"aggressive":{}, "normal":{}, "passive":{}}}, "6":{"open":{"aggressive":{}, "normal":{}, "passive":{}}}}

OPENRANGE['9']['open']['aggressive'].update({   "UTG":      "22+, AJs+, KQs, QJs, JTs, T9s, 98s, AQo+", 
                                                "UTG+1":    "22+, AJs+, KQs, QJs, JTs, T9s, 98s, AQo+", 
                                                "MP":       "22+, ATs+, KQs, QJs, JTs, T9s, 98s, AJo+",
                                                "MP+1":     "22+, ATs+, KQs, QJs, JTs, T9s, 98s, 87s, 76s, ATo+, KQo",
                                                "HJ":       "22+, A9s+, KJs+, QTs+, J9s+, T8s+, 97s+, 87s, 76s, 65s, ATo+, KQo",
                                                "CO":       "22+, A2s+, K9s+, Q9s+, J8s+, T8s+, 97s+, 86s+, 75s+, 64s+, 54s, A2o+, KTo+, QTo+, J9o+, T8o+, 98o, 87o, 76o",
                                                "BU":       "22+, A2s+, K2s+, Q4s+, J7s+, T8s+, 97s+, 86s+, 75s+, 64s+, 54s, A2o+, K7o+, Q8o+, J8o+, T8o+, 97o+, 87o, 76o, 65o, 54o",
                                                "SB":       "22+, A2s+, K9s+, Q9s+, J9s+, T8s+, 97s+, 86s+, 76s, 65s, 54s, ATo+, KTo+, QTo+, J9o+, T8o+, 98o",
                                                "BB":      "22+, AJs+, KQs, QJs, JTs, T9s, 98s, AQo+"
                                            })

OPENRANGE['9']['open']['normal'].update({       "UTG":      "77+, AQs+, AKo", 
                                                "UTG+1":    "77+, AQs+, AKo", 
                                                "MP":       "77+, AQs+, AQo+",
                                                "MP+1":     "66+, AJs+, KQs, QJs, JTs, T9s, 98s, 87s, AJo+, KQo",
                                                "HJ":       "55+, ATs+, KJs+, QJs, JTs, T9s, 98s, 87s, 76s, ATo+, KQo",
                                                "CO":       "22+, A7s+, K9s+, Q9s+, J9s+, T8s+, 97s+, 87s, 76s, 65s, 54s, A9o+, KTo+, QTo+, JTo, T9o, 98o, 87o",
                                                "BU":       "22+, A2s+, K9s+, Q9s+, J8s+, T8s+, 97s+, 86s+, 75s+, 65s, 54s, A2o+, KTo+, QTo+, J9o+, T8o+, 98o, 87o, 76o, 65o",
                                                "SB":       "22+, A2s+, K9s+, Q9s+, J9s+, T9s, 98s, 87s, 76s, 65s, 54s, ATo+, KTo+, QTo+, JTo",
                                                "BB":      "22+, AJs+, KQs, QJs, JTs, T9s, 98s, AQo+"
                                            })

OPENRANGE['9']['open']['passive'].update({      "UTG":      "TT+, AQs+, AKo", 
                                                "UTG+1":    "TT+, AQs+, AKo", 
                                                "MP":       "TT+, AQs+, AKo",
                                                "MP+1":     "99+, AJs+, AQo+",
                                                "HJ":       "88+, AJs+, KQs, AJo+, KQo",
                                                "CO":       "22+, A9s+, KTs+, QTs+, JTs, T9s, 98s, 87s, ATo+, KTo+, QTo+, JTo",
                                                "BU":       "22+, A2s+, K9s+, Q9s+, J8s+, T9s, 98s, 87s, 76s, 65s, 54s, A8o+, KTo+, QTo+, JTo, T9o, 98o",
                                                "SB":       "77+, A7s+, K9s+, Q9s+, J9s+, T9s, 98s, 87s, ATo+, KTo+, QTo+, JTo",
                                                "BB":      "22+, AJs+, KQs, QJs, JTs, T9s, 98s, AQo+"
                                            })

OPENRANGE['6']['open']['aggressive'].update({   "UTG":      "22+, ATs+, KJs+, QJs, JTs, T9s, 98s, 87s, 76s, 65s, ATo+, KJo+",                                                 
                                                "MP":       "22+, A9s+, KJs+, QTs+, JTs, T9s, 98s, 87s, 76s, 65s, A9o+, KJo+, QJo", 
                                                "CO":       "22+, A2s+, K7s+, Q7s+, J7s+, T8s+, 97s+, 87s, 76s, 65s, 54s, A2o+, K8o+, Q8o+, J8o+, T8o+, 97o+, 87o, 76o, 65o, 54o", 
                                                "BU":       "22+, A2s+, K2s+, Q2s+, J5s+, T8s+, 97s+, 87s, 76s, 65s, 54s, A2o+, K2o+, Q8o+, J8o+, T8o+, 97o+, 87o, 76o, 65o, 54o", 
                                                "SB":       "22+, A2s+, K7s+, Q7s+, J7s+, T8s+, 97s+, 87s, 76s, 65s, 54s, A2o+, K8o+, Q8o+, J8o+, T8o+, 97o+, 87o, 76o, 65o, 54o",
                                                "BB":      "22+, AJs+, KQs, QJs, JTs, T9s, 98s, AQo+"
                                            })

OPENRANGE['6']['open']['normal'].update({       "UTG":      "22+, ATs+, KQs, AJo+, KQo",                                                 
                                                "MP":       "22+, ATs+, KJs+, ATo+, KJo+", 
                                                "CO":       "22+, A6s+, K9s+, Q9s+, J9s+, T8s+, 98s, 87s, 76s, 65s, A9o+, K9o+, Q9o+, J9o+, T9o, 98o, 87o", 
                                                "BU":       "22+, A2s+, K7s+, Q7s+, J7s+, T8s+, 97s+, 87s, 76s, 65s, 54s, A2o+, K8o+, Q8o+, J8o+, T8o+, 97o+, 87o, 76o, 65o, 54o", 
                                                "SB":       "22+, A6s+, K9s+, Q9s+, J8s+, T9s, 98s, 87s, 76s, ATo+, K9o+, Q9o+, J9o+",
                                                "BB":      "22+, AJs+, KQs, QJs, JTs, T9s, 98s, AQo+"
                                            })

OPENRANGE['6']['open']['passive'].update({      "UTG":      "66+, AJs+, AQo+",                                                 
                                                "MP":       "66+, AJs+, KQs, AJo+, KQo", 
                                                "CO":       "22+, A7s+, K9s+, Q9s+, J9s+, T8s+, A9o+, K9o+, QTo+, J9o+", 
                                                "BU":       "22+, A2s+, K8s+, Q8s+, J7s+, T9s, 98s, 87s, 76s, 65s, 54s, A9o+, K9o+, Q9o+, J8o+, T8o+, 97o+, 87o, 76o, 65o, 54o", 
                                                "SB":       "22+, ATs+, KTs+, QTs+, JTs, ATo+, KTo+, QTo+, JTo",
                                                "BB":      "22+, AJs+, KQs, QJs, JTs, T9s, 98s, AQo+"
                                            })


BETvsPFR = {"9":{"aggressive":{}, "normal":{}, "passive":{}}, "6":{"aggressive":{}, "normal":{}, "passive":{}}}

BETvsPFR['9']['aggressive'].update({   "UTG":      "JJ+, AKs, AKo", 
                                        "UTG+1":    "JJ+, AKs, AKo",
                                        "MP":       "99+, AJs+, AQo+",
                                        "MP+1":     "88+, ATs+, KQs, ATo+, KQo",
                                        "HJ":       "88+, ATs+, KQs, ATo+, KQo",
                                        "CO":       "33+, A2s+, KTs+, A8o+, KTo+",
                                        "BU":       "22+, A2s+, K8s+, QTs+, A2o+, K9o+",
                                        "SB":       "QQ+, AKs",
                                        "BB":       "QQ+, AKs"
                                    })

BETvsPFR['9']['normal'].update({       "UTG":      "JJ+, AKs, AKo", 
                                        "UTG+1":    "JJ+, AKs, AKo",
                                        "MP":       "99+, AJs+, AQo+",
                                        "MP+1":     "88+, ATs+, KQs, ATo+, KQo",
                                        "HJ":       "88+, ATs+, KQs, ATo+, KQo",
                                        "CO":       "33+, A2s+, KTs+, A8o+, KTo+",
                                        "BU":       "22+, A2s+, K8s+, QTs+, A2o+, K9o+",
                                        "SB":       "QQ+, AKs",
                                        "BB":       "QQ+, AKs"
                                    })

BETvsPFR['9']['passive'].update({      "UTG":      "JJ+, AKs, AKo", 
                                        "UTG+1":    "JJ+, AKs, AKo",
                                        "MP":       "99+, AJs+, AQo+",
                                        "MP+1":     "88+, ATs+, KQs, ATo+, KQo",
                                        "HJ":       "88+, ATs+, KQs, ATo+, KQo",
                                        "CO":       "33+, A2s+, KTs+, A8o+, KTo+",
                                        "BU":       "22+, A2s+, K8s+, QTs+, A2o+, K9o+",
                                        "SB":       "QQ+, AKs",
                                        "BB":       "QQ+, AKs"
                                    })

BETvsPFR['6']['aggressive'].update({   "UTG":      "JJ+, AKs, AKo", 
                                        "MP":       "99+, AJs+, AQo+",
                                        "CO":       "33+, A2s+, KTs+, A8o+, KTo+",
                                        "BU":       "22+, A2s+, K8s+, QTs+, A2o+, K9o+",
                                        "SB":       "QQ+, AKs",
                                        "BB":       "QQ+, AKs"
                                    })

BETvsPFR['6']['normal'].update({       "UTG":      "JJ+, AKs, AKo", 
                                        "MP":       "99+, AJs+, AQo+",
                                        "CO":       "33+, A2s+, KTs+, A8o+, KTo+",
                                        "BU":       "22+, A2s+, K8s+, QTs+, A2o+, K9o+",
                                        "SB":       "QQ+, AKs",
                                        "BB":       "QQ+, AKs"
                                    })

BETvsPFR['6']['passive'].update({      "UTG":      "JJ+, AKs, AKo", 
                                        "MP":       "99+, AJs+, AQo+",
                                        "CO":       "33+, A2s+, KTs+, A8o+, KTo+",
                                        "BU":       "22+, A2s+, K8s+, QTs+, A2o+, K9o+",
                                        "SB":       "QQ+, AKs",
                                        "BB":       "QQ+, AKs"
                                    })




CALLING = {     "9":{"UTG":{}, "UTG+1":{}, "MP":{}, "MP+1":{}, "HJ":{}, "CO":{}, "BU":{},"SB":{}, "BB":{}},  
                "6":{"UTG":{}, "MP":{}, "CO":{}, "BU":{},"SB":{}, "BB":{}}}

CALLING['9']['UTG'].update({    "UTG":      "NONE", 
                                "UTG+1":    "QQ+, AKs, AKo", 
                                "MP":       "QQ+, AJs+, KQs, AQo+", 
                                "MP+1":     "QQ+, AJs+, KQs, AQo+", 
                                "HJ":       "QQ+, AJs+, KQs, AQo+", 
                                "CO":       "QQ+, AJs+, KQs, AQo+", 
                                "BU":       "QQ+, AJs+, KQs, AQo+", 
                                "SB":       "JJ+, AQs+, KQs, AKo",
                                "BB":       "JJ+, AQs+, KQs, AKo"
                            })
CALLING['9']['UTG+1'].update({  "UTG":      "88+, A7s+, KJs+, QJs, AQo+", 
                                "UTG+1":    "NONE", 
                                "MP":       "QQ+, AKs, AKo", 
                                "MP+1":     "QQ+, AJs+, KQs, AQo+", 
                                "HJ":       "QQ+, AJs+, KQs, AQo+", 
                                "CO":       "QQ+, AJs+, KQs, AQo+", 
                                "BU":       "QQ+, AJs+, KQs, AQo+", 
                                "SB":       "JJ+, AQs+, KQs, AKo",
                                "BB":       "JJ+, AQs+, KQs, AKo"
                            })
CALLING['9']['MP'].update({     "UTG":      "55+, A2s+, KJs+, QJs, JTs, T9s, 98s, 87s, 76s, 65s, AQo+, JTo", 
                                "UTG+1":    "88+, A7s+, KJs+, QJs, AQo+", 
                                "MP":       "NONE", 
                                "MP+1":     "JJ+, ATs+, KQs, AQo+", 
                                "HJ":       "JJ+, ATs+, KQs, AQo+", 
                                "CO":       "JJ+, ATs+, KQs, AQo+", 
                                "BU":       "JJ+, ATs+, KQs, AQo+", 
                                "SB":       "55+, A2s+, KJs+, QJs, JTs, T9s, 98s, 87s, 76s, 65s, AQo+, JTo",
                                "BB":       "55+, A2s+, KJs+, QJs, JTs, T9s, 98s, 87s, 76s, 65s, AQo+, JTo"
                            })
CALLING['9']['MP+1'].update({   "UTG":      "55+, A2s+, KJs+, QJs, JTs, T9s, 98s, 87s, 76s, 65s, AQo+, JTo", 
                                "UTG+1":    "55+, A2s+, KJs+, QJs, JTs, T9s, 98s, 87s, 76s, 65s, AQo+, JTo", 
                                "MP":       "88+, A7s+, KJs+, QJs, AQo+", 
                                "MP+1":     "NONE", 
                                "HJ":       "JJ+, ATs+, KQs, AQo+", 
                                "CO":       "JJ+, ATs+, KQs, AQo+", 
                                "BU":       "JJ+, ATs+, KQs, AQo+", 
                                "SB":       "55+, A2s+, KJs+, QJs, JTs, T9s, 98s, 87s, 76s, 65s, AQo+, JTo",
                                "BB":       "55+, A2s+, KJs+, QJs, JTs, T9s, 98s, 87s, 76s, 65s, AQo+, JTo"
                            })
CALLING['9']['HJ'].update({     "UTG":      "55+, A2s+, KJs+, QJs, JTs, T9s, 98s, 87s, 76s, 65s, AQo+, JTo", 
                                "UTG+1":    "55+, A2s+, KJs+, QJs, JTs, T9s, 98s, 87s, 76s, 65s, AQo+, JTo", 
                                "MP":       "55+, A2s+, KJs+, QJs, JTs, T9s, 98s, 87s, 76s, 65s, AQo+, JTo", 
                                "MP+1":     "88+, A7s+, KJs+, QJs, AQo+",  
                                "HJ":       "NONE", 
                                "CO":       "JJ+, ATs+, KQs, AQo+", 
                                "BU":       "JJ+, ATs+, KQs, AQo+", 
                                "SB":       "55+, A2s+, KJs+, QJs, JTs, T9s, 98s, 87s, 76s, 65s, AQo+, JTo",
                                "BB":       "55+, A2s+, KJs+, QJs, JTs, T9s, 98s, 87s, 76s, 65s, AQo+, JTo"
                            })
CALLING['9']['CO'].update({     "UTG":      "55+, A2s+, KJs+, QJs, JTs, T9s, 98s, 87s, 76s, 65s, AQo+, JTo", 
                                "UTG+1":    "55+, A2s+, KJs+, QJs, JTs, T9s, 98s, 87s, 76s, 65s, AQo+, JTo", 
                                "MP":       "55+, A2s+, KJs+, QJs, JTs, T9s, 98s, 87s, 76s, 65s, AQo+, JTo", 
                                "MP+1":     "88+, A7s+, KJs+, QJs, AQo+", 
                                "HJ":       "88+, A7s+, KJs+, QJs, AQo+", 
                                "CO":       "NONE", 
                                "BU":       "QQ+, AKs, AKo", 
                                "SB":       "55+, A2s+, KJs+, QJs, JTs, T9s, 98s, 87s, 76s, 65s, AQo+, JTo",
                                "BB":       "55+, A2s+, KJs+, QJs, JTs, T9s, 98s, 87s, 76s, 65s, AQo+, JTo"
                            })
CALLING['9']['BU'].update({     "UTG":      "55+, A2s+, KJs+, QJs, JTs, T9s, 98s, 87s, 76s, 65s, AQo+, JTo", 
                                "UTG+1":    "55+, A2s+, KJs+, QJs, JTs, T9s, 98s, 87s, 76s, 65s, AQo+, JTo", 
                                "MP":       "55+, A2s+, KJs+, QJs, JTs, T9s, 98s, 87s, 76s, 65s, AQo+, JTo", 
                                "MP+1":     "88+, A7s+, KJs+, QJs, AQo+", 
                                "HJ":       "88+, A7s+, KJs+, QJs, AQo+", 
                                "CO":       "TT+, ATs+, KQs, QJs, AQo+", 
                                "BU":       "NONE", 
                                "SB":       "55+, A2s+, KJs+, QJs, JTs, T9s, 98s, 87s, 76s, 65s, AQo+, JTo",
                                "BB":       "55+, A2s+, KJs+, QJs, JTs, T9s, 98s, 87s, 76s, 65s, AQo+, JTo"
                            })
CALLING['9']['SB'].update({     "UTG":      "QQ+, AKs, AKo", 
                                "UTG+1":    "QQ+, AKs, AKo", 
                                "MP":       "JJ+, AJs+, KQs, AKo", 
                                "MP+1":     "JJ+, AJs+, KQs, AKo", 
                                "HJ":       "TT+, AJs+, KQs, QJs, AKo", 
                                "CO":       "TT+, AJs+, KQs, QJs, AKo", 
                                "BU":       "TT+, AJs+, KQs, QJs, AKo", 
                                "SB":       "NONE",
                                "BB":       "22+, A2s+, KTs+, QTs+, JTs, T9s, 98s, 87s, 76s, 65s, 54s, 43s, 32s, A8o+, KTo+, QTo+, JTo, T9o, 98o"
                            })
CALLING['9']['BB'].update({     "UTG":      "QQ+, AKs, AKo", 
                                "UTG+1":    "QQ+, AKs, AKo", 
                                "MP":       "JJ+, AJs+, KQs, AKo", 
                                "MP+1":     "JJ+, AJs+, KQs, AKo", 
                                "HJ":       "TT+, AJs+, KQs, QJs, AKo", 
                                "CO":       "TT+, AJs+, KQs, QJs, AKo", 
                                "BU":       "TT+, AJs+, KQs, QJs, AKo", 
                                "BB":       "NONE"
                            })

CALLING['6']['UTG'].update({    "UTG":      "NONE", 
                                "MP":       "99+, ATs+, KQs, ATo+, KQo", 
                                "CO":       "66+, A8s+, KQs, QJs, JTs, ATo+, KQo",
                                "BU":       "JJ+, ATs+, KQs, AQo+", 
                                "SB":       "99+, ATs+, KQs, ATo+, KQo",
                                "BB":       "99+, ATs+, KQs, ATo+, KQo"
                            })
CALLING['6']['MP'].update({     "UTG":      "99+, ATs+, KQs, ATo+, KQo", 
                                "MP":       "NONE",
                                "CO":       "66+, A8s+, KQs, QJs, JTs, ATo+, KQo", 
                                "BU":       "JJ+, ATs+, KQs, AQo+", 
                                "SB":       "99+, ATs+, KQs, ATo+, KQo",
                                "BB":       "99+, ATs+, KQs, ATo+, KQo"
                            })
CALLING['6']['CO'].update({     "UTG":      "99+, ATs+, KQs, ATo+, KQo", 
                                "MP":       "66+, A8s+, KQs, QJs, JTs, ATo+, KQo", 
                                "CO":       "NONE", 
                                "BU":       "JJ+, ATs+, KQs, AQo+", 
                                "SB":       "99+, ATs+, KQs, ATo+, KQo",
                                "BB":       "99+, ATs+, KQs, ATo+, KQo"
                            })
CALLING['6']['BU'].update({     "UTG":      "66+, A8s+, KQs, QJs, JTs, ATo+, KQo", 
                                "MP":       "22+, A6s+, KQs, QJs, JTs, T9s, 98s, 87s, 76s, 65s, 54s, 43s, 32s, A8o+, KQo", 
                                "CO":       "22+, A5s+, KJs+, QJs, JTs, T9s, 98s, 87s, 76s, 65s, 54s, 43s, 32s, A2o+, KQo, JTo", 
                                "BU":       "NONE", 
                                "SB":       "22+, A2s+, KTs+, QTs+, JTs, T9s, 98s, 87s, 76s, 65s, 54s, 43s, 32s, A2o+, KTo+, QTo+, JTo, T9o, 98o, 87o, 76o, 65o, 54o, 43o, 32o",
                                "BB":       "22+, A2s+, KTs+, QTs+, JTs, T9s, 98s, 87s, 76s, 65s, 54s, 43s, 32s, A2o+, KTo+, QTo+, JTo, T9o, 98o, 87o, 76o, 65o, 54o, 43o, 32o"
                            })
CALLING['6']['SB'].update({     "UTG":      "99+, ATs+, KQs, AQo+", 
                                "MP":       "44+, A8s+, KQs, QJs, JTs, T9s, 98s, 87s, AQo+", 
                                "CO":       "22+, A6s+, KJs+, QJs, JTs, T9s, 98s, 87s, 76s, 65s, AQo+", 
                                "BU":       "22+, A2s+, KTs+, QTs+, JTs, T9s, 98s, 87s, 76s, 65s, 54s, 43s, 32s, AQo+", 
                                "SB":       "NONE", 
                                "BB":       "55+, A2s+, KTs+, QTs+, J9s+, T8s+, 98s, 87s, 76s, 65s, ATo+, KTo+, QTo+, JTo"
                            })
CALLING['6']['BB'].update({     "UTG":      "22+, A7s+, K9s+, Q9s+, J9s+, T8s+, 98s, 87s, 76s, 65s, 54s, A9o+, KTo+, QTo+, JTo", 
                                "MP":       "22+, A5s+, K4s+, Q8s+, J7s+, T8s+, 97s+, 86s+, 75s+, 65s, 54s, A6o+, K9o+, QTo+, JTo, T9o", 
                                "CO":       "22+, A2s+, K2s+, Q5s+, J5s+, T6s+, 96s+, 86s+, 75s+, 64s+, 54s, A2o+, K7o+, Q9o+, J9o+, T8o+, 98o", 
                                "BU":       "22+, A2s+, K2s+, Q2s+, J4s+, T5s+, 95s+, 85s+, 75s+, 64s+, 53s+, A2o+, K2o+, Q4o+, J7o+, T7o+, 97o+, 86o+, 76o", 
                                "SB":       "22+, A2s+, K2s+, Q2s+, J2s+, T2s+, 94s+, 85s+, 74s+, 63s+, 53s+, 43s, A2o+, K2o+, Q2o+, J2o+, T4o+, 96o+, 86o+, 75o+, 65o", 
                                "BB":       "NONE"
                            })


def InRange(re, hand=False): 
    # DIC = {'2':1,'3':2,'4':3,'5':4,'6':5,'7':6,'8':7,'9':8,'T':9,'J':10,'Q':11,'K':12,'A':13}  USANDO CARDS_DIC

    if re == 'NONE':
        print("RE ERROR")
        time.sleep(999)

    temp = re.split(", ")

    for r in temp:
        if hand[0] == hand[1] and r[0] == r[1] and CARDS_DIC[hand[0]] >= CARDS_DIC[r[0]]: # if hand value >= range value
            return True
        else:
            if hand[0] == r[0]: # 1st card from hand same as 1st from range - check 2nd card
                if hand[2] == r[2]: # check if suited or off from hand == range
                    if hand[1] == r[1]:
                        return True
                    elif len(r) == 4 and r[3] == "+" and CARDS_DIC[hand[1]] > CARDS_DIC[r[1]]:
                        return True
                        
    else:
        return False

#                                                               TO USE LATER
# def DecideAgression(t, check):
#     setup_agr = [[2, 2], [4, 4], [4, 12], [8, 24]]
#     max_points = 0

#     for p in t.players_in_hand:
#         if t.pl[p].points_total > max_points:
#             max_points = t.pl[p].points_total
        
#     if max_points < setup_agr[t.breakcounter][0]:
#         return 'aggressive' #'agressive'
#     elif max_points > setup_agr[t.breakcounter][1]:
#         return 'passive' #'passive'
#     else:
#         return 'normal' #'normal'
#                                                               TO USE LATER

def SetupVars(t, check): # acao, self_pos, rank, pos, max_points
    # DICT_FINAL = {0:'NOTHING', 1:'PAIR', 2:'2 PAIRS', 3:'TRIPLE', 4:'STRAIGHT', 5:'FLUSH', 6:'FULLHOUSE', 7:'QUAD', 8:'STRAIGHT FLUSH', 9:'ROYAL FLUSH'}  # DECLARADA NO COMECO
    return check.p.position, DICT_FINAL[check.h.hand_rank], check.p.position, 0



#                                                                                                                                                                   LOG
def DecisionLogs(t, check):
    fnk(t.fnkfile, '*************************************************\n\nHand Number: '+str(t.hand_num)+
    '\nHole Cards: '+str(check.h.hole_full[0])+' '+str(check.h.hole_full[1])+' - '+str(check.h.hole_percent)+'%\tPos: '+str(POSITION_DICT[check.p.position])+
    '\nPlayers at Start: '+str(t.players_in_hand)+
    '\n\n--------------------\n\n'+str(STREET[t.breakcounter])+
    '\nMax points: '+str(max_points)+'\tPlay Style: '+str(agr)+
    '\nLimps: '+str(t.limp)+'\tCount: '+str(t.limp_count)+
    '\nBets: '+str(t.bets)+'\tPosition: '+str(t.bets_pos)+'\n')

def FullDecisionLogs(t, check):
    fnk(t.fnkfile, '\n\n--------------------\n\n'+str(STREET[t.breakcounter])+
    '\nin_hand: '+
    '\nMax points: '+str(max_points)+'\tPlay Style: '+str(agr)+
    '\nLimps: '+str(t.limp)+'\tCount: '+str(t.limp_count)+
    '\nBets: '+str(t.bets)+'\tPosition: '+str(t.bets_pos)+
    '\nBoard: '+str(t.board.board[1])+' '+str(t.board.board[2])+' '+str(t.board.board[3])+' '+str(t.board.board[4])+' '+str(t.board.board[5])+'\t Hole: '+str(check.h.hole_full[0])+' '+str(check.h.hole_full[1])+
    '\n\tBoard Flush:\t'+str(DICT_STR[check.b.suit_check[check.t.breakcounter]])+' ( '+str(str(check.b.suit_check[check.t.breakcounter]))+
    ' )\n\tBoard Straight:\t'+str(DICT_STR[check.b.str_check[check.t.breakcounter]])+' ( '+str(str(check.b.str_check[check.t.breakcounter]))+
    ' )\n\tBoard Pair:\t'+str(DICT_STR[check.b.pair_check[check.t.breakcounter]])+' ( '+str(str(check.b.pair_check[check.t.breakcounter]))+
    ' )\n\n\tHand Rank: '+str(DICT_FINAL[check.h.hand_rank])+' ( '+str(check.h.hand_rank)+
    ' )\n\nMy_outs: '+str(t.outs_hand)+' ( '+str(check.h.outs_por)+
    '% )\nPot size: '+str(t.pot)+
    '\nTo call: '+str(t.to_call)+
    '\nPot odds: '+str(t.pot_odds_str)+' ( '+str(t.pot_odds_per)+'% )\n')
#                                                                                                                                                                   LOG

def jogada(acao, log, t=False):
    if t.logging:
        log += str(acao)

        with open(DIR+'logs/'+t.hand_num+'.fnk', 'a') as fl:
            fl.write(log+'\n')
            fl.flush()

    return str(acao)


# DECISIION MAKING DECISIION MAKING DECISIION MAKING DECISIION MAKING DECISIION MAKING DECISIION MAKING DECISIION MAKING DECISIION MAKING DECISIION MAKING DECISIION MAKING 
# DECISIION MAKING DECISIION MAKING DECISIION MAKING DECISIION MAKING DECISIION MAKING DECISIION MAKING DECISIION MAKING DECISIION MAKING DECISIION MAKING DECISIION MAKING 

# PRE FLOP  PRE FLOP  PRE FLOP  PRE FLOP  PRE FLOP  PRE FLOP  PRE FLOP  PRE FLOP  PRE FLOP  PRE FLOP  PRE FLOP  PRE FLOP  PRE FLOP  PRE FLOP  PRE FLOP  PRE FLOP 
def PreFlopDecision(t, check, agr):
    if not t.bets:
        if NPLAYERS == 9:
            if InRange(re=OPENRANGE['9']['open']['normal'][POSITION_DICT[check.p.position]], hand=check.FullHole()):
                log = 'preflop - no bets - n=9 - openrange 3x'
                return jogada('3x', log, t)
        elif NPLAYERS == 6:
            if InRange(re=OPENRANGE['6']['open']['normal'][POSITION_DICT[check.p.position]], hand=check.FullHole()):
                log = 'preflop - no bets - n=6 - openrange 3x'
                return jogada('3x', log, t)
    else:
        if NPLAYERS == 9:
            if InRange(re=BETvsPFR['9']['normal'][POSITION_DICT[check.p.position]], hand=check.FullHole()):
                log = 'preflop - bets - n=9 - 3betvspfr 55'
                return jogada('55', log, t)
            if InRange(re=CALLING['9'][POSITION_DICT[check.p.position]][POSITION_DICT[check.t.bets_pos]], hand=check.FullHole()): # CALLING RANGE
                log = 'preflop - bets - n=9 call'
                return jogada('call', log, t)

        elif NPLAYERS == 6:
            if InRange(re=BETvsPFR['6']['normal'][POSITION_DICT[check.p.position]], hand=check.FullHole()):
                log = 'preflop - bets - n=6 - 3betvspfr 55'
                return jogada('55', log, t)
            if InRange(re=CALLING['6'][POSITION_DICT[check.p.position]][POSITION_DICT[check.t.bets_pos]], hand=check.FullHole()): # CALLING RANGE
                log = 'preflop - bets - n=6 - call'
                return jogada('call', log, t)
# PRE FLOP  PRE FLOP  PRE FLOP  PRE FLOP  PRE FLOP  PRE FLOP  PRE FLOP  PRE FLOP  PRE FLOP  PRE FLOP  PRE FLOP  PRE FLOP  PRE FLOP  PRE FLOP  PRE FLOP  PRE FLOP 
# POS FLOP  POS FLOP  POS FLOP  POS FLOP  POS FLOP  POS FLOP  POS FLOP  POS FLOP  POS FLOP  POS FLOP  POS FLOP  POS FLOP  POS FLOP  POS FLOP  POS FLOP  POS FLOP 
def FlopOne(t, check, agr):
    log = 'HeadsUp Flop - '
    if check.h.hand_rank >= 4: 
        log += 'HAVE STRAIGHT OR HIGHER, GET VALUE.'
        return jogada('55', log, t)

    elif check.h.hand_outs >= 13:
        log += 'outs >= 13'
        return jogada('55', log, t)

    elif check.HavePotOddsToCall() and check.p.can_call:
        log += 'Have pot odds to call'
        return jogada('call', log, t)

    elif check.h.hand_rank == 3:
        if not check.PossibleFlushDraw() and not check.PossibleStrDraw():
            log += 'Have 3 of a kind and no flush/straight draw on board'
            return jogada('45', log, t)

        elif check.PossibleStrDraw() or check.PossibleFlushDraw():
            log += 'Have 3 of a kind and possible flush/straight draw on board'
            return jogada('55', log, t)

    elif check.h.hand_rank == 2:
        if check.IsPocketPair():
            log += '2 pairs with one pocket'
            return jogada('45', log, t)
        else:
            log += '2 pairs without pocket pair'
            return jogada('55', log, t)

    elif check.h.hand_rank == 1 and not check.PairOnBoard():
        if check.IsPocketPair() and check.IsOverpair():
            log += 'Have one pair being pocket and Overpair'
            return jogada('55', log, t)
        
        elif check.IsPocketPair() and not check.IsOverpair() and check.InPosition() and check.p.can_check:
            log += 'InPosition PocketPair but its not an Overpair and villan checked'
            return jogada('55', log, t)

        elif not check.IsPocketPair() and check.InPosition() and check.p.can_check:
            log += 'not PocketPair InPosition villan checked'
            return jogada('55', log, t)

        elif not check.IsPocketPair() and check.InPosition() and check.p.can_call and not check.HavePotOddsToCall() and check.HowManyAboveMyHole() < 1:
            log += 'its not a PocketPair InPosition villanBet DontHavePotOddsToCall and HowManyAboveMyHole < 1'
            return jogada('55', log, t)
    
    elif check.h.hand_rank == 1 and check.PairOnBoard():
        if check.HaveTwoOverCards():
            log += 'PairOnBoard and have 2OverCards'
            return jogada('55', log, t)

        elif check.HaveOneOverCard() and check.p.can_check and check.InPosition():
            log += 'PairOnBoard and have 1 OverCard and villan checked'
            return jogada('45', log, t)

        elif check.HaveOneOverCard() and check.p.can_call and check.InPosition():
            log += 'PairOnBoard and have 1 OverCard and villan bet'
            return jogada('fold', log, t)

        else:
            log += 'PairOnBoard and no overcards.'
            return jogada('fold', log, t)

    elif check.h.hand_rank == 0:
        if check.InPosition() and check.p.can_check and check.p.action[0]:
            log += 'InPosition and villan checked and i have bet preflop. CBET'
            return jogada('45', log, t)

        elif check.HaveFshDraw():
            log += 'VER MELHOR ESSE ODDS PRA STRAIGHT AND FLUSH DRAWS '*20
            return jogada('fold', log, t)

        elif check.HavePotOddsToCall():
            log += 'HavePotOddsToCall'*20
            return jogada('call', log, t)

        else:
            log += 'have nothing'
            return jogada('fold', log, t)

def FlopMult(t, check, agr):
    log = 'Multway Flop - '
    if check.h.hand_rank >= 4: 
        log += 'HAVE STRAIGHT OR HIGHER, GET VALUE.'
        return jogada('55', log, t)

    if check.h.hand_outs >= 13:
        log += 'outs >= 13'
        return jogada('55', log, t)
        
    elif check.HavePotOddsToCall() and check.p.can_call:
        log += 'Have pot odds to call'
        return jogada('call', log, t)

    elif check.h.hand_rank == 3:
        if not check.PossibleFlushDraw() and not check.PossibleStrDraw():
            log += 'Have 3 of a kind and no flush/straight draw on board'
            return jogada('45', log, t)

        elif check.PossibleStrDraw() or check.PossibleFlushDraw():
            log += 'Have 3 of a kind and possible flush/straight draw on board'
            return jogada('55', log, t)

    elif check.h.hand_rank == 2:
        if check.IsPocketPair():
            log += '2 pairs with one pocket'
            return jogada('45', log, t)
        else:
            log += '2 pairs without pocket pair'
            return jogada('55', log, t)

    elif check.h.hand_rank == 1 and not check.PairOnBoard():
        if check.IsPocketPair() and check.IsOverpair():
            log += 'Have one pair being pocket and Overpair'
            return jogada('55', log, t)
        
        elif check.IsPocketPair() and not check.IsOverpair() and check.InPosition() and check.p.can_check:
            log += 'InPosition PocketPair but its not an Overpair and villan checked'
            return jogada('55', log, t)

        elif not check.IsPocketPair() and check.InPosition() and check.p.can_check:
            log += 'not PocketPair InPosition villan checked'
            return jogada('55', log, t)

        elif not check.IsPocketPair() and check.InPosition() and check.p.can_call and not check.HavePotOddsToCall() and check.HowManyAboveMyHole() < 1:
            log += 'its not a PocketPair InPosition villanBet DontHavePotOddsToCall and HowManyAboveMyHole < 1'
            return jogada('55', log, t)
    
    elif check.h.hand_rank == 1 and check.PairOnBoard():
        if check.HaveTwoOverCards():
            log += 'PairOnBoard and have 2OverCards'
            return jogada('55', log, t)

        elif check.HaveOneOverCard() and check.p.can_check and check.InPosition():
            log += 'PairOnBoard and have 1 OverCard and villan checked'
            return jogada('45', log, t)

        elif check.HaveOneOverCard() and check.p.can_call and check.InPosition():
            log += 'PairOnBoard and have 1 OverCard and villan bet'
            return jogada('fold', log, t)

        else:
            log += 'PairOnBoard and no overcards.'
            return jogada('fold', log, t)

    elif check.h.hand_rank == 0:
        if check.InPosition() and check.p.can_check and check.p.action[0]:
            log += 'InPosition and villan checked and i have bet preflop. CBET'
            return jogada('45', log, t)

        elif check.HaveFshDraw():
            log += 'VER MELHOR ESSE ODDS PRA STRAIGHT AND FLUSH DRAWS '*20
            return jogada('fold', log, t)

        elif check.HavePotOddsToCall():
            log += 'HavePotOddsToCall'*20
            return jogada('call', log, t)

        else:
            log += 'have nothing'
            return jogada('fold', log, t)

def FlopDecision(t, check, agr):
    if check.HowManyInHand() == 1:
        return FlopOne(t, check, agr)
    if check.HowManyInHand() >= 2:
        return FlopMult(t, check, agr)
# POS FLOP  POS FLOP  POS FLOP  POS FLOP  POS FLOP  POS FLOP  POS FLOP  POS FLOP  POS FLOP  POS FLOP  POS FLOP  POS FLOP  POS FLOP  POS FLOP  POS FLOP  POS FLOP

# TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN 
def TurnOne(t, check, agr):
    log = 'HeadsUp Turn - '
    if check.h.hand_rank >= 4:
        log += 'HAVE STRAIGHT OR HIGHER, GET VALUE'
        return jogada('55', log, t)

    elif check.h.hand_outs >= 13:
        log += 'outs >= 13'
        return jogada('55', log, t)

    elif check.HavePotOddsToCall() and check.p.can_call:
        log += 'Have pot odds to call'
        return jogada('call', log, t)

    elif check.h.hand_rank == 3:
        if not check.PossibleFlushDraw() and not check.PossibleStrDraw():
            log += 'Have 3 of a kind and no flush/straight draw on board'
            return jogada('45', log, t)

        elif check.PossibleStrDraw() or check.PossibleFlushDraw():
            log += 'Have 3 of a kind and possible flush/straight draw on board'
            return jogada('55', log, t)

    elif check.h.hand_rank == 2:
        if check.IsPocketPair():
            log += '2 pairs with one pocket'
            return jogada('45', log, t)
        else:
            log += '2 pairs without pocket pair'
            return jogada('55', log, t)

    elif check.StrImproved() or check.FlsImproved() or check.PairImproved():
        if check.InPosition() and not check.VillanMoreAggressive():
            log += 'straight/flush/pair Improved, in position and villan more passive'
            return jogada('45', log, t)

        elif not check.InPosition() and check.VillanMoreAggressive():
            log += 'straight/flush/pair Improved, out of position and villan more aggressive'
            return jogada('fold', log, t)

    elif check.h.hand_rank == 1 and not check.PairOnBoard():
        if check.IsPocketPair() and check.IsOverpair():
            log += 'Have one pair being pocket and Overpair'
            return jogada('55', log, t)
        
        elif check.IsPocketPair() and not check.IsOverpair() and check.InPosition() and check.p.can_check:
            log += 'InPosition PocketPair but its not an Overpair and villan checked'
            return jogada('55', log, t)

        elif not check.IsPocketPair() and check.InPosition() and check.p.can_check:
            log += 'not PocketPair InPosition villan checked'
            return jogada('55', log, t)

        elif not check.IsPocketPair() and check.InPosition() and check.p.can_call and not check.HavePotOddsToCall() and check.HowManyAboveMyHole() < 1:
            log += 'its not a PocketPair InPosition villanBet DontHavePotOddsToCall and HowManyAboveMyHole < 1'
            return jogada('55', log, t)
    
    elif check.h.hand_rank == 1 and check.PairOnBoard():
        if check.HaveTwoOverCards():
            log += 'PairOnBoard and have 2OverCards'
            return jogada('55', log, t)

        elif check.HaveOneOverCard() and check.p.can_check and check.InPosition():
            log += 'PairOnBoard and have 1 OverCard and villan checked'
            return jogada('45', log, t)

        elif check.HaveOneOverCard() and check.p.can_call and check.InPosition():
            log += 'PairOnBoard and have 1 OverCard and villan bet'
            return jogada('fold', log, t)

        else:
            log += 'PairOnBoard and no overcards.'
            return jogada('fold', log, t)

    elif check.h.hand_rank == 0:
        if check.InPosition() and check.p.can_check and check.p.action[0]:
            log += 'InPosition and villan checked and i have bet preflop. CBET'
            return jogada('45', log, t)

        elif check.HaveFshDraw():
            log += 'VER MELHOR ESSE ODDS PRA STRAIGHT AND FLUSH DRAWS '*20
            return jogada('fold', log, t)

        elif check.HavePotOddsToCall():
            log += 'HavePotOddsToCall'*20
            return jogada('call', log, t)

        else:
            log += 'have nothing'
            return jogada('fold', log, t)

def TurnMult(t, check, agr):
    log = 'Multway Turn - '
    if check.h.hand_rank >= 4:
        log += 'HAVE STRAIGHT OR HIGHER, GET VALUE'
        return jogada('55', log, t)

    elif check.h.hand_outs >= 13:
        log += 'outs >= 13'
        return jogada('55', log, t)

    elif check.HavePotOddsToCall() and check.p.can_call:
        log += 'Have pot odds to call'
        return jogada('call', log, t)

    elif check.h.hand_rank == 3:
        if not check.PossibleFlushDraw() and not check.PossibleStrDraw():
            log += 'Have 3 of a kind and no flush/straight draw on board'
            return jogada('45', log, t)

        elif check.PossibleStrDraw() or check.PossibleFlushDraw():
            log += 'Have 3 of a kind and possible flush/straight draw on board'
            return jogada('55', log, t)

    elif check.h.hand_rank == 2:
        if check.IsPocketPair():
            log += '2 pairs with one pocket'
            return jogada('45', log, t)
        else:
            log += '2 pairs without pocket pair'
            return jogada('55', log, t)

    elif check.StrImproved() or check.FlsImproved() or check.PairImproved():
        if check.InPosition() and not check.VillanMoreAggressive():
            log += 'straight/flush/pair Improved, in position and villan more passive'
            return jogada('45', log, t)

        elif not check.InPosition() and check.VillanMoreAggressive():
            log += 'straight/flush/pair Improved, out of position and villan more aggressive'
            return jogada('fold', log, t)

    elif check.h.hand_rank == 1 and not check.PairOnBoard():
        if check.IsPocketPair() and check.IsOverpair():
            log += 'Have one pair being pocket and Overpair'
            return jogada('55', log, t)
        
        elif check.IsPocketPair() and not check.IsOverpair() and check.InPosition() and check.p.can_check:
            log += 'InPosition PocketPair but its not an Overpair and villan checked'
            return jogada('55', log, t)

        elif not check.IsPocketPair() and check.InPosition() and check.p.can_check:
            log += 'not PocketPair InPosition villan checked'
            return jogada('55', log, t)

        elif not check.IsPocketPair() and check.InPosition() and check.p.can_call and not check.HavePotOddsToCall() and check.HowManyAboveMyHole() < 1:
            log += 'its not a PocketPair InPosition villanBet DontHavePotOddsToCall and HowManyAboveMyHole < 1'
            return jogada('55', log, t)
    
    elif check.h.hand_rank == 1 and check.PairOnBoard():
        if check.HaveTwoOverCards():
            log += 'PairOnBoard and have 2OverCards'
            return jogada('55', log, t)

        elif check.HaveOneOverCard() and check.p.can_check and check.InPosition():
            log += 'PairOnBoard and have 1 OverCard and villan checked'
            return jogada('45', log, t)

        elif check.HaveOneOverCard() and check.p.can_call and check.InPosition():
            log += 'PairOnBoard and have 1 OverCard and villan bet'
            return jogada('fold', log, t)

        else:
            log += 'PairOnBoard and no overcards.'
            return jogada('fold', log, t)

    elif check.h.hand_rank == 0:
        if check.InPosition() and check.p.can_check and check.p.action[0]:
            log += 'InPosition and villan checked and i have bet preflop. CBET'
            return jogada('45', log, t)

        elif check.HaveFshDraw():
            log += 'VER MELHOR ESSE ODDS PRA STRAIGHT AND FLUSH DRAWS '*20
            return jogada('fold', log, t)

        elif check.HavePotOddsToCall():
            log += 'HavePotOddsToCall'*20
            return jogada('call', log, t)

        else:
            log += 'have nothing'
            return jogada('fold', log, t)

def TurnDecision(t, check, agr):
    if check.HowManyInHand() == 1:
        return TurnOne(t, check, agr)
    if check.HowManyInHand() >= 2:
        return TurnMult(t, check, agr)
# TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN  TURN 

# RIVER  RIVER  RIVER  RIVER  RIVER  RIVER  RIVER  RIVER  RIVER  RIVER  RIVER  RIVER  RIVER  RIVER  RIVER  RIVER  RIVER  RIVER  RIVER  RIVER  RIVER  RIVER  RIVER  
def RiverOne(t, check, agr):
    log = ''
    if check.h.hand_rank >= 4:
        log += 'HAVE A STRAIGHT OR HIGHER, GO TO TOWN'
        return jogada('55', log, t)

    elif check.h.hand_rank == 3 or check.h.hand_rank == 2:
        if not check.PossibleFlushDraw() and not check.PossibleStrDraw():
            log += 'Have 3 of a kind and no flush/straight draw on board'
            return jogada('55', log, t)

        elif check.PossibleStrDraw() and check.StrImproved() and check.VillanMoreAggressive():
            log += 'out of position, StrImproved now PossibleStrDraw and VillanMoreAggressive'
            return jogada('fold', log, t)

        elif check.PossibleStrDraw() and check.StrImproved() and not check.VillanMoreAggressive() and check.InPosition():
            log += 'InPosition PossibleStrDraw and StrImproved - not VillanMoreAggressive'
            return jogada('45', log, t)

        elif check.PossibleFlushDraw() and check.FlsImproved() and check.VillanMoreAggressive():
            log += 'out of position, FlsImproved now PossibleFlushDraw and VillanMoreAggressive'
            return jogada('fold', log, t)

        elif check.PossibleFlushDraw() and check.FlsImproved() and not check.VillanMoreAggressive() and check.InPosition():
            log += 'InPosition PossibleFlushDraw and FlsImproved - not VillanMoreAggressive'
            return jogada('45', log, t)

        elif check.PossibleFlushDraw() or check.PossibleStrDraw():
            if check.p.can_check:
                log += 'PossibleFlushDraw or PossibleStrDraw and can check with 3 of a kind'
                return jogada('fold', log, t)

    elif check.h.hand_rank == 1:
        if check.PairOnBoard():
            if check.p.can_check:
                log += 'hand_rank == 1, PairOnBoard. check/fold'
                return jogada('fold', log, t)
        else:
            if check.IsPocketPair() and check.IsOverpair():
                log += 'Overpair PocketPair'
                return jogada('45', log, t)
            
            elif check.IsPocketPair() and not check.IsOverpair():
                log += 'PocketPair and not Overpair'
                return jogada('fold', log, t)

            elif not check.IsPocketPair() and not check.PossibleStrDraw() and not check.PossibleFlushDraw() and not check.PairOnBoard():
                log += 'one pair not PocketPair and dry board. '
                if check.VillanMoreAggressive() and check.InPosition():
                    log += 'VillanMoreAggressive'
                    return jogada('fold', log, t)
                
                elif not check.InPosition():
                    log += 'out of position'
                    return jogada('fold', log, t)

                elif check.InPosition() and not check.VillanMoreAggressive():
                    log += 'InPosition and not VillanMoreAggressive '
                    return jogada('fold', log, t) 

    elif check.h.hand_rank == 0:
        log += 'hand_rank == 0'
        return jogada('fold', log, t)

def RiverMult(t, check, agr):
    log = ''
    if check.h.hand_rank >= 4:
        log += 'HAVE A STRAIGHT OR HIGHER, GO TO TOWN'
        return jogada('55', log, t)

    elif check.h.hand_rank == 3 or check.h.hand_rank == 2:
        if not check.PossibleFlushDraw() and not check.PossibleStrDraw():
            log += 'Have 3 of a kind and no flush/straight draw on board'
            return jogada('55', log, t)

        elif check.PossibleStrDraw() and check.StrImproved() and check.VillanMoreAggressive():
            log += 'out of position, StrImproved now PossibleStrDraw and VillanMoreAggressive'
            return jogada('fold', log, t)

        elif check.PossibleStrDraw() and check.StrImproved() and not check.VillanMoreAggressive() and check.InPosition():
            log += 'InPosition PossibleStrDraw and StrImproved - not VillanMoreAggressive'
            return jogada('45', log, t)

        elif check.PossibleFlushDraw() and check.FlsImproved() and check.VillanMoreAggressive():
            log += 'out of position, FlsImproved now PossibleFlushDraw and VillanMoreAggressive'
            return jogada('fold', log, t)

        elif check.PossibleFlushDraw() and check.FlsImproved() and not check.VillanMoreAggressive() and check.InPosition():
            log += 'InPosition PossibleFlushDraw and FlsImproved - not VillanMoreAggressive'
            return jogada('45', log, t)

        elif check.PossibleFlushDraw() or check.PossibleStrDraw():
            if check.p.can_check:
                log += 'PossibleFlushDraw or PossibleStrDraw and can check with 3 of a kind'
                return jogada('fold', log, t)

    elif check.h.hand_rank == 1:
        if check.PairOnBoard():
            if check.p.can_check:
                log += 'hand_rank == 1, PairOnBoard. check/fold'
                return jogada('fold', log, t)
        else:
            if check.IsPocketPair() and check.IsOverpair():
                log += 'Overpair PocketPair'
                return jogada('45', log, t)
            
            elif check.IsPocketPair() and not check.IsOverpair():
                log += 'PocketPair and not Overpair'
                return jogada('fold', log, t)

            elif not check.IsPocketPair() and not check.PossibleStrDraw() and not check.PossibleFlushDraw() and not check.PairOnBoard():
                log += 'one pair not PocketPair and dry board. '
                if check.VillanMoreAggressive() and check.InPosition():
                    log += 'VillanMoreAggressive'
                    return jogada('fold', log, t)
                
                elif not check.InPosition():
                    log += 'out of position'
                    return jogada('fold', log, t)

                elif check.InPosition() and not check.VillanMoreAggressive():
                    log += 'InPosition and not VillanMoreAggressive '
                    return jogada('fold', log, t) 

    elif check.h.hand_rank == 0:
        log += 'hand_rank == 0'
        return jogada('fold', log, t)
    
def RiverDecision(t, check, agr):
    if check.HowManyInHand() == 1:
        return RiverOne(t, check, agr)
    if check.HowManyInHand() >= 2:
        return RiverMult(t, check, agr)
# RIVER  RIVER  RIVER  RIVER  RIVER  RIVER  RIVER  RIVER  RIVER  RIVER  RIVER  RIVER  RIVER  RIVER  RIVER  RIVER  RIVER  RIVER  RIVER  RIVER  RIVER  RIVER  RIVER 

def DecisionMaking(t, check):
    self_pos, rank, pos, max_points = SetupVars(t, check)
    agr = 0 #DecideAgression(t, check)
    acao = ''
    # GPIO_DICT = {'fold':11, '':11, 'call':12, 'check':12, 'bet':13, '3x':40, '40':16, '45':18, '55':38, '65':38, 'None':11}

    if t.breakcounter == 0:
        acao = PreFlopDecision(t, check, agr)
    elif t.breakcounter == 1:
        acao = FlopDecision(t, check, agr)
    elif t.breakcounter == 2:
        acao = TurnDecision(t, check, agr)
    elif t.breakcounter == 3:
        acao = RiverDecision(t, check, agr)
    else:
        print("WTF??? STREET AFTER RIVER?????")
        time.sleep(999)

    if acao == '' or acao =='None' or acao not in GPIO_DICT or acao == 'fold' or acao == 'FOLD':
        if check.p.can_check:
            acao = 'check'
        else:
            acao = 'fold'
    
    if AUTOPLAYING:
        if acao == 'raise':
            acao = 'bet'
            
        if not check.p.can_check:
            if acao == 'check':
                acao = 'call'
        
        if not check.p.can_call:
            if acao == 'call':
                acao = 'check'
        
        time.sleep(random.uniform(0.001, 0.084))
        print('ACAO: ',acao)
        sendAcao(GPIO_DICT[acao])
    return acao

# DECISIION MAKING DECISIION MAKING DECISIION MAKING DECISIION MAKING DECISIION MAKING DECISIION MAKING DECISIION MAKING DECISIION MAKING DECISIION MAKING DECISIION MAKING 
# DECISIION MAKING DECISIION MAKING DECISIION MAKING DECISIION MAKING DECISIION MAKING DECISIION MAKING DECISIION MAKING DECISIION MAKING DECISIION MAKING DECISIION MAKING 




# CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES 
# CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES 
class Hole():
    def __init__(self):
        self.hole_percent = 0
        self.hole_full = ['', '']
        self.hole_suit = ['', '']
        self.hole_card = ['', '']
        self.pos = hole_pos()

        self.hand_val = 0

        
        self.hand_rank = 0
        self.hand_outs = 0
        self.outs_por = 0
        
        self.flush_naipe = 0
        self.str_min_card = 0
        self.pair_card = 0

    def HolePercent(self):
        s = list()
        c = list()
        
        if CARDS_DIC[self.hole_full[0][0]] > CARDS_DIC[self.hole_full[1][0]]:
            c.append(self.hole_full[0][0])
            c.append(self.hole_full[1][0])
        else:
            c.append(self.hole_full[1][0])
            c.append(self.hole_full[0][0])

        s.append(self.hole_full[0][1])
        s.append(self.hole_full[1][1])

        f_hand = ''
        f_hand += c[0]
        f_hand += c[1]
        if s[0] == s[1]:
            f_hand += 's'
        else:
            f_hand += 'o'
        
        f_hand = np.where(HOLE_RANK == f_hand)  
        
        self.hole_percent = int(100 - (f_hand[0][0] * 0.5952))

    def UpdateOutsPercent(self, breakcounter):
        self.outs_por = (float(self.hand_outs)/float(CARDS_LEFT[breakcounter]))*100
 
class Board():
    def __init__(self):
        self.suit_check = [0, 0, 0, 0]              # suit_check    -   suit_check
        self.suit_index = [0, 0, 0, 0]              # suit_index  -   suit_index
        self.str_check = [0, 0, 0, 0]               # str_check     -   str_check
        self.str_index = [0, 0, 0, 0]               # str_index   -   str_index
        self.pair_check = [0, 0, 0, 0]              # pair_check          -   pair_check
        self.pair_index = [0, 0, 0, 0]              # pair_index        -   pair_index
        self.board = ['', '', '', '', '', '']
        self.board_pos = board_pos()
    
    def UpdateBoard(self, breakcounter):
        if breakcounter == 1:
            for x in range(1,4):   
                while(self.board[x] == False or self.board[x] == ''):
                    self.board[x] = read_hole(crop_img(GetFrame(coor=TABLECOOR), self.board_pos[x]))
                    
        if breakcounter > 1 and breakcounter < 4:
            x = breakcounter + 2
            print('board ',x)
            while(self.board[x] == False or self.board[x] == ''):
                self.board[x] = read_hole(crop_img(GetFrame(coor=TABLECOOR), self.board_pos[x]))
    
    def UpdateBoardStats(self, t):
        count = 0
        pair_check = 0
        pair_index = 0
        suit_check = 0
        board_cards, board_suit = break_cards(self.board)
        suit_count = [0, 0, 0, 0]
        pair_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        cards_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        board_str_counter = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        board_cards_val = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        gutshot = [False, False, False, False, False, False, False, False, False, False, False, False, False, False]
        
        if True:                            # BOARD SUIT VALUES.
            for x in board_suit:
                if x != '':
                    suit_count[SUIT_DIC[x]] += 1
                    count += 1
            
            if count == 3:                # FLOP
                if suit_count.count(1) == 3:
                    suit_check = 0
                elif suit_count.count(2) == 1:
                    suit_check = 1
                elif suit_count.count(3) == 1:
                    suit_check = 3
            
            if count == 4:                # TURN
                if suit_count.count(2) == 2:
                    suit_check = 2
                elif suit_count.count(3) == 1:
                    suit_check = 3
                elif suit_count.count(4) == 1:
                    suit_check = 4
                elif suit_count.count(2) == 1:
                    suit_check = 1
                else:
                    suit_check = 0
                    
            if count == 5:                # RIVER
                if suit_count.count(5) == 1:
                    suit_check = 5
                elif suit_count.count(4) == 1:
                    suit_check = 4
                elif suit_count.count(3) == 1:
                    suit_check = 3
                else:
                    suit_check = 0
        # BOARD SUIT VALUES END.
        
        if True:                            # BOARD CARDS VALUES.
            for x in board_cards:
                if x != '':
                    cards_count[CARDS_DIC[x]] += 1
                    if x == 'A':
                        cards_count[0] += 1
                    
            for x in range(len(cards_count)):
                count = 0
                if cards_count[x] > 0:
                    board_str_counter[x] += 1 
                    while True:
                        count += 1
                        if (x+count) < 13:
                            if cards_count[x+count] > 0:
                                board_str_counter[x] += 1   
                            elif cards_count[x+count+1] > 0 and not gutshot[x]:
                                board_str_counter[x] += 2 
                                count += 1
                                gutshot[x] = True
                            else:
                                break
                        elif (x+count) == 13:
                            if cards_count[x+count] > 0:
                                board_str_counter[x] += 1 
                        else:
                            break
                            
            max = fmax(board_str_counter)

            for x in range(len(board_str_counter)):
                if board_str_counter[x] >= 5:
                    if not gutshot[x]:
                        board_cards_val[x] = 5
                    if gutshot[x]:
                        board_cards_val[x] = 2
     
                elif board_str_counter[x] == 4:
                    if not gutshot[x]: 
                        if x < 11 and x > 0:
                            board_cards_val[x] = 4   
                        else:
                            board_cards_val[x] = 3
                    elif len(self.board) < 4:
                        board_cards_val[x] = 1
                    else:
                        board_cards_val[x] = 0
                        
                        
                elif board_str_counter[x] == 3 and not gutshot[x]: # > 0 and < 11 == espaco pra up down    
                    board_cards_val[x] = 1   
        # BOARD CARDS VALUES END.
        
        if True:                            # BOARD PAIRS VALUES.   pair_check
            for x in board_cards:
                if x != '':
                    pair_count[CARDS_DIC[x]] += 1
                    if x == 'A':
                        pair_count[0] += 1
                        
            if fmax(pair_count) == 4:
                pair_check = 7            # quad
                pair_index = fmax_index(pair_count)
            
            elif fmax(pair_count) == 3 and fmax(pair_count, 2) == 2:
                pair_check = 6            # FULLHOUSE
                pair_index = fmax_index(pair_count)
            elif fmax(pair_count) == 3:
                pair_check = 3            # TRIPLE
                pair_index = fmax_index(pair_count)
            
            elif fmax(pair_count) == 2 and pair_count.count(2) == 2:
                pair_check = 2            # 2 PAIRS
                pair_index = fmax_index(pair_count)
            
            elif fmax(pair_count) == 2 and pair_count.count(2) == 1:
                pair_check = 1               # 1 PAIR
                pair_index = fmax_index(pair_count)
            
            else:
                pair_check = 0            # NOTHING
                pair_index = False

                
        #                   MAX INDEX 13
        #       STRAIGHT                    SUITS    
        # 0 = RAINBOW                   RAINBOW
        # 1 = BACKDOOR                  BACKDOOR
        # 2 = GUTSHOT                   DOUBLE BACKDOOR
        # 3 = STRAIGHT DRAW 1 SIDE      FLUSHDRAW 2 NEEDED
        # 4 = UPDOWN STRAIGHT DRAW      FLUSHDRAW 1 NEEDED
        # 5 = STRAIGHT MADE             FLUSH MADE

        if fmax(board_cards_val) == -1:
            board_cards_val[fmax_index(board_str_counter)] = 0
        
        self.suit_check[t.breakcounter] = suit_check
        self.suit_index[t.breakcounter] = fmax_index(suit_count)
        self.str_check[t.breakcounter] = board_cards_val[fmax_index(board_str_counter)]
        self.str_index[t.breakcounter] = fmax_index(board_str_counter)
        self.pair_check[t.breakcounter] = pair_check
        self.pair_index[t.breakcounter] = pair_index
    
    def ResetBoard(self):
        self.suit_check = [0, 0, 0, 0]
        self.suit_index = [0, 0, 0, 0]
        self.str_check = [0, 0, 0, 0]
        self.str_index = [0, 0, 0, 0]
        self.pair_check = [0, 0, 0, 0]
        self.pair_index = [0, 0, 0, 0]
        self.board = ['', '', '', '', '', '']
   
    def MaxCard(self): # return CARDS_DIC code
        c, s = break_cards(self.board)
        max = 0
        for x in c:
            if x != '':
                if CARDS_DIC[x] > max:
                    max = CARDS_DIC[x]
                
        return max
 
class Player(): # NEW
    def __init__(self, p):
        self.p = p
        self.position = False                                   # position of the player in the hand. 0=bb, 9=utg
        self.setor = ''                                         # position of the player in the hand. early, mid, late, blinds
        self.last_action = 0                                    # timer since last action
        self.action = ['None','None','None','None']             # timer since last action
        self.points = ['None','None','None','None']             # timer since last action
        self.points_total = 1                                   # calc point of actions
        self.decisao_calculada = ''                                                  # put in another class to self play
        
        
        self.is_allin = False                                   #se all in pula a tomada de decisao.
        self.have_cards = False                                 #se all in pula a tomada de decisao.
        self.is_playing = False                                 #se all in pula a tomada de decisao.
        
        
        self.cards_pos = cards_pos(p)                   # POSITIONS
        self.action_pos = action_pos(p)                 # POSITIONS
        self.hud_pos = hud_pos(p)                       # POSITIONS
        self.allin_pos = allin_pos(p)                   # POSITIONS
        
    
        if self.p == SELF_SEAT and SELF_PLAYING == True:     # SELF ESPECIAL VARS
            self.cards = Hole()
            self.can_fold = False
            self.can_check = False
            self.can_call = False
            self.can_raise = False
            self.raise_size_pos = raise_size_pos()

    def __str__(self):
        return "%s\t%s\t\t%s\t%s" % (self.p, self.action, self.points, POSITION_DICT[self.position])        
            
    def UpdateHole(self):
        if self.p == SELF_SEAT and SELF_PLAYING == True:
            self.cards.hole_full = [False, False]
            temp = False
            
            while temp == False or temp == 'nada':
                frame = GetFrame(coor=TABLECOOR)
                temp = read_hole(crop_img(frame, self.cards.pos[0]))
            
            self.is_playing = True
            self.have_cards = True
            self.cards.hole_full[0] = temp
            self.cards.hole_full[1] = read_hole(crop_img(frame, self.cards.pos[1]))
            
    def CheckCards(self):
        result = FindTemplate(template_input=DIR+CHECKCARDS, frame=crop_img(GetFrame(coor=TABLECOOR), self.cards_pos), wanted='filename_or_nada')

        if SELF_PLAYING:
            if self.p != SELF_SEAT:
                if result != 'nada':
                    self.have_cards = True
                    self.is_playing = True
                else:
                    self.have_cards = False
                    self.is_playing = False
                
            elif self.cards.hole_full[0] == '':
                self.UpdateHole()
                self.cards.hole_card, self.cards.hole_suit = break_hole(self.cards.hole_full)
                self.have_cards = True
                self.is_playing = True
        
        else: # if NOT SELF_PLAYING:
            if result != 'nada':
                self.have_cards = True
                self.is_playing = True
            else:
                self.have_cards = False
                self.is_playing = False
                
    def reset(self):
        #self.allin_count = 0
        self.limp = False
        self.limp_count = 0
        self.bet = False
        self.bet_ammount = 0
        self.bet_position = -1
    
    def AllinCheck(self, breakcounter):
        if FindTemplate(frame=crop_img(GetFrame(coor=TABLECOOR), self.allin_pos), template_input=DIR+'template/allin/*.png', wanted='filename_or_nada') == 'allin':
            self.is_allin = True
            self.action[breakcounter] = 'allin'        
    
    def UpdatePossibleActions(self):
        if DecisionRead(0) == 'fold':
            self.can_fold = True
            self.can_check = False
            self.can_call = True
            
        if DecisionRead(1) == 'check':
            self.can_check = True
            self.can_call = False

        
        self.can_raise = True
        print('\n\n\nFold: ',self.can_fold,'\nCheck: ',self.can_check,'\nCall: ',self.can_call)
        
    def WaitAction(self, t, check):
        ti = time.time() 
        
        if self.p != SELF_SEAT or not SELF_PLAYING:         # if its not SELF
            if pl_thinking() == self.p:
                wait_pl_think(self.p)


            frame = self.GetActionFrame()

            self.action[t.breakcounter] = pl_action(frame)
            if self.action[t.breakcounter] == 'nada':
                self.AllinCheck(t.breakcounter)
                if not self.is_allin:
                    self.CheckCards()
                    if self.have_cards:
                        if self.action[t.breakcounter] == 'nada':
                            self.action[t.breakcounter] = 'check'
                    else:
                        self.action[t.breakcounter] = 'fold'


            # COD_ACTION = {'fold':0, 'check':1, 'call':2, 'bet':3, 'raise':3, 'allin':30}
                        
            # self.points[t.breakcounter] = COD_ACTION[self.action[t.breakcounter]]
            # self.points_total = self.points_total * COD_ACTION[self.action[t.breakcounter]]

            #print '\n1st action check fora de loop ',self.action[t.breakcounter],'\n\n'
                
        else:       #if its SELF wait for shortcuts.
            while not FindTemplate(template_input=DIR+WAITOWNACTION, wanted='bool', threshold=0.95, tp=TABLECOOR):
                pass
                
            self.UpdatePossibleActions()
            if self.can_fold:
                t.UpdateToCall()
   
 
 
     

            #self.decisao_calculada = escolha_decisao(a, t, players_in_hand, index, t.breakcounter, in_hand)
            self.decisao_calculada = DecisionMaking(t, check)

            
            
            
            #frame = self.WaitOwnAction()
            while FindTemplate(template_input=DIR+WAITOWNACTION, wanted='bool', threshold=0.95, tp=TABLECOOR):
                frame = GetFrame(coor=TABLECOOR)[self.raise_size_pos[1]:(self.raise_size_pos[1] + self.raise_size_pos[3]), self.raise_size_pos[0]:(self.raise_size_pos[0] + self.raise_size_pos[2])]
                


            if self.decisao_calculada != 'fold' and self.decisao_calculada != 'check' and self.decisao_calculada != 'call':
                bet = ReadRaise(template_input='C:/Users/RoKeR/Dropbox/poker/2019/template/raise/numbers/*.png', frame_input=frame)
                try:
                    self.raise_amount = int(bet)
                except:
                    cv2.imwrite('C:/Users/RoKeR/Dropbox/poker/2019/logs/hands/ERROR-BET-'+str(int(time.time()))+'.png', frame)
                    self.raise_amount = 1
                    
                self.action[t.breakcounter] = 'raise'
                self.AllinCheck(t.breakcounter)

                
            # elif self.decisao_calculada == 'fold'and not self.can_fold:
            #         self.action[t.breakcounter] = 'check'
                    
            # elif self.decisao_calculada == 'call'and not self.can_call:
            #         self.action[t.breakcounter] = 'check'
                    
            # elif self.decisao_calculada == 'check'and self.can_call:
            #         self.action[t.breakcounter] = 'call'
            elif self.decisao_calculada == 'fold':
                if self.can_fold:
                    self.action[t.breakcounter] = 'fold'
                else:
                    self.action[t.breakcounter] = 'check'
                    
            elif self.decisao_calculada == 'call':
                if self.can_call:
                    self.action[t.breakcounter] = 'call'
                else:
                    self.action[t.breakcounter] = 'check'
                    
            elif self.decisao_calculada == 'check':
                if self.can_call:
                    self.action[t.breakcounter] = 'call'
                else:
                    self.action[t.breakcounter] = 'check'

            
        COD_ACTION = {'fold':0, 'check':1, 'call':2, 'bet':3, 'raise':3, 'allin':30}
        self.points[t.breakcounter] = COD_ACTION[self.action[t.breakcounter]]
        self.points_total = self.points_total * COD_ACTION[self.action[t.breakcounter]]
                    
    def GetActionFrame(self):
        frame = np.array(ImageGrab.grab(bbox=((TP[0]+self.action_pos[0]), (TP[1]+self.action_pos[1]), (TP[0]+self.action_pos[0]+self.action_pos[2]), (TP[1]+self.action_pos[1]+self.action_pos[3]))))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    
    def WaitOwnAction(self):
        template = cv2.imread(DIR+WAITOWNACTION) #dir pra templates
        result = True

        while(result == True):   
            img = GetFrame()
            h, w = template.shape[:-1]
            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.8
            loc = np.where(res >= threshold)#check loc[1][0]

            for pt in zip(*loc[::-1]):  # pt contem coordenadas  **** (pt[0] + (w/2), pt[1] + (h/2)) = CENTRO ****
                frame = img[self.raise_size_pos[1]:(self.raise_size_pos[1] + self.raise_size_pos[3]), self.raise_size_pos[0]:(self.raise_size_pos[0] + self.raise_size_pos[2])]
                result = False
                break
            else:
                result = True

        return frame
       
class Table(): # NEW
    def __init__(self):
        self.bb = 100
        self.pot_pos = pot_pos()
        self.dealer_pos = dealer_pos()
        self.handnumber_pos = (28, 34, 225, 14) # NO NEDDED
        self.dealer = 0
        print('start')
        self.UpdatePot()    # MOD
        print('UpdatePot: '+str(self.pot))
        self.UpdateDealer() # OK
        print('UpdateDealer: '+str(self.dealer))
        self.UpdateHandNum()
        print('UpdateHandNum: '+str(self.hand_num))
        
        self.to_call = 1
        self.pot_odds_str = ''
        self.pot_odds_per = 0
        self.allin_count = 0
        
        self.board = Board()
        self.logging = True
        
        
        # POSITION SETUP
        self.to_call_pos = to_call_pos()
        self.to_allin_call_pos = to_allin_call_pos()

    def __str__(self):
        return "Board\nFlop: %s %s %s\nTurn: %s\nRiver: %s\n" % (self.board.board[1], self.board.board[2], self.board.board[3], self.board.board[4], self.board.board[5])
    
    def GetFrame(self):
        img = np.array(ImageGrab.grab(bbox=TABLECOOR))
        #show(img)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #ajusta RGB
        
    def UpdatePot(self):
        self.pot = False
        while self.pot == False:
            if PLAYING_MONEY:
                template = cv2.imread(DIR+POT['REALMONEY'])
            else:
                template = cv2.imread(DIR+POT['PLAYMONEY'])

            self.pot = int(ReadPot(template_input=template, frame_input=self.GetFrame()))
        
    def UpdateDealer(self): 
        # UPDATE self.dealer
        # USING funcs.check_coor()
        r = False
        while(r == False):            
            xx = FindTemplate(template_input=DIR+DEALER, tp=TABLECOOR, wanted='center', threshold=0.9)     
            for t_pos in range(1, (NPLAYERS+1)):
                try:
                    if check_coor(xx, self.dealer_pos[t_pos]):
                        r = t_pos
                        break
                except:
                    pass

        self.dealer = r
    
    def UpdateHandNum(self): # Only used to detect when new hand is dealt. cant be used in new pokerstars versions. 
        result = False
        while result == False:
            result = ReadHandNumber(template_input=DIR+HANDNUMBER, frame_input=TABLECOOR)
        
        self.hand_num = result
        #print('Hand num = ',self.hand_num)

    def GetHandNum(self):
        result = False
        while result == False:
            result = ReadHandNumber(template_input=DIR+HANDNUMBER, frame_input=TABLECOOR)
        
        return result

    def WaitForCards(self):
        while self.card_count < 2:
            self.card_count = 0
            self.pl = ['', '', '', '', '', '', '', '', '', '']
            for x in range(1, (NPLAYERS+1)): # loop all players
                self.pl[x] = Player(x)

    def ResetAllinCount(self):
        self.allin_count = 0

    def ResetData(self):
        self.allin_count = 0
        self.limp = False
        self.limp_count = 0
        self.bets = False
        self.bet_quantia = 0
        self.bets_pos = -1

    def ResetStreet(self):
        self.to_call = 1
        self.pot_odds_str = ''
        self.pot_odds_per = 0
          
    def WaitFastFold(self):
        temp_list = glob.glob(DIR+WAITFASTFOLD) #dir pra templates
        #check pra ver ql tipo de input.
        
        tempcount = 0 # count pra rodar tds templates.
        result = False

        while(result == False):
            if(tempcount > (len(temp_list)-1)):
                tempcount = 0
                
            template = cv2.imread(temp_list[tempcount])
            h, w = template.shape[:-1]
            res = cv2.matchTemplate(GetFrame(), template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.8
            loc = np.where(res >= threshold)#check loc[1][0]

            for pt in zip(*loc[::-1]):  # pt contem coordenadas  **** (pt[0] + (w/2), pt[1] + (h/2)) = CENTRO ****
                result = True
                break

            tempcount += 1
        return result
          
    def UpdateToCall(self): # (1096, 886, 201, 42) to allin call
        self.to_call = DecisionRead(1) # to_call == 'check' if check
        if self.to_call != 'check':
            self.to_call = int(self.to_call)
            
        self.UpdatePot()
        if self.to_call != 'check':
            self.UpdatePotOdds()

    def UpdatePotOdds(self):
        self.pot_odds_str = '1:',(float(self.pot)/float(self.to_call))
        self.pot_odds_per = ((self.to_call/(self.pot+self.to_call))*100)
   
class Check():
    def __str__(self):
        return "\n\n\nTable()\nPot: "+str(self.t.pot)+"\nDealer: "+str(self.t.dealer)+"\nLimp: "+str(self.t.limp)+"\nLimpCount: "+str(self.t.limp_count)+"\nBets: "+str(self.t.bets)+"\nBetPos: "+str(self.t.bets_pos)+"\nTo call: "+str(self.t.to_call)+"\nPot odds %: "+str(self.t.pot_odds_per)+"\nAllin Count: "+str(self.t.allin_count)+"\n\nBoard()\nBoardSuitVal: "+str(self.b.suit_check)+"\nBoardStrVal: "+str(self.b.str_check)+"\nBoardPairVal: "+str(self.b.pair_check)+"\nBoardSuitIndex: "+str(self.b.suit_index)+"\nBoardStrIndex: "+str(self.b.str_index)+"\nBoardPairIndex: "+str(self.b.pair_index)+"\nBoardFull: "+str(self.b.board)+"\n\nPlayer()\nposition: "+str(self.p.position)+"\n\nHole()\nHoleFull: "+str(self.h.hole_full)+"\nHoleSuit: "+str(self.h.hole_suit)+"\nHoleCard: "+str(self.h.hole_card)+"\nHole_%: "+str(self.h.hole_percent)+"\nHandRank: "+str(self.h.hand_rank)+"\nHandOuts: "+str(self.h.hand_outs)+"\nFlushNaipe: "+str(self.h.flush_naipe)+"\nStrMinCard: "+str(self.h.str_min_card)+"\nPairCard: "+str(self.h.pair_card)+"\nOuts%: "+str(self.h.outs_por)
        
    def __init__(self, t, p, b, h):
            # t.pot = POT AMMOUNT
            # t.dealer = DEALER POSITION
            # t.to_call = TO CALL AMMOUNT
            # t.pot_odds_per = POT ODDS %
            # t.allin_count
            # t.limp = Bool if there is limps/calls
            # t.limp_count = how many limp/calls
            # t.bets = Bool if there is bets/reraise
            # t.bets_pos = earliest bet position
            # t.players_in_hand = in_hand after hand_order()
        self.t = t

            
            # b.suit_check = {0:'RAINBOW', 1:'BACKDOOR', 2:'DOUBLE BACKDOOR', 3:'FSH DRAW 2 NEEDED', 4:'FSH DRAW 1 NEEDED', 5:'FLUSH MADE'}
            # b.str_check = {0:'NOTHING', 1:'BACKDOOR', 2:'GUTSHOT', 3:'1 SIDE STR DRAW', 4:'UPDOWN STR DRAW', 5:'STRAIGHT MADE'}
            # b.pair_check = {0:'NOTHING', 1:'PAIR', 2:'2 PAIRS', 3:'TRIPLE', 4:'FULLHOUSE', 5:'QUAD'}
    
            # b.suit_index = {0:'c', 1:'d', 2:'s', 3:'h'}
            # b.str_index = {0:'A', 1:'2', 2:'3', 3:'4', 4:'5', 5:'6', 6:'7', 7:'8', 8:'9', 9:'T', 10:'J', 11:'Q', 12:'K', 13:'A'}
            # b.pair_index = {0:'A', 1:'2', 2:'3', 3:'4', 4:'5', 5:'6', 6:'7', 7:'8', 8:'9', 9:'T', 10:'J', 11:'Q', 12:'K', 13:'A'}
        
            # b.board = ['', '', '', '', ''] Board Cards
        self.b = b
            
            
            # p.position = 0=BB 9=UTG
            # POSITION_DICT = {0:'BB', 1:'SB', 2:'BU', 3:'CO', 4:'HJ', 5:'MP+1', 6:'MP', 7:'UTG+1', 8:'UTG'}
            # p.setor = blinds, late, mid, early
            # p.can_call
            # p.can_fold
            # p.can_check
            # p.can_raise
        self.p = p
        
            
            # h.hole_full = ['', '']  HOLE CARDS FULL STRING 'Ad'
            # h.hole_suit = ['', '']  HOLE CARDS SUIT ONLY. 'd'
            # h.hole_card = ['', '']  HOLE CARDS CARD ONLY. 'A'
            # h.hole_percent = HAND % (27o  0 - 100 AA )
            # h.hand_rank = HAND's RANK  {0:'NOTHING', 1:'PAIR', 2:'2 PAIRS', 3:'TRIPLE', 4:'STRAIGHT', 5:'FLUSH', 6:'FULLHOUSE', 7:'QUAD', 8:'STRAIGHT FLUSH', 9:'ROYAL FLUSH'}
            # h.hand_outs = HOW MANY OUTS WE HAVE
            # h.flush_naipe = FMAX_INDEX NAIPE OF THE HAND
            # h.str_min_card = MIN CARD FROM STRAIGHT - IF NOT STRAIGHT, USELESS
            # h.pair_card = CARD THAT PAIRED
            # h.outs_por = % of hitting outs
        self.h = h
    
    def ToCallBB(self):
        try:
            return int(int(self.t.to_call)/int(self.t.bb))
        except:
            return 0
   
    def FullHole(self):
        r = ''
        DIC = {'2':1,'3':2,'4':3,'5':4,'6':5,'7':6,'8':7,'9':8,'T':9,'J':10,'Q':11,'K':12,'A':13}   

        if DIC[self.h.hole_card[0]] > DIC[self.h.hole_card[1]]:
            r += self.h.hole_card[0]
            r += self.h.hole_card[1]
        else:
            r += self.h.hole_card[1]
            r += self.h.hole_card[0]

        if self.h.hole_suit[0] == self.h.hole_suit[1]:
            r += 's'
        else:
            r += 'o'

        return r

    def InBlinds(self):
        if self.p.setor == 'BLINDS':
            return True
        else:
            return False

    def InLate(self):
        if self.p.setor == 'LATE':
            return True
        else:
            return False

    def InMid(self):
        if self.p.setor == 'MID':
            return True
        else:
            return False

    def InEarly(self):
        if self.p.setor == 'EARLY':
            return True
        else:
            return False  
    # Hole Cards Check List
    def IsPocketPair(self):
        if self.h.hole_card[0] == self.h.hole_card[1]:
            return True
        else:
            return False
    
    def IsSuited(self):
        if self.h.hole_suit[0] == self.h.hole_suit[1]:
            return True
        else:
            return False
    
    def IsConnected(self):
        s = (CARDS_DIC[self.h.hole_card[0]] - CARDS_DIC[self.h.hole_card[1]])
        if s == 1 or s == -1:
            return True
        elif self.h.hole_card[0] == 'A' or self.h.hole_card[1] == 'A':
            if self.h.hole_card[0] == '2' or self.h.hole_card[1] == '2':
                return True
            else:
                return False

    def IsBroadway(self):
        if CARDS_DIC[self.h.hole_card[0]] >= CARDS_DIC['T'] and CARDS_DIC[self.h.hole_card[1]] >= CARDS_DIC['T']:
            return True
        else:
            return False
    
    def IsOverpair(self):
        if self.h.hand_rank > 0 and CARDS_DIC[self.b.pair_index] > self.b.MaxCard():
            return True
        else:
            return False
    
    def HaveOneOverCard(self):
        if CARDS_DIC[self.h.hole_card[0]] > self.b.MaxCard() or CARDS_DIC[self.h.hole_card[1]] > self.b.MaxCard():
            return True
        else:
            return False
    
    def HaveTwoOverCards(self):
        if CARDS_DIC[self.h.hole_card[0]] > self.b.MaxCard() and CARDS_DIC[self.h.hole_card[1]] > self.b.MaxCard():
            return True
        else:
            return False
    
    def HowManyAboveMyHole(self):
        c, s = break_cards(self.b.board)
        counter = 0
        for x in c:
            if x != ''and CARDS_DIC[x] > CARDS_DIC[self.h.hole_card[0]] or CARDS_DIC[x] > CARDS_DIC[self.h.hole_card[1]]:
                counter += 1
                
        return counter

    def InPosition(self):
        Pos = [2, 1, 9, 8, 7, 6, 5, 4, 3] # Pos > == BETTER position
        max = 0
        max_seat = 0 # SEAT NUMBER IN THE TABLE 
        
        for p in self.t.players_in_hand:
            if Pos[self.t.pl[p].position] > max and p != SELF_SEAT:
                max = Pos[self.t.pl[p].position]
                max_seat = p
        
        if Pos[self.t.pl[SELF_SEAT].position] > max:
            return True
        else:
            return False

    def FirstAction(self):
        if self.p.action[self.t.breakcounter] == 'None':
            return True
        else:
            return False

    def CanCheck(self):
        if self.p.can_check:
            return True
        else:
            return False

    def CanCall(self):
        if self.p.can_call:
            return True
        else:
            return False

    def CanFold(self):
        if self.p.can_fold:
            return True
        else:
            return False

    def SBAgainstBB(self):
        Pos = [2, 1, 9, 8, 7, 6, 5, 4, 3]
        for p in self.t.players_in_hand:
            if Pos[self.t.pl[p].position] > 2 and p != SELF_SEAT and self.t.pl[p].is_playing:
                return True
        else:
            return False
    
    def VillanMoreAggressive(self):
        if self.t.breakcounter > 0 and len(self.t.players_in_hand) > 1:
            for p in self.t.players_in_hand:
                if p != SELF_SEAT and self.t.pl[p].points[self.t.breakcounter] > self.t.pl[p].points[self.t.breakcounter-1]:
                    return True
            
            return False

    def AnyAllin(self):
        for p in self.t.players_in_hand:
            if self.t.pl[p].is_allin:
                return True
                
        return False

    def HowManyInHand(self):
        return len(self.t.players_in_hand)-1
    # Board Check List
    def StrImproved(self):
        if self.t.breakcounter >= 2:   # only checks on turn or river.
            if self.b.str_check[self.t.breakcounter] > self.b.str_check[self.t.breakcounter-1]:
                return True
            else:
                return False
                
    def FlsImproved(self):
        if self.t.breakcounter >= 2:   # only checks on turn or river.
            if self.b.suit_check[self.t.breakcounter] > self.b.suit_check[self.t.breakcounter-1]:
                return True
            else:
                return False
   
    def PairImproved(self):
        if self.t.breakcounter >= 2:   # only checks on turn or river.
            if self.b.pair_check[self.t.breakcounter] > self.b.pair_check[self.t.breakcounter-1]:
                return True
            else:
                return False

    def PossibleFlushDraw(self):
        if self.b.suit_check[self.t.breakcounter] >= 3:
            return True
        else:
            return False

    def PossibleStrDraw(self):
        if self.b.str_check[self.t.breakcounter] >= 3:
            return True
        else:
            return False

    def PairOnBoard(self):
        if self.b.pair_check[self.t.breakcounter] >= 1:
            return True
        else:
            return False

    def CallOnce(self):
        if self.FirstAction():
            return 'call'
        else:
            return 'fold'

    def RaiseOnce(self, r):
        if self.FirstAction():
            return str(r)
        else:
            return 'fold'

    def HavePotOddsToCall(self):
        if self.t.pot_odds_per <= self.h.outs_por and not self.p.can_check:
            return True
        else:
            return False

    def StrDraw(self):
        if self.b.str_check[self.t.breakcounter] == 3 or self.b.str_check[self.t.breakcounter] == 4:
            return True
        else:
            return False
    
    def FshDraw(self):
        if self.t.suit_check == 3 or self.t.suit_check == 4:
            return True
        else:
            return False

    def HaveFshDraw(self):
        if self.b.suit_check[self.t.breakcounter] == 3 or self.b.suit_check[self.t.breakcounter] == 4 and self.b.suit_index == self.h.flush_naipe:
            return True
        else:
            return False

    def HaveStrDraw(self):
        pass


            
    # why = ''
    # acao = {'passive':'', 'normal':'', 'agressive':''}
    
    # {'fold', 'call', 'bet', '3x', '40%', '45%', '55%', '65%'}
    
    
    
    # Position Check
    # POSITION_DICT = {0:'BB', 1:'SB', 2:'BU', 3:'CO', 4:'HJ', 5:'MP+1', 6:'MP', 7:'UTG+1', 8:'UTG'}
# CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES 
# CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES CLASSES 



time.sleep(1)
print('STARTING...')
t = Table()
print("table set")
start = time.time()
t.fnkfile = 'logs/'+str(int(time.time()))+'.fnk'
t.dealer_old = t.dealer
t.hand_num_old = 0
t.hand_num_old = copy.deepcopy(t.hand_num)
hand_list = []
AddHandList(hand_list, t.hand_num)
t.breakcounter = -1
while True: # LOOP TOTAL                        new version = RunTable()          
    t.UpdateHandNum()
    #print(hand_list)
    #print('Hand number ',t.hand_num,' - ',t.hand_num_old)
    if t.breakcounter is -1:
        print('Esperando proxima mao')
        # print('Time: ',(time.time()-start))
    t.breakcounter = 0        # counter pras streets

    # NEW VERSION USAR Table.WaitNextHand()
    
    if t.hand_num not in hand_list: # ESPERA PROXIMO DEALER. PRA NAO COMECAR NO MEIO DA MAO
        #print('GOT In')
        #os.system('cls')
        t.UpdateDealer()
        t.board.ResetBoard()
        t.hand_num_old = copy.deepcopy(t.hand_num)
        AddHandList(hand_list, t.hand_num)
        t.dealer_old = t.dealer
        card_count = 0
        
        while card_count < 2:   # ESPERA DEALER DAR CARTAS
            card_count = 0
            t.pl = ['', '', '', '', '', '', '', '', '', '']
            for x in range(1,(NPLAYERS+1)): # loop all players
                t.pl[x] = Player(x)
                t.pl[x].CheckCards()
                if t.pl[x].have_cards:     # break na primeira vez q achar hole cards
                    card_count += 1
        
        zoombreak = False
        in_hand = []
        t.players_in_hand = []
        t.pl = ['', '', '', '', '', '', '', '', '', '']
        
        for x in range(1,(NPLAYERS+1)):   # SETA TODOS PLAYERS
            t.pl[x] = Player(x)
            t.pl[x].CheckCards()     # ALTERA SELF.CARDS

            if t.pl[x].have_cards:              # SE tiver cards.
                in_hand.append(x)
                t.pl[x].is_allin = False
     
        #print(t.dealer)
        #print(in_hand)
        #print('-'*20)
        in_hand = hand_order(t.dealer, in_hand) # ordena a mao
        in_use = 0                              # list in use
        position_counter = 0
        
        t.players_in_hand = list(in_hand[in_use])
        for x in in_hand[in_use][::-1]:
            # POSITION_DICT = {0:'BB', 1:'SB', 2:'BU', 3:'CO', 4:'HJ', 5:'MP+1', 6:'MP', 7:'UTG+1', 8:'UTG'}
            # 0-1 == Blinds     2 - 3 == Late       4 - 6 == mid        7-8 == Early
            t.pl[x].position = position_counter
            t.pl[x].setor = SETOR_DICT[position_counter]
            position_counter += 1

        jump = 99                               # jump pra qndo arrumar lista

        t.ResetData()
        

        #print('SETUP COMPLET')
        # STREET LOOP                       new version = StreetLoop()
        while len(in_hand[1]) > 1 and t.breakcounter <= 3 and t.allin_count <= (len(in_hand[1])-1):    # t.breakcounter [0] = pre-flop / [1] = flop / [2] = turn / [3] = river / [4+] = hands over     
            t.ResetStreet()
            t.UpdateDealer()      # update dealer
            
            if t.hand_num == t.hand_num_old:        # CHECKA PRA VER SE ESTA AINDA NA MESMA MAO
                while True:             # so sai desse loop com break
                    for index in range(0,len(t.players_in_hand)):          # loop de cada street   
                        if t.players_in_hand[index] != jump and len(in_hand[1]) > 1 and t.pl[t.players_in_hand[index]].is_playing == True and t.pl[t.players_in_hand[index]].is_allin == False:    
                            t.ResetStreet()
                            t.UpdatePot()
                            os.system('cls')
                            print('Hand number ',t.hand_num,' - ',t.hand_num_old)
                            print('$'*20,' ',STREET[t.breakcounter],' ','$'*20)
                            print(t)
                            print('Pot: ',t.pot)
                            print('limps ',t.limp,' - ',t.limp_count)
                            print('bets ',t.bets,' - ',t.bets_pos)
                            #print('\n\n',t.players_in_hand,'\n\n')
                            
                            
                            

                            if SELF_PLAYING and t.pl[SELF_SEAT].is_playing: # if im playing
                                t.pl[SELF_SEAT].cards.HolePercent()
                                print('Hole Cards: ',t.pl[SELF_SEAT].cards.hole_full[0],' - ',t.pl[SELF_SEAT].cards.hole_full[1],'  ',t.pl[SELF_SEAT].cards.hole_percent,'\n')
                                
                                if t.breakcounter > 0 and t.breakcounter < 3 and t.pl[SELF_SEAT].is_playing: #nao calcula odds no pre flop
                                        giving_hand = [t.pl[SELF_SEAT].cards.hole_full[0], t.pl[SELF_SEAT].cards.hole_full[1]]
                                        for x in range(1,t.breakcounter+3):
                                            giving_hand.append(t.board.board[x])
                                            
                                        t.pl[SELF_SEAT].cards.hand_rank, t.pl[SELF_SEAT].cards.outs_hand, t.pl[SELF_SEAT].cards.flush_naipe, t.pl[SELF_SEAT].cards.str_min_card, t.pl[SELF_SEAT].cards.pair_card = hand_outs(giving_hand)
                                        t.pl[SELF_SEAT].cards.UpdateOutsPercent(t.breakcounter)
                                        print('HAND RANK ',DICT_FINAL[t.pl[SELF_SEAT].cards.hand_rank])
                                        print('OUTS ',t.pl[SELF_SEAT].cards.outs_hand,' = ',t.pl[SELF_SEAT].cards.outs_por,'%')
                                    
                            if t.breakcounter > 0:# and t.players_in_hand[index] == SELF_SEAT:
                                t.board.UpdateBoardStats(t)
                                print('BOARD RANK STR', DICT_STR[t.board.str_check[t.breakcounter]],' ',t.board.str_check[t.breakcounter])
                                print('BOARD RANK FSH', DICT_FSH[t.board.suit_check[t.breakcounter]],' ',t.board.suit_check[t.breakcounter])
                                print('BOARD RANK PAIR', DICT_PAIR[t.board.pair_check[t.breakcounter]],' ',t.board.pair_check[t.breakcounter])
                            
                            #print 'ALLIN COUNTER: ',t.allin_count
                            
                            if ZOOM and t.pl[SELF_SEAT].cards.hole_percent < MIN_RANGE and Action() and t.pl[SELF_SEAT].position > 0 and AUTOPLAYING:
                                if Action():
                                    while Action():
                                        SendFastFold()
                                        print('FastFoldSending.....')
                                        time.sleep(0.5)
 
                                    
                                zoombreak = True
                                break 
                                
                            else:    
                                print('\nPlayer\tAction\tPosition')
                                for x in range(0,len(t.players_in_hand)):
                                    #if t.pl[t.players_in_hand[x]].have_cards:
                                    if index == x:
                                        print('--> ',t.pl[t.players_in_hand[x]])
                                    else:
                                        print(t.pl[t.players_in_hand[x]] )

                                jump = 99
                                print('\tesperando ',POSITION_DICT[t.pl[t.players_in_hand[index]].position])
                                
                                
                                
                                if SELF_PLAYING:
                                    check = Check(t, t.pl[SELF_SEAT], t.board, t.pl[SELF_SEAT].cards)
                                    t.pl[t.players_in_hand[index]].WaitAction(t, check)          # wait for action
                                else:
                                    check = ''
                                    t.pl[t.players_in_hand[index]].WaitAction(t, check)          # wait for action

                                #print('in hand len ',len(in_hand[1]))
                                #print('allin ',t.allin_count)
                             
                                
                                
                                
                                if ZOOM and t.players_in_hand[index] == SELF_SEAT and t.pl[t.players_in_hand[index]].action[t.breakcounter] == 'fold' and SELF_PLAYING: 
                                    zoombreak = True
                                    break

                                if t.pl[t.players_in_hand[index]].action[t.breakcounter] == 'fold' or t.pl[t.players_in_hand[index]].action[t.breakcounter] == 'empty' or t.pl[t.players_in_hand[index]].action[t.breakcounter] == 'sitout':
                                    in_hand[0].remove(t.players_in_hand[index])
                                    in_hand[1].remove(t.players_in_hand[index])
                                    #print(in_hand,' inhandn after remove')
                                    t.pl[t.players_in_hand[index]].is_playing = False
                                    t.pl[t.players_in_hand[index]].have_cards = False
                                    if SELF_PLAYING and t.players_in_hand[index] == SELF_SEAT:
                                        t.pl[t.players_in_hand[index]].reset()
                                        zoombreak = True        # set zoombreak as true to end this hand and wait for the next.(dont waste time/less bug)
                                        break

                                if t.pl[t.players_in_hand[index]].action[t.breakcounter] == 'call':
                                    t.limp = True
                                    t.limp_count += 1
                                    #print('allin check when calling')
                                    t.pl[t.players_in_hand[index]].AllinCheck(t.breakcounter)
                                    #print('allin check when calling ',t.pl[t.players_in_hand[index]].is_allin)
                                    
                                    if t.pl[t.players_in_hand[index]].is_allin == True:
                                        t.allin_count += 1

                                elif(t.pl[t.players_in_hand[index]].action[t.breakcounter] == 'raise' or t.pl[t.players_in_hand[index]].action[t.breakcounter] == 'bet' or t.pl[t.players_in_hand[index]].action[t.breakcounter] == 'allin'):
                                    t.bets = True
                                    t.bet_quantia += 1
                                    
                                    if t.bets_pos < t.pl[t.players_in_hand[index]].position:
                                        t.bets_pos = t.pl[t.players_in_hand[index]].position
                                    
                                    jump = t.players_in_hand[index]       # seta jumper pro proximo FOR LOOP
                                    t.pl[t.players_in_hand[index]].AllinCheck(t.breakcounter)
                                    
                                    if t.pl[t.players_in_hand[index]].is_allin == True:
                                        t.allin_count += 1

                                    index_temp = t.players_in_hand[index]
                                    t.players_in_hand = arruma_list(t.players_in_hand[index], list(in_hand[in_use])) 
                                    index = t.players_in_hand.index(index_temp)
                                    del index_temp
                                    break           # break pra comecar o index do comeco. break o for loop e nao o while.
                        
                        #print(in_hand,' inhand')
                        
                    if zoombreak:
                        print("ZOOMBREAK TRUE 1")
                        break
                        
                    if len(in_hand[1]) <= 1: # so tem um players na mao
                        break
                                                                                    # dava bug qndo eu contra + 1 e villan called allin, ficava esperando minha acao.
                    if t.allin_count >= (len(in_hand[1])-1) and t.allin_count >= 1: # and not t.pl[SELF_SEAT].is_playing: # pl > 1.
                        zoombreak = True
                        break
                        
                        
                    if t.players_in_hand[index] == t.players_in_hand[-1]:       # BREAK DEPOIS DO ULTIMO PLAYER. SE ELE NAO RAISE / BET 
                        t.breakcounter += 1
                        t.board.UpdateBoard(t.breakcounter)
                        for x in range(0,len(t.players_in_hand)):
                            if t.pl[t.players_in_hand[x]].have_cards:
                                t.pl[t.players_in_hand[x]].reset()

                        in_use = 1
                        t.players_in_hand = list(in_hand[in_use])

                        break           # break o while True loop.
                
                if zoombreak:
                    print("ZOOMBREAK TRUE 2")
                    break
            
            else:   # A MAO ACAOU. DEALER != OLD DEALER.
                break            # break while len(t.players_in_hand) > 1
                

        del in_hand
        del t.players_in_hand
        del jump
        del in_use          
        zoombreak = False

    #os.system('cls')
    #print('\n')




# *********************************************************************************************************************************************
# *********************************************************************************************************************************************

# *********************************************************************************************************************************************
# *********************************************************************************************************************************************

# *********************************************************************************************************************************************
# *********************************************************************************************************************************************

# *********************************************************************************************************************************************
# *********************************************************************************************************************************************

# *********************************************************************************************************************************************
# *********************************************************************************************************************************************







# def show(img): # DONE
#     #SHOW IMG, 'Q' TO DESTROY
#     while(True):
#         cv2.imshow('output',img)
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             cv2.destroyAllWindows()
#             break


# def inittopleft():
#     TABLE_X_SIZE = 1322
#     TABLE_Y_SIZE = 942

#     temp_list = glob.glob(DIR+TOPLEFT) #dir pra templates
#     tempcount = -1 # count pra rodar tds templates.
#     result = False

#     while result is False:
#         img = GetFrame()

#         if(tempcount >= (len(temp_list)-1)):
#             tempcount = -1

#         tempcount += 1
#         template = cv2.imread(temp_list[tempcount])
            
#         h, w = template.shape[:-1]
#         res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
#         threshold = 0.9
#         loc = np.where(res >= threshold)

#         for pt in zip(*loc[::-1]):
#             #cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)
#             result = pt
#             #show(img)
#             break
            
#     tp = result
#     tablecoor = (tp[0], tp[1], TABLE_X_SIZE, TABLE_Y_SIZE)
#     return tp, tablecoor




#TP, TABLECOOR = inittopleft()
#print(ReadHandNumber(frame_input=GetFrame()))
#print(FindTemplate(DIR+"template/read_hole/cards/*.png", wanted="show", threshold=0.9))


# *********************************************************************************************************************************************
# *********************************************************************************************************************************************

            # from funcs import *

# from constants import *
# from c import FindTemplate

# def show(img): # DONE
#     #SHOW IMG, 'Q' TO DESTROY
#     while(True):
#         cv2.imshow('output',img)
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             cv2.destroyAllWindows()
#             break

# def GetFrame(x1=0, y1=0, x2=1920, y2=1080, coor=False):
#     if not coor:
#         coordenadas = bbox=(x1, y1, (x1+x2), (y1+y2))
#     else:
#         coordenadas = bbox=(coor[0], coor[1], (coor[0]+coor[2]), (coor[1]+coor[3]))

#     img = np.array(ImageGrab.grab(coordenadas)) 
#     return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #ajusta RGB

# def fnprint():
#     print(TP,' *** ',TABLECOOR)


# def queryMousePosition(): # DONE
#     pt = POINT()
#     windll.user32.GetCursorPos(byref(pt))
#     return {"x": pt.x, "y": pt.y}

# def mpos(): # DONE
#     class POINT(Structure):
#         _fields_ = [("x", c_ulong), ("y", c_ulong)]

#     def queryMousePosition():
#         pt = POINT()
#         windll.user32.GetCursorPos(byref(pt))
#         return {"x": pt.x, "y": pt.y}

#     pos = queryMousePosition()
#     return pos

# def crop_img(img, cpc): # USING IN THE NEW VERSION - DONE
#     # cpc[0] = x inicial, [1] = y inicial, [2] = x added, [3] = y added
#     imgg = img
#     crop = imgg[cpc[1]:(cpc[1] + cpc[3]), cpc[0]:(cpc[0] + cpc[2])]
#     return crop  
       
            # def pre_process(img): # ???
            #     show(img)
            #     img = cv2.resize(img, (0,0), fx=10, fy=10) 
            #     kernel = np.ones((1, 1), np.uint8)
            #     img = cv2.dilate(img, kernel, iterations=1)
            #     img = img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #ajusta RGB
            #     #show(img)
            #     #img = cv2.erode(img, kernel, iterations=1)
            #     #img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            #     lol, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                
            #     return img

            # def tess_read(img, t=2): # ???
            #     tess_config = ("-c tessedit_char_whitelist=0123456789- -psm 4 -oem 3",
            #                    "-c tessedit_char_whitelist=123456789abcdefghijklmnopqrstuvxwyzABCDEFGHIJKLMNOPQRSTUVXWYZ0 -psm 4 -oem 3",
            #                    "-c tessedit_char_whitelist=1234567890-@# -psm 4 -oem 3",
            #                    "-c tessedit_char_whitelist=,.1234567890 -psm 4 -oem 3")
            #     result = pytesseract.image_to_string(Image.fromarray(img), config=tess_config[t])
            #     result = str(result)
            #     result = result.replace(" ", "")
            #     return result

# def tess(img=False, t=2): # ????
#     #img = pre_process(img)
#     print('WTF, TRYING TO USE TESSERACT!!!!!!!!!\n'*50)
#     time.sleep(10)
#     return False



# def find_template(img_in, temp_in): # USING IN THE NEW VERSION - DONE
#     # RETURN TEMPLATE FILE NAME ELSE: RETURN 'nada'
    
#     temp_list = glob.glob(temp_in) #dir pra templates
#     img = img_in
#     tempcount = 0 # count pra rodar tds templates.
#     result = False

#     while(result == False):
#         template = cv2.imread(temp_list[tempcount])
#         h, w = template.shape[:-1]
#         res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF_NORMED)
#         threshold = 0.8
#         loc = np.where(res >= threshold)#check loc[1][0]

#         for pt in zip(*loc[::-1]):  # pt contem coordenadas  **** (pt[0] + (w/2), pt[1] + (h/2)) = CENTRO ****
#             result = os.path.splitext(os.path.basename(temp_list[tempcount]))[0]
#             break

#         if(tempcount >= (len(temp_list)-1) and result == False):
#             result = 'nada' #print 'nada encontrado'
#             break

#         tempcount += 1

#     return result



# def add_pos(pos):
#     #TP = top_left()
#     r = ((TP[0] + pos[0]), (TP[1] + pos[1]), (TP[0] + pos[2]), (TP[1] + pos[3]))
#     return r

# def read_pot(imgg):
#     template = cv2.imread(DIR+'imgs/2.png')

#     if type(imgg) == str: #check type do input
#         # print 'str'
#         img = rgb_img(imgg)
#     if type(imgg) == np.ndarray: #check type do input
#         # print 'np array'
#         img = imgg
            
#     result = False
    
#     h, w = template.shape[:-1]
#     res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF_NORMED)
#     threshold = 0.9
#     loc = np.where(res >= threshold)
#     t = 0
#     for pt in zip(*loc[::-1]):
#         #cv2.rectangle(img, (pt[0] + w, 0), ((img.shape[1]), img.shape[0]), (0, 0, 255), 1)
        
#         img = crop_img(img, ((pt[0] + w), (pt[1]-2), (w*4), (h+4)))
#         #show(img)
#         break

    # t = tess(img, 3)
    # t = t.replace(",", "")
    # t = t.replace(".", "")
    
    # return float(t)

            # def read_hud(imgg='imgs/1.png'):
            #     temp_list = glob.glob('imgs/imgs/hud.png') #seta 1 template, pode ter mais pra frente
                
            #     tempcount = -1 # count pra rodar tds templates.
            #     result = False
            #     resultt = [['nada', 'nada'], ['nada', 'nada']]
            #     while(result == False):
            #         if type(imgg) == str: #check type do input
            #             # print 'str'
            #             img = rgb_img(imgg)
            #         if type(imgg) == np.ndarray: #check type do input
            #             # print 'np array'
            #             img = imgg
            #         tempcount += 1

            #         template = cv2.imread(temp_list[tempcount]) #seta o template
            #         if(tempcount >= (len(temp_list)-1)):
            #             result = 'nada'
                        
            #         h, w = template.shape[:-1]
            #         res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF_NORMED)
            #         #show(img)
            #         threshold = 0.9
            #         loc = np.where(res >= threshold)
            #         t = 0
            #         for pt in zip(*loc[::-1]):
            #             #cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)
            #             resultt[t] = pt
            #             #print temp_list[tempcount]
            #             t = t+1

                        
            #     tt = resultt
                


            #     if type(tt[0][0]) is not str and type(tt[1][0]) is not str:
            #         t = [0, 0, 0, 0]
            #         t[0] = tt[0][0] - 5
            #         t[1] = tt[0][1] - 5
            #         t[2] = tt[1][0] + 25 # +20 pra pegar final bot right
            #         t[3] = tt[1][1] + 25

            #         #img = get_img('imgs/0.png')
            #         #show(img)
            #         #show(img)
            #         #print t
            #         crop = img[t[1]:t[3], t[0]:t[2]] #cropa so o hud usando os @@@@ [top left, bot right]
            #         #show(crop)
            #         r = tess(crop) #string do hud

            #         # trata a string pra lista do hud
            #         r = str(r)
            #         r = r.replace("\n", "")
            #         r = r.replace("@", "")
            #         rr = r.split("#")

            #         for x in range(len(rr)): #replace null e ' - ' por 0
            #             if rr[x] == '' or rr[x] == '-':
            #                 rr[x] = 0

            #         #retorna rr  = array = ordem do hud.
            #         # [ hands, VPIP, PFR, BB/100, WTSD, WSD, LIVE BB STACK  ]
            #         #print rr
            #         #show(img)
            #     return rr
 
'''
    tempcount = 0 # count pra rodar tds templates.
    result = 'nada'
    
    while(result == 'nada'):
        img = GetFrame()
        if(tempcount > (len(temp_list)-1)):
            tempcount = 0
            
        template = cv2.imread(temp_list[tempcount])
        h, w = template.shape[:-1]
        res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF_NORMED)
        threshold = 0.95
        loc = np.where(res >= threshold)#check loc[1][0]
        
        for pt in zip(*loc[::-1]):  # pt contem coordenadas  **** (pt[0] + (w/2), pt[1] + (h/2)) = CENTRO ****
            #result = os.path.splitext(os.path.basename(temp_list[tempcount]))[0]
            if check_coor(pt, wait[p]) != True:
                result = True
                break
  
        tempcount += 1
        #retorna string da acao. [ bet call check fold ]
        #retorn 'nada' if nothing found.
    return result
'''



# def GetHandNumber():
#     result = False
#     template = cv2.imread(DIR+'d/f.png')
#     size = (28, 34, 225, 14)
#     while result == False:
#         img = GetFrame()
        
#         h, w = template.shape[:-1]
#         res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF_NORMED)
#         threshold = 0.9
#         loc = np.where(res >= threshold)
        
#         for pt in zip(*loc[::-1]):
#             #cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)
#             #result = os.path.splitext(os.path.basename(template))[0]
#             #img_to_read = crop_img(img, ((0), (pt[1] + h), (img.shape[1]), (img.shape[0])))
#             img_to_read = crop_img(img, ((pt[0]+w+1), (pt[1]-1), (size[2]-(pt[0]+w)), size[3]))
#             result = tess(img_to_read, 3)
#             #show(tamanho(img_to_read,10))
#             break
        
#     return result


'''
def calc_por_from_outs(outs, breakcounter):
    return (float(outs)/float(CARDS_LEFT[breakcounter]))*100
'''
    

# def FoldConfirmation():
#     temp_list = glob.glob(DIR+WAITFASTFOLD) #dir pra templates
#     #check pra ver ql tipo de input.
    
#     result = False
#     for temp in temp_list:
#         print('FOLDCONFIRMATION')
#         template = cv2.imread(temp)
#         h, w = template.shape[:-1]
#         res = cv2.matchTemplate(GetFrame(), template, cv2.TM_SQDIFF_NORMED)
#         threshold = 0.8
#         loc = np.where(res >= threshold)#check loc[1][0]

#         for pt in zip(*loc[::-1]):  # pt contem coordenadas  **** (pt[0] + (w/2), pt[1] + (h/2)) = CENTRO ****
#             result = True
#             break
            
#     return result
    
# def Action():
#     temp_list = glob.glob(DIR+WAITOWNACTION) #dir pra templates
#         #check pra ver ql tipo de input.

#     result = False
#     for temp in temp_list:
#         template = cv2.imread(temp)
#         h, w = template.shape[:-1]
#         res = cv2.matchTemplate(GetFrame(), template, cv2.TM_SQDIFF_NORMED)
#         threshold = 0.8
#         loc = np.where(res >= threshold)#check loc[1][0]

#         for pt in zip(*loc[::-1]):  # pt contem coordenadas  **** (pt[0] + (w/2), pt[1] + (h/2)) = CENTRO ****
#             result = True
#             break

#     return result













# def DecideAgression(t, check):
#     setup_agr = [[2, 2], [4, 4], [4, 12], [8, 24]]
#     max_points = 0

#     for p in t.players_in_hand:
#         if t.pl[p].points_total > max_points:
#             max_points = t.pl[p].points_total
        
#     if max_points < setup_agr[t.breakcounter][0]:
#         return 'aggressive' #'agressive'
#     elif max_points > setup_agr[t.breakcounter][1]:
#         return 'passive' #'passive'
#     else:
#         return 'normal' #'normal'


