import cv2
import numpy as np
from PIL import ImageGrab
import pyautogui
import time
import glob
from PIL import Image
import pytesseract
import os
import string
import keyboard
from ctypes import windll, Structure, c_ulong, byref
from posfunc import *
from constants import *

import  itertools
from math import sqrt

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'

def GetFrame(tablecoor):
    img = np.array(ImageGrab.grab(bbox=tablecoor))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #ajusta RGB

def fnprint():
    print DIR

def fnk(file, string):
    DIR = 'C:/Users/RoKeR/Dropbox/poker/beta/'

    with open(DIR+file, 'a') as fl:
        fl.write(string+'\n')
        fl.flush()

def get_screen(x1=0, y1=0, x2=1280, y2=720): # DONE
    #screen grab and convert to numpyArray.
    img = np.array(ImageGrab.grab(bbox=(x1, y1, (x1+x2), (y1+y2)))) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #ajusta RGB
    return img  #return NUMPYARRAY
''' MOVED TO NEW TABLE CLASS
def get_table(tp): #get once per frame. 
    img = np.array(ImageGrab.grab(bbox=(tp[0], tp[1], (tp[0]+1323), (tp[1]+945)))) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #ajusta RGB
    return img  #return NUMPYARRAY
'''
def pos_screen(imgg, pos): # DONE
    # cpc[0] = x inicial, [1] = y inicial, [2] = x added, [3] = y added
    img = crop_img(imgg, pos)
    return img  #return NUMPYARRAY

def screen(pt, x, y, xx, yy): # DONE
    #screen grab and convert to numpyArray.
    img = np.array(ImageGrab.grab(bbox=((pt[0]+x), (pt[1]+y), (pt[0]+xx), (pt[1]+yy)))) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #ajusta RGB
    return img  #return NUMPYARRAY

def get_img(filename): # DONE
    #default = 'C:\Python27\python-projects\poker\'
    img = cv2.imread(filename, 0)
    return img

def rgb_img(filename): # DONE
    #default = 'C:\Python27\python-projects\poker\'
    img = cv2.imread(filename)
    return img
    
def show(img): # DONE
    #SHOW IMG, 'Q' TO DESTROY
    while(True):
        cv2.imshow('output',img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

def queryMousePosition(): # DONE
    pt = POINT()
    windll.user32.GetCursorPos(byref(pt))
    return {"x": pt.x, "y": pt.y}

def mpos(): # DONE
    class POINT(Structure):
        _fields_ = [("x", c_ulong), ("y", c_ulong)]

    def queryMousePosition():
        pt = POINT()
        windll.user32.GetCursorPos(byref(pt))
        return {"x": pt.x, "y": pt.y}

    pos = queryMousePosition()
    return pos

def crop_img(img, cpc): # USING IN THE NEW VERSION - DONE
    # cpc[0] = x inicial, [1] = y inicial, [2] = x added, [3] = y added
    imgg = img
    crop = imgg[cpc[1]:(cpc[1] + cpc[3]), cpc[0]:(cpc[0] + cpc[2])]
    return crop  
       
def pre_process(img): # ???
    img = resize(img)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = gray(img)
    #show(img)
    #img = cv2.erode(img, kernel, iterations=1)
    #img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    lol, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    return img

def tess_read(img, t=2): # ???
    tess_config = ("-c tessedit_char_whitelist=0123456789- -psm 4 -oem 3",
                   "-c tessedit_char_whitelist=123456789abcdefghijklmnopqrstuvxwyzABCDEFGHIJKLMNOPQRSTUVXWYZ0 -psm 4 -oem 3",
                   "-c tessedit_char_whitelist=1234567890-@# -psm 4 -oem 3",
                   "-c tessedit_char_whitelist=,.1234567890 -psm 4 -oem 3")
    result = pytesseract.image_to_string(Image.fromarray(img), config=tess_config[t])
    result = str(result)
    result = result.replace(" ", "")
    return result

def tess(img, t=2): # ????
    img = pre_process(img)
    return tess_read(img, t)

def resize(img): # DONE
    #img = np.array(int(img))
    img = cv2.resize(img, (0,0), fx=10, fy=10) 
    return img

def tamanho(img, ratio):
    img = cv2.resize(img, (0,0), fx=ratio, fy=ratio) 
    return img

def gray(img): # DONE
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #ajusta RGB
    return img

def pl_action(img): #retorna string da acao. [ bet call check fold ]   ESPERAR 3 SECS PRA LER DENOVO PRA NAO LER ACAO DA RODADA ANTERIOR
    # 14/05 DONE
    result = False
    #print '$$$$ ENTROU PL_ACTION $$$$'
    #img =  'imgs/11.png' #image input. screen cap or file.
    temp_list = glob.glob(DIR+'template/actions/*.png')
    tempcount = -1

    if type(img) == str: #check type do input
        # print 'str'
        img = rgb_img(img)
        
    elif type(img) == np.ndarray: #check type do input
        # print 'np array'
        img = img
        
    while(result == False):
        if(tempcount >= (len(temp_list)-1)):
            result = 'nada'
            break
            
        tempcount += 1
        template = cv2.imread(temp_list[tempcount])

        h, w = template.shape[:-1]                       
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.95
        loc = np.where(res >= threshold)#check loc[1][0]

        for pt in zip(*loc[::-1]):  # pt contem coordenadas  **** (pt[0] + (w/2), pt[1] + (h/2)) = CENTRO ****
            result = os.path.splitext(os.path.basename(temp_list[tempcount]))[0]
            #retorna string da acao. [ bet call check fold ]
            break
    if result == 'sitout' or result == 'empty' or result == 'take' or result == 'r':
        result = 'fold'

    #if result != 'nada':       #Salva img da action
        #cv2.imwrite(DIR+'action_read/'+str(result)+'/'+str(time.time())+'.png', img)
    return result
''' MOVED TO TABLE NEW CLASS
def top_left(): # DONE
    # loop until find template in the screen
    # RETURN (x, y) top left
    temp_list = glob.glob(DIR+TOPLEFT) #dir pra templates
    tempcount = -1 # count pra rodar tds templates.
    result = False

    while(result == False):
        img = get_screen(0, 0, 1920, 1080)
        tempcount += 1
        template = cv2.imread(temp_list[tempcount])
        if(tempcount >= (len(temp_list)-1)):
            tempcount = -1
            
        h, w = template.shape[:-1]
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.9
        loc = np.where(res >= threshold)

        for pt in zip(*loc[::-1]):
            result = pt
            break

    return result
'''
def check_flop(img):
    temp_list = glob.glob(DIR+'cards/*.png')
    tempcount = 0
    
    if type(img) == str: #check type do input
        #print 'str'
        img = rgb_img(img)
        
    elif type(img) == np.ndarray: #check type do input
        #print 'np array'
        img = img
        
    r = False
    
    while(r == False):
        #print os.path.splitext(os.path.basename(temp_list[tempcount]))[0]
        template = cv2.imread(temp_list[tempcount])
        h, w = template.shape[:-1]							
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.95
		#show(img)
        loc = np.where(res >= threshold)#check loc[1][0

        for pt in zip(*loc[::-1]):  # pt contem coordenadas  **** (pt[0] + (w/2), pt[1] + (h/2)) = CENTRO ****
			#cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)
            r = os.path.splitext(os.path.basename(temp_list[tempcount]))[0]
            break

        if(tempcount >= (len(temp_list)-1) and r == False):
            r = 'nada'
            break
			
        tempcount += 1

    if r == 'nada':
        r = False
        
    return r

def find_template(img_in, temp_in): # USING IN THE NEW VERSION - DONE
    # RETURN TEMPLATE FILE NAME ELSE: RETURN 'nada'
    
    temp_list = glob.glob(temp_in) #dir pra templates
    img = img_in
    tempcount = 0 # count pra rodar tds templates.
    result = False

    while(result == False):
        template = cv2.imread(temp_list[tempcount])
        h, w = template.shape[:-1]
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold)#check loc[1][0]

        for pt in zip(*loc[::-1]):  # pt contem coordenadas  **** (pt[0] + (w/2), pt[1] + (h/2)) = CENTRO ****
            result = os.path.splitext(os.path.basename(temp_list[tempcount]))[0]
            break

        if(tempcount >= (len(temp_list)-1) and result == False):
            result = 'nada' #print 'nada encontrado'
            break

        tempcount += 1

    return result

def wait_fastfold(tp):
    temp_list = glob.glob(DIR+'imgs/decision/fastfold/*.png') #dir pra templates
    #check pra ver ql tipo de input.
    
    tempcount = 0 # count pra rodar tds templates.
    result = False

    while(result == False):
        print 'WAITING FAST FOLD'
        img = get_table(tp)
        
        if(tempcount > (len(temp_list)-1)):
            tempcount = 0
        template = cv2.imread(temp_list[tempcount])
        h, w = template.shape[:-1]
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold)#check loc[1][0]

        for pt in zip(*loc[::-1]):  # pt contem coordenadas  **** (pt[0] + (w/2), pt[1] + (h/2)) = CENTRO ****
            result = os.path.splitext(os.path.basename(temp_list[tempcount]))[0]
            #result = temp_list[tempcount]
            break

        
 
        tempcount += 1
        #retorna string da acao. [ bet call check fold ]
        #retorn 'nada' if nothing found.
    
    return result
    
def read_hole(img='imgs/hole.png'):
    temp_list = glob.glob(DIR+'cards/*.png')
    tempcount = 0
    
    if type(img) == str: #check type do input
        #print 'str'
        img = rgb_img(img)
        
    elif type(img) == np.ndarray: #check type do input
        #print 'np array'
        img = img
        
    r = False
    while(r == False):
        #print os.path.splitext(os.path.basename(temp_list[tempcount]))[0]
        template = cv2.imread(temp_list[tempcount])
        h, w = template.shape[:-1]							
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.95
		#show(img)
        loc = np.where(res >= threshold)#check loc[1][0

        for pt in zip(*loc[::-1]):  # pt contem coordenadas  **** (pt[0] + (w/2), pt[1] + (h/2)) = CENTRO ****
			#cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)
            r = os.path.splitext(os.path.basename(temp_list[tempcount]))[0]
            break

        if(tempcount >= (len(temp_list)-1) and r == False):
            r = 'nada'
            break
			
        tempcount += 1

    if r == 'nada':
        r = False
        
    return r

def add_pos(pos, tp):
    #tp = top_left()
    r = ((tp[0] + pos[0]), (tp[1] + pos[1]), (tp[0] + pos[2]), (tp[1] + pos[3]))
    return r

def read_pot(imgg):
    template = cv2.imread(DIR+'imgs/2.png')

    if type(imgg) == str: #check type do input
        # print 'str'
        img = rgb_img(imgg)
    if type(imgg) == np.ndarray: #check type do input
        # print 'np array'
        img = imgg
            
    result = False
    
    h, w = template.shape[:-1]
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where(res >= threshold)
    t = 0
    for pt in zip(*loc[::-1]):
        #cv2.rectangle(img, (pt[0] + w, 0), ((img.shape[1]), img.shape[0]), (0, 0, 255), 1)
        
        img = crop_img(img, ((pt[0] + w), (pt[1]-2), (w*4), (h+4)))
        #show(img)
        break

    t = tess(img, 3)
    t = t.replace(",", "")
    t = t.replace(".", "")
    
    return float(t)
    
def find_dealer(frame, dealer_pos, tp): 
    temp_list = glob.glob(DIR+'teste/dealer/*.png') #dir pra templates
    r = False
    dp = dealer_pos
    while(r == False):
        #tp = top_left()
        frame = get_table(tp)

        for temp_x in temp_list:
            template = rgb_img(temp_x)
            img = frame
            h, w = template.shape[:-1]							
            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.6
            loc = np.where(res >= threshold)#check loc[1][0

            for xx in zip(*loc[::-1]):  # pt contem coordenadas  **** (pt[0] + (w/2), pt[1] + (h/2)) = CENTRO ****
                break
        
        for t_pos in range(1,10):
            try:
                if check_coor(xx, dp[t_pos]):
                    r = t_pos
                    break
            except:
                pass
                
    return r
    
def decision_read(tp, d): # d = [ 0 = left, 1 = mid, 2 = right ] !!!!!!!!!! MELHORAR !!!!!!!!!!
    # 0 = return FOLD or 'nada'
    # 1 = return check ou the call amount
    # 2 = return the min_raise amount
    
    temp_dic = {'0':DIR+'imgs/decision/fold.png', '1':DIR+'imgs/decision/*.png', '2':DIR+'imgs/decision/raise.png'}
    coor = (0,0,0,0)
    temp_list = glob.glob(temp_dic[d])
    tempcount = 0
    
    img = get_table(tp)
    #show(img)
    result = False
    while result == False:
        template = cv2.imread(temp_list[tempcount])
        
        h, w = template.shape[:-1]
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.9
        loc = np.where(res >= threshold)
        
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)
            result = os.path.splitext(os.path.basename(temp_list[tempcount]))[0]
            if d != '0' and result != 'check': # to read numb
                #img_to_read = crop_img(img, ((0), (pt[1] + h), (img.shape[1]), (img.shape[0])))
                img_to_read = crop_img(img, (pt[0], (pt[1]+h), w, h))
                coor = [(pt[0]), (pt[1]+h), (w), (h)]
                result = tess(img_to_read, 3)
                #show(img_to_read)
            break
        
        if(tempcount >= (len(temp_list)-1) and result == False):
            result = 'nada'
            break
        
        tempcount += 1
        
    frame = img[coor[1]:(coor[1] + coor[3]), coor[0]:(coor[0] + coor[2])]
    if result == 'nada':
        return False, False
        
    else:
        return result, frame

def wait_action(pos, tp):
    last = pl_thinking()
    #tp = top_left()
    action = 'nada'
    #pos = (800,55,110,110)
    while (action == 'nada'): # check cards a cada 3 sec or so. if false remove pl
        if pl_thinking() != last:
            try:
                '''
                if keyboard.is_pressed('0'):
                    action = 'fold'
                if keyboard.is_pressed('.'):
                    action = 'call'
                '''
                frame = np.array(ImageGrab.grab(bbox=((tp[0]+pos[0]), (tp[1]+pos[1]), (tp[0]+pos[0]+pos[2]), (tp[1]+pos[1]+pos[3]))))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #frame = rgb_img('imgs/2.png')
                #cv2.imshow('output',frame)
                #cv2.waitKey(1)
                action = pl_action(frame)
            except:
                pass

    return action
 
def wait_cards(tp):
    cards_wait = True
    
    while cards_wait == True:
        print 'waiting cards'
        frame = get_table(tp)
        for x in range(1,10):
            a[x] = Player(frame, x)
            a[x].check_cards(frame)
            if a[x].cards:
                cards_wait = True
                break

def hand_order(d, in_hand):
    b = []
    c = []
    if d > 0 and d < 10 and len(in_hand) > 1:
        b = list(in_hand[((in_hand.index(d)+3)-(((in_hand.index(d)+2)/len(in_hand))*len(in_hand))):]) + list(in_hand[:((in_hand.index(d)+3)-(((in_hand.index(d)+2)/len(in_hand))*len(in_hand)))])
        c = list(in_hand[((in_hand.index(d)+1)-(((in_hand.index(d)+1)/len(in_hand))*len(in_hand))):]) + list(in_hand[:((in_hand.index(d)+1)-(((in_hand.index(d)+1)/len(in_hand))*len(in_hand)))])  
        return [b, c]

    else:
        return False

def arruma_list(d, lista):
    #print 'LISTA ',lista
    #print 'ARRUMADA ', list(lista[lista.index(d):]) + list(lista[:lista.index(d)])
    return list(lista[lista.index(d):]) + list(lista[:lista.index(d)])

def read_hud(imgg='imgs/1.png'):
    temp_list = glob.glob('imgs/imgs/hud.png') #seta 1 template, pode ter mais pra frente
    
    tempcount = -1 # count pra rodar tds templates.
    result = False
    resultt = [['nada', 'nada'], ['nada', 'nada']]
    while(result == False):
        if type(imgg) == str: #check type do input
            # print 'str'
            img = rgb_img(imgg)
        if type(imgg) == np.ndarray: #check type do input
            # print 'np array'
            img = imgg
        tempcount += 1

        template = cv2.imread(temp_list[tempcount]) #seta o template
        if(tempcount >= (len(temp_list)-1)):
            result = 'nada'
            
        h, w = template.shape[:-1]
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        #show(img)
        threshold = 0.9
        loc = np.where(res >= threshold)
        t = 0
        for pt in zip(*loc[::-1]):
            #cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)
            resultt[t] = pt
            #print temp_list[tempcount]
            t = t+1

            
    tt = resultt
    


    if type(tt[0][0]) <> str and type(tt[1][0]) <> str:
        t = [0, 0, 0, 0]
        t[0] = tt[0][0] - 5
        t[1] = tt[0][1] - 5
        t[2] = tt[1][0] + 25 # +20 pra pegar final bot right
        t[3] = tt[1][1] + 25

        #img = get_img('imgs/0.png')
        #show(img)
        #show(img)
        #print t
        crop = img[t[1]:t[3], t[0]:t[2]] #cropa so o hud usando os @@@@ [top left, bot right]
        #show(crop)
        r = tess(crop) #string do hud

        # trata a string pra lista do hud
        r = str(r)
        r = r.replace("\n", "")
        r = r.replace("@", "")
        rr = r.split("#")

        for x in range(len(rr)): #replace null e ' - ' por 0
            if rr[x] == '' or rr[x] == '-':
                rr[x] = 0

        #retorna rr  = array = ordem do hud.
        # [ hands, VPIP, PFR, BB/100, WTSD, WSD, LIVE BB STACK  ]
        #print rr
        #show(img)
    return rr

def twice(tp):
    frame = crop_img(get_table(tp), run_twice())
    r = find_template(frame, DIR+'template/twice/*.png')
    if r == 'nada':
        return False
        
    else:
        return True
        
def pl_thinking(tp):
    temp_list = glob.glob(DIR+'template/waiting/0.png') #dir pra templates
    #tp = top_left()
    img = get_table(tp)
    tempcount = 0 # count pra rodar tds templates.
    result = False

    while(result == False):
        if(tempcount > (len(temp_list)-1)):
            tempcount = 0
            break
            
        template = cv2.imread(temp_list[tempcount])
        h, w = template.shape[:-1]
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.95
        loc = np.where(res >= threshold)#check loc[1][0]

        for pt in zip(*loc[::-1]):  # pt contem coordenadas  **** (pt[0] + (w/2), pt[1] + (h/2)) = CENTRO ****
            #result = os.path.splitext(os.path.basename(temp_list[tempcount]))[0]
            result = pt
            #result = temp_list[tempcount]
            break
  
        tempcount += 1
        #retorna string da acao. [ bet call check fold ]
        #retorn 'nada' if nothing found.

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
    for x in range(1,10):
        if check_coor(p, r[x]):
            r = x
            break
    return r

def wait_pl_think(p, tp):
    temp_list = glob.glob(DIR+'template/waiting/0.png') #dir pra templates
    #tp = top_left()

    tempcount = 0 # count pra rodar tds templates.
    result = 'nada'
    wait = wait_pos()
    
    while(result == 'nada'):
        img = get_table(tp)
        if(tempcount > (len(temp_list)-1)):
            tempcount = 0
            
        template = cv2.imread(temp_list[tempcount])
        h, w = template.shape[:-1]
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
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
    pair_val = 0
    card_val = 0
    board_suit_val = 0
    hand_val = 0
    # out vals
    pair_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    suit_count = [0, 0, 0, 0]
    cards_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    board_str_counter = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    board_cards_val = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    gutshot = [False, False, False, False, False, False, False, False, False, False, False, False, False, False]

    if True:                            # BOARD SUIT VALUES.    board_suit_val
        for x in hand_suit:
            suit_count[SUIT_DIC[x]] += 1
        
        
        if len(hand_suit) == 5:                # FLOP
            if suit_count.count(5) == 1:
                board_suit_val = 5
            elif suit_count.count(4) == 1:
                board_suit_val = 4
                outs += 9
            elif suit_count.count(3) == 1:
                board_suit_val = 3
            elif suit_count.count(2) == 2:
                board_suit_val = 2
            elif suit_count.count(2) == 1:
                board_suit_val = 1
            else:
                board_suit_val = 0
                
        
        if len(hand_suit) == 6:                # TURN
            if suit_count.count(5) == 1 or suit_count.count(6) == 1:
                board_suit_val = 5
            elif suit_count.count(4) == 1:
                board_suit_val = 4
                outs += 9
            else:
                board_suit_val = 0
                
        if len(hand_suit) == 7:                # RIVER
            if suit_count.count(5) == 1 or suit_count.count(6) == 1 or suit_count.count(7) == 1:
                board_suit_val = 5
            else:
                board_suit_val = 0
    # BOARD SUIT VALUES END.

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
                       
        max = fmax(board_str_counter)
        
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

    if True:                            # BOARD PAIRS VALUES.   pair_val
        for x in hand_cards:
            pair_count[CARDS_DIC[x]] += 1


        if fmax(pair_count) == 4:
            pair_val = 5
            pair_card = fmax_index(pair_count)
        elif fmax(pair_count) == 3 and fmax(pair_count, 2) == 2:
            pair_val = 4
            pair_card = fmax_index(pair_count)
        elif fmax(pair_count) == 3:
            pair_val = 3
            pair_card = fmax_index(pair_count)
        elif fmax(pair_count) == 2 and pair_count.count(2) == 2:
            pair_val = 2
            pair_card = fmax_index(pair_count)
        elif fmax(pair_count) == 2 and pair_count.count(2) == 1:
            pair_val = 1
            pair_card = fmax_index(pair_count)
        else:
            pair_val = 0
            pair_card = fmax_index(pair_count)
    # BOARD PAIRS VALUES END.
    
    if board_suit_val == 5 and card_val == 5:
        if n_str == 9:
            hand_val = 9
        elif n_str < 9:
            hand_val = 8
    elif pair_val == 5:
        hand_val = 7
    elif pair_val == 4:
        hand_val = 6
    elif board_suit_val == 5:
        hand_val = 5
    elif card_val == 5:
        hand_val = 4
    elif pair_val == 3:
        hand_val = 3
    elif pair_val == 2:
        hand_val = 2
    elif pair_val == 1:
        hand_val = 1
    else:
        hand_val = 0
    
    if pair_card != '':
        return_val = pair_card
        
    if outs == 17:
        outs -= 2
    elif outs == 13:
        outs -= 1

    return hand_val, outs, return_val

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

 

def escolha_decisao(a, t, players_in_hand, index, breakcounter, in_hand=''):
    acao = {'passive':'', 'normal':'', 'agressive':''}
    setup_agr = [[2, 2], [4, 4], [4, 12], [8, 24]] # MIN / MAX OF NORMAL RANGE. < = PASSIVE PLAYER = PLAY AGGRO
    SELF_SEAT = 5
    pos = a[SELF_SEAT].position
    # HAND_RANGE[''][agr][pos]
    max_points = 0
    why = ''
    
    # {'fold', 'call', 'bet', '3x', '40%', '45%', '55%', '65%'}
    
    if True:            # SETUP
        if decision_read(t.tp, '0')[0]:
            canfold = True
            cancall = True
        else:
            canfold = False
            
        if type(decision_read(t.tp, '2')[0]) == str:
            canbet = True
            if type(decision_read(t.tp, '1')[0]) == str:
                if decision_read(t.tp, '1')[0] == 'check':
                    cancheck = True
                    cancall = False
                elif type(decision_read(t.tp, '1')[0]) == str:
                    cancheck = False
                    cancall = True
        else:
            canbet = False
            cancall = True
            cancheck = False
        
        for p in players_in_hand:
            if a[p].points > max_points:
                max_points = a[p].points
        
        if max_points < setup_agr[t.breakcounter][0]:
            agr = 'agressive'
        elif max_points > setup_agr[t.breakcounter][1]:
            agr = 'passive'
        else:
            agr = 'normal'
        
        print 'PLAY STYLE = ',agr
    
    
    if breakcounter == 0:   # PRE FLOP
        
        fnk(t.fnkfile, '*************************************************\n\nHand Number: '+str(t.hand_num)+
        '\nHole Cards: '+str(a[SELF_SEAT].hole[0])+' '+str(a[SELF_SEAT].hole[1])+' - '+str(a[SELF_SEAT].hole_per)+'%\tPos: '+str(POSITION_DICT[a[SELF_SEAT].position])+
        '\nPlayers at Start: '+str(players_in_hand)+
        '\n\n--------------------\n\n'+str(street[t.breakcounter])+
        '\nin_hand: '+str(in_hand)+
        '\nMax points: '+str(max_points)+'\tPlay Style: '+str(agr)+
        '\nLimps: '+str(t.limp)+'\tCount: '+str(t.limp_count)+
        '\nBets: '+str(t.bets)+'\tPosition: '+str(t.bets_pos)+'\n')

        if t.bets:      # BETS BEFORE
            if a[SELF_SEAT].hole_per >= HAND_RANGE['reraise'][agr][pos] and canbet:
                why = 'RE-RAISE'
                acao = {'passive':'40%', 'normal':'45%', 'agressive':'55%'}
            
            elif a[SELF_SEAT].hole_per >= HAND_RANGE['call'][agr][pos] or t.pot_odds_per < 15:
                why = 'CALL'
                acao = {'passive':'call', 'normal':'call', 'agressive':'call'}

        else:                                   # NO BETS
            if not canfold:    # if u CAN ONLY fold. check or raise
                if a[SELF_SEAT].hole_per >= HAND_RANGE['open'][agr][pos]:
                    why = 'DEFAULT OPEN'
                    acao = {'passive':'3x', 'normal':'3x', 'agressive':'3x'}

                else:
                    why = 'FREE CHECK'
                    acao = {'passive':'check', 'normal':'check', 'agressive':'check'}

            else:   # if u can fold = FOLD CALL RAISE
                if a[SELF_SEAT].hole_per >= HAND_RANGE['open'][agr][pos]:
                    why = 'DEFAULT OPEN'
                    acao = {'passive':'3x', 'normal':'3x', 'agressive':'3x'}

                elif a[SELF_SEAT].hole_per >= HAND_RANGE['call'][agr][pos] or t.pot_odds_per < 15:
                    why = 'CALL'
                    acao = {'passive':'call', 'normal':'call', 'agressive':'call'}

    # END OF PRE FLOP DECISION MAKING

    if breakcounter == 1:   # FLOP
        fnk(t.fnkfile, '\n\n--------------------\n\n'+str(street[t.breakcounter])+
        '\nin_hand: '+str(in_hand)+
        '\nMax points: '+str(max_points)+'\tPlay Style: '+str(agr)+
        '\nLimps: '+str(t.limp)+'\tCount: '+str(t.limp_count)+
        '\nBets: '+str(t.bets)+'\tPosition: '+str(t.bets_pos)+
        '\nBoard: '+str(t.board[1])+' '+str(t.board[2])+' '+str(t.board[3])+' '+str(t.board[4])+' '+str(t.board[5])+'\t Hole: '+str(a[SELF_SEAT].hole[0])+' '+str(a[SELF_SEAT].hole[1])+
        '\n\tBoard Flush:\t'+str(DICT_STR[t.board_suit_val])+' ( '+str(str(t.board_suit_val))+
        ' )\n\tBoard Straight:\t'+str(DICT_STR[t.board_str_val])+' ( '+str(str(t.board_str_val))+
        ' )\n\tBoard Pair:\t'+str(DICT_STR[t.pair_val])+' ( '+str(str(t.pair_val))+
        ' )\n\n\tHand Rank: '+str(DICT_FINAL[t.hand_rank])+' ( '+str(t.hand_rank)+
        ' )\n\nMy_outs: '+str(t.outs_hand)+' ( '+str(t.outs_por)+
        '% )\nPot size: '+str(t.pot)+
        '\nTo call: '+str(t.to_call)+
        '\nPot odds: '+str(t.pot_odds_str)+' ( '+str(t.pot_odds_per)+'% )\n')
        
        
    
    
        if t.hand_rank >= 4: # SE HAND RANK > STRAIGHT NO FLOP == GO TO TOWN
            why = 'Hand rank >= 4'
            acao = {'passive':'40%', 'normal':'45%', 'agressive':'55%'}
       
        elif DICT_FINAL[t.hand_rank] == 'TRIPLE':
            if (t.board_str_val < 4 and t.board_suit_val < 4):
                why = 'Triple with no flush/straight draw'
                acao = {'passive':'40%', 'normal':'45%', 'agressive':'55%'}
            
            else:
                if t.outs_por > t.pot_odds_per:
                    why = 'Triple ODDS CALL'
                    acao = {'passive':'call', 'normal':'call', 'agressive':'call'}
   
        elif DICT_FINAL[t.hand_rank] == 'PAIRS':
            if (t.board_str_val < 4 and t.board_suit_val < 4):
                why = '2 pairs with no flush/straight draw'
                acao = {'passive':'40%', 'normal':'45%', 'agressive':'55%'}
            
            else:
                if t.outs_por > t.pot_odds_per:
                    why = '2 pairs ODDS CALL'
                    acao = {'passive':'call', 'normal':'call', 'agressive':'call'}
               
        elif DICT_FINAL[t.hand_rank] == 'PAIR':
            if (t.board_str_val < 4 and t.board_suit_val < 4):
                if t.pair_val < 1: # SE EU TENHO 1 PAIR MAS NAO ESTA NO BOARD. = MEU PAR E NAO DO BOARD
                    if t.hand_val > 9:
                        why = 'A pair that isnt in the board and no flush/straight draw. pair > 10'
                        acao = {'passive':'40%', 'normal':'45%', 'agressive':'55%'}
                        
                    else:
                        why = 'A pair that isnt in the board and no flush/straight draw. pair < 10'
                        acao = {'passive':'call', 'normal':'call', 'agressive':'call'}
                
        else:  # t.hand_rank == 0
            if t.outs_por > t.pot_odds_per:
                why = 'Nothing ODDS CALL'
                acao = {'passive':'call', 'normal':'call', 'agressive':'call'}

    # END OF POS FLOP DECISION MAKING

    if breakcounter == 2:   # TURN
        fnk(t.fnkfile, '\n\n--------------------\n\n'+str(street[t.breakcounter])+
        '\nin_hand: '+str(in_hand)+
        '\nMax points: '+str(max_points)+'\tPlay Style: '+str(agr)+
        '\nLimps: '+str(t.limp)+'\tCount: '+str(t.limp_count)+
        '\nBets: '+str(t.bets)+'\tPosition: '+str(t.bets_pos)+
        '\nBoard: '+str(t.board[1])+' '+str(t.board[2])+' '+str(t.board[3])+' '+str(t.board[4])+' '+str(t.board[5])+'\t Hole: '+str(a[SELF_SEAT].hole[0])+' '+str(a[SELF_SEAT].hole[1])+
        '\n\tBoard Flush:\t'+str(DICT_STR[t.board_suit_val])+' ( '+str(str(t.board_suit_val))+
        ' )\n\tBoard Straight:\t'+str(DICT_STR[t.board_str_val])+' ( '+str(str(t.board_str_val))+
        ' )\n\tBoard Pair:\t'+str(DICT_STR[t.pair_val])+' ( '+str(str(t.pair_val))+
        ' )\n\n\tHand Rank: '+str(DICT_FINAL[t.hand_rank])+' ( '+str(t.hand_rank)+
        ' )\n\nMy_outs: '+str(t.outs_hand)+' ( '+str(t.outs_por)+
        '% )\nPot size: '+str(t.pot)+
        '\nTo call: '+str(t.to_call)+
        '\nPot odds: '+str(t.pot_odds_str)+' ( '+str(t.pot_odds_per)+'% )\n')
        
        if t.hand_rank >= 4: # SE HAND RANK > STRAIGHT NO FLOP == GO TO TOWN
            why = 'Hand rank >= 4'
            acao = {'passive':'40%', 'normal':'45%', 'agressive':'55%'}
       
        elif DICT_FINAL[t.hand_rank] == 'TRIPLE':
            if (t.board_str_val < 4 and t.board_suit_val < 4):
                why = 'Triple with no flush/straight draw'
                acao = {'passive':'40%', 'normal':'45%', 'agressive':'55%'}
            
            else:
                if t.outs_por > t.pot_odds_per:
                    why = 'Triple ODDS CALL'
                    acao = {'passive':'call', 'normal':'call', 'agressive':'call'}
   
        elif DICT_FINAL[t.hand_rank] == 'PAIRS':
            if (t.board_str_val < 4 and t.board_suit_val < 4):
                why = '2 pairs with no flush/straight draw'
                acao = {'passive':'40%', 'normal':'45%', 'agressive':'55%'}
            
            else:
                if t.outs_por > t.pot_odds_per:
                    why = '2 pairs ODDS CALL'
                    acao = {'passive':'call', 'normal':'call', 'agressive':'call'}
               
        elif DICT_FINAL[t.hand_rank] == 'PAIR':
            if (t.board_str_val < 4 and t.board_suit_val < 4):
                if t.pair_val < 1: # SE EU TENHO 1 PAIR MAS NAO ESTA NO BOARD. = MEU PAR E NAO DO BOARD
                    if t.hand_val > 9:
                        why = 'A pair that isnt in the board and no flush/straight draw. pair > 10'
                        acao = {'passive':'40%', 'normal':'45%', 'agressive':'55%'}
                        
                    else:
                        why = 'A pair that isnt in the board and no flush/straight draw. pair < 10'
                        acao = {'passive':'call', 'normal':'call', 'agressive':'call'}
                
        else:  # t.hand_rank == 0
            if t.outs_por > t.pot_odds_per:
                why = 'Nothing ODDS CALL'
                acao = {'passive':'call', 'normal':'call', 'agressive':'call'}
    
    # END OF TURN DECISION MAKING

    if breakcounter == 3:   # RIVER
        fnk(t.fnkfile, '\n\n--------------------\n\n'+str(street[t.breakcounter])+
        '\nin_hand: '+str(in_hand)+
        '\nMax points: '+str(max_points)+'\tPlay Style: '+str(agr)+
        '\nLimps: '+str(t.limp)+'\tCount: '+str(t.limp_count)+
        '\nBets: '+str(t.bets)+'\tPosition: '+str(t.bets_pos)+
        '\nBoard: '+str(t.board[1])+' '+str(t.board[2])+' '+str(t.board[3])+' '+str(t.board[4])+' '+str(t.board[5])+'\t Hole: '+str(a[SELF_SEAT].hole[0])+' '+str(a[SELF_SEAT].hole[1])+
        '\n\tBoard Flush:\t'+str(DICT_STR[t.board_suit_val])+' ( '+str(str(t.board_suit_val))+
        ' )\n\tBoard Straight:\t'+str(DICT_STR[t.board_str_val])+' ( '+str(str(t.board_str_val))+
        ' )\n\tBoard Pair:\t'+str(DICT_STR[t.pair_val])+' ( '+str(str(t.pair_val))+
        ' )\n\n\tHand Rank: '+str(DICT_FINAL[t.hand_rank])+' ( '+str(t.hand_rank)+
        ' )\n\nMy_outs: '+str(t.outs_hand)+' ( '+str(t.outs_por)+
        '% )\nPot size: '+str(t.pot)+
        '\nTo call: '+str(t.to_call)+
        '\nPot odds: '+str(t.pot_odds_str)+' ( '+str(t.pot_odds_per)+'% )\n')
        
        if t.hand_rank >= 4: # SE HAND RANK > STRAIGHT NO FLOP == GO TO TOWN
            why = 'Hand rank >= 4'
            acao = {'passive':'40%', 'normal':'45%', 'agressive':'55%'}
       
        elif DICT_FINAL[t.hand_rank] == 'TRIPLE':
            if (t.board_str_val < 4 and t.board_suit_val < 4):
                why = 'Triple with no flush/straight draw'
                acao = {'passive':'40%', 'normal':'45%', 'agressive':'55%'}
            
            else:
                if t.outs_por > t.pot_odds_per:
                    why = 'Triple ODDS CALL'
                    acao = {'passive':'call', 'normal':'call', 'agressive':'call'}
   
        elif DICT_FINAL[t.hand_rank] == 'PAIRS':
            if (t.board_str_val < 4 and t.board_suit_val < 4):
                why = '2 pairs with no flush/straight draw'
                acao = {'passive':'40%', 'normal':'45%', 'agressive':'55%'}
            
            else:
                if t.outs_por > t.pot_odds_per:
                    why = '2 pairs ODDS CALL'
                    acao = {'passive':'call', 'normal':'call', 'agressive':'call'}
               
        elif DICT_FINAL[t.hand_rank] == 'PAIR':
            if (t.board_str_val < 4 and t.board_suit_val < 4):
                if t.pair_val < 1: # SE EU TENHO 1 PAIR MAS NAO ESTA NO BOARD. = MEU PAR E NAO DO BOARD
                    if t.hand_val > 9: # if self.pair > 9
                        why = 'A pair that isnt in the board and no flush/straight draw. pair > 10'
                        acao = {'passive':'40%', 'normal':'45%', 'agressive':'55%'}
                        
                    else:
                        why = 'A pair that isnt in the board and no flush/straight draw. pair < 10'
                        acao = {'passive':'call', 'normal':'call', 'agressive':'call'}
                
        else:  # t.hand_rank == 0
            if t.outs_por > t.pot_odds_per:
                why = 'Nothing ODDS CALL'
                acao = {'passive':'call', 'normal':'call', 'agressive':'call'} 
    # END OF RIVER DECISION MAKING

    for x in range(1, 10):
        fnk(t.fnkfile, 'Player '+str(a[x].p)+' '+str(a[x].action)+' - '+str(a[x].points))
    
    
    if acao[agr] == '':
        acao[agr] = 'fold'
    
    #fnk(t.fnkfile, '')
    
    fnk(t.fnkfile, '\nAction: '+str(acao[agr]))
    if why != '':
        fnk(t.fnkfile, str(why))
        
    
    if AUTOPLAYING:
        if acao[agr] == 'raise':
            acao[agr] = 'bet'
            
        if acao[agr] == 'check':
            acao[agr] = 'call'
        
        sendAcao(GPIO_DICT[acao[agr]])
    
    return acao[agr]

 
def GetHandNumber(tp):
    result = False
    template = cv2.imread(DIR+'d/f.png')
    size = (28, 34, 225, 14)
    while result == False:
        img = get_table(tp)
        
        h, w = template.shape[:-1]
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.9
        loc = np.where(res >= threshold)
        
        for pt in zip(*loc[::-1]):
            #cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)
            #result = os.path.splitext(os.path.basename(template))[0]
            #img_to_read = crop_img(img, ((0), (pt[1] + h), (img.shape[1]), (img.shape[0])))
            img_to_read = crop_img(img, ((pt[0]+w+1), (pt[1]-1), (size[2]-(pt[0]+w)), size[3]))
            result = tess(img_to_read, 3)
            #show(tamanho(img_to_read,10))
            break
        
    return result

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
    return 

def board_read(board): # return type of boards. use to alter actions
    # RETURN  [ SUIT_VAL, SUIT_INDEX ],[ STR_VAL, STR_INDEX ]

    count = 0
    pair_val = 0
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
                board_suit_val = 0
            elif suit_count.count(2) == 1:
                board_suit_val = 1
            elif suit_count.count(3) == 1:
                board_suit_val = 3
        
        if count == 4:                # TURN
            if suit_count.count(2) == 2:
                board_suit_val = 2
            elif suit_count.count(3) == 1:
                board_suit_val = 3
            elif suit_count.count(4) == 1:
                board_suit_val = 4
            elif suit_count.count(2) == 1:
                board_suit_val = 1
            else:
                board_suit_val = 0
                
        if count == 5:                # RIVER
            if suit_count.count(5) == 1:
                board_suit_val = 5
            elif suit_count.count(4) == 1:
                board_suit_val = 4
            elif suit_count.count(3) == 1:
                board_suit_val = 3
            else:
                board_suit_val = 0
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
    
    if True:                            # BOARD PAIRS VALUES.   pair_val
        for x in board_cards:
            if x != '':
                pair_count[CARDS_DIC[x]] += 1
                if x == 'A':
                    pair_count[0] += 1
                    
        if fmax(pair_count) == 4:
            pair_val = 7            # quad
            pair_index = fmax_index(pair_count)
        
        elif fmax(pair_count) == 3 and fmax(pair_count, 2) == 2:
            pair_val = 6            # FULLHOUSE
            pair_index = fmax_index(pair_count)
        elif fmax(pair_count) == 3:
            pair_val = 3            # TRIPLE
            pair_index = fmax_index(pair_count)
        
        elif fmax(pair_count) == 2 and pair_count.count(2) == 2:
            pair_val = 2            # 2 PAIRS
            pair_index = fmax_index(pair_count)
        
        elif fmax(pair_count) == 2 and pair_count.count(2) == 1:
            pair_val = 1               # 1 PAIR
            pair_index = fmax_index(pair_count)
        
        else:
            pair_val = 0            # NOTHING
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

    return [board_suit_val, fmax_index(suit_count)], [board_cards_val[fmax_index(board_str_counter)], fmax_index(board_str_counter)], [pair_val, pair_index]

def calc_por_from_outs(outs, breakcounter):
    return (float(outs)/float(CARDS_LEFT[breakcounter]))*100

    
