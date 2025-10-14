from funcs import *
from posfunc import *
import time


class Hole():
    def __init__(self):
        self.percent = 0
        self.full = ['', '']
        self.suit = ['', '']
        self.card = ['', '']
        self.pos = hole_pos()

class Player(): # NEW
    def __init__(self, p, tp):
        self.p = p
        self.tp = tp
        
        self.cards_pos = cards_pos(p)                   # POSITIONS
        self.action_pos = action_pos(p)                 # POSITIONS
        self.hud_pos = hud_pos(p)                       # POSITIONS
        self.allin_pos = allin_pos(p)                   # POSITIONS
    
        if p == SELF_SEAT and SELF_PLAYING == True:     # SELF ESPECIAL VARS
            self.hole = Hole()

    def UpdateHole(self, tablecoor):
        if p == SELF_SEAT and SELF_PLAYING == True:
            self.hole.full = [False, False]
            temp = False
            
            while temp == False:
                self.frame = GetFrame(tablecoor)
                temp = read_hole(crop_img(self.frame, self.hole.pos[0]))
            
            self.is_playing = True
            self.have_cards = True
            self.hole.full[0] = temp
            self.hole.full[1] = read_hole(crop_img(self.frame, self.hole.pos[1]))
            
    def CheckCards(self, tablecoor):
        self.frame = GetFrame()
        temp = find_template(crop_img(self.frame, self.cards_pos), DIR+CHECKCARDS)
        
        if SELF_PLAYING:
            if self.p != SELF_SEAT:
                if temp != 'nada':
                    self.have_cards = True
                    self.is_playing = True
                else:
                    self.have_cards = False
                    self.is_playing = False
                
            elif self.hole.full[0] == '':
                self.UpdateHole(tablecoor)
                self.hole.card, self.hole.suit = break_hole(self.hole.full)
        
        else: # if NOT SELF_PLAYING:
            if temp != 'nada':
                self.have_cards = True
                self.is_playing = True
            else:
                self.have_cards = False
                self.is_playing = False
                
    def reset(self):
        self.hole.board = ['', '', '', '', '', '']
        self.allin_count = 0
        self.limp = False
        self.limp_count = 0
        self.is_bet = False
        self.bet_ammount = 0
        self.bet_position = -1
    
    
    
    
    
class Table(): # NEW
    def __init__(self):
        self.TopLeft()             # GET TOP LEFT COORD
        self.frame = self.GetFrame()
        self.UpdatePot()
        self.UpdateDealer()
        self.UpdateHandNum()
        
        # POSITION SETUP
        self.pos = Positions()
        # POSITION SETUP

    def TopLeft(self):
        # UPDATE self.tp, self.tablecoor
        # USING funcs.get_screen()
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
                
        self.tp = result
        self.tablecoor = (self.tp[0], self.tp[1], (self.tp[0]+TABLE_X_SIZE), (self.tp[1]+TABLE_Y_SIZE))
        
    def GetFrame(self):
        img = np.array(ImageGrab.grab(bbox=self.tablecoor))) 
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #ajusta RGB
    
    def UpdateFrame(self):
        self.frame = GetFrame()
        
    def UpdatePot(self):
        # UPDATE self.pot 
        # USING FUNCS.CROP_IMG()
        if PLAYING_MONEY:
            template = cv2.imread(DIR+POT['REALMONEY'])
        else:
            template = cv2.imread(DIR+POT['PLAYMONEY'])
        
        img = crop_img(self.frame, self.pos.pot)
        h, w = template.shape[:-1]
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.9
        loc = np.where(res >= threshold)
        
        for pt in zip(*loc[::-1]):
            img = crop_img(img, ((pt[0] + w), (pt[1]-2), (w*4), (h+4)))
            break
            
        t = tess(img, 3)
        t = t.replace(",", "")
        t = t.replace(".", "")
        
        self.pot = float(t)
        
    def UpdateDealer(self): 
        # UPDATE self.dealer
        # USING funcs.check_coor()
        temp_list = glob.glob(DIR+DEALER) #dir pra templates
        r = False
        while(r == False):
            self.UpdateFrame()
            
            for temp_x in temp_list:
                template = cv2.imread(temp_x)
                h, w = template.shape[:-1]							
                res = cv2.matchTemplate(self.frame, template, cv2.TM_CCOEFF_NORMED)
                threshold = 0.6
                loc = np.where(res >= threshold)
                
                for xx in zip(*loc[::-1]):  # pt contem coordenadas  **** (pt[0] + (w/2), pt[1] + (h/2)) = CENTRO ****
                    break
                    
            for t_pos in range(1,10):
                try:
                    if check_coor(xx, self.pos.dealer[t_pos]):
                        r = t_pos
                        break
                except:
                    pass
        
        self.dealer = r
        
    def UpdateHandNum(self):
        # UPDATE self.hand_num
        # USING funcs.crop_img()
        result = False
        template = cv2.imread(DIR+HANDNUMBER)
        while result == False:
            self.UpdateFrame()
            h, w = template.shape[:-1]
            res = cv2.matchTemplate(self.frame, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.9
            loc = np.where(res >= threshold)
            for pt in zip(*loc[::-1]):
                img_to_read = crop_img(img, ((pt[0]+w+1), (pt[1]-1), (self.pos.handnumber[2]-(pt[0]+w)), self.pos.handnumber[3]))
                break
                
            result = tess(img_to_read, 3)
    
        self.hand_num = result

    def WaitForCards(self):
        while self.card_count < 2:
            self.card_count = 0
            self.pl = ['', '', '', '', '', '', '', '', '', '']
            self.UpdateFrame()
            for x in range(1,10): # loop all players
                self.pl[x] = Player(x, self.tp)


class Positions():
    def __init__(self):
        self.pot = pot_pos()
        self.dealer = dealer_pos()
        self.handnumber = (28, 34, 225, 14)
        self.to_call_pos = to_call_pos()
        self.to_allin_call_pos = to_allin_call_pos()
        self.raise_size = raise_size_pos()

        
        
        
        
        
        