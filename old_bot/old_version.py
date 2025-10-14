from funcs import *
from oop import *



#
#   06/05 
#   
#   obs pra testes ok.
#
#   
#


#fnk(txt, '\t\t\t\t\t mods. if pra terminar a street from index == len() pra players_in_hand[index] == players_in_hand[-1].\n\t\t\t\t\t\tpra nao mudar de street qndo player[-2] raise')


#time.sleep(3)
wait_dealer_time = time.time()

time.sleep(2)
start_time = time.time()

street = ['pre-flop', 'flop', 'turn', 'river',' HAND IS OVER ', ' HAND IS OVER ',' HAND IS OVER ',' HAND IS OVER ',' HAND IS OVER ',' HAND IS OVER ',' HAND IS OVER ',' HAND IS OVER ',' HAND IS OVER ',' HAND IS OVER ']

# PREP TO TEST 1
hand_counter = 0
file_counter = 0


card_count = 0
finalbool = False
folder_counter = 0
tp = top_left()
frame = get_table(tp)
t = Table(frame, tp)
t.dealer_old = t.dealer
t.hand_num_old = t.hand_num
breakcounter = 0
#fnk(txt, '')
#fnk(txt, 'File started')
while True: # LOOP TOTAL
    
    t.update_hand_num(tp)
    print 'Hand number ',t.hand_num,' - ',t.hand_num_old
    #time.sleep(0.2)
    
    breakcounter = 0        # counter pras streets
    print 'waiting dealer', (time.time() - wait_dealer_time)

    t.update_dealer(get_table(tp), tp)
    print t.dealer,' - ',t.dealer_old
    
    if t.dealer != t.dealer_old: # ESPERA PROXIMO DEALER. PRA NAO COMECAR NO MEIO DA MAO
        #t.update_dealer(get_table(tp), tp)
        print 'main loop'
        wait_cards = time.time()
        t.hand_num_old = t.hand_num
        t.dealer_old = t.dealer
        #old_dealer = t.dealer                      nao esta sendo usada?
        
        
        card_count = 0
        #cards_wait = True           # pra esperar cartas
        while card_count < 2:   # ESPERA DEALER DAR CARTAS
            card_count = 0
            a = ['', '', '', '', '', '', '', '', '', '']
            print 'waiting cards', (time.time() - wait_cards)
            frame = get_table(tp)
            for x in range(1,10): # loop all players
                a[x] = Player(frame, x, tp)
                a[x].check_cards(tp)
                if a[x].cards:     # break na primeira vez q achar hole cards
                    #cards_wait = False             substituido pelo card_count > 2
                    card_count += 1
                    #break              nao funciona com o card_count
        
        print 'got card'
        # ja achou cartas. setar a mao corretamente apartir daqui
        folder_counter += 1
        #folder = DIR+'alpha/'+str(time.time())
        #try:
        #    os.mkdir(folder, 0755)
        #except:
        #    pass
            
        #filename = str(folder)+'/'+str(time.time())+'.#fnk'

        #txt = open(filename, 'w')
        frame = get_table(tp)
        in_hand = []
        players_in_hand = []
        a = ['', '', '', '', '', '', '', '', '', '']
        finalbool = True
        
        for x in range(1,10):   # SETA TODOS PLAYERS
            a[x] = Player(frame, x, tp)
            a[x].check_cards(tp)     # ALTERA SELF.CARDS
            if PRINTS:
                print 'x ',x
                print 'a[x] ',a[x].cards
            if a[x].cards:              # SE tiver cards.
                in_hand.append(x)
                a[x].allin = False
                a[x].is_playing = True  # prob inutil

                
        in_hand = hand_order(t.dealer, in_hand) # ordena a mao
        in_use = 0                              # list in use
        position_counter = 0
        
        players_in_hand = list(in_hand[in_use])
        for x in in_hand[in_use][::-1]:
            a[x].position = position_counter
            position_counter += 1
            
            
        jump = 99                               # jump pra qndo arrumar lista
        #fnk(txt, 'Hand SET')
        #cv2.imwrite(DIR+'alpha/'+filename+'.png', get_table(top_left())) 
        if PRINTS:    
            print 'Hand Set'
        hand_counter += 1
        file_counter = 0
        t.bet = True
        t.reset()
        while len(in_hand[1]) > 1 and breakcounter <= 3 and t.allin_count <= (len(in_hand[1])-1):    # breakcounter [0] = pre-flop / [1] = flop / [2] = turn / [3] = river / [4+] = hands over     
        # loop players in hand > 1. tds streets sao aqui dentro. so quebra com break ou folds.
            t.reset_data()
            if PRINTS:    
                print 'hand loop'
            #fnk(txt, 'len(players_in_hand) > 1 ')
            #fnk(txt, '************************************** '+str(street[breakcounter])+' **************************************')
            t.update_dealer(get_table(tp), tp)      # update dealer
            if t.dealer == t.dealer_old:        # CHECKA PRA VER SE ESTA AINDA NA MESMA MAO
                
                if PRINTS:    
                    print 'same hand'
                #fnk(txt, 'DEALER CHECKS ')
                #players_in_hand = list(PLAYERS_CONS)    # faz lista de 1-9 mas so vai entrar se is_playing == True
                while True:             # so sai desse loop com break
                    # only breaks when 
                    # loop pra lista de players
                    if PRINTS:    
                        print 'street loop'
                    for index in range(0,len(players_in_hand)):          # loop de cada street    
                        if PRINTS:    
                            print 'index '+str(index)+' len '+str(len(players_in_hand))
                        #os.system('cls')
                        
                        
                        # check se pl != jumper ///  se tem mais de 1 player na list  /// if is_playing  ///  if not all in
                        if players_in_hand[index] != jump and len(in_hand[1]) > 1 and a[players_in_hand[index]].is_playing == True and a[players_in_hand[index]].allin == False:    
                            t.reset_street()
                            if not PRINTS:
                                os.system('cls')
                            
                            print '$'*20,' ',street[breakcounter],' ','$'*20
                            print t
                            frame_1 = get_table(tp)
                            t.update_pot(frame_1)
                            print 'Pot: ',t.pot
                            print 'limps ',t.limp,' - ',t.limp_count
                            print 'bets ',t.bets,' - ',t.bets_pos
                            
                            
                            if SELF_PLAYING and a[SELF_SEAT].is_playing:
                                a[SELF_SEAT].hole_per = hole_percent([a[SELF_SEAT].hole[0],a[SELF_SEAT].hole[1]])
                                print 'Hole Cards: ',a[SELF_SEAT].hole[0],' - ',a[SELF_SEAT].hole[1],'  ',a[SELF_SEAT].hole_per
                                if a[SELF_SEAT].raise_amount > 0:
                                    print 'Ammount raised: ',a[SELF_SEAT].raise_amount
                                
                                
                                print ''
                                if breakcounter > 0:
                                    if breakcounter < 3: #nao calcula odds no pre flop
                                        giving_hand = []
                                        giving_hand.append(a[SELF_SEAT].hole[0])
                                        giving_hand.append(a[SELF_SEAT].hole[1])
                                        for x in range(1,breakcounter+3):
                                            giving_hand.append(t.board[x])
                                            
                                        t.hand_rank, t.outs_hand = hand_outs(giving_hand)
                                        t.outs_por = calc_por_from_outs(t.outs_hand, breakcounter)
                                        print 'HAND RANK ',DICT_FINAL[t.hand_rank]
                                        print 'OUTS ',t.outs_hand,' = ',t.outs_por,'%'
                                    
                                    print ''
                            if breakcounter > 0:
                                [t.board_suit_val, t.board_suit_index], [t.board_str_val, t.board_str_index], [t.pair_val, t.pair_index] = board_read(t.board)
                                print 'BOARD RANK STR', DICT_STR[t.board_str_val],' ',t.board_str_val
                                print 'BOARD RANK FSH', DICT_FSH[t.board_suit_val],' ',t.board_suit_val
                                    
                                    
                            print '\nPlayer\t\tAction\t\tPosition'
                            for x in range(0,len(players_in_hand)):
                                if a[players_in_hand[x]].cards:
                                    if index == x:
                                        print '--> ',a[players_in_hand[x]]
                                    else:
                                        print a[players_in_hand[x]]     

                                    
                            #fnk(txt, '\tPlayer '+str(players_in_hand[index])+' valido, waiting action now')
                            jump = 99
                            
                            #print '[1]', in_hand[1]                            
                            #print 'players_in_hand ',players_in_hand
                            #print 'dealer', t.dealer, ' - player: ', players_in_hand[index]
                            
                            #print '\n[0]', in_hand[0]
                            #print '[1]', in_hand[1]                            
                            #print 'players_in_hand ',players_in_hand
                            
                            
                            print '\tesperando ',POSITION_DICT[a[players_in_hand[index]].position]
                            
                            if players_in_hand[index] == SELF_SEAT: # if its my turn to act.
                                if PRINTS:
                                    print '-'*20
                                    print 'break\t\t',breakcounter
                                    print 'bets_pos\t\t',t.bets_pos
                                    print 'limp c\t\t',t.limp_count
                                    print 'hole_per\t\t',a[SELF_SEAT].hole_per
                                    print '-'*20
                                
                                #a[SELF_SEAT].decisao_calculada = escolha_decisao(a, t, players_in_hand, index, breakcounter)
 

 
 
                            a[players_in_hand[index]].wait_action(frame, t, tp, a, players_in_hand, index, breakcounter)          # wait for action
                           


                            if PRINTS:
                                print a[players_in_hand[index]],' -------'
                                print players_in_hand[index],' == ',players_in_hand[-1]
                                print t.allin_count,' = ALL IN = ',(len(in_hand[1])-1)
                            file_counter += 1
                            #cv2.imwrite('Z:/poker/alpha/'+str(hand_counter)+'-'+str(file_counter)+'.png', get_screen(0, 0, 1920, 1080))
                            #fnk(txt, '\t\t'+str(a[players_in_hand[index]].action))
                            # filtra acoes. fold = remove from list.
                            #               check / call = whatever
                            #               raise / bet = arrumar_lista
                            if a[players_in_hand[index]].action == 'fold' or a[players_in_hand[index]].action == 'empty' or a[players_in_hand[index]].action == 'sitout':
                                #print 'fold if'
                                in_hand[0].remove(players_in_hand[index])
                                in_hand[1].remove(players_in_hand[index])
                                a[players_in_hand[index]].is_playing = False
                                a[players_in_hand[index]].cards = False
                                if SELF_PLAYING and players_in_hand[index] == SELF_SEAT:
                                    a[players_in_hand[index]].reset()
                            # nao remove da lista q esta sendo usada. senao new index 1 vira 2(BUG)
                            if a[players_in_hand[index]].action == 'call':
                                t.limp = True
                                t.limp_count += 1

                            
                            elif(a[players_in_hand[index]].action == 'raise' or a[players_in_hand[index]].action == 'bet' or a[players_in_hand[index]].action == 'allin'):
                                if PRINTS:
                                    print 'raise if'
                                t.bet = True
                                t.bets = True
                                if t.bets_pos < a[players_in_hand[index]].position:
                                    t.bets_pos = a[players_in_hand[index]].position
                                jump = players_in_hand[index]       # seta jumper pro proximo FOR LOOP
                                
                                #2nd check, ja faz no oop. confiavel so 1?
                                if find_template(crop_img(get_table(tp), a[players_in_hand[index]].allin_pos), DIR+'template/allin/*.png') == 'allin' or a[players_in_hand[index]].allin == True:
                                    a[players_in_hand[index]].allin = True
                                    t.allin_count += 1
                                    print 'ALLIN*'*5
                                
                                # MUDAR ISSO
                                # SE SETAR A PLAYERS_IN_HAND USANDO A IN_HAND, PODE DAR BUG.
                                # ARRUMAR UMA LISTA FULL [1-9 ANYWAY] E REMOVE IS_PLAYING == FALSE

                                index_temp = players_in_hand[index]
                                players_in_hand = arruma_list(players_in_hand[index], list(in_hand[in_use])) 
                                index = players_in_hand.index(index_temp)

                                del index_temp
                                # MUDAR ISSO
                                break           # break pra comecar o index do comeco. break o for loop e nao o while.
                            
                            #fnk(txt, '\t\t\t[0] '+str(in_hand[0]))
                            #fnk(txt, '\t\t\t[1] '+str(in_hand[1]))
                            #fnk(txt, '\t\t\t using '+str(players_in_hand))
            
                        if not PRINTS:
                            os.system('cls')
                        
                    # SE O ULTIMO PLAYER DER CHECK / CALL. BREAK O LOOP
                    # headsup [0] = checks / [1] bet. players_in_hand[1] ==> [0] e nao entra nesse if.
                    #print 'players_in_hand[index] == players_in_hand[-1]'
                    #print players_in_hand
                    #print index
                    #print players_in_hand[index]
                    #print ' == '
                    #print players_in_hand[-1]
                    if len(in_hand[1]) <= 1: # so tem um players na mao
                        print 'only one player'
                        #fnk(txt, '\t\t\t '+str(in_hand[1])+' ==> 1')
                        #fnk(txt, '\t\t\t\t antes '+str(players_in_hand)+'')
                        #fnk(txt, '*** len < 1 ***')
                        #cv2.imwrite(filename[:-4]+'-Folded.png', get_table(tp)) 
                        #txt.close()
                        break
                    
                    
                    # t.allin_count >= (len(in_hand[1])-1)   ==   todos de allin e um call. pra
                    if t.allin_count >= (len(in_hand[1])-1) and t.allin_count >= 1: # pl > 1.   
                        print 'TODOS PLAYERS ESTAO DE ALLIN'
                        break
                        
                        
                    if players_in_hand[index] == players_in_hand[-1]:       # BREAK DEPOIS DO ULTIMO PLAYER. SE ELE NAO RAISE / BET 
                        #cv2.imwrite(filename[:-4]+'-'+str(street[breakcounter])+'.png', get_table(tp)) 
                        breakcounter += 1
                        t.update_board(breakcounter, tp)
                        for x in range(0,len(players_in_hand)):
                            if a[players_in_hand[x]].cards:
                                a[players_in_hand[x]].reset()
                        #fnk(txt, '\t\t\t '+str(in_use)+' ==> 1')
                        #fnk(txt, '\t\t\t\t antes '+str(players_in_hand)+'')
                        #fnk(txt, '*** BREAK COUNTER '+str(breakcounter)+'**********'+street[breakcounter])
                        in_use = 1
                        players_in_hand = list(in_hand[in_use])
                        #fnk(txt, '\t\t\t\t depois '+str(players_in_hand)+'')
                        if breakcounter >= 4:
                            print 'last player on the river acted'
                            #cv2.imwrite(filename[:-4]+'-HandsOver.png', get_table(tp)) 
                            #txt.close()
                        break           # break o while True loop.
                     
                    
            
            
                    
                
            else:   # A MAO ACAOU. DEALER != OLD DEALER.
                print 'dealer != old dealer'
                #fnk(txt, '*'*50)
                #fnk(txt, 'MAO ACABOU - DEALER != OLD DEALER')
                #fnk(txt, '*'*50)
                #cv2.imwrite(filename[:-4]+'-Else.png', get_table(tp)) 
                #txt.close()
                break            # break while len(players_in_hand) > 1
                
            t.bet = False   
             
        # so deleta depois de break while len(players in hand) > 1

        print 'HAND IS OVER'  
        del a
        del in_hand
        del players_in_hand
        del jump
        del in_use

        
    #if (time.time() - finaltime) > 5 and finalbool == True:            
        #cv2.imwrite(DIR+'final/'+str(time.time())+'.png', get_table(tp)) 
        #finaltime = time.time()
        #finalbool = False
    
    if not PRINTS:
        os.system('cls')
