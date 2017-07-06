#!/usr/bin/python

# !/bin/python
def next_move(posr, posc, board):
    move = 0
    n = 5
    pos_d=[]
    out=[]
    pos_b = [posr, posc]
    to_back=[]

    lt,rt,lb,rb=[posr-1,posc-1],[posr-1,posc+1],[posr+1,posc-1],[posr+1,posc+1]
    if posr==0:
        lt[0],rt[0]=0,0
    if posr==n-1:
        lb[0],rb[0]=n-1,n-1
    if posc==0:
        lt[1],lb[1]=0,0
    if posc==n-1:
        rt[1],rb[1]=n-1,n-1

    filename = "to_reset.txt"
    file2="to_back.txt"
    with open(filename) as f:
         # file read can happen here
         # print "file exists"
    #     f.readlines()
    to_reset = False
    try:
        with open(filename) as f:
            to_reset=f.readline()

    except:

        with open(filename, "w+") as f:
            # print "file write happening here"
            f.write('False')
            to_reset = 'False'

    #print(lt,rt,)

    #print(lb,rb)
    #print range(lt[0],rb[0]+1)
    for x in range(lt[0],rb[0]+1):
        for y in range(lt[1],rb[1]+1):
            #print "xy",x,y
            # if board[x][y] == 'b':
            #     pos_b = [x, y]
            if board[x][y] == 'd':
                #print (board[x][y])
                pos_d.append([x,y])
            #print "hello",x,y
    #print pos_d
    if(len(pos_d)!=0 and to_reset=='False' ):
        to_clean=pos_d[0]
        minimun=abs(pos_d[0][0]-pos_b[0])+abs(pos_d[0][1]-pos_b[1])
        for i in range(len(pos_d)):
            diff_x=abs(pos_d[i][0]-pos_b[0])
            diff_y=abs(pos_d[i][1]-pos_b[1])
            if((diff_x+diff_y)<minimun):
                minimun=diff_x+diff_y
                to_clean=pos_d[i]
        h_steps = to_clean[1] - pos_b[1]
        v_steps = to_clean[0] - pos_b[0]
        #print pos_d


        if h_steps > 0 and move==0:
            # for i in range(abs(h_steps)):
            print 'RIGHT'
            move='RIGHT'
            to_back.append(move)
            with open(file2, "w+") as f:
                # print "file write happening here"
                f.write('RIGHT')


        elif h_steps < 0 and move==0:
            # for i in range(abs(h_steps)):
            print 'LEFT'
            move='LEFT'
            to_back.append(move)
            with open(file2, "a+") as f:
                # print "file write happening here"
                f.write('LEFT')

        if v_steps < 0 and move==0:
            # for i in range(abs(v_steps)):
            print 'UP'
            move='UP'
            to_back.append(move)
            with open(file2, "a+") as f:
                # print "file write happening here"
                f.write('UP')
        elif v_steps > 0 and move==0:
            # for i in range(abs(v_steps)):
            print 'DOWN'
            move='DOWN'
            to_back.append(move)
            with open(file2, "a+") as f:
                # print "file write happening here"
                f.write('DOWN')
        if h_steps == 0 and v_steps == 0 and move==0:
            print 'CLEAN'
            with open(filename, "w") as f:
                # print "file write happening here"
                f.write("True")


    if(to_reset=='True'):
        with open(file2, "r") as f:
            # print "file write happening here"
            d= (f.readlines())

    else:
        #prev_move='No'
        if posc!=n-1:
            print "RIGHT"
            #prev_mov='LEFT'
        elif posr!=n-1:
            print "DOWN"
        elif posc!=0:
            print "LEFT"
        elif posr!=0:
            print "UP"
    for x in range(n):
        out.append()
    print "oui ",out




import sys

if __name__ == "__main__":
    # pos = [int(i) for i in raw_input().strip().split()]
    # board = [[j for j in raw_input().strip()] for i in range(5)]
    # next_move(pos[0], pos[1], board)
    fo=open("E:/datasets/bot.txt","r+")
    pos = [int(i) for i in fo.readline().strip().split()]
    board = [[j for j in fo.readline().strip()] for i in range(5)]
    next_move(pos[0], pos[1], board)
    fo.seek(0)
    print fo.read()
    fo.seek(0)

