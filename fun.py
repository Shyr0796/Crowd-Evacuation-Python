# @Time    : 2023/10/3 10:27
# @Author  : Chun Song
#function that used in all py files
import math
import random
import vtk

import numpy as np

import tkinter as tk
from tkinter import *

def Name(string, i):
    '''return Name and code num'''
    return '_'.join([str(string), str(i),'.csv'])

#@【VECTOR】---------------------------------------------------------------------------------------------------------------------------------
def Vector_len(v):
    '''vector module/length'''
    len = np.sqrt(v[0]**2 + v[1]**2)
    return len

def Vector_sum(v1, v2):
    '''vector sum: v1+v2'''
    '''sum of v1 and v2'''
    sum = [v1[0]+v2[0], v1[1]+v2[1]]
    return sum

def Vector_sub(v1, v2):
    '''vector sub: v1-v2'''
    sub = [v1[0]-v2[0], v1[1]-v2[1]]
    return sub

def Vector_dot(v1, v2):
    '''vector dot: v1*v2'''
    dot = v1[0]*v2[0]+v1[1]*v2[1]
    return dot

def Vecter_offset(Area, SP):
    '''set the cooradite of E, B'''
    Area[0] +=SP[0]
    Area[1] += SP[1]

def Vecter_offset2(Areas, SP):
    for Area in Areas:
        Vecter_offset(Area, SP)

def Vector_norm_dot(v, n):
    '''vector dot normal: '''
    V_shape = np.shape(v)
    V = v
    for i in range(V_shape[0]):
        if len(V_shape) ==2:
            for j in range(V_shape[1]):
                V[i][j] =v[i][j]*n
        else:
            V[i] =v[i]*n
    return V

#@【Vector and Angle】---------------------------------------------------------------------------------------------------------------------------------
def Vector_Projection(self, v1, v2):
    '''vector projection: v1's projection on v2'''
    dot = Vector_dot(v1, v2)
    len2 = v2[0]**2+v2[1]**2
    if len2 == 0:
        len2=0.1
    ratio = dot/len2
    projection =[ratio*v2[0], ratio*v2[1]]
    return projection

def Vector_norm(v):
    '''vector normalizing'''
    v = np.array(v)
    sum_1 = np.sum(v ** 2)
    if sum_1 != 0:
        norm_1 = v / np.sqrt(sum_1)
    else:
        norm_1 = [0, 0]
    return norm_1

def Vecter_angle(Vector):
    '''caculate the angle and length of arrraw p0 to p1, 0-2*pi'''
    pi = 3.14159
    vx, vy = Vector[0], Vector[1]
    Radian_v = 0.0
    if vx == 0:
        if vy > 0:
            Radian_v = 90.0 / 180 * pi
        elif vy < 0:
            Radian_v = 270.0 / 180 * pi
    elif vy == 0:
        if vx >= 0:
            Radian_v = 0.0
        elif vx < 0:
            Radian_v = pi
    else:
        if vx > 0 and vy > 0:
            Radian_v = math.atan(vy / vx)
        elif vx > 0 and vy < 0:
            Radian_v = math.atan(vy / vx) + 2*pi
        elif vx < 0 and vy > 0:
            Radian_v = math.atan(vy / vx) + pi
        elif vx < 0 and vy < 0:
            Radian_v = math.atan(vy / vx) + pi
    return Radian_v

#@【Degree and Points】---------------------------------------------------------------------------------------------------------------------------------
pie = 10 #int degree
global n_p
global n_f
global pi
global piece
global pieces
global Per_Ps0
n_f = int(240/pie)
n_p = int(360/pie)
pi=3.14159
piece =2*pi/n_p

def Cos(degree):
    '''cos(degree)'''
    return math.cos(degree)

def Sin(degree):
    '''sin(degree)'''
    return math.sin(degree)

def Check_In_n_p(i):
    '''check i in (0, n_p), and make sure in that range'''
    global n_p
    if i >= n_p:
        i -= n_p
    if i < 0:
        i += n_p
    return i

def Generate_degree():
    '''make 360 into n pieces'''
    global pi
    global n_p
    global piece
    piece_s =[]
    for i in range(n_p):
        piece_s.append(i*piece)
    return piece_s

def Generate_PerPoints():
    '''generate perception points'''
    global  pi
    global n_p
    global pieces
    global Per_Ps0
    Per_Ps0 = []
    for i in range(n_p):
        Co, Si = Cos(pieces[i]), Sin(pieces[i])
        Per_Ps0.append([Co, Si])
    return Per_Ps0

#@【angle pieces & Point pieces】---------------------------------------------------------------------------------------------------------------------------------
pieces = Generate_degree()
Per_Ps0 = Generate_PerPoints()

def Get_direction(P, degree, r):
    '''get the direction values from all directions, r = [r1, r2, r3......]'''
    global piece
    global pieces
    global Per_Ps0
    global n_f
    global  n_p
    n_2 = int(n_f/2)
    name = 'Per_Ps'
    Per_Ps1 = []
    for k in range(len(r)):
        Per_Psi = []
        ri = r[k]
        if degree%piece == 0:
            b=int(degree/piece)
            for i in range(b-n_2, b+n_2+1):
                i = Check_In_n_p(i)
                Per_Psi.append([P[0]+ri*Per_Ps0[i][0], P[1]+ri*Per_Ps0[i][1], i*piece])  #[x, y, degree]
                #Per_Ps2.append([P[0] + r2 * Per_Ps0[i][0], P[1] + r2 * Per_Ps0[i][1], i * piece])  # [x, y, degree]
                #Per_Ps3.append([P[0] + r3 * Per_Ps0[i][0], P[1] + r3 * Per_Ps0[i][1], i * piece])  # [x, y, degree]
        else:
            b = int(degree/piece)
            for i in range(b-n_2+1, b+n_2):
                i = Check_In_n_p(i)
                Per_Psi.append([P[0]+ri*Per_Ps0[i][0], P[1]+ri*Per_Ps0[i][1], i*piece])
            #Per_Ps2.append([P[0] + r2 * Per_Ps0[i][0], P[1] + r2 * Per_Ps0[i][1], i * piece])  # [x, y, degree]
            #Per_Ps3.append([P[0] + r3 * Per_Ps0[i][0], P[1] + r3 * Per_Ps0[i][1], i * piece])  # [x, y, degree]
        Per_Ps1.append(Per_Psi) #shape: [num of ri, num of point the r =ri]
    #return [Per_Ps1, Per_Ps2, Per_Ps3]
    return Per_Ps1

def Get_value(ratio, P, A, R_value):
    '''get four points of P, return the max values'''
    x, y = int(P[0]*ratio), int(P[1]*ratio)
    In = False
    P_max = 0
    if  A[0]<=x<=A[0]+A[2]-1 and A[1]<=y<=A[1]+A[3]-1:  #R1, one Room
        #P_Value = [R_value[x][y], R_value[x][y + 1], R_value[x + 1][y], R_value[x + 1][y + 1]] #4 points
        #P_max = max(P_Value)
        P_max = R_value[x][y] #a simple way
        In = True
    else:
        P_max = 3
    return In, P_max

def Check_PerValues(ratio, Per_Ps, A, R_value):
    '''based on the Per_Ps(generated perception points), get values from room types value'''
    global pieces
    r_num, d_num, j_num = np.shape(Per_Ps)
    NewPer_Ps = []
    for i in range(d_num):
        linshi = []
        for j in range(r_num):
            linshi.append(Per_Ps[j][i])
            if j == r_num-1:
                NewPer_Ps.append(linshi)
    To_delate = []
    for d in range(d_num):
        TRUE_d = True
        for r in range(r_num):
            In_room, P_value = Get_value(ratio, NewPer_Ps[d][r], A, R_value)
            if In_room:
                if P_value == 3:
                    TRUE_d =False
                else:
                    NewPer_Ps[d][r].append(P_value)  # P_value should be 0
            else:
                TRUE_d =False
        if not TRUE_d:
            To_delate.append(NewPer_Ps[d])
    for d0 in To_delate:
            NewPer_Ps.remove(d0)
    return NewPer_Ps

def Choose_PerValues(ratio, Per_Ps, Field):
    '''choose the directest direction to Exit in the rest directions after check'''
    global pi
    diff_values=2 * pi
    dg = 0
    # the shape might be 1, means the one has no P_points
    P_shape = np.shape(Per_Ps)
    if len(P_shape) == 3:
        L1, L2, L3 = P_shape
    else:
        L1, L2, L3 = 1, 0, 0
    for i in range(L1):
        j = 0
        if L2 == 0:
            continue
        x, y = Per_Ps[i][j][0]*ratio, Per_Ps[i][j][1]*ratio
        value_f = Field[int(x)][int(y)]
        value_p = Per_Ps[i][j][2]
        diff = abs(value_p-value_f)
        if diff < diff_values:
            dg = value_p
            diff_values = diff
    return dg

#@【ROOM】---------------------------------------------------------------------------------------------------------------------------------

global ROOM
ROOM =  np.full((1000 ,1000), 3, dtype=int)

def Room_set_types(A, W_b = 1):
    '''generate an empty room space, length ,width, width of boundary'''
    global  ROOM
    #room = np.full((L ,W), 3, dtype=int)  # boundary=1
    ROOM[A[0]+W_b:A[2]+A[0]-W_b, A[1]+W_b:A[3]+A[1]-W_b] = 0 #basic/empty space = 0
    return ROOM

def Room_crowd_value(area, b):
    '''set the crowd value in the room, area = [x, y, r]'''
   #b : buffer width for ones
    x_l, x_r, y_l, y_r = int(area[0]-area[2]-b), int(area[0]+area[2]+b), int(area[1]-area[2]-b), int(area[1]+area[2]+b)
    #room[x_l: x_r, y_l: y_r] = int(value)
    return x_l, x_r, y_l, y_r

def Room_set_value(room, area, value):
    '''based on the room, set the area[x0,y0,l,w] with value, for barriers'''
    b = 8 #buffer width for barriers
    room[area[0]-b: area[0]+area[2]+2*b , area[1]-b: area[1]+area[3]+2*b] = int(value)

def Room_set_value2(room, areas, value):
    '''for more than one area set the same value, in the room areas'''
    for area in areas:
        Room_set_value(room, area, value)

def Move_B(SP, B):
    '''move barrier'''
    return [B[0]+SP[0], B[1]+SP[1], B[2], B[3]]

def Move_B2(SP, Bs):
    '''move barriers'''
    for i  in range(len(Bs)):
        Bs[i] = Move_B(SP, Bs[i])
    return Bs

def Room_check_object(room, objects):
    '''check objects, judgy whether the object is in the room, and change'''
    result = False
    for ob in objects:
        if ob[0] >= room[0] and ob[1] >= room[1]:
            if ob[0]+ob[2]<=room[2]+room[0] and ob[1]+ob[3]<=room[3]+room[1]:
                result = True
                continue
        else:
            print("Check the object: [{0}, {1}, {2}, {3}]".format(ob[0], ob[1], ob[2], ob[3]))
    return result

def Room_list(area, L=[], P =0):
    '''generate the area points list '''
    list = []
    for i in range(area[0], area[0]+area[2]):
        for j in range(area[1], area[1]+area[3]):
            list.append([i, j])
            if P==1 and [i, j] in L:
                L.remove([i, j])
    return list

def Room_list2(areas, L=[], P=0):
    '''generate the list including areas as the same type'''
    lists =[]
    for area in areas:
        list = Room_list(area, L, P)
        lists.extend(list)
    return  lists

def Room_boundary(A, W_b = 1):
    '''return the boundary of the room'''
    B= [[A[0], A[1], W_b, A[3]], [A[0]+A[2] - W_b, A[1], W_b, A[3]], [A[0]+W_b, A[1], A[2] - 2 * W_b, W_b], [A[0]+W_b, A[1]+A[3] - W_b, A[2] - 2 * W_b, W_b]]
    return  B

def Room_Connect(R, i, j):
    '''room connect between rooms, Send an entrance location to another space.'''
    if len(R[i].De_ones)>0:
        for De_one in R[i].De_ones:
            R[j].One_add(De_one)

#@【FIELD】---------------------------------------------------------------------------------------------------------------------------------
global field_ar
global field_dg
field_ar = np.zeros((1000, 1000, 2), dtype=float)
field_dg = np.zeros((1000, 1000), dtype=float)

def Field_velocity(Area, Point, V=[0, 0]):
    '''generate the field with the area to the point'''
    global field_ar
    global field_dg
    for i in range(Area[0], Area[0]+Area[2]):
        for j in range(Area[1], Area[1]+Area[3]):
            m, n = i,  j
            p = Vector_norm([Point[0] - m, Point[1] - n])
            if V[0]==0 and V[1]==0:
                field_ar[int(m)][int(n)] = p
                field_dg[int(m)][int(n)] = Vecter_angle(p)
            else:
                field_ar[int(m)][int(n)] = V
                field_dg[int(m)][int(n)] = Vecter_angle(p)
    return field_ar, field_dg

global field_h
field_h = np.zeros((1000, 1000), dtype=float)

def Field_hight(A, P, value_max = 100):
    '''based on P0, generate a linear potential field'''
    global field_h
    L, W = A[2], A[3]
    len = [Vector_len(P),Vector_len(Vector_sub(P,[L, 0])), Vector_len(Vector_sub(P,[0, W])), Vector_len(Vector_sub(P,[L, W]))]
    len_max = max(len)
    ratio1 = 100/len_max
    for i in range(L):
        for j in range(W):
            field_h[i+A[0]][j+A[1]] = Vector_len([P[0] - i, P[1] - j])*ratio1
    return field_h

def Field_add(field1, field2, p0):
    '''add field1 on field2, and field1<field2'''
    f1_l, f1_w = np.shape(field1)
    f2_l, f2_w = np.shape(field2)
    if f1_l + p0[0] <= f2_l and p0[1] + f1_w <= f2_w:
        for i in range(f1_l):
            for j in range(f1_w):
                field2[i+int(p0[0])][j+int(p0[1])] += field2[i][j]
    else:
        print("The size of field1:[{0},{1}] is out of field2:[{2},{3}, p0:{4} range".format(f1_l, f1_w,f2_l, f2_w,p0))
    return field2

#@【Building】---------------------------------------------------------------------------------------------------------------------------------
#based on 4 or more points, get the points in the rectangle circled by the points
#def Building

#@【CROWD】---------------------------------------------------------------------------------------------------------------------------------
#【尺寸】50cm*30cm
# 【速度】1.22m/s，标准差0.29
#【速度、密度关系】v = 1.26 - 0.28*p
# 【视线范围】90-120°
# 【单步耗时】0.52s>仿真时间步长，0.5s
def Exit_check(O, E):
    '''chekck one's position on exit list and delete if TRUE'''
    # add the Exit buffer area and other operations, to do......
    x, y = O.loca[0], O.loca[1]
    IN = False
    if E[0]<=x<=E[0]+E[2]+1 and E[1]<=y<=E[1]+E[3]:
        IN = True
    return IN

def Crowd_mass(num):
    '''#generate num people's Mass'''
    mass = []
    for n in range(num):
        a = random.gauss(1.0, 0.3)
        mass.append(a)
    return mass

def Crowd_velo(num):
    '''generate num people's Velocity Amplitude (m/s)'''
    velo = []
    for i in range(num):
        a = random.gauss(1.22, 0.29)
        velo.append(a)
        i+=1
    return velo

def Crowd_loca(num, empty, List =[]):
    '''set the inital location of  the crowd in the room'''
    '''if len(List) ==0:
        for i in range(1, 39):
            for j in range(40, 69):
                List.append([i, j])'''
    Locas = random.sample(empty, num)
    return Locas

#@【Information Output】---------------------------------------------------------------------------------------------------------------------------------
#@【VTK Output】---------------------------------------------------------------------------------------------------------------------------------
#def VTK_basic()
    #cube =

#@【Figure Generate】---------------------------------------------------------------------------------------------------------------------------------

#@【Tkinter Visualization】---------------------------------------------------------------------------------------------------------------------------------

def Visual_basic(ratio):
    '''show background of the room'''
    global R
    global BG
    R = tk.Tk()
    BG = tk.Canvas(R, width=1560, height=820,bg="white" )
    R.title("Evacuation simulation")
    R.geometry("{0}x{1}".format(1560, 820))
    R.resizable(True, True)

def Visual_objects(Bd, Ex, Bs, Cd, ratio):
    #Show barriers、exit、boundary、crowd
    Visual_barrier(Bs, ratio)
    Visual_exit_s(Ex, ratio)
    Visual_boundary(Bd, ratio)
    Visual_crowd(Cd, ratio)
    BG.pack()

def Mainloop():
    '''mainloop() function'''
    global BG
    BG.mainloop()

def Visual_barrier(Bs, ratio):
    '''show barriers in the room'''
    global BG
    for B in Bs:
        x1, y1, x2, y2 = B[0], B[1], B[0]+B[2]+1, B[1]+B[3]+1
        #[x1, y1, x2, y2] = map(lambda x: x * ratio, [x1, y1, x2, y2])
        [y1, x1, y2, x2] = map(lambda x: x * ratio, [x1, y1, x2, y2])
        BG.create_rectangle(x1, y1, x2, y2, fill="red", outline="black")
        #BG.create_polygon()
        BG.pack()

def Visual_one(O, ratio):
    '''show crowd in the room'''
    global BG
    r = O.r
    Ox, Oy = int(O.loca[0]*ratio), int(O.loca[1]*ratio)
    r *= ratio
    #print(r)
    #print(Oy - r)
    #BG.create_oval(Ox - r, Oy - r, Ox + r, Oy + r, fill="green", tag='p')
    BG.create_oval(Oy - r, Ox - r, Oy + r, Ox + r, fill="green", tag='p')
    BG.pack()

def Visual_crowd(Os, ratio):
    '''show crowds in the room'''
    global BG
    for O in Os:
        Visual_one(O, ratio)

def Visual_Update_crowd(Os, ratio):
    global  BG
    BG.delete('p')
    Visual_crowd(Os, ratio)
    BG.update()

def Visual_exit_s(Es, ratio):
    for E in Es:
        Visual_exit(E, ratio)

def Visual_exit(E, ratio):
    '''show the exit in the room'''
    global BG
    x1, y1, x2, y2 = E[0], E[1], E[0]+E[2], E[1]+E[3]
    #[x1, y1, x2, y2] = map(lambda x: x * ratio, [x1, y1, x2, y2])
    [y1, x1, y2, x2] = map(lambda x: x * ratio, [x1, y1, x2, y2])
    BG.create_rectangle(x1, y1, x2, y2, fill="green", outline="black")
    BG.pack()

def Visual_boundary(Bd, ratio):
    '''show boundary of the room'''
    global BG
    for B in Bd:
        x1, y1, x2, y2 = B[0], B[1], B[0]+B[2], B[1]+B[3]
        #[x1, y1, x2, y2] = map(lambda x: x * ratio, [x1, y1, x2, y2])
        [y1, x1, y2, x2] = map(lambda x: x * ratio, [x1, y1, x2, y2])
        BG.create_rectangle(x1, y1, x2, y2, fill="black", outline="black")
        BG.pack()

def Visual_line(Ps, ratio, width=1):
    '''show the line of ones'''
    global BG
    Ps = list(Ps)
    BG.create_line(Ps, width, fill="black")
    BG.pack()
