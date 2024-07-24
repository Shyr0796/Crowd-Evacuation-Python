# @Time    : 2023/10/4 19:49
# @Author  : Chun Song
#【space】set class Room(room information and field information)
#【people】set class One and Crowd(all people informaton and interaction )
#【run】

import time
import math
import random
import numpy as np
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=10000)

import pandas as pd
import tkinter as tk

from fun import *
#from VTK import *

class Time:
    def __init__(self):
        self.time = 0.0
        self.delta = 0.5

class One:
    # def __init__(self):
    def __init__(self, mass, velo_a, loca, step =1):
        self.s = step
        self.r = 0.25           #radius o f one
        self.m = mass     #mass of one
        self.v_a =velo_a*step  #velocity amplitude and time step
        self.v = [0, 0]       #velocity diraction [array,array] or degree(radian: 0-2*pi)
        self.v_dg = 0
        self.acce = 1        #accelaration
        self.loca = loca #location now

    def Delta(self):
        return [self.v_a*Cos(self.v_dg), self.v_a*Sin(self.v_dg)]

    def Move(self, A, W_b):
        self.v = self.Delta()
        self.Check_In_room(A, W_b)
        self.loca = Vector_sum(self.loca, self.v)

    def Move_try(self):
        return Vector_sum(self.loca, self.v*self.a)

    def V_update(self, V_field):
        '''update velocity diraction based on location and Velocity field'''
        #x, y = int(self.loca[0]), int(self.loca[1])
        #self.v = V_field[x][y]
        self.v = [0, -1]

    def Check_In_room(self, A, W_b):
        '''check weather self is in the room, and collision'''
        InRoom = True
        #x, y = Vector_sum(self.loca, self.v*2)
        x, y = self.loca
        W_buffer = 0
        W_b += W_buffer
        if x<A[0]+W_b or x>=A[0]+A[2]-W_b:
            self.v = [-self.v[0], self.v[1]]
            #x += self.v[0]
            #self.loca = [self.loca[0]- self.v[0]*2, self.loca[1]+self.v[1]*2]
            #self.v_dg = np.arctan(self.v[1]/- self.v[0])
            InRoom=False
            #print("Clock!")
        if y< A[1]+W_b or y>=A[1]+A[3] -W_b:
            self.v = [self.v[0], -self.v[1]]
            #self.loca = [self.loca[0] + self.v[0]*2, self.loca[1] - self.v[1]*2]
            #self.v_dg = np.arctan(self.v[1] / - self.v[0])
            InRoom=False
            #print("Clock!")
        return InRoom

    def Perception_points(self):
        '''Get the perception points based on the location NOW'''
        Per_Ps1= Get_direction(self.loca, self.v_dg, r= [0.1, 0.2, 0.4, 0.8])
        #Per_Ps2 = Get_direction(self.loca, self.v_dg, r=3)
        return Per_Ps1

    def Check_Others(self):
        '''check others in the room and interaction'''

class Room:
    #def __init__(self, SP, L, W, E, N, ratio):

    #def __init__(self, SP=[10, 10], L=40, W=60, B= [[3, 3, 1, 1]], E=[19, 1, 3, 2, 1, 0], N =40, ratio=16):
    def __init__(self, SP, L, W, B, E, N, ratio):
        self.st = 0.2 #update time step
        self.SP = SP #start point, move the whole room to Start Point
        self.L = L #lenth
        self.W = W #width
        #【Barriers】
        self.B = B
        #self.B = [[3, 3, 1, 1]]
        #self.B =[[16,5,2,4],[4,8,4,2],[8,19,20,2],[13,29,2,2], [10,43,24,2], [16,25,2,2], [18,32,2,2]] #barriers
        #self.B = [[16,8,4,2],[20,18,9,3],[4,28,4,2],[8,19,6,2],[13,29,2,4], [10,43,8,8], [22,34,12,5]]
        #self.B = [[1,16, 15, 1], [1,34, 15, 1], [1,52, 15, 1],[24,16, 15, 1], [24,34, 15, 1], [24,52, 15, 1],
                  #[15,3,1,12], [15,21,1,13],[15,38,1,13],[15,57,1,12], [24,1,1,12], [24,18,1,13],[24,36,1,13],[24,54,1,12]]
        self.E =E #exit [m, n], 0<=m<=1, 0<=n<=1
        self.En = [20, 60, 4, 1] #entrance
        self.C = [] #crowd of the crowd
        self.T = 0.0 #time of the room/field
        self.N = N #number of ones in the room

        self.ratio = ratio #ratio showing the room in the screen
        self.Field_ratio = 10

        self.Room_offset()
        self.Room_generate()
        self.List_generate()
        self.Field_generate()
        self.Crowd_generate()

    def Room_offset(self):
        '''set the off set of the room in the building'''
        self.A0 = Move_B(self.SP, [0, 0, self.L, self.W])
        self.B0 = Move_B2(self.SP, self.B)
        self.E0 = Move_B(self.SP, self.E)
        self.En0 = Move_B(self.SP, self.En)
        self.RB0 = Room_boundary(self.A0, 1)
        self.A1 = np.array(self.A0)*self.Field_ratio
        self.B1 = np.array(self.B0)* self.Field_ratio
        self.E1 = np.array(self.E0)* self.Field_ratio
        self.RB1 = np.array(self.RB0)* self.Field_ratio

    def Room_generate(self):
        '''generate the room with start point, lenth, width, barriers, exits'''
        room = Room_set_types(self.A1, W_b = 1*self.Field_ratio) #generate room with boundary = 3, empty space = 0
        if Room_check_object(self.A1, self.B1): #check barriers in the space
            Room_set_value2(room, self.B1, 3) #barriers = 3
        self.R = room #room values
        #print(self.R[200:600, 300:1000])
        return room

    def List_generate(self):
        '''generate the list of barriers, boundary, exit'''
        self.List_all = Room_list(self.A0)
        self.List_barrier = Room_list2(self.B0, self.List_all, P=1)
        self.List_exit = Room_list(self.E0, self.List_all, P=1)
        self.List_boundary = Room_list2(Room_boundary(self.A0, 1), self.List_all, P=1)
        self.List_empty = self.List_all
        #return List_barrier, List_exit, List_boundary

    def Field_generate(self):
        '''generate velocity field, barriers field and exit field'''
        d_point = [self.E1[0]+0.5*self.E1[2], self.E1[1]+0.5*self.E1[3]]  #【buffer area】
        self.Field_velocity_1, self.Field_dg = Field_velocity(self.A1, d_point)
        #print(self.Field_velocity_1[550][950])
        self.Field_height_1 = Field_hight(self.A1, d_point)

    def Room_update(self):
        '''Update the crowd values in the room, based on the Room_generate, which has on add crowd values'''
        R_1 = self.Room_generate()
        for one in self.C:
            x, y = one.loca[0]*self.Field_ratio, one.loca[1]*self.Field_ratio
            #r = one.r*self.Field_ratio
            r = 1
            x_l, x_r, y_l, y_r = Room_crowd_value([x, y, r], b=0)
            R_1[x_l: x_r, y_l: y_r] = 2 #set the crowd value as 3
        self.R1 = R_1

    def Crowd_generate(self):
        '''generate the crowd in the room'''
        mass = Crowd_mass(self.N)
        velo = Crowd_velo(self.N)
        #loca = Crowd_loca(self.N, self.List_empty)
        loca = Crowd_loca(self.N, self.List_empty)
        for i in range(self.N):
            self.C.append(One(mass[i], velo[i], loca[i], step=self.st))

    def Crowd_VHF(self, one):
        '''VHF Method, update the crowd velocity '''
        #Per_Ps1, Per_Ps2, Per_Ps3 = one.Perception_points() #[x, y, degree]
        Per_Ps1 = one.Perception_points()  # [x, y, degree]
        #print(Per_Ps1)
        Per_Ps2 = Check_PerValues(self.Field_ratio, Per_Ps1, self.A1, self.R) #check and return [x, y, degree, value]
        #print(Per_Ps2)
        Per_P = Choose_PerValues(self.Field_ratio, Per_Ps2, self.Field_dg)
        one.v_dg = Per_P

    def Crowd_update(self):
        '''update the crowd loca and move、check exit 、show'''
        self.locas = []
        self.De_ones = []
        #self.Room_update()
        #print(np.shape(self.room))
        for one in self.C:
            #【velocity update】
            #one.V_update(self.Field_velocity_1) #velocity field method
            # 【VHF Method】
            self.Crowd_VHF(one)
            # 【make the move/check the boundary】
            one.Move(self.A0, W_b=1)
            #【Check the one in the exit】
            self.locas.append(one.loca)
            if Exit_check(one, self.E0):
                self.One_delete(one)
                self.De_ones.append(one)
                #self.One_add(one)
                self.N -=1
                # add one in another room......
        #【Time add】
        self.T += 1
        #【Visualization of the updated crowd】
        #Visual_Update_crowd(self.C, ratio=self.ratio)

    def Room_visualization(self):
        '''room visualization based on room and tkinter'''
        Bd = Room_boundary(self.A0, 1)
        #Visual_basic(ratio= self.ratio)
        Visual_objects(self.RB0, self.E0, self.B0, self.C)

    def Room_visual(self):
        '''visualization the room and move based on start points'''

    def One_add(self, one):
        '''add One object in the room'''
        #x = self.En0[0]+random.uniform(0, self.En0[2])
        #y = self.En0[1]+random.uniform(0, self.En0[3])
        #one.loca = [x, y]
        self.N +=1
        self.C.append(one)

    def One_delete(self, one):
        '''delate One object in the room'''
        one.loca = [one.loca[0]+self.E[4]*2, one.loca[1]+self.E[5]*2]
        #one.loca = [one.loca[0] , one.loca[1] + 1* 2]
        self.C.remove(one)

global R
global Connect
global ratio
#Connect = [[0,1], [2,1]]
ratio = 20

def Room_s():
    '''generate more rooms'''
    #Room[StartPoint, Length, Width, Barrier, Exit, CrowNumber, ratio = 16]
    #(self, SP=[10, 10], L=40, W=60, B=[[3, 3, 1, 1]], E=[19, 1, 3, 2, 1, 0], N=40, ratio=16):
    global  R
    global ratio
    global Connect

    '''R1 = Room([0,0],10,10,[[2,3,1,1]], [4,8,2,1,0,1], 18,ratio)
    R2 = Room([0,9],10,10,[[4,4,1,1]], [4,8,2,1,0,1], 18, ratio)
    R3 = Room([9, 9],10,10, [[4,4,1,1]], [1,4,1,2,-1,0], 24, ratio)
    R = [R1, R2, R3]'''

    R0 = Room([0, 0], 8, 8, [[2, 3, 1, 1]], [4, 6, 2, 1, 0, 1], 18, ratio)
    R1 = Room([7, 0], 8, 8, [[2, 3, 1, 1]],[4, 6, 2, 1, 0, 1], 18,ratio)
    R2 = Room([14, 0], 10, 8, [[2, 3, 1, 1]], [2, 6, 2, 1, 0,1], 24, ratio)
    R3 = Room([0, 20], 8, 28, [[2, 13, 1, 1]], [4, 1, 2, 1, 0, -1], 18, ratio)
    R4 = Room([7, 20], 8, 24, [[4, 4, 1, 1]],[6, 11, 1, 3, 1, 0], 18, ratio)
    R5 = Room([7, 43], 8, 16,[[4, 4, 1, 1]],[6, 7, 1, 3, 1,0], 24, ratio)
    R6 = Room([23, 7], 8, 12, [[2, 3, 1, 1]], [1, 6, 1, 2, -1,0], 18, ratio)
    R7 = Room([23, 18], 8, 18, [[2, 4, 1, 1]],[1, 8, 1, 2, -1, 0], 18,ratio)
    R8 = Room([23, 35], 8, 18,[[4, 4, 1, 1]],[1, 6, 1, 3,-1,0], 24, ratio)
    R9 = Room([0, 7], 15, 14, [], [13, 6, 1, 2, 1, 0], 18, ratio)
    R10 = Room([14, 7], 10, 54, [[2,3,1,1],[2,15,2,2],[3,22,3,2],[5, 29,1, 6],[4, 44, 2,2]], [2, 52, 5, 1, 0, 1], 18, ratio)
    R = [R0, R1, R2, R3, R4, R5, R6, R7, R8,R9, R10]
    Connect = [[0, 9], [1, 9], [3, 9], [1, 9], [4, 10], [2, 10], [5, 10], [6, 10], [7, 10], [8, 10], [9,10]]
    RB0, E0, B0, C = [], [], [], []
    for Ri in R:
        RB0.extend(Ri.RB0)
        E0.append(Ri.E0)
        B0.extend(Ri.B0)
        C.extend(Ri.C)
    Visual_basic(ratio)
    Visual_objects(RB0, E0, B0, C, ratio)
    return R

def Room_0():
    '''main function, check '''
    global R
    global ratio
    global  Connect
    R = Room_s()
    while R[-1].N >0:
        #Create_file('1023/', Name(T, R1.T),AS=[R1.A0])
        Crow_new = []
        for Ri in R:
            Ri.Crowd_update()
            Crow_new.extend(Ri.C)
        for con in Connect:
            Room_Connect(R, con[0], con[1])
        Visual_Update_crowd(Crow_new, ratio)
        #time.sleep(R1.st)
        time.sleep(0.1)
    print('疏散用时: {:.3f} S'.format(R1.T*R1.st))
    print("Evacuation success! ")
    #Mainloop()

time_start = time.time()
#SP=[40, 30], L=40, W=50, E=[19, 1, 3, 2, 1, 0], N =30, ratio=10
Room_0()
time_end = time.time()
print('运行用时: {:.3f} S'.format(time_end-time_start))

'''if __name__ == '__main__':
    main()
    print ('now __name__ is %s' %__name__)'''