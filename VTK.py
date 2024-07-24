# @Time    : 2023/10/22 11:05
# @Author  : Chun Song
#【Write a VTK file, base on building coordinate, crowd coordinate】
import os

def Title():
    return  "# vtk DataFile Version 2.0 \nTry VTK 20231019 \nASCII \nDATASET POLYDATA\n"

def Element_num(element, num, type='double'):
    return ' '.join([str(element), str(num),str(type),'\n'])

def Poltgons(num):
    return ' '.join([str('POLYGONS'), str(num), str(num*5), '\n'])

def A_color(num):
    #l1 = ' '.join([str('CELL_DATA'), str(num)])
    #l2 = 'SCALARS Damage double 1'
    #l3 = 'LOOKUP_TABLE default'
    return 'CELL_DATA {0} \nSCALARS Damage double 1\nLOOKUP_TABLE default\n'.format(num)
    #return  ' '.join([str(l1),'\n',str(l2),'\n',str(l3)])

def List_Str(List):
    '''list to str'''
    str0 = ''
    for Li in List:
        str0 = ' '.join([str0, str(Li)])
    str0 = ' '.join([str0, '\n'])
    #str0.lstrip()
    return str0

def Color(value):
    '''set the area one color'''
    return ' '.join([str(value), '\n'])

def Color_s(value1 = 1, value2 =2 , num = 7):
    '''set areas colors'''
    colors = ''
    for i in range(num):
        if i == 6:
            colors = ''.join([colors, str(value2), '\n'])
        else:
            colors = ''.join([colors, str(value1), '\n'])
    return colors

def Rename(path):
    # 获取该目录下所有文件，存入列表中
    fileList = os.listdir(path)
    for file in fileList:
        # 设置旧文件名（就是路径+文件名）
        ext = os.path.splitext(file)
        oldname = ext[0]+'.csv'  # os.sep添加系统分隔符
        newname = ext[0]+'.vtk'
        os.rename(path+oldname, path+newname)  # 用os模块中的rename方法对文件改名

def Vuboid(od, A = [20,30,30,40], E=[1, 0.5, 1], H=[3, 6]):
    '''return Vuboid points and areas'''
    Ps = []
    As = []
    Ps.extend([[A[0], A[1],H[0]], [A[0],A[1]+A[3],H[0]],  [A[0]+A[2],A[1]+A[3],H[0]],  [A[0]+A[2], A[1],H[0]]])
    Ps.extend([[A[0], A[1],H[1]],  [A[0], A[1]+A[3], H[1]], [A[0]+A[2], A[1]+A[3],H[1]],  [A[0]+A[2], A[1],H[1]]])
    As.extend([[4, od,od+1,od+2,od+3], [4, od,od+1,od+5,od+4], [4, od+1,od+2,od+6,od+5],
              [4, od+2,od+3,od+7,od+6], [4, od+3,od,od+4,od+7], [4, od+4,od+5,od+6,od+7]])
    x, y, z = Ps[int(E[0])]
    if E[0]==0:
        a = A[3]*E[1]
        b = E[2]
        Ps.extend([[x, y+a-b, H[0]], [x, y+a+b, H[0]], [x, y+a+b, H[1]], [x, y+a-b, H[1]]])
    if E[0] == 1:
        a = A[2]*E[1]
        b = E[2]
        Ps.extend([[x+a-b, y,H[0]], [x+a+b, y,H[0]], [x+a+b, y,H[1]],[x+a-b, y,H[1]]])
    if E[0] == 2:
        a = A[3] * E[1]
        b = E[2]
        Ps.extend([[x, y-a-b , H[0]], [x, y-a+b, H[0]], [x, y-a+b , H[1]], [x, y-a-b, H[1]]])
    if E[0] == 3:
        a = A[2] * E[1]
        b = E[2]
        Ps.extend([[x-a-b, y , H[0]], [x-a+b, y, H[0]], [x-a+b, y , H[1]], [x-a-b, y, H[1]]])
    #else:
        #print('Check the Exit Coordinate: {0}'.format(E))
    As.append([4, od+8,od+9,od+10,od+11])

    Ps_n = len(Ps)
    As_n = len(As)
    return Ps_n, Ps, As_n, As

def Vuboid_s(AA):
    '''more Vuboids'''
    od, A_n = 0, 0
    PPs = []
    AAs = []
    for A in AA:
        Ps_n, Ps, As_n, As = Vuboid(od, A)
        od += Ps_n
        A_n += As_n
        PPs.extend(Ps)
        AAs.extend(As)
    return od, PPs, A_n, AAs

def Create_file(file_path, file_name, AS =[[20,30,30,40], [70,80,30,40], [70,30,30,40],[20,80,30,40]]):
    '''check the file path'''
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    #JOIN TH EFILE PATH AND FILE NAME
    file_PN = os.path.join(file_path, file_name)
    with open(file_PN, 'w') as file:
        title = Title()
        file.write(title)
        #Ps_n, Ps, As_n, As = Vuboid(0,[20,30,30,40],10)
        Ps_n, Ps, As_n, As = Vuboid_s([[20,30,30,40], [70,80,30,40], [70,30,30,40],[20,80,30,40]])
        P_title = Element_num('POINTS', Ps_n)
        file.write(P_title)
        for point in Ps:
            point = List_Str(point)
            file.write(point)
        file.write(Poltgons(As_n))
        for area in As:
            area = List_Str(area)
            file.write(area)
        file.write(A_color(As_n))
        print(As_n)
        for i in range(int(As_n/7)):
            #c = Color(2)
            c = Color_s()
            file.write(c)
        print("Write DONE!!!")
    Rename('New/')
    print('Rename DONE!!!')

Create_file('New/', 'New_3.csv')
