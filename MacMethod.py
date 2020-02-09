import numpy as np

def simulate_cavity_by_mac_method():
    '''計算の設定をする'''
    x_range = 31 #mesh size + 1（植木算）
    y_range = 31 #mesh size + 1（植木算）

    dt = 0.1
    re = 100

    dx = 1.0 / (x_range - 1)#mesh size
    dy = 1.0 / (y_range - 1)

    u_wall = 1.0 #上部の壁面の境界条件
    
    '''初期条件の設定'''
    #流速の初期条件を設定する
    u, v = _set_ini_u_v(x_range, y_range)

    #圧力の初期条件を設定する
    p = _set_ini_p(x_range, y_range)

    #loop
    for istep in range(0, 100):

        '''STEP 0 : 境界条件の更新'''
        #流速の境界条件を更新する
        u, v = _update_u_v_by_left_right_boundary(u, v, x_range, y_range)
        u, v = _update_u_v_by_up_down_boudary(u, v, x_range, y_range, u_wall)

        '''STEP 1 : 圧力のポアソン方程式'''
        #div Vを求める
        div_velocity = _calc_div_velocity(u, v, x_range, y_range, dx, dy)

        #移流項を求める
        adv_velocity = _calc_advection(u, v, x_range, y_range, dx, dy)

        #圧力のポアソン方程式を解き、圧力を更新
        _solve_poison(p, adv_velocity, div_velocity, u, v, re, x_range, y_range, dx, dy)

        '''STEP 2 : 速度を更新'''

        #流速を更新する


    return

#スタガード格子
def _set_ini_u_v(x_range, y_range):
    ini_u = np.zeros(((x_range+2), (y_range+1)))
    ini_v = np.zeros(((x_range+1), (y_range+2)))
    return ini_u, ini_v

def _set_ini_p(x_range, y_range):
    ini_p = np.zeros(((x_range+1), (y_range+1)))
    return ini_p

def _update_u_v_by_left_right_boundary(u, v, x_range, y_range):
    for y in range(0, y_range+1):
        u[1][y] = 0.0
        u[0][y] = u[2][y] #教科書P.70 圧力の境界条件を計算する為に求める（壁の向こう側）
        v[0][y] = -v[1][y]

        u[x_range][y] = 0.0
        u[x_range+1][y] = u[x_range-1][y]
        v[x_range][y] = -v[x_range-1][y]
    
    v[0][y_range+1] = -v[1][y_range+1]
    v[x_range][y_range+1] = -v[x_range-1][y_range+1]

    return u, v

def _update_u_v_by_up_down_boudary(u, v, x_range, y_range, u_wall_top):
    for x in range(0, x_range+1):
        u[x][0] = -u[x][1]
        v[x][1] = 0.0
        v[x][0] = v[x][2]

        u[x][y_range] = 2.0*u_wall_top - u[x][y_range-1] #(u_yrange+u_yrange-1)/2=u_wall
        v[x][y_range] = 0.0
        v[x][y_range+1] = v[x][y_range-1]
    
    u[x_range+1][0] = -u[x_range+1][0]
    u[x_range+1][y_range] = 2.0*u_wall_top - u[x_range+1][y_range-1]
    
    return u, v

def _calc_div_velocity(u, v, x_range, y_range, dx, dy):
    div = np.zeros((x_range+1, y_range+1)) #端っこはここでは入れない
    for x in range(1, x_range):
        for y in range(1, y_range):
            du_x = (u[x+1][y] - u[x][y])/dx
            dv_y = (v[x][y+1] - v[x][y])/dy
            div[x][y] = du_x + dv_y
    return div

def _calc_advection(u, v, x_range, y_range, dx, dy):
    adv = np.zeros((x_range+1, y_range+1)) #端っこはここでは入れない 
    for x in range(1, x_range):
        for y in range(1, y_range):
            u_x = (u[x+1][y] - u[x][y])/dx
            v_y = (v[x][y+1] - v[x][y])/dy
            v_x = (v[x+1][y]+v[x+1][y+1] - v[x-1][y] - v[x-1][y+1])/4
            u_y = (u[x][y+1]+u[x+1][y+1] - u[x][y-1] - u[x+1][y+1])/4

            adv[x][y] = u_x**2 + v_y**2 + 2*v_x*u_y
    return adv

def _solve_poison(p, advection_velocity, div_velocity, u, v, re, x_range, y_range, dx, dy):
    '''境界条件：ノイマン'''
    for x in range(0, x_range+1):
        p[x][0] = p[x][1] - 1.0/re*2.0*v[x][2] #上端(中心差分)
        p[x][y_range] = p[x][y_range-1] + 1.0/re*2.0*v[x][y_range-1] #下端（中心差分）

    for y in range(0, y_range+1):
        p[0][y] = p[1][y] - 1.0/re*2.0*u[2][y] #左端
        p[x_range][y] = p[x_range-1][y] + 1.0/re*2.0*u[x_range-1][y]

    '''ポアソン方程式をとく'''
    
    return