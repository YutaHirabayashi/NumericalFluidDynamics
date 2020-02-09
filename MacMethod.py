import numpy as np
import numba

def simulate_cavity_by_mac_method(loop_number = 1000):
    '''計算の設定をする'''
    x_range = 10 #mesh size + 1（植木算）
    y_range = 10 #mesh size + 1（植木算）

    dt = 0.001
    re = 100

    dx = 1.0 / (x_range - 1)#mesh size
    dy = 1.0 / (y_range - 1)

    #loop_number = 1000

    u_wall = 1.0 #上部の壁面の境界条件
    
    '''初期条件の設定'''
    #流速の初期条件を設定する
    u, v = _set_ini_u_v(x_range, y_range)

    #圧力の初期条件を設定する
    p = _set_ini_p(x_range, y_range)

    #結果格納用(格子点の結果をもつ)
    result_u = np.zeros((loop_number, x_range-1, y_range-1))
    result_v = np.zeros((loop_number, x_range-1, y_range-1))
    result_p = np.zeros((loop_number, x_range-1, y_range-1))

    #loop
    for index_main_loop in range(0, loop_number):

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
        p = _solve_poison(p, adv_velocity, div_velocity, u, v, re, x_range, y_range, dx, dy, dt)

        '''STEP 2 : 速度を更新'''
        u, v = _update_uv(u, v, p, re, dt, dx, dy, x_range, y_range)

        '''STEP 3 : 結果を格納'''
        result_u[index_main_loop] = _change_u_size(u, x_range, y_range)
        result_v[index_main_loop] = _change_v_size(v, x_range, y_range)
        result_p[index_main_loop] = _change_p_size(p, x_range, y_range)



    return result_u, result_v, result_p

#スタガード格子
@numba.jit
def _set_ini_u_v(x_range, y_range):
    ini_u = np.zeros(((x_range+2), (y_range+1)))
    ini_v = np.zeros(((x_range+1), (y_range+2)))
    return ini_u, ini_v

@numba.jit
def _set_ini_p(x_range, y_range):
    ini_p = np.zeros(((x_range+1), (y_range+1)))
    return ini_p

@numba.jit
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

@numba.jit
def _update_u_v_by_up_down_boudary(u, v, x_range, y_range, u_wall_top):
    for x in range(0, x_range+1):
        u[x][0] = -u[x][1]
        v[x][1] = 0.0
        v[x][0] = v[x][2]

        u[x][y_range] = 2.0*u_wall_top - u[x][y_range-1] #(u_yrange+u_yrange-1)/2=u_wall
        v[x][y_range] = 0.0
        v[x][y_range+1] = v[x][y_range-1]
    
    u[x_range+1][0] = -u[x_range][0]
    u[x_range+1][y_range] = - u[x_range][y_range]
    
    return u, v

@numba.jit
def _calc_div_velocity(u, v, x_range, y_range, dx, dy):
    #教科書P.69 (3.47) 右辺第１項（の1/dt
    div = np.zeros((x_range+1, y_range+1)) #端っこはここでは入れない
    for x in range(1, x_range):
        for y in range(1, y_range):
            du_x = (u[x+1][y] - u[x][y])/dx
            dv_y = (v[x][y+1] - v[x][y])/dy
            div[x][y] = du_x + dv_y
    return div

@numba.jit
def _calc_advection(u, v, x_range, y_range, dx, dy):
    #教科書P.69 (3.47) 右辺第二項＋第三項
    adv = np.zeros((x_range+1, y_range+1)) #端っこはここでは入れない 
    for x in range(1, x_range):
        for y in range(1, y_range):
            u_x = (u[x+1][y] - u[x][y])/dx
            v_y = (v[x][y+1] - v[x][y])/dy
            v_x = (v[x+1][y]+v[x+1][y+1] - v[x-1][y] - v[x-1][y+1])/(4*dx)
            u_y = (u[x][y+1]+u[x+1][y+1] - u[x][y-1] - u[x+1][y-1])/(4*dy)

            adv[x][y] = u_x**2 + v_y**2 + 2*v_x*u_y
    return adv

@numba.jit
def _solve_poison(p, advection_velocity, div_velocity, u, v, re, x_range, y_range, dx, dy, dt):
    #教科書P69 (3.47)
    
    C1 = 0.5*dy*dy/(dx*dx+dy*dy)
    C2 = 0.5*dx*dx/(dx*dx+dy*dy)
    C3 = 0.5*dy*dy/(1.+dy*dy/(dx*dx))

    for _ in range(0, 1000):

        '''境界条件：ノイマン'''
        for x in range(0, x_range+1):
            p[x][0] = p[x][1] - 1.0/re*2.0*v[x][2] #上端(中心差分)
            p[x][y_range] = p[x][y_range-1] + 1.0/re*2.0*v[x][y_range-1] #下端（中心差分）

        for y in range(0, y_range+1):
            p[0][y] = p[1][y] - 1.0/re*2.0*u[2][y] #左端
            p[x_range][y] = p[x_range-1][y] + 1.0/re*2.0*u[x_range-1][y]

        '''方程式をとく(SOR)'''
        err = 0
        for ix in range(1, x_range):
            for iy in range(1, y_range):
                r = div_velocity[ix][iy] / dt - advection_velocity[ix][iy]
                pres=C1*(p[ix+1][iy]+p[ix-1][iy])+C2*(p[ix][iy+1]+p[ix][iy-1])-C3*r-p[ix][iy]
                err+=(pres*pres)
                p[ix][iy]=pres+p[ix][iy]

        if err<=0.000005:
            break

    return p

@numba.jit
def _update_uv(u, v, p, re, dt, dx, dy, x_range, y_range):
    '''u''' #教科書P69 (3.48)
    new_u = u
    for ix in range(2, x_range):
        for iy in range(1, y_range):
            vmid=(v[ix][iy]+v[ix][iy+1]+v[ix-1][iy+1]+v[ix-1][iy])/4.0
            uad=u[ix][iy]*(u[ix+1][iy]-u[ix-1][iy])/2.0/dx+vmid*(u[ix][iy+1]-u[ix][iy-1])/2.0/dy
            udif=(u[ix+1][iy]-2.0*u[ix][iy]+u[ix-1][iy])/dx/dx+(u[ix][iy+1]-2.0*u[ix][iy]+u[ix][iy-1])/dy/dy
            new_u[ix][iy]=u[ix][iy]+dt*(-uad-(p[ix][iy]-p[ix-1][iy])/dx+1.0/re*udif)
        
    '''v''' #教科書P69 (3.49)
    new_v = v
    for ix in range(1, x_range):
        for iy in range(2, y_range):
            umid=(u[ix][iy]+u[ix+1][iy]+u[ix+1][iy-1]+u[ix][iy-1])/4.0
            vad=umid*(v[ix+1][iy]-v[ix-1][iy])/2.0/dx+v[ix][iy]*(v[ix][iy+1]-v[ix][iy-1])/2.0/dy
            vdif=(v[ix+1][iy]-2.0*v[ix][iy]+v[ix-1][iy])/dx/dx+(v[ix][iy+1]-2.0*v[ix][iy]+v[ix][iy-1])/dy/dy
            new_v[ix][iy]=v[ix][iy]+dt*(-vad-(p[ix][iy]-p[ix][iy-1])/dy+1.0/re*vdif)

    return new_u, new_v

@numba.jit
def _change_u_size(u, x_range, y_range):
    changed_u = np.zeros((x_range-1, y_range-1))

    for ix in range(0, x_range-1):
        for iy in range(0, y_range-1):
            changed_u[ix][iy] = (u[ix+1][iy+1] + u[ix+2][iy+1])/2
    return changed_u

@numba.jit
def _change_v_size(v, x_range, y_range):
    changed_v = np.zeros((x_range-1, y_range-1))

    for ix in range(0, x_range-1):
        for iy in range(0, y_range-1):
            changed_v[ix][iy] = (v[ix+1][iy+2] + v[ix+1][iy+1])/2
    return changed_v

@numba.jit
def _change_p_size(p, x_range, y_range):
    changed_p = np.zeros((x_range-1, y_range-1))

    for ix in range(0, x_range-1):
        for iy in range(0, y_range-1):
            changed_p[ix][iy] = p[ix+1][iy+1]
    return changed_p