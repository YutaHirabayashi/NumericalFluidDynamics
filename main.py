
import matplotlib.pyplot as plt
import numpy as np
import pickle

from MacMethod import simulate_cavity_by_mac_method


def create_vector_image(u, v, p, index):
    u = u.T
    v = -v.T
    p = p.T #画像化の為

    lx = u.shape[1]
    ly = u.shape[0]

    X, Y= np.meshgrid(np.arange(0, lx), np.arange(0, ly))

    plt.figure(dpi = 300)

    #大きさは色
    # r = (u**2+v**2)**0.5
    # plt.imshow(r)
    # plt.colorbar()

    #方向だけを矢印で表現
    # plt.quiver(X,Y,u/r,v/r, scale = 15, color = "white")
    quiver_fig_name = "uv_{:0=3}.png".format(index)
    # plt.savefig(quiver_fig_name)
    # plt.close()

    # #圧力
    # plt.imshow(p)
    # plt.colorbar()
    # p_fig_name = "p_{:0=3}.png".format(index)
    # plt.savefig(p_fig_name)
    # plt.close()

    plt.quiver(X,Y,u,v,scale=2,color="white")
    plt.imshow(p)
    plt.colorbar()
    plt.savefig(quiver_fig_name)

    return


def main():
    loop_number = 100
    # u, v, p = simulate_cavity_by_mac_method(loop_number)
    # pickle.dump(u, open("u.pkl", "wb"))
    # pickle.dump(v, open("v.pkl", "wb"))
    # pickle.dump(p, open("p.pkl", "wb"))

    u = pickle.load(open("u.pkl","rb"))
    v = pickle.load(open("v.pkl","rb"))
    p = pickle.load(open("p.pkl","rb"))

    for index_loop in range(0, loop_number, 10):
        create_vector_image(u[index_loop], v[index_loop], p[index_loop], index_loop)


    
    return


if __name__ == "__main__":
    main()
    pass