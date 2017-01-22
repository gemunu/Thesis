for i in range(len(norm_tr_flux)):
    plt.clf()
    plt.plot(wl,norm_tr_flux[i])
    plt.draw()
    plt.pause(2)
