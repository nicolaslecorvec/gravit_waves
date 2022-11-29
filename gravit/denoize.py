



def denoize(a, b, offx, offy, rate, flipv=False, data_type=["test","test"]):
    plt.figure(figsize=(24,4))
    x = load_example(a, data_type[0])
    y = load_example(b, data_type[1])
    if flipv:
        x = x[::-1,:]

    z = x.copy()
    z[offy:,:z.shape[1]-offx] -= y[:y.shape[0]-offy,offx:]

    x = x - x.mean()
    x = np.mean(x.reshape(360, 128, 32), axis=2)
    x = img_clipping(x)

    print(z.shape)
    z = z - z.mean()
    z = np.mean(z.reshape(360, 128, 32), axis=2)
    z = img_clipping(z)

    ax = plt.subplot(1,4,1)
    ax.set_xticks(np.arange(1,16)*360)
    ax.set_yticks([])
    ax.set_title(a)
    ax.imshow(x, aspect="auto", cmap="Greys", norm=plt.Normalize(vmin=-256, vmax=256))

    ax = plt.subplot(1,4,2)
    ax.set_xticks(np.arange(1,16)*360)
    ax.set_yticks([])
    ax.set_title(f"{a} - {b}")
    ax.imshow(z, aspect="auto", cmap="Greys", norm=plt.Normalize(vmin=-256, vmax=256))
