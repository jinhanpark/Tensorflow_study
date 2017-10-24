def show_16_pics(lst):
    nx = ny = 4
    canvas = np.empty((28*ny, 28*nx))
    for j in range(ny):
        for i in range(nx):
            ind = ny * j + i
            canvas[28*j:28*j+28, 28*i:28*i+28] = lst[ind].reshape(28, 28)
    plt.imshow(canvas, cmap="gray")
    plt.show()

data = mnist.test
n = data.num_examples


rand_arr = np.random.randint(0, n-1, size=100)

print("AE : test image enc-dec")
show_16_pics(ae_trainer.get_reconstructed_imgs(rand_arr[:16]))

print("VAE : test image enc-dec")
input_imgs = data.images[rand_arr]
output_imgs = vae_20d.reconstruct(input_imgs)
show_16_pics(output_imgs[:16])


print("AE : random latent vector")
lst = ae_trainer.get_random_latent_imgs()
show_16_pics(lst)

print("VAE : random latent vector")
lst = []
for i in range(16):
    out_img = vae_20d.generate()
    lst += [out_img]
show_16_pics(lst)


two_rand = np.random.randint(0, n-1, size=2)

print("AE: latent space walking")
lst = ae_trainer.get_lin_space_between_two_imgs(two_rand)
show_16_pics(lst)

print ("VAE: latent space walking")
zs = sess.run(vae_20.z,
              feed_dict={vae_20.x: data.images[two_rand]})
a = zs[0]
b = zs[1]
lst = []
for i in range(16):
    mid_points = []
    for j in range(vae_20d.network_architecture["n_z"]):
        mid_point = [a[j] + (b[j] - a[j]) * i / 15]
    out_img = vae_20d.generate(mid_point)
    lst += [out_img]
show_16_pics(lst)

