
#from skimage.feature import corner_peaks, plot_matches


# black_mask = color.rgb2gray(image) < 0.2
# image[black_mask] = 1
# fig, ax = plt.subplots()
# ax.imshow(image, cmap='gray')
# ax.set_title('Microscopy image of human cells stained for nuclear DNA')
# plt.show()

# fig, ax = plt.subplots()
# ax.imshow(image, cmap='gray')
# ax.set_title('Microscopy image of human cells stained for nuclear DNA')
# plt.show()

#Compute the Canny filter for two values of sigma


# image_resized = resize(image, (image.shape[0] // 4, image.shape[1] // 4),
#                        anti_aliasing=True)
# image_downscaled = downscale_local_mean(image, (4, 3))

# fig, axes = plt.subplots(nrows=2, ncols=2)

# ax = axes.ravel()

# ax[0].imshow(image, cmap='gray')
# ax[0].set_title("Original image")

# ax[1].imshow(image_rescaled, cmap='gray')
# ax[1].set_title("Rescaled image (aliasing)")

# ax[2].imshow(image_resized, cmap='gray')
# ax[2].set_title("Resized image (no aliasing)")

# ax[3].imshow(image_downscaled, cmap='gray')
# ax[3].set_title("Downscaled image (no aliasing)")

# ax[0].set_xlim(0, 512)
# ax[0].set_ylim(512, 0)
# plt.tight_layout()
# plt.show()
