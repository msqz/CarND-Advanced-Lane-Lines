import matplotlib.pyplot as plt


def show(img, gray=0):
    plt.figure(figsize=(10, 6))
    if gray is 1:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)

    plt.show()
