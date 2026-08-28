import matplotlib.pyplot as plt

def plot_fig(loss,label,predict):
    plt.figure()
    plt.plot(loss,label="loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("loss")
    plt.legend()

    plt.figure()
    plt.plot(label,label="label")
    plt.plot(predict,label="predict")
    plt.xlabel("feature")
    plt.ylabel("temp")
    plt.title("compare")
    plt.legend()
    
    plt.show()