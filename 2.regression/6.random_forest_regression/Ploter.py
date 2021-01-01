import matplotlib.pyplot as plt


# plot function to draw data figures
def plot(x_scat, y_scat, x_plt=None, y_plt=None, title="Figure", xlabel="X axis",
         ylabel="Y axis", plot_color="blue", scat_color="red", draw_plot=True, figure=1):
    plt.figure(figure)
    plt.title(title)
    plt.scatter(x_scat, y_scat, color=scat_color)
    if draw_plot:
        plt.plot(x_plt, y_plt, color=plot_color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
