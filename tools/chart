import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def full(image, label, prediction, classes, epochs=None, train_loss=None, test_loss=None, title=None):
    display_loss = epochs and train_loss and test_loss
    
    if display_loss:
        gs = gridspec.GridSpec(2, 2)
    else:
        gs = gridspec.GridSpec(1, 2)

    plt.subplot(gs[0, 0])
    img_plot(image)

    plt.subplot(gs[0, 1])
    prediction_plot(prediction, classes)

    if display_loss:
        plt.subplot(gs[1, :])
        loss_plot(train_loss, test_loss)

    if title:
        plt.figure().canvas.set_window_title(title)
    else:
        plt.figure().canvas.set_window_title(f'Prediction for {label}')

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()

def img_plot(img, show=False):
    plt.title('Image')
    plt.imshow(img.numpy().squeeze(), cmap="Greys")
    if show:
        plt.show()

def prediction_plot(prediction, classes, show=False):
    plt.title('Prediction')
    plt.barh(range(len(classes)), prediction.numpy())
    plt.yticks(range(len(classes)), classes)
    if show:
        plt.show()

def loss_plot(train_loss, test_loss, show=False):
    x = range(len(train_loss))
    plt.title('Loss')
    plt.plot(x, train_loss, x, test_loss)
    plt.legend(('Training loss', 'Test loss'))
    if show:
        plt.show()
