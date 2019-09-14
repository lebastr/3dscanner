import matplotlib.pyplot as plt
import lenet


def plot_target_prediction(img, target, prediction, ax=None):
    colors = ['r', 'g', 'm', 'w']
    img = img.permute([1, 2, 0])

    gc_target, nb_target, coords_target = lenet.Net.extract_data(target[None, :])
    gc_pred, nb_pred, coords_pred = lenet.Net.extract_data(prediction[None, :])

    xc_target, yc_target, xn_target, yn_target = [coords_target[i][0].cpu().numpy() for i in range(4)]
    xc_pred, yc_pred, xn_pred, yn_pred = [coords_pred[i][0].cpu().numpy() for i in range(4)]

    xc_target, xc_pred, xn_target, xn_pred = [img.shape[0] * v for v in [xc_target, xc_pred, xn_target, xn_pred]]
    yc_target, yc_pred, yn_target, yn_pred = [img.shape[1] * v for v in [yc_target, yc_pred, yn_target, yn_pred]]

    if ax is None:
        _, ax = plt.subplots(1, 1)

    ax.imshow(img)

    ax.scatter(xc_target, yc_target, c=colors, s=100, marker='+')
    ax.scatter(xn_target, yn_target, c=colors, s=100, marker='x')

    ax.scatter(xc_pred, yc_pred, c=colors, s=50, marker='+')
    ax.scatter(xn_pred, yn_pred, c=colors, s=50, marker='x')

    return ax.figure