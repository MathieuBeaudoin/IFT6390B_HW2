import matplotlib.pyplot as plt
from solution import SVM

def plot_learning(C_vec: list,
                  data: tuple,
                  common_args: dict = {
                      "niter": 200,
                      "eta": 0.0001,
                      "batch_size": 100,
                      "verbose": False},
                  verbose: bool = True):
    models = {}
    for C in C_vec:
        if verbose:
            print(f"Training model with C={C}")
        model = SVM(**common_args, C=C)
        model.fit(*data)
        models[C] = model

    axes = plt.subplots(ncols=4, figsize=(16, 4))[1]
    for c, model in models.items():
        to_plot = {
            "Training loss": model.train_losses,
            "Training accuracy": model.train_accs,
            "Test loss": model.test_losses,
            "Test accuracy": model.test_accs
        }
        for ax, (title, value) in zip(axes, to_plot.items()):
            ax.plot(value, label=str(c))
            ax.set_xlabel("Epoch")
            ax.set_title(title)
            ax.legend()
    plt.tight_layout()
    plt.show()