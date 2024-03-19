import torch
import matplotlib.pyplot as plt

def plot_scheduler(scheduler, epochs, title):
    """
    Plots the learning rate schedule.

    :param scheduler: Learning rate scheduler.
    :param epochs: Total number of epochs.
    :param title: Title for the plot.
    """
    lrs = []
    for epoch in range(epochs):
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        lrs.append(lr)
    plt.plot(range(epochs), lrs)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.show()

# Parameters
initial_lr = 1
epochs = 500

# Example optimizer
optimizer = torch.optim.SGD([torch.zeros(1)], lr=initial_lr)

# # StepLR Scheduler
# step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
# plot_scheduler(step_scheduler, epochs, 'StepLR Scheduler')

# Reset optimizer
optimizer = torch.optim.SGD([torch.zeros(1)], lr=initial_lr)

# CosineAnnealingLR Scheduler
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
plot_scheduler(cosine_scheduler, epochs, 'CosineAnnealingLR Scheduler')
