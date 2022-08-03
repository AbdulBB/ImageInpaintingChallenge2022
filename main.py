# -*- coding: utf-8 -*-

import glob
import os
import random

import albumentations as A
import cv2
import numpy as np
import torch
import torch.utils.data
import wandb
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torchsummary

from architectures import SimpleCNN
from datasets import ImageDataset
from utils import plot


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn,
    device: torch.device,
):
    """Function for evaluation of a model `model` on the data in `dataloader` on device `device`,
    using the specified `loss_fn` loss function"""
    model.eval()
    # We will accumulate the mean loss in variable `loss`
    loss = 0
    with torch.no_grad():  # We do not need gradients for evaluation
        # Loop over all samples in `dataloader`
        for data in tqdm(dataloader, desc="scoring", position=0):
            # Get a sample and move inputs and targets to device
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Get outputs of the specified model
            outputs = model(inputs)

            # Here we could clamp the outputs to the minimum and maximum values of inputs for better performance

            # Add the current loss, which is the mean loss over all minibatch samples
            # (unless explicitly otherwise specified when creating the loss function!)
            loss += loss_fn(outputs, targets).item()
    # Get final mean loss by dividing by the number of minibatch iterations (which
    # we summed up in the above loop)
    loss /= len(dataloader)
    model.train()
    return loss


def main(
    results_path,  # path to save results
    network_config: dict,  # network configuration
    learningrate: int = 1e-3,  # learning rate
    weight_decay: float = 1e-5,  # weight decay
    n_updates: int = 50_000,  # number of updates
    device: torch.device = torch.device("cuda:0"),  # device to use
):
    """Main function that takes hyperparameters and performs training and evaluation of model"""
    # Set a known random seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Prepare a path to plot to
    plotpath = os.path.join(results_path, "plots")
    os.makedirs(plotpath, exist_ok=True)

    train_data_path = r"/mnt/d/Data/training"

    train_image_paths = []  # to store image paths in list
    classes = []  # to store class values

    # Loop over all subdirectories in the training data path
    image_paths = glob.glob(f"{train_data_path}/**/*.jpg", recursive=True)

    def flatten(exp):
        def sub(exp, res):
            if type(exp) == dict:
                for k, v in exp.items():
                    yield from sub(v, res + [k])
            elif type(exp) == list:
                for v in exp:
                    yield from sub(v, res)
            else:
                yield "/".join(res + [exp])

        yield from sub(exp, [])

    train_image_paths = list(flatten(image_paths))  # flatten the list of lists
    random.shuffle(train_image_paths)  # shuffle the list of image paths

    print(
        "train_image_path example: ", len(train_image_paths)
    )  # print the length of the list of image paths

    # 2.
    # split train valid from train paths (80,20)
    train_image_paths, valid_image_paths = (
        train_image_paths[
            : int(0.9 * len(train_image_paths))
        ],  # 90% of the list of image paths
        train_image_paths[
            int(0.9 * len(train_image_paths)) :
        ],  # 10% of the list of image paths
    )
    print(
        "Train size: {}\nValid size: {}\n".format(
            len(train_image_paths),
            len(
                valid_image_paths
            ),  # print the length of the train and valid image paths
        )
    )
    train_transforms = A.Compose(
        [
            A.RandomCrop(
                100, 100, always_apply=True, p=1
            ),  # resize the image to 100x100
            A.HorizontalFlip(p=0.5),  # flip the image horizontally
            A.VerticalFlip(p=0.5),  # flip the image vertically
            A.Transpose(p=0.5),  # transpose the image
            ToTensorV2(),  # convert the image to a tensor
        ]
    )

    validation_transforms = A.Compose(
        [
            A.Resize(
                100, 100, interpolation=cv2.INTER_LINEAR, always_apply=True, p=1
            ),  # resize the image to 100x100
            ToTensorV2(),  # convert the image to a tensor
        ]
    )

    train_dataset = ImageDataset(
        train_image_paths, train_transforms, random_offset=True, random_spacing=True
    )  # create a train dataset
    valid_dataset = ImageDataset(
        valid_image_paths, validation_transforms
    )  # create a valid dataset

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=16,  # create a train dataloader
        pin_memory=True
    )
    valloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=16,  # create a valid dataloader
        pin_memory=True
    )

    # Create Network
    net = SimpleCNN(**network_config)  # create a network
    print(torchsummary.summary(net, (4, 100, 100)))  # print the summary of the network
    net.to(device)  # move the network to the device

    # Get mse loss function
    mse = torch.nn.MSELoss()

    # Get adam optimizer
    optimizer = torch.optim.Adam(
        net.parameters(), lr=learningrate, weight_decay=weight_decay
    )

    print_stats_at = 100  # print status to tensorboard every x updates
    plot_at = 1_000  # plot every x updates
    validate_at = 5000  # evaluate model on validation set and check for new best model every x updates
    update = 0  # current update counter
    best_validation_loss = np.inf  # best validation loss so far
    update_progress_bar = tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)

    # Save initial model as "best" model (will be overwritten later)
    saved_model_file = os.path.join(results_path, "best_model.pt")
    torch.save(net, saved_model_file)

    # Train until n_updates updates have been reached
    while update < n_updates:  # while the update is less than the number of updates
        for data in trainloader:
            # Get next samples
            inputs, targets = data  # get the inputs and targets
            inputs = inputs.to(device)  # move the inputs to the device
            targets = targets.to(device)  # move the targets to the device

            # Reset gradients
            optimizer.zero_grad()

            # Get outputs of our network
            outputs = net(inputs)

            # Calculate loss, do backward pass and update weights
            loss = mse(outputs, targets)
            loss.backward()
            optimizer.step()

            # Print current status and score
            if (update + 1) % print_stats_at == 0:
                wandb.log({"training/loss": loss})
            # Plot output
            if (update + 1) % plot_at == 0:
                plot(
                    inputs.detach().cpu().numpy(),  # detach the inputs from the graph
                    targets.detach().cpu().numpy(),  # detach the targets from the graph
                    outputs.detach().cpu().numpy(),  # detach the outputs from the graph
                    plotpath,
                    update,
                )

            # Evaluate model on validation set
            if (
                update + 1
            ) % validate_at == 0:  # if the update is divisible by the number of updates
                val_loss = evaluate_model(
                    net,
                    dataloader=valloader,
                    loss_fn=mse,
                    device=device,  # evaluate the model
                )
                wandb.log({"validation/loss": val_loss})  # log the validation loss

                # Save best model for early stopping
                if val_loss < best_validation_loss:
                    best_validation_loss = val_loss
                    torch.save(net, saved_model_file)

            update_progress_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
            update_progress_bar.update()

            # Increment update counter, exit if maximum number of updates is reached
            # Here, we could apply some early stopping heuristic and also exit if its
            # stopping criterion is met
            update += 1
            if update >= n_updates:
                break

    update_progress_bar.close()  # close the progress bar
    print("Finished Training!")  # print that the training is finished

    # Load best model and compute score on test set
    print(f"Computing scores for best model")
    net = torch.load(saved_model_file)  # load the best model
    train_loss = evaluate_model(
        net, dataloader=trainloader, loss_fn=mse, device=device
    )  # evaluate the model on the train set
    val_loss = evaluate_model(
        net, dataloader=valloader, loss_fn=mse, device=device
    )  # evaluate the model on the valid set

    print(f"Scores:")
    print(f"  training loss: {train_loss}")  # print the training loss
    print(f"validation loss: {val_loss}")  # print the validation loss

    # write the results to a text file
    with open(os.path.join(results_path, "results.txt"), "w") as rf:
        print(f"Scores:", file=rf)
        print(f"  training loss: {train_loss}", file=rf)
        print(f"validation loss: {val_loss}", file=rf)


if __name__ == "__main__":
    import argparse
    import json

    wandb.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to JSON config file")
    args = parser.parse_args()

    with open(args.config_file) as cf:
        config = json.load(cf)
        wandb.config.update(config)
    main(**config)
