import torch
from tqdm import tqdm
import os


def train(
    data_loader: torch.utils.data.DataLoader, 
    model: torch.nn.Module, 
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epoch: int = 20, 
    device: str = 'cuda'
):
    """
    Training function for 
    """
    model.to(device)
    model.train()

    for epoch in range(num_epoch):
        loop = tqdm(data_loader, leave=True)
        running_loss = 0.0
        
        for batch_idx, (image, label) in enumerate(loop):
            # remember the index starts from 0
            image = image.to(device)
            label = label.to(device)

            out = model(image, label)

            loss = criterion(out.reshape(-1, out.shape[2]), label.reshape(-1))
            loss.backward()

            optimizer.zero_grad()
            optimizer.step()

            running_loss += loss.item()

            loop.set_description(f"Epoch [{epoch+1}/{num_epoch}]")
            loop.set_postfix(loss=loss.item())

        # TODO: Add training checkpointing to continue training from the checkpoint
            
        avg_epoch_loss = running_loss / len(data_loader)
        print(f"Epoch {epoch+1} Complete. Average Loss: {avg_epoch_loss:.6f}")

    torch.save(model, os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp", "models", "ImageCaptionv1.pt"))
            
    return model
