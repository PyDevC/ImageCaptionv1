import torch
from tqdm import tqdm


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
    model.train() # TODO: Let me check if I need it or not

    for epoch in range(num_epoch):
        loop = tqdm(data_loader, leave=True)
        running_loss = 0.0
        
        for batch_idx, (image, label) in enumerate(loop):
            # remember the index starts from 0
            image = image.to(device)
            label = label.to(device)

            out = model(image)

            loss = criterion(out, label) # TODO: Check if the order is correct
            loss.backward()

            optimizer.zero_grad()
            optimizer.step()

            running_loss += loss.item()

            loop.set_description(f"Epoch [{epoch+1}/{num_epoch}]")
            loop.set_postfix(loss=loss.item())

        # TODO: Add training checkpointing to continue training from the checkpoint
            
        avg_epoch_loss = running_loss / len(data_loader)
        print(f"Epoch {epoch+1} Complete. Average Loss: {avg_epoch_loss:.6f}")
            
    return model
