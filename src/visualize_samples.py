import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import ShanghaiTechImageDataset


def show_batch(num_images: int = 4, part: str = "A", split: str = "train") -> None:
    """
    Display a small batch of images from the preprocessed dataset.

    Args:
        num_images: Number of images to display (max per batch).
        part: "A" or "B"
        split: "train" or "test"
    """
    dataset = ShanghaiTechImageDataset(part=part, split=split)
    loader = DataLoader(dataset, batch_size=num_images, shuffle=True)

    images, paths = next(iter(loader))  # images: (B, C, H, W)

    # Undo normalization for display
    mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images = images * std + mean
    images = images.clamp(0.0, 1.0)

    batch_size = images.size(0)
    cols = min(batch_size, num_images)
    rows = 1

    plt.figure(figsize=(4 * cols, 4 * rows))
    for i in range(cols):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(paths[i].split("/")[-1])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_batch(num_images=4, part="A", split="train")

