import matplotlib.pyplot as plt
import torch


def show_image(img_tensor):
    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()

    plt.figure()
    plt.imshow(img)
    plt.title("Input Image (256x256)")
    plt.axis("off")
    plt.show()


def show_density_map(density_tensor, title="Density Map"):
    density = density_tensor.detach().cpu().squeeze().numpy()

    plt.figure()
    plt.imshow(density, cmap="jet")
    plt.title(title)
    plt.axis("off")
    plt.show()


def compare_batch(images, gt_maps, pred_maps, max_samples=5):

    import matplotlib.pyplot as plt

    batch_size = images.shape[0]
    n = min(batch_size, max_samples)

    plt.figure(figsize=(10, 3 * n))

    for i in range(n):

        image = images[i].detach().cpu().permute(1, 2, 0).numpy()
        gt = gt_maps[i].detach().cpu().squeeze().numpy()
        pred = pred_maps[i].detach().cpu().squeeze().numpy()

        gt_count = gt.sum()
        pred_count = pred.sum()

        # 🔹 Column 1: Input
        plt.subplot(n, 3, i * 3 + 1)
        plt.imshow(image)
        if i == 0:
            plt.title("Input")
        plt.axis("off")

        # 🔹 Column 2: Ground Truth
        plt.subplot(n, 3, i * 3 + 2)
        plt.imshow(gt, cmap="jet", vmin=0, vmax=gt.max())
        if i == 0:
            plt.title("Ground Truth")
        plt.axis("off")

        # 🔹 Column 3: Prediction
        plt.subplot(n, 3, i * 3 + 3)
        plt.imshow(pred, cmap="jet", vmin=0, vmax=pred.max())
        if i == 0:
            plt.title("Predicted")
        plt.axis("off")

        # 🔹 Add text on right side
        plt.text(
            1.05, 0.5,
            f"GT: {gt_count:.0f}\nPred: {pred_count:.0f}",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='center'
        )

    plt.tight_layout()
    plt.show()

    # Input Image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis("off")

    # Ground Truth
    plt.subplot(1, 3, 2)
    plt.imshow(gt, cmap="jet")
    plt.title(f"GT Density\nCount: {gt_count:.2f}")
    plt.axis("off")

    # Prediction
    plt.subplot(1, 3, 3)
    plt.imshow(pred, cmap="jet")
    plt.title(f"Predicted Density\nCount: {pred_count:.2f}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
