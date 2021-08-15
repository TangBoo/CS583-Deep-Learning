import torch
from torch.autograd import Variable as V
from tqdm import tqdm
import config
from model import YOLOv3
from loss import YoloLoss
import torch.optim as optim
from dataset import YoloDataset
import Meta as meta
from torch.utils.data import DataLoader
from utils import get_evaluation_bboxes, mAP

table = meta.meta_table
anchors = meta.anchors


def train_fn(train_loader, model, optimizer, loss_fn, scaler, anchors):
    loop = tqdm(train_loader, leave=True)
    losses = []

    for idx, (x, y) in enumerate(loop):
        x = V(x.to("cuda", dtype=torch.float))
        y2, y1, y0 = (
            V(y[0].to("cuda", dtype=torch.float)),
            V(y[1].to("cuda", dtype=torch.float)),
            V(y[2].to("cuda", dtype=torch.float))
        )

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (
                    loss_fn(out[0], y0, anchors[0]) + loss_fn(out[1], y1, anchors[1]) + loss_fn(out[2], y2, anchors[2])
            )

        losses.append(loss.item())
        optimizer.zero_grad()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)
        print(mean_loss)


def main():
    model = YOLOv3(num_classes = config.NUM_CLASSES).to(config.DEVICE, dtype=torch.float)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_fn = YoloLoss().to(config.DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    dataset = YoloDataset(table=table, anchors = anchors, transform=config.transform)
    train_loader = DataLoader(dataset, batch_size = 2, shuffle = True)

    for epoch in (range(20)):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, anchors)
        # if epoch > 0 and epoch % 3 == 0:
        #     #             check_class_accuracy(model, test_loader, threshold = 0.05)
        #     pred_boxes, true_boxes = get_evaluation_bboxes(
        #         test_loader,
        #         model,
        #         iou_threshold=0.45,
        #         anchors=anchors,
        #         threshold=0.05,
        #     )
        #     mapval = mAP(
        #         pred_boxes,
        #         true_boxes,
        #         iou_threshold=0.5,
        #         box_format="midpoint",
        #         num_classes=3,
        #     )
        #     print(f"MAP: {mapval.item()}")
        #     model.train()

if __name__ == "__main__":
    main()