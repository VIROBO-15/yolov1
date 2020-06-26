import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LAMBDA_COORD = 5
LAMBDA_NOOBJ = 0.5


def calc_loss(inp , target, opt):
    if inp.size(0) != target.size(0):
        raise Exception("Batch size does not match")

    total_loss = torch.tensor(0.0)
    #total_loss = total_loss.dtype(tensor)

    for i in range(inp.size(0)):
        inp = inp[i]
        target = target[i]
        Q = predict_one_bbox(inp, target, opt)
        total_loss = total_loss + calc_loss_single(Q, target, opt)
        return total_loss

def predict_one_bbox(inp, target,  opt):
    Q = torch.zeros(opt.S, opt.S, 5 + opt.C)

    select = torch.tensor(0).to(device)

    for i in range(opt.S):
        for j in range(opt.S):
            for b in range(opt.B):
                if b==0:
                    boxes = inp[i, j, b*5 : b*5+5].to(device)
                else:
                    boxes = torch.stack((boxes, inp[i, j, b*5 : b*5+5])).to(device)

            if len(target[i, j, :].nonzero()) > 1:
                max_iou = torch.tensor([0.]).to(device)


                groundtruth_box = target[i, j, :4].clone()

                for b in range(opt.B):
                    iou = calc_IOU(groundtruth_box, boxes[b][:-1], device)

                    if iou > max_iou:
                        max_iou = iou
                        select = torch.tensor(b).to(device)

            else:
                max_confidence = torch.tensor(0.).to(device)

                for b in range(opt.B):
                    confidence = boxes[b][-1]

                    if confidence > max_confidence:
                        max_confidence = confidence
                        select = torch.tensor(b).to(device)

            Q[i, j, :5] = boxes[select]
            Q[i, j, 5:] = inp[i, j, -opt.C:]
    return Q

def calc_loss_single(inp, target, opt):

    loss = torch.zeros(1)
    for i in range(opt.S):
        for j in range(opt.S):
            # case 1: grid cell HAS object
            if len(target[i, j, :].nonzero()) > 1:
                # localization
                loss = loss + LAMBDA_COORD * (torch.pow(inp[i, j, 0] - target[i, j, 0], 2) + torch.pow(inp[i, j, 1] - target[i, j, 1], 2))

                loss = loss + LAMBDA_COORD * (torch.pow(torch.sqrt(torch.abs(inp[i, j, 2])) - torch.sqrt(torch.abs(target[i, j,2])), 2) \
                        + torch.pow(torch.sqrt(torch.abs(inp[i, j, 3])) - torch.sqrt(torch.abs(target[i, j, 3])), 2))  # org
                # loss = loss + LAMBDA_COORD * (torch.sqrt(torch.abs(P[i, j, 2] - G[i, j, 2])) +
                #                               torch.sqrt(torch.abs(P[i, j, 3] - G[i, j, 3])))  # ZZ

                loss = loss + torch.pow(inp[i, j, 4]-1, 2)   # Ground truth confidence is constant 1
                # classification
                true_cls = target[i, j, -1].type(torch.int64)
                true_cls_vec = torch.zeros(opt.C)
                true_cls_vec[true_cls] = torch.tensor(1)
                pred_cls_vec = inp[i, j, -opt.C:]
                loss = loss + torch.sum(torch.pow(pred_cls_vec - true_cls_vec, 2))

            # case 2: grid cell NO object
            # classification
            else:
                loss = loss + LAMBDA_NOOBJ * torch.pow(inp[i, j, 4] - 0, 2)  # Ground truth confidence is constant 0

    return loss






























def calc_IOU(box_1, box_2, device=torch.device('cpu'), use_float64=False):
    """
    Tensor version of calc_IOU()
    compute IOU between two bounding boxes
    :param box_1: Detection x, y, w, h image coordinates in [0, 1]
    :param box_2: GroundTruth x, y, w, h image coordinates in [0, 1]
    :return:
    """
    '''
    x_min_1 = torch.clamp((box_1[0] - box_1[2] / 2), 0, 1).to(device)
    x_max_1 = torch.clamp((box_1[0] + box_1[2] / 2), 0, 1).to(device)
    y_min_1 = torch.clamp((box_1[1] - box_1[3] / 2), 0, 1).to(device)
    y_max_1 = torch.clamp((box_1[1] + box_1[3] / 2), 0, 1).to(device)
    '''

    x_min_1 = torch.clamp((abs(box_1[0]) - abs(box_1[2]) / 2), 0, 1).to(device)
    x_max_1 = torch.clamp((abs(box_1[0]) + abs(box_1[2]) / 2), 0, 1).to(device)
    y_min_1 = torch.clamp((abs(box_1[1]) - abs(box_1[3]) / 2), 0, 1).to(device)
    y_max_1 = torch.clamp((abs(box_1[1]) + abs(box_1[3]) / 2), 0, 1).to(device)

    x_min_2 = torch.clamp((box_2[0] - box_2[2] / 2), 0, 1).to(device)
    x_max_2 = torch.clamp((box_2[0] + box_2[2] / 2), 0, 1).to(device)
    y_min_2 = torch.clamp((box_2[1] - box_2[3] / 2), 0, 1).to(device)
    y_max_2 = torch.clamp((box_2[1] + box_2[3] / 2), 0, 1).to(device)


    # z = torch.tensor(0, dtype=torch.float).to(device)
    z = torch.tensor(0.).to(device)

    a = torch.min(x_max_1, x_max_2)
    b = torch.max(x_min_1, x_min_2)
    c = torch.min(y_max_1, y_max_2)
    d = torch.max(y_min_1, y_min_2)

    overlap_width = torch.max(a-b, z)
    overlap_height = torch.max(c-d, z)
    overlap_area = overlap_width * overlap_height

    union_area = (x_max_1 - x_min_1) * (y_max_1 - y_min_1) \
                 + (x_max_2 - x_min_2) * (y_max_2 - y_min_2) \
                 - overlap_area
    intersection_over_union = overlap_area / union_area
    return intersection_over_union


