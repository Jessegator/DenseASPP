import matplotlib.pyplot as plt
import torch
from utils import Adder, get_miou, get_biou

def evaluation(args, model, test_loader):

    state = torch.load(args.load)
    model.load_state_dict(state['model'])

    mIoU = Adder()
    bIoU = Adder()

    model.eval()
    with torch.no_grad():
        for idx, (name_id, img, inputs, masks) in enumerate(test_loader):
            inputs, masks = inputs.to(args.device), masks.to(args.device)

            outputs = model(inputs)
            mIoU(get_miou(outputs, masks))
            bIoU(get_biou(outputs, masks))

            outputs = outputs.cpu().squeeze()
            outputs = torch.argmax(outputs,dim=0)
            fig = plt.figure(figsize=(12, 8))
            plt.subplot(1, 3, 1), plt.imshow(img.squeeze()), plt.axis('off')
            plt.title('Original')

            plt.subplot(1, 3, 2), plt.imshow(masks.cpu().squeeze(), 'gray'), plt.axis('off')
            plt.title('GroundTruth')

            plt.subplot(1, 3, 3), plt.imshow(outputs, 'gray'), plt.axis('off')
            plt.title('Output')

            plt.show()

    print('mIOU:%.3f' % mIoU.average())
    print('bIOU:%.3f' % bIoU.average())
    print('Done!')

    return