import torch
from models import pretrained
import dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
import time
import sys
import cvtransforms


def reload_model(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cpu')
        pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        print('load {} parameters'.format(len(pretrained_dict)))
        model.load_state_dict(model_dict)
        return model


def extract_feature(args, source_dir, target_dir):

    if source_dir[-1] == '/':
        source_dir = source_dir[:-1]
    if target_dir[-1] == '/':
        target_dir = target_dir[:-1]


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model  = pretrained.Lipreading(mode='temporalConv', nClasses=500)
    model = reload_model(model, args.path)
    model = model.float()
    model.eval()
    model.to(device)


    # vox = dataset.Voceleb2Raw(args.source_dir)
    vox = dataset.LRS2Raw(args.source_dir)

    print('load dataset, len: {}'.format(len(vox)))


    data_loader = DataLoader(dataset=vox,
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers,
                             drop_last=False, collate_fn=dataset.PadCollate(dim=0))

    total_passed = 0


    for batch_idx, (inputs, lens ,soure_path) in enumerate(data_loader):
        if batch_idx == 0:
            since = time.time()
        if len(inputs) == 0 :
            continue
        sys.stdout.flush()
        target_path = soure_path[0].replace(source_dir, target_dir)
        dir = '/'.join(target_path.split('/')[0:-1])
        filename = target_path.split('/')[-1].replace('.mp4', '.npy')
        if os.path.exists(dir+'/'+filename):
            print("Skip file {}, note batch_size is 1".format(soure_path[0]))
            continue
        inputs = cvtransforms.CenterCrop(inputs.numpy(), (args.crop_size, args.crop_size))
        inputs = cvtransforms.ColorNormalize(inputs)
        inputs = torch.from_numpy(inputs)
        inputs = inputs.unsqueeze(1).float()
        with torch.no_grad():
            outputs = model(inputs.to(device))
        try:
            outputs = outputs.cpu()
            outputs = outputs.view(len(lens), max(lens), 512)
        except:
            print(lens)
            print(outputs)


        for i in range(len(soure_path)):
            target_path = soure_path[i].replace(source_dir, target_dir)
            dir = '/'.join(target_path.split('/')[0:-1])
            filename = target_path.split('/')[-1].replace('.mp4', '.npy')
            if not os.path.exists(dir):
                os.makedirs(dir)
            # with lzma.open(dir+'/'+filename, 'w') as f:
            np.save(dir+'/'+filename, outputs[i].numpy()[0:lens[i], ...])

        total_passed += len(lens)
        if batch_idx == 0:
            since = time.time()
        elif batch_idx % args.interval == 0 or (batch_idx == len(data_loader)-1):
            print('Process [{:5.0f}/{:5.0f}] \tCost time: {:5.0f}s\tEstimated time:{:5.0f}s\r'.format(
                total_passed,
                len(vox),
                time.time()-since,
                (time.time() - since) * (len(data_loader) - 1) / batch_idx - (time.time() - since),
            ))
        sys.stdout.flush()
        pass

#
# a = np.load('/Users/chenda/SpeechLab/projects/avse/data/VoCeleb2_feature_mini/Videos/dev/mp4/id00129/yB82Acj5hV0/00104.npy')
#
# import matplotlib.pyplot as plt
#
# plt.imshow(a, interpolation='nearest')
# plt.show()


def main():
    parser = argparse.ArgumentParser(description='feature extract')
    parser.add_argument('--path', default='', help='path to model')
    parser.add_argument('--batch-size', default=36, type=int, help='mini-batch size (default: 36)')
    parser.add_argument('--workers', default=4, help='number of data loading workers (default: 4)', type=int)
    parser.add_argument('--source-dir', default='', help='source data dir')
    parser.add_argument('--target-dir', default='', help='target data dir')
    parser.add_argument('--interval', default=10, help='display interval', type=int)
    parser.add_argument('--crop-size', default=112, help='center crop size', type=int)

    args = parser.parse_args()

    extract_feature(args, args.source_dir, args.target_dir)
    print('Finished')



if __name__ == '__main__':
    main()






