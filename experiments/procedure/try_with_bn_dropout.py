""" batch normalizationとdropout"""

# Official packages
import sys
import os
from os import getcwd, makedirs
from os.path import join, isdir
import datetime
import gc  # garbage collection
import argparse
import pickle

# PyPI packages
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

sys.path.append("..")  # If you get Rotational-Update from PyPI, you don't need this line
from rotational_update import Rotatable

# Local modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.batchnorm_dropout_with_vgg16 import BatchnormDropoutWithVGG16 as vgg16
from procedure import preprocess


# 引数を受け取る
parser = argparse.ArgumentParser(description='Training using Rotational-update')
parser.add_argument('--convolution',
                    choices=['vgg_with_maxpool', 'vgg_without_maxpool'],
                    help='convolution type', required=True)
parser.add_argument('-p', '--pooling', choices=['max', 'average'],
                    help='pooling method', required=True)
parser.add_argument('--rotational', choices=['false', 'true'],
                    help='use of rotational update in full connected layers', required=True)
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--deepness', default=2, type=int, metavar='N',
                    help='number of fc layer deepness')
parser.add_argument('--seed', default=0, type=int, metavar='N',
                    help='random seed', required=True)

parser.add_argument('--cnn_bn_flag', action='store_true', help='flag for cnn bn')
parser.add_argument('--fc_bn_flag', action='store_true', help='flag for fc bn')
parser.add_argument('--fc_do_flag', action='store_true', help='flag for fc dropout')

args = parser.parse_args()

now = datetime.datetime.now()
MODEL_NAME = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_batchnom_dropout".format(
    now.strftime("%Y-%m-%d_%H-%M-%S"),
    args.convolution,
    args.pooling,
    args.rotational,
    args.epochs,
    args.deepness,
    args.cnn_bn_flag,
    args.fc_bn_flag,
    args.fc_do_flag,
    args.seed
)
writer = SummaryWriter('runs/' + MODEL_NAME)

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

# DataFrame for CSV
train_record_df = pd.DataFrame(columns=['epoch', 'train_loss', 'train_accuracy'])
validate_record_df = pd.DataFrame(columns=['epoch', 'validate_loss', 'validate_accuracy'])


def conduct(model: nn.Module, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader):
    def rotate_all():
        global args, model

        for layer in model.fc:
            if isinstance(layer, Rotatable):
                layer.rotate()

    loss_layer = nn.CrossEntropyLoss(reduction='none')
    loss_layer_reduce = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Prepare for saving
    global MODEL_NAME
    save_dir = join(getcwd(), "log/" + MODEL_NAME)
    mkdirs(save_dir)

    # validation of pre_training
    validate(model, test_loader, -1)

    # training
    for epoch in range(args.epochs):
        model.train()
        outputs_list = []
        answers_list = []
        total_loss = 0.0
        total_correct = 0
        item_counter = 0

        for i, mini_batch in enumerate(train_loader):
            input_data, label_data = mini_batch

            in_tensor = input_data.to(DEVICE)
            label_tensor = label_data.to(DEVICE)

            optimizer.zero_grad()  # Optimizer を0で初期化

            # forward - backward - optimize
            outputs = model(in_tensor)
            loss_vector = loss_layer(outputs, label_tensor)  # for evaluation
            reduced_loss = loss_layer_reduce(
                outputs, label_tensor)  # for backward
            _, predicted = torch.max(outputs.data, 1)

            reduced_loss.backward()
            optimizer.step()

            rotate_all()

            total_loss += loss_vector.data.sum().item()
            total_correct += (predicted.to('cpu') == label_data).sum().item()
            item_counter += len(outputs)
            outputs_list.append(outputs.to('cpu'))
            answers_list.append(label_tensor.to('cpu'))
            # debug
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss:.4f}\t'
                  'Accuracy: {accuracy:.3f}'.format(
                   epoch, i, len(train_loader),
                   loss=loss_vector.data.sum().item(),
                   accuracy=(predicted.to('cpu') == label_data).sum().item()/len(outputs)))

        """ 記録"""
        writer.add_scalar('train loss',
                          total_loss/item_counter,
                          epoch)
        writer.add_scalar('train Accuracy',
                          total_correct/item_counter,
                          epoch)
        d = {
            "outputs": outputs_list,
            "answers": answers_list
        }
        n = "train_{}".format(epoch)
        save(data=d, name=n, type="progress")

        # Save to CSV
        save_csv(
            str(epoch),
            "train",
            total_loss/item_counter,
            total_correct/item_counter
        )

        """ メモリ解放"""
        del outputs_list
        del answers_list
        gc.collect()

        """ 評価"""
        validate(model, test_loader, epoch)

    print('Finished Training')


def validate(model: nn.Module, test_loader: torch.utils.data.DataLoader, epoch: int):
    model.eval()
    with torch.no_grad():
        loss_layer = nn.CrossEntropyLoss(reduction='none')

        outputs_list = []
        answers_list = []
        total_correct = 0
        total_loss = 0.0
        item_counter = 0

        for i, mini_batch in enumerate(test_loader):
            input_data, label_data = mini_batch
            mini_batch_size = list(input_data.size())[0]

            in_tensor = input_data.to(DEVICE)
            label_tensor = label_data.to(DEVICE)

            outputs = model(in_tensor)
            loss_vector = loss_layer(outputs, label_tensor)
            _, predicted = torch.max(outputs.data, 1)

            assert list(loss_vector.size()) == [mini_batch_size]

            total_correct += (predicted.to('cpu') == label_data).sum().item()
            total_loss += loss_vector.sum().item()
            item_counter += len(outputs)
            outputs_list.append(outputs.to('cpu'))
            answers_list.append(label_tensor.to('cpu'))
            # debug
            print('progress: [{0}/{1}]\t'
                  'Loss: {loss:.3f}\t'
                  'Accuracy: {accuracy:.3f}'.format(
                      i, len(test_loader),
                      loss=loss_vector.sum().item(),
                      accuracy=(predicted.to('cpu') == label_data).sum().item()/len(outputs)))

        """ 記録"""
        writer.add_scalar('validate loss',
                          total_loss/item_counter,
                          epoch)
        writer.add_scalar('validate Accuracy',
                          total_correct/item_counter,
                          epoch)
        d = {
            "outputs": outputs_list,
            "answers": answers_list
        }
        n = "validate_{}".format(epoch)
        save(data=d, name=n, type="progress")

        # Save to CSV
        save_csv(
            "学習前" if epoch == -1 else str(epoch),
            "validate",
            total_loss/item_counter,
            total_correct/item_counter
        )

        """ メモリ解放"""
        del outputs_list
        del answers_list
        gc.collect()


def mkdirs(path):
    """ ディレクトリが無ければ作る """
    if not isdir(path):
        makedirs(path)


def save(data, name, type):
    """ 保存

    data: 保存するデータ
    name: ファイル名
    type: データのタイプ
    """
    global MODEL_NAME

    save_dir = join(getcwd(), "log/" + MODEL_NAME)
    mkdirs(save_dir)

    if type == "model":
        """ モデルを保存

        Memo: ロードする方法
        model = TheModelClass(*args, **kwargs)
        model.load_state_dict(torch.load(PATH))
        model.eval()
        """
        torch.save(data.state_dict(), join(save_dir, name+'.model'))
    elif type == "progress":
        """ 予測の途中経過

        Memo: ロードする方法
        data = None
        with open(PATH, 'rb') as f:
            data = pickle.load(f)
        """
        with open(join(save_dir, name+'.dump'), 'wb') as f:
            pickle.dump(data, f)

def save_csv(epoch: str, mode: str, loss: float, accuracy: float):
    # mode
    if mode == "train":
        global train_record_df
        train_record_df.loc[len(train_record_df)] = [epoch, loss, accuracy]
        record_df = train_record_df
        filename = "train.csv"
    elif mode == "validate":
        global validate_record_df
        validate_record_df.loc[len(validate_record_df)] = [epoch, loss, accuracy]
        record_df = validate_record_df
        filename = "validate.csv"
    else:
        raise ValueError("mode must be 'train' or 'validate'")
    
    # Save
    record_df.to_csv(join(getcwd(), "log/" + MODEL_NAME + "/" + filename), index=False)


if __name__ == '__main__':
    torch.manual_seed(args.seed)
    model = vgg16(
        num_classes=10, model_type=args.convolution, pooling=args.pooling,
        rotational=args.rotational, deepness=args.deepness, poolingshape=7,
        cnn_bn_flag=args.cnn_bn_flag, fc_bn_flag=args.fc_bn_flag,
        fc_do_flag=args.fc_do_flag
    )

    model.to(DEVICE)

    conduct(model, *(preprocess.cifar_10_for_vgg_loaders()))

    """ 学習後のモデルをdumpする"""
    save(data=model, name=MODEL_NAME, type="model")
