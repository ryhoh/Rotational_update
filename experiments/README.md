# Rotational-update experiment
Rotational-update experiment with batchnormalization and dropout.

## Usage

```sh
$ python3 procedure/try_with_bn_dropout.py --help
usage: try_with_bn_dropout.py [-h] --convolution
                              {vgg_with_maxpool,vgg_without_maxpool} -p
                              {max,average} --rotational {false,true}
                              [--epochs N] [--deepness N] --seed N
                              [--cnn_bn_flag] [--fc_bn_flag] [--fc_do_flag]

Training using Rotational-update

optional arguments:
  -h, --help            show this help message and exit
  --convolution {vgg_with_maxpool,vgg_without_maxpool}
                        convolution type
  -p {max,average}, --pooling {max,average}
                        pooling method
  --rotational {false,true}
                        use of rotational update in full connected layers
  --epochs N            number of total epochs to run
  --deepness N          number of fc layer deepness
  --seed N              random seed
  --cnn_bn_flag         flag for cnn bn
  --fc_bn_flag          flag for fc bn
  --fc_do_flag          flag for fc dropout
  
# sample
$ python3 procedure/try_with_bn_dropout.py --convolution vgg_with_maxpool -p max --rotational true --epochs 50 --seed 0
```
