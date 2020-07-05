from torchvision.models import vgg

from rotational_update import RotationalLinear


if __name__ == '__main__':
    myvgg16 = vgg.vgg16()

    myvgg16.classifier[0] = RotationalLinear(myvgg16.classifier[0])
    myvgg16.classifier[3] = RotationalLinear(myvgg16.classifier[3])

    # RotationalLinear has rotate() function
    # Call it after every minibatch spending
    assert("rotate" in dir(myvgg16.classifier[0]))

    print(myvgg16)
