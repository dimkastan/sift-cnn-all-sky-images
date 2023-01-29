import sys
import os
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if not path in sys.path:
    sys.path.insert(1, path)

from deepsky.models.classification import return_resnet_18

def test_return_resnet_18():
    model = return_resnet_18('cpu', NUM_CLASSES=7, model_file='weights/rgb-after-simclr/checkpoint_train_eval_other0001_99.83552551269531.pth.tar', feature_only=False)
    assert model.fc.out_features == 7