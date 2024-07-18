export MODELNAME="Thyroid_Benign&PTC"
export ARCH="resnet"
export IMAGEPATH="./dataset"
export TRAIN="Benign&PTC_train"
export VALID="Benign&PTC_valid"
export TEST="Benign&PTC_test" 
export BATCH=64

python main_train.py \
        --modelname $MODELNAME \
        --architecture $ARCH   \
        --imagepath $IMAGEPATH   \
        --train_data $TRAIN   \
        --valid_data $VALID   \
        --test_data $TEST  \
        --learning_rat 0.0005  \
        --batch_size $BATCH   \
        --num_epochs 200 \
        --Class 2



export MODELNAME="Thyroid_Benign&FTC"
export ARCH="resnet"
export IMAGEPATH="./dataset"
export TRAIN="Benign&FTC_train"
export VALID="Benign&FTC_valid"
export TEST="Benign&FTC_test" 
export BATCH=64

python main_train.py \
        --modelname $MODELNAME \
        --architecture $ARCH   \
        --imagepath $IMAGEPATH   \
        --train_data $TRAIN   \
        --valid_data $VALID   \
        --test_data $TEST  \
        --learning_rat 0.0005  \
        --batch_size $BATCH   \
        --num_epochs 200 \
        --Class 2

export MODELNAME="Thyroid_Benign&MTC"
export ARCH="resnet"
export IMAGEPATH="./dataset"
export TRAIN="Benign&MTC_train"
export VALID="Benign&MTC_valid"
export TEST="Benign&MTC_test" 
export BATCH=64

python main_train.py \
        --modelname $MODELNAME \
        --architecture $ARCH   \
        --imagepath $IMAGEPATH   \
        --train_data $TRAIN   \
        --valid_data $VALID   \
        --test_data $TEST  \
        --learning_rat 0.0005  \
        --batch_size $BATCH   \
        --num_epochs 200 \
        --Class 2
