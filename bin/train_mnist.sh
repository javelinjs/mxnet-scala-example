#!/bin/bash
CURR_DIR=$(cd $(dirname $0); pwd)
PROJ_DIR=$(cd $(dirname $0)/../; pwd)
DATA_DIR=$PROJ_DIR/data

# get mnist dataset
$CURR_DIR/get_mnist_data.sh $DATA_DIR

CLASSPATH=$CLASSPATH:$PROJ_DIR/target/*:$PROJ_DIR/target/classes/lib/*
java -Xmx2G -cp $CLASSPATH \
  -Dlog4j.configuration=file://$PROJ_DIR/conf/log4j.properties \
  me.yzhi.mxnet.example.classification.TrainMnist \
  --data-dir=$DATA_DIR/ \
  --num-epochs=5 \
  --network=mlp \
  --cpus=0,1,2,3 \
  --optimizer=SGD
