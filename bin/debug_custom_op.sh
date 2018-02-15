#!/bin/bash
CURR_DIR=$(cd $(dirname $0); pwd)
PROJ_DIR=$(cd $(dirname $0)/../; pwd)
DATA_DIR=$PROJ_DIR/data

# get mnist dataset
$CURR_DIR/get_mnist_data.sh $DATA_DIR

CLASSPATH=$CLASSPATH:$PROJ_DIR/target/*:$PROJ_DIR/target/classes/lib/*
java -Xmx16g -cp $CLASSPATH \
  -Dlog4j.configuration=file://$PROJ_DIR/conf/log4j.properties \
  me.yzhi.mxnet.example.classification.CustomOpDebug \
  --data-path=$DATA_DIR/
