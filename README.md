# MXNet Scala Examples
Examples for Using MXNet Scala Package

Run with `maven` and `scala` installed:

```bash
mvn clean package
bin/train_mnist.sh
```

This will download MNIST dataset automatically and train a simple 3-layer MLP on it.

Note that Scala 2.10.x is required for current binary. If you want to use 2.11, please change the scala version defined in mxnet/scala-package's pom settings.
