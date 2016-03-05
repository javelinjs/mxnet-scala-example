# MXNet Scala Examples
Examples for Using MXNet Scala Package

Run with `ant` and `scala` installed:

```bash
ant clean run
```

This will download MNIST dataset automatically and train a simple 3-layer MLP on it.

Note that the jar I put in the lib directory is built on OSX. If you are running Linux, please build your own version and replace the one `lib/mxnet_2.10-osx-x86_64-0.1-SNAPSHOT-full.jar`. And please make sure you've set the right `scala.home` in `build.xml`.

We'll provide pre-built binary package on [Maven Repository](http://mvnrepository.com) soon, to make life easier.