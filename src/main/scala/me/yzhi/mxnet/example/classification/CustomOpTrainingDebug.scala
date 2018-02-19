package me.yzhi.mxnet.example.classification

import ml.dmlc.mxnet.Callback.Speedometer
import ml.dmlc.mxnet.DType.DType
import ml.dmlc.mxnet.optimizer.{RMSProp, SGD}
import ml.dmlc.mxnet.{Accuracy, Context, CustomOp, CustomOpProp, NDArray, Operator, Shape, Symbol, Xavier}
import org.kohsuke.args4j.{CmdLineParser, Option}
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._

object CustomOpTrainingDebug {
  private val logger = LoggerFactory.getLogger(classOf[CustomOpTrainingDebug])

  var fwInData: Array[NDArray] = null
  var fwOutData: Array[NDArray] = null
  var fwAux: Array[NDArray] = null
  var bwOutGrad: Array[NDArray] = null
  var bwInData: Array[NDArray] = null
  var bwOutData: Array[NDArray] = null
  var bwInGrad: Array[NDArray] = null
  var bwAux: Array[NDArray] = null

  def disposeAll(): Unit = {
    if (fwInData != null) {
      fwInData.foreach(_.disposeDeps())
      fwInData.foreach(_.dispose())
      fwInData = null
    }
    if (fwOutData != null) {
      fwOutData.foreach(_.disposeDeps())
      fwOutData.foreach(_.dispose())
      fwOutData = null
    }
    if (fwAux != null) {
      fwAux.foreach(_.disposeDeps())
      fwAux.foreach(_.dispose())
      fwAux = null
    }
    if (bwOutGrad != null) {
      bwOutGrad.foreach(_.disposeDeps())
      bwOutGrad.foreach(_.dispose())
      bwOutGrad = null
    }
    if (bwInData != null) {
      bwInData.foreach(_.disposeDeps())
      bwInData.foreach(_.dispose())
      bwInData = null
    }
    if (bwOutData != null) {
      bwOutData.foreach(_.disposeDeps())
      bwOutData.foreach(_.dispose())
      bwOutData = null
    }
    if (bwInGrad != null) {
      bwInGrad.foreach(_.disposeDeps())
      bwInGrad.foreach(_.dispose())
      bwInGrad = null
    }
    if (bwAux != null) {
      bwAux.foreach(_.disposeDeps())
      bwAux.foreach(_.dispose())
      bwAux = null
    }
  }

  class Softmax(_param: Map[String, String]) extends CustomOp {

    override def forward(sTrain: Boolean, req: Array[String],
                         inData: Array[NDArray], outData: Array[NDArray], aux: Array[NDArray]): Unit = {
      val xShape = inData(0).shape
      val x = inData(0).toArray.grouped(xShape(1)).toArray
      val yArr = x.map { it =>
        val max = it.max
        val tmp = it.map(e => Math.exp(e.toDouble - max).toFloat)
        val sum = tmp.sum
        tmp.map(_ / sum)
      }.flatten
      val y = NDArray.empty(xShape, outData(0).context)
      y.set(yArr)
      this.assign(outData(0), req(0), y)
      y.dispose()

      fwInData = inData
      fwOutData = outData
      fwAux = aux
    }

    override def backward(req: Array[String], outGrad: Array[NDArray],
                          inData: Array[NDArray], outData: Array[NDArray],
                          inGrad: Array[NDArray], aux: Array[NDArray]): Unit = {
      val l = inData(1).toArray.map(_.toInt)
      val oShape = outData(0).shape
      val yArr = outData(0).toArray.grouped(oShape(1)).toArray
      l.indices.foreach { i =>
        yArr(i)(l(i)) -= 1.0f
      }
      val y = NDArray.empty(oShape, inGrad(0).context)
      y.set(yArr.flatten)
      this.assign(inGrad(0), req(0), y)
      y.dispose()

      bwOutGrad = outGrad
      bwInData = inData
      bwOutData = outData
      bwInGrad = inGrad
      bwAux = aux
    }
  }

  class SoftmaxProp(needTopGrad: Boolean = false)
    extends CustomOpProp(needTopGrad) {

    override def listArguments(): Array[String] = Array("data", "label")

    override def listOutputs(): Array[String] = Array("output")

    override def inferShape(inShape: Array[Shape]):
    (Array[Shape], Array[Shape], Array[Shape]) = {
      val dataShape = inShape(0)
      val labelShape = Shape(dataShape(0))
      val outputShape = dataShape
      (Array(dataShape, labelShape), Array(outputShape), null)
    }

    override def inferType(inType: Array[DType]):
    (Array[DType], Array[DType], Array[DType]) = {
      (inType, inType.take(1), null)
    }

    override def createOperator(ctx: String, inShapes: Array[Array[Int]],
                                inDtypes: Array[Int]): CustomOp = new Softmax(this.kwargs)
  }

  Operator.register("softmax", new SoftmaxProp)

  def main(args: Array[String]): Unit = {
    val leop = new CustomOpTrainingDebug
    val parser: CmdLineParser = new CmdLineParser(leop)
    try {
      parser.parseArgument(args.toList.asJava)
      assert(leop.dataPath != null)

      val ctx = Context.cpu()

      val dataName = Array("data")
      val labelName = Array("softmax_label")

      val data = Symbol.Variable("data")
      val label = Symbol.Variable("label")
      val fc1 = Symbol.FullyConnected("fc1")()(Map("data" -> data, "num_hidden" -> 128))
      val act1 = Symbol.Activation("relu1")()(Map("data" -> fc1, "act_type" -> "relu"))
      val fc2 = Symbol.FullyConnected("fc2")()(Map("data" -> act1, "num_hidden" -> 64))
      val act2 = Symbol.Activation("relu2")()(Map("data" -> fc2, "act_type" -> "relu"))
      val fc3 = Symbol.FullyConnected("fc3")()(Map("data" -> act2, "num_hidden" -> 10))
      val mlp = Symbol.Custom("softmax")()(Map("data" -> fc3,
        "label" -> label, "op_type" -> "softmax"))
//      val mlp = Symbol.SoftmaxOutput(name = "softmax")()(Map("data" -> fc3, "label" -> label))

      val (trainIter, testIter) =
        Data.mnistIterator(leop.dataPath, batchSize = 100, inputShape = Shape(784))

      val datasAndLabels = trainIter.provideData ++ trainIter.provideLabel
      val (argShapes, outputShapes, auxShapes) = mlp.inferShape(datasAndLabels)

      val initializer = new Xavier(factorType = "in", magnitude = 2.34f)
      val argNames = mlp.listArguments()
      val argDict = argNames.zip(argShapes.map(s => NDArray.empty(s, ctx))).toMap

      val gradDict = argNames.zip(argShapes).filter { case (name, shape) =>
        !datasAndLabels.contains(name)
      }.map(x => x._1 -> NDArray.empty(x._2, ctx) ).toMap

      argDict.foreach { case (name, ndArray) =>
        if (!datasAndLabels.contains(name)) {
          initializer.initWeight(name, ndArray)
        }
      }

      val executor = mlp.bind(ctx, argDict, gradDict)
      val lr = 0.001f
      val opt = new SGD(learningRate = lr, wd = 0.00001f)
//      val opt = new RMSProp(learningRate = lr, wd = 0.00001f)
      val paramsGrads = gradDict.toList.zipWithIndex.map { case ((name, grad), idx) =>
        (idx, name, grad, opt.createState(idx, argDict(name)))
      }

      val evalMetric = new Accuracy
      val batchEndCallback = new Speedometer(100, 100)
      val numEpoch = 10000000

      for (epoch <- 0 until numEpoch) {
        val tic = System.currentTimeMillis
        evalMetric.reset()
        var nBatch = 0
        var epochDone = false

        trainIter.reset()
        while (!epochDone) {
          var doReset = true
          while (doReset && trainIter.hasNext) {
            val dataBatch = trainIter.next()
            argDict("data").set(dataBatch.data(0))
            argDict("label").set(dataBatch.label(0))
            executor.forward(isTrain = true)
            executor.backward()
            paramsGrads.foreach { case (idx, name, grad, optimState) =>
              opt.update(idx, argDict(name), grad, optimState)
            }
            evalMetric.update(dataBatch.label, executor.outputs)
            nBatch += 1
            batchEndCallback.invoke(epoch, nBatch, evalMetric)
            // TODO
            disposeAll()
            dataBatch.dispose()
          }
          if (doReset) {
            trainIter.reset()
          }
          epochDone = true
        }
        val (name, value) = evalMetric.get
        name.zip(value).foreach { case (n, v) =>
          logger.info(s"Epoch[$epoch] Train-accuracy=$v")
        }
        val toc = System.currentTimeMillis
        logger.info(s"Epoch[$epoch] Time cost=${toc - tic}")

        evalMetric.reset()
        testIter.reset()
        while (testIter.hasNext) {
          val evalBatch = testIter.next()
          argDict("data").set(evalBatch.data(0))
          argDict("label").set(evalBatch.label(0))
          executor.forward(isTrain = true)
          evalMetric.update(evalBatch.label, executor.outputs)
          evalBatch.dispose()
          disposeAll()
        }
        val (names, values) = evalMetric.get
        names.zip(values).foreach { case (n, v) =>
          logger.info(s"Epoch[$epoch] Validation-accuracy=$v")
        }
      }
      executor.dispose()
    } catch {
      case ex: Exception => {
        logger.error(ex.getMessage, ex)
        parser.printUsage(System.err)
        sys.exit(1)
      }
    }
  }
}

class CustomOpTrainingDebug {
  @Option(name = "--data-path", usage = "the mnist data path")
  private val dataPath: String = null
}
