package de.longuyen.neuronalnetwork.losses

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms

class MSE : LossFunction {
    override fun forward(yTrue: INDArray, yPrediction: INDArray): INDArray {
        val diff = yPrediction.sub(yTrue)
        val squared = Transforms.pow(diff, 2)
        return squared.sum(true, 0).mean()
    }

    override fun backward(yTrue: INDArray, yPrediction: INDArray): INDArray {
        TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
    }

}