package de.longuyen.trainer

import de.longuyen.core.Model
import de.longuyen.core.NeuronalNetwork
import de.longuyen.data.HousePriceDataEncoder
import de.longuyen.data.HousePriceDataGenerator

class HousePriceModelTrainer {
    fun train() : Model{
        val dataGenerator = HousePriceDataGenerator()
        val trainingData = dataGenerator.getTrainingData()
        val testingData = dataGenerator.getTestingData()

        val dataEncoder = HousePriceDataEncoder(trainingData)


        val model = NeuronalNetwork()
        return model
    }
}

fun main(){

}