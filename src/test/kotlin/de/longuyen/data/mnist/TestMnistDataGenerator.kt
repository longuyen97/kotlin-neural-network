package de.longuyen.data.mnist

import org.junit.Test
import java.util.*
import kotlin.test.assertTrue

class TestMnistDataGenerator {
    @Test
    fun `Test method read raw csv to map with training data`(){
        val dataGenerator = MnistDataGenerator()
        val trainingData = dataGenerator.getTrainingData()
        val testingData = dataGenerator.getTestingDataWithLabels()
        assertTrue { Arrays.equals(trainingData.first.shape(), longArrayOf(784, 60000)) }
        assertTrue { Arrays.equals(trainingData.second.shape(), longArrayOf(10, 60000)) }
        assertTrue { Arrays.equals(testingData.first.shape(), longArrayOf(784, 10000)) }
        assertTrue { Arrays.equals(testingData.second.shape(), longArrayOf(10, 10000)) }
    }

    @Test
    fun `Test validation data`(){
        val dataGenerator = MnistDataGenerator()
        dataGenerator.getTestingData()
    }
}