import de.longuyen.trainer.HousePriceModelTrainer
import org.knowm.xchart.BitmapEncoder
import org.knowm.xchart.XYChart
import org.knowm.xchart.XYChartBuilder
import java.io.FileOutputStream
import java.io.ObjectOutputStream


fun main() {
    val trainer = HousePriceModelTrainer(intArrayOf(318, 32, 1), 0.001, 10000)
    val losses = trainer.train()
    val xData = DoubleArray(losses.first.size)
    for (i in 0 until losses.first.size) {
        xData[i] = i.toDouble()
    }
    val yTrain = losses.first.toDoubleArray()
    val yTest = losses.second.toDoubleArray()

    val chart: XYChart =
        XYChartBuilder().width(600).height(500).title("Two-layer-network's performance").xAxisTitle("X").yAxisTitle("Y")
            .build()
    chart.addSeries("Training", xData, yTrain)
    chart.addSeries("Validating", xData, yTest)

    BitmapEncoder.saveBitmap(chart, "target/Sample_Chart", BitmapEncoder.BitmapFormat.PNG)


    FileOutputStream("target/model.ser").use { fos ->
        ObjectOutputStream(fos).use { os ->
            os.writeObject(trainer)
        }
    }
}