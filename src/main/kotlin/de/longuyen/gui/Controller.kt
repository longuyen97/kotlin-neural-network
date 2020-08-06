package de.longuyen.gui

import de.longuyen.neuronalnetwork.NeuronalNetwork
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.factory.Nd4j
import java.awt.BorderLayout
import java.awt.Color
import java.awt.Dimension
import java.awt.GridLayout
import java.awt.image.BufferedImage
import java.io.File
import java.io.ObjectInputStream
import java.util.*
import javax.imageio.ImageIO
import javax.swing.*
import javax.swing.event.ChangeEvent
import javax.swing.event.ChangeListener
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt


class Controller(width: Int, height: Int) {
    private val view = View()
    private val clearButton = JButton("Clear")
    private val processButton = JButton("Process")
    private val thicknessStat = JLabel("40")
    private val thicknessSlider = JSlider(JSlider.HORIZONTAL, 10, 50, 40)
    private val modelsOptions = mutableMapOf<String, NeuronalNetwork>()
    private lateinit var predictive: NeuronalNetwork
    private lateinit var autoencoder: NeuronalNetwork
    private var currentModel: NeuronalNetwork
    private val outputDistribution = mutableListOf<JProgressBar>()
    private val modelComboBox: JComboBox<String>

    init {
        ObjectInputStream(this.javaClass.getResourceAsStream("/models/predictive.ser")).use {
            predictive = it.readObject() as NeuronalNetwork
        }
        ObjectInputStream(this.javaClass.getResourceAsStream("/models/autoencoder.ser")).use {
            autoencoder = it.readObject() as NeuronalNetwork
        }
        currentModel = autoencoder
        modelsOptions["Autoencoder"] = autoencoder
        modelsOptions["Predictive"] = predictive
        modelComboBox = JComboBox(Vector(modelsOptions.keys.toList()))
        modelComboBox.addActionListener {
            val cb = it.source as JComboBox<*>
            val modelName = cb.selectedItem as String
            currentModel = modelsOptions[modelName]!!
        }


        // Main frame
        val mainFrame = JFrame("Neuronal network interaction")
        val mainFrameContainer = mainFrame.contentPane
        mainFrameContainer.layout = BorderLayout()
        mainFrameContainer.add(view, BorderLayout.CENTER)

        // Result panel in the south
        val resultPanel = JPanel(GridLayout(2, 2))
        for (i in 0..9) {
            val progressBar = JProgressBar(0, 100);
            progressBar.value = 0;
            progressBar.isStringPainted = true
            progressBar.string = "$i"
            outputDistribution.add(progressBar)
            resultPanel.add(progressBar)
        }
        mainFrameContainer.add(resultPanel, BorderLayout.SOUTH)

        // Controlling Panel in the north
        val controllingPanel = JPanel()
        thicknessSlider.majorTickSpacing = 25
        thicknessSlider.paintTicks = true
        thicknessSlider.preferredSize = Dimension(40, 40)
        view.setThickness(thicknessSlider.value)
        val thicknessChangeListener = ChangeListener { e: ChangeEvent? ->
            thicknessStat.text = String.format("%s", thicknessSlider.value)
            view.setThickness(thicknessSlider.value)
        }
        thicknessSlider.addChangeListener(thicknessChangeListener)
        processButton.addActionListener {
            if (currentModel == predictive) {
                predictiveModelActivated()
            } else {
                autoencoderModelActivated()
            }
        }
        clearButton.addActionListener {
            view.clear()
        }

        controllingPanel.add(modelComboBox)
        controllingPanel.add(processButton)
        controllingPanel.add(clearButton)
        controllingPanel.add(thicknessSlider)
        controllingPanel.add(thicknessStat)
        mainFrameContainer.add(controllingPanel, BorderLayout.NORTH)

        // Others stuff
        mainFrame.isVisible = true
        mainFrame.setSize(width, height + 11)
        mainFrame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
    }

    private fun autoencoderModelActivated() {
        val image = resize(view.getImage(), 28, 28)
        val nativeImageMatrix = arrayOfNulls<DoubleArray>(28)
        for (y in 0 until image.height) {
            val doubleArray = DoubleArray(28)
            for (x in 0 until image.width) {
                val color = Color(image.getRGB(x, y))
                doubleArray[x] = 255.0 - ((color.red + color.green + color.blue) / 3.0)
            }
            nativeImageMatrix[y] = doubleArray
        }
        val ndarray = ((Nd4j.createFromArray(nativeImageMatrix).reshape(
            intArrayOf(
                784,
                1
            )
        ).castTo(DataType.DOUBLE)).div(255.0))
        val resultNdarray = autoencoder.inference(ndarray).mul(255.0).reshape(28, 28)
        val result = resultNdarray.toIntMatrix()
        var decodedImage = BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB)
        for (y in result.indices) {
            for (x in result[y].indices) {
                val grayColor = 255 - min(255, max(result[y][x], 0))
                decodedImage.setRGB(x, y, Color(grayColor, grayColor, grayColor).rgb)
            }
        }
        decodedImage = resize(decodedImage, 500, 500)
        ImageIO.write(decodedImage, "PNG", File("test.png"))
        view.setImage(decodedImage)
    }

    private fun predictiveModelActivated() {
        val image = resize(view.getImage(), 28, 28)
        val nativeImageMatrix = arrayOfNulls<DoubleArray>(28)
        for (y in 0 until image.height) {
            val doubleArray = DoubleArray(28)
            for (x in 0 until image.width) {
                val color = Color(image.getRGB(x, y))
                doubleArray[x] = 255.0 - ((color.red + color.green + color.blue) / 3.0)
            }
            nativeImageMatrix[y] = doubleArray
        }
        val ndarray = ((Nd4j.createFromArray(nativeImageMatrix).reshape(
            intArrayOf(
                784,
                1
            )
        ).castTo(DataType.DOUBLE)).div(255.0))
        val result = predictive.inference(ndarray).mul(100).toDoubleVector()
        for (i in 0..9) {
            outputDistribution[i].value = result[i].roundToInt()
        }
    }
}
