package de.longuyen.gui

import de.longuyen.neuronalnetwork.NeuronalNetwork
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.factory.Nd4j
import java.awt.*
import java.awt.event.ActionEvent
import java.awt.event.ActionListener
import java.io.ObjectInputStream
import javax.swing.*
import javax.swing.event.ChangeEvent
import javax.swing.event.ChangeListener
import kotlin.math.roundToInt


class Controller(width: Int, height: Int) {
    private val view = View()
    private val clearButton = JButton("Clear")
    private val processButton = JButton("Process")
    private val thicknessStat = JLabel("40")
    private val thicknessSlider = JSlider(JSlider.HORIZONTAL, 10, 50, 40)
    private lateinit var predictive: NeuronalNetwork

    init {
        ObjectInputStream(this.javaClass.getResourceAsStream("/models/predictive.ser")).use {
            predictive = it.readObject() as NeuronalNetwork
        }

        // Main frame
        val mainFrame = JFrame("Neuronal network interaction")
        val mainFrameContainer = mainFrame.contentPane
        mainFrameContainer.layout = BorderLayout()
        mainFrameContainer.add(view, BorderLayout.CENTER)

        // Result panel in the south
        val resultPanel = JPanel(GridLayout(2, 2))
        val outputDistribution = mutableListOf<JProgressBar>()
        for(i in 0..9){
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
        val mouseClickedActionListener = ActionListener { event: ActionEvent ->
            if (event.source === clearButton) {
                view.clear()
            } else if (event.source === processButton) {
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
                val ndarray = ((Nd4j.createFromArray(nativeImageMatrix).reshape(intArrayOf(784, 1)).castTo(DataType.DOUBLE)).div(255.0))
                val result = predictive.inference(ndarray).mul(100).toDoubleVector()
                for(i in 0..9){
                    outputDistribution[i].value = result[i].roundToInt()
                }
            }
        }

        processButton.addActionListener(mouseClickedActionListener)
        clearButton.addActionListener(mouseClickedActionListener)

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
}
