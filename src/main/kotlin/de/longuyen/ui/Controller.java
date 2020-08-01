package de.longuyen.ui;

import javax.swing.*;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.ActionListener;

public class Controller {
    private final View view = new View();
    private final JButton clearButton = new JButton("Clear");
    private final JButton processButton = new JButton("Process");
    private final JLabel thicknessStat = new JLabel("10");
    private final JLabel resultLabel = new JLabel("", SwingConstants.CENTER);
    private final JSlider thicknessSlider = new JSlider(JSlider.HORIZONTAL, 10, 50, 10);

    public Controller(int width, int height) {
        JFrame mainFrame = new JFrame("Neuronal network interaction");
        Container mainFrameContainer = mainFrame.getContentPane();
        mainFrameContainer.setLayout(new BorderLayout());
        mainFrameContainer.add(view, BorderLayout.CENTER);
        mainFrameContainer.add(resultLabel, BorderLayout.SOUTH);

        JPanel panel = new JPanel();
        thicknessSlider.setMajorTickSpacing(25);
        thicknessSlider.setPaintTicks(true);
        thicknessSlider.setPreferredSize(new Dimension(40, 40));
        ChangeListener thicknessChangeListener = e -> {
            thicknessStat.setText(String.format("%s", thicknessSlider.getValue()));
            view.setThickness(thicknessSlider.getValue());
        };
        thicknessSlider.addChangeListener(thicknessChangeListener);

        ActionListener mouseClickedActionListener = event -> {
            if (event.getSource() == clearButton) {
                view.clear();
            } else if (event.getSource() == processButton) {
                // TODO
                System.out.println("Processing image");
            }
        };
        processButton.addActionListener(mouseClickedActionListener);
        clearButton.addActionListener(mouseClickedActionListener);

        panel.add(processButton);
        panel.add(clearButton);
        panel.add(thicknessSlider);
        panel.add(thicknessStat);

        mainFrameContainer.add(panel, BorderLayout.NORTH);
        mainFrame.setVisible(true);
        mainFrame.setSize(width, height + 11);
        mainFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }
}
