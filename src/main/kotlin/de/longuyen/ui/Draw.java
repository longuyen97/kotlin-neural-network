package de.longuyen.ui;

import javax.swing.*;
import javax.swing.UIManager.LookAndFeelInfo;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class Draw {
    Canvas canvas;
    JButton clearButton;
    JButton processButton;
    private final JLabel thicknessStat;
    private final JSlider thicknessSlider;

    ChangeListener thicknessChangeListener = new ChangeListener() {
        public void stateChanged(ChangeEvent e) {
            thicknessStat.setText(String.format("%s", thicknessSlider.getValue()));
            canvas.setThickness(thicknessSlider.getValue());
        }
    };
    ActionListener mouseClickedActionListener = new ActionListener() {
        public void actionPerformed(ActionEvent event) {
            if (event.getSource() == clearButton) {
                canvas.clear();
            } else if (event.getSource() == processButton) {
                System.out.println("Processing image");
            }
        }
    };

    public Draw(int width,int height) {
        for (LookAndFeelInfo info : UIManager.getInstalledLookAndFeels()) {
            if ("Nimbus".equals(info.getName())) {
                try {
                    UIManager.setLookAndFeel(info.getClassName());
                } catch (ClassNotFoundException | InstantiationException
                        | IllegalAccessException
                        | UnsupportedLookAndFeelException e) {
                    e.printStackTrace();
                }
                break;
            }
        }
        JFrame frame = new JFrame("Neuronal network interaction");
        Container container = frame.getContentPane();
        container.setLayout(new BorderLayout());
        canvas = new Canvas();

        container.add(canvas, BorderLayout.CENTER);

        JPanel panel = new JPanel();
        thicknessSlider = new JSlider(JSlider.HORIZONTAL, 10, 50, 10);
        thicknessSlider.setMajorTickSpacing(25);
        thicknessSlider.setPaintTicks(true);
        thicknessSlider.setPreferredSize(new Dimension(40, 40));
        thicknessSlider.addChangeListener(thicknessChangeListener);
        processButton = new JButton();
        processButton.addActionListener(mouseClickedActionListener);
        clearButton = new JButton("Clear");
        clearButton.addActionListener(mouseClickedActionListener);

        thicknessStat = new JLabel("10");

        panel.add(clearButton);
        panel.add(thicknessSlider);
        panel.add(thicknessStat);


        container.add(panel, BorderLayout.NORTH);
        frame.setVisible(true);
        frame.setSize(width+79,height+11);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }
}
