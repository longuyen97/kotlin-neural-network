package de.longuyen.ui;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;


public class View extends JComponent {
    private int X1, Y1, X2, Y2;
    private final Graphics2D globalGraphics;
    private final BufferedImage image = new BufferedImage(500, 500, BufferedImage.TYPE_INT_RGB);

    @Override
    protected void paintComponent(Graphics g1) {
        g1.drawImage(image, 0, 0, null);
    }

    public View() {
        setBackground(Color.WHITE);
        setDoubleBuffered(false);
        globalGraphics = (Graphics2D) image.getGraphics();
        globalGraphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        clear();
        this.print(globalGraphics);
        addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                X2 = e.getX();
                Y2 = e.getY();
            }
        });
        addMouseMotionListener(new MouseMotionAdapter() {
            @Override
            public void mouseDragged(MouseEvent e) {
                X1 = e.getX();
                Y1 = e.getY();
                globalGraphics.drawLine(X2, Y2, X1, Y1);
                repaint();
                X2 = X1;
                Y2 = Y1;
            }
        });
    }

    public void clear() {
        globalGraphics.setPaint(Color.white);
        globalGraphics.fillRect(0, 0, 500, 500);
        globalGraphics.setPaint(Color.black);
        repaint();
    }

    public void setThickness(int thick) {
        globalGraphics.setStroke(new BasicStroke(thick));
    }

    public BufferedImage getImage() {
        return image;
    }
}
