package de.longuyen.ui;

import javax.swing.*;
import javax.swing.event.MouseInputAdapter;
import java.awt.*;
import java.awt.event.*;


public class Canvas extends JComponent {
    private int X1, Y1, X2, Y2;
    private Graphics2D g;
    private Image img;
    private Rectangle shape;
    private MouseMotionListener motion;
    private MouseListener listener;

    protected void paintComponent(Graphics g1) {
        if (img == null) {
            img = createImage(getSize().width, getSize().height);
            g = (Graphics2D) img.getGraphics();
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            clear();
        }
        g1.drawImage(img, 0, 0, null);
        if (shape != null) {
            Graphics2D g2d = g;
            g2d.draw(shape);
        }
    }

    public Canvas() {
        setBackground(Color.WHITE);
        defaultListener();
    }

    public void defaultListener() {
        setDoubleBuffered(false);
        listener = new MouseAdapter() {
            public void mousePressed(MouseEvent e) {
                X2 = e.getX();
                Y2 = e.getY();
            }
        };

        motion = new MouseMotionAdapter() {
            public void mouseDragged(MouseEvent e) {
                X1 = e.getX();
                Y1 = e.getY();

                if (g != null) {
                    g.drawLine(X2, Y2, X1, Y1);
                    repaint();
                    X2 = X1;
                    Y2 = Y1;
                }
            }
        };
        addMouseListener(listener);
        addMouseMotionListener(motion);
    }

    public void addRectangle(Rectangle rectangle, Color color) {
        Graphics2D g2d = (Graphics2D) img.getGraphics();
        g2d.setColor(color);
        g2d.draw(rectangle);
        repaint();
    }

    public void clear() {
        g.setPaint(Color.white);
        g.fillRect(0, 0, getSize().width, getSize().height);
        g.setPaint(Color.black);
        repaint();
    }

    public void rect() {
        removeMouseListener(listener);
        removeMouseMotionListener(motion);
        MyMouseListener ml = new MyMouseListener();
        addMouseListener(ml);
        addMouseMotionListener(ml);
    }


    public void setThickness(int thick) {
        g.setStroke(new BasicStroke(thick));
    }

    class MyMouseListener extends MouseInputAdapter {
        private Point startPoint;

        public void mousePressed(MouseEvent e) {
            startPoint = e.getPoint();
            shape = new Rectangle();
        }

        public void mouseDragged(MouseEvent e) {
            int x = Math.min(startPoint.x, e.getX());
            int y = Math.min(startPoint.y, e.getY());
            int width = Math.abs(startPoint.x - e.getX());
            int height = Math.abs(startPoint.y - e.getY());

            shape.setBounds(x, y, width, height);
            repaint();
        }

        public void mouseReleased(MouseEvent e) {
            if (shape.width != 0 || shape.height != 0) {
                addRectangle(shape, e.getComponent().getForeground());
            }
            shape = null;
        }
    }
}
