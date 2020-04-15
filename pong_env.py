import turtle
import keyboard  # using module keyboard

win = turtle.Screen()    # Create a screen
win.title('Paddle')      # Set the title to paddle
win.bgcolor('black')     # Set the color to black
win.tracer(0)
win.setup(width=600, height=600)   # Set the width and height to 600

# Paddle
paddle = turtle.Turtle()    # Create a turtle object
paddle.shape('square')      # Select a square shape
paddle.speed(0)
paddle.shapesize(stretch_wid=1, stretch_len=5)   # Streach the length of square by 5
paddle.penup()
paddle.color('white')       # Set the color to white
paddle.goto(0, -275)        # Place the shape on bottom of the screen

# Ball
ball = turtle.Turtle()      # Create a turtle object
ball.speed(0)
ball.shape('circle')        # Select a circle shape
ball.color('red')           # Set the color to red
ball.penup()
ball.goto(0, 100)           # Place the shape in middle

def paddle_right():
    x = paddle.xcor()        # Get the x position of paddle
    if x < 225:
        paddle.setx(x+20)    # increment the x position by 20

def paddle_left():
    x = paddle.xcor()        # Get the x position of paddle
    if x > -225:
        paddle.setx(x-20)    # decrement the x position by 20

# Keyboard Control
win.listen()
win.onkeyrelease(paddle_right, 'Right')   # call paddle_right on right arrow key
win.onkeyrelease(paddle_left, 'Left')     # call paddle_left on right arrow key

ball.dx = 3   # ball's x-axis velocity
ball.dy = -3  # ball's y-axis velocity

while True:   # same loop

     win.update()
     ball.setx(ball.xcor() + ball.dx)  # update the ball's x-location using velocity
     ball.sety(ball.ycor() + ball.dy)  # update the ball's y-location using velocity
     # Ball-Walls collision
     if ball.xcor() > 290:    # If ball touch the right wall
        ball.setx(290)
        ball.dx *= -1        # Reverse the x-axis velocity

     if ball.xcor() < -290:   # If ball touch the left wall
        ball.setx(-290)
        ball.dx *= -1        # Reverse the x-axis velocity

     if ball.ycor() > 290:    # If ball touch the upper wall
        ball.sety(290)
        ball.dy *= -1        # Reverse the y-axis velocity

     # Ball-Paddle collision
     if abs(ball.ycor() + 250) < 2 and abs(paddle.xcor() - ball.xcor()) < 55:
        ball.dy *= -1

     # Ball-Ground collison
     if ball.ycor() < -290:   # If ball touch the ground
        ball.goto(0, 100)    # Reset the ball position
