# GHI Prediciton using Neural Networks

This implementation is designed to use forecasted weather metrics such as temperature, cloud coverage, humidity, rainfall, and more to predict the power output of a solar array. We use GHI prediction to estimate the power output of solar arrays and send those predictions to grid managers to use how they see fit. We envision hourly, daily, and even weekly predicted measurements to help efficiently schedule local electric grids in Pennsylvania.

## GEA

Goals
- Analyze weather patterns in Pennsylvania and use forecasted metrics to predict Global Horizontal Irradiance
- Use GHI prediction to calculate PV output in kW
- Take in user input to specialize prediction for different arrays
- Continously evaluate and refine networks to ensure accuracy of predictions
- Aid grid managers in schedulding power reliance efficiently
- Help reduce reliance of non-renewable energy sources in PA

Environment
- weather patterns in PA change from season to season and are more unpredictable on a daily basis than other regions
- used for many different sizes and configurations of solar arrays
- used at grid locations by users not necessarily fluent in neural networks
- many weather metrics will be used to train the network, will need to consider temperature, cloud coverage, humidity, rainfall, etc.

Adaptation
- Must adapt to rapid changes in weather patterns, hourly, daily, and weekly
- Must adapt to different solar array sizes
- Must adapt to different solar array efficiency's
- Must be able to satisfy grid managers needs
- Must adapt to energy consumption variation

## Implementation Diagram

![image](https://user-images.githubusercontent.com/61809423/234450247-7b60d1f8-ddca-449d-a87a-77a365c99634.png)


## Installation 

To run this project you will need to have the following installed: 

- Python 3.x
- TensorFlow
- Keras
- Pandas
- NumPy

## Usage

To use this project, first clone the repository to your local machine. Then, navigate to the project directory and run the `final_proj.py` file:

You can modify the parameters of the neural networks by adjusting these values: 

- `epochs`: The number of training epochs to run.
- `learning_rate`: The learning rate for the optimizer.
- `activation`: The activation function of the layer in the neural network. 
- `frac`: The fraction of training data to test data
- `loss`: The loss function of the neural network
- `optimizer`: The optimizer function of the neural network
- And more depending on your comfort with neural networks in TensorFlow and Keras

You can make predictions using the Neural Network by inputting your own data into each category in the new_data assignment

You will be asked to input data about the solar array you are predicting will, it will look like this:

![image](https://user-images.githubusercontent.com/61809423/234453792-7d166fa9-0466-4868-aa15-89460991fd14.png)


## Contributing

If you'd like to contribute to this project, feel free to fork the repository and submit a pull request.

## Thank You
