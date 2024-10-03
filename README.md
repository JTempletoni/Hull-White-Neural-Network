This contains 4 files, each one trains a different neural network (NN). Each one with the aim of predicting european call option prices.

BS Model trains a NN with the Black-Scholes model using time to expiry (T) and moneyness (S/K)

HW Model trains a NN with the Hull-White model using time to expiry (T) and moneyness (S/K)

HW based on BS and greek sort of accurate does as it says - it uses T, S/K, the Black-Scholes price and some of the greeks to predict the Hull-White price.

calibrated NN works completely uses real data to calibrate part of the model, then trains the network using the greeks and T and S/K. The file that contains the real data is too large to upload to github, so if required please contact me.

To use some of the code you may need to install additional packages, e.g. if in colab use !pyDOE to install the pyDOE package.
