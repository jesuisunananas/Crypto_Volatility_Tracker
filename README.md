# Crypto Volatility Tracker

### Description
This project is a real time cryptocurrency volatility tracker and predictor that uses an online ML approach to learn and predict market volatility on the fly. The model used is a LSTM model, which is a specialized version of an RNN that solves the vanishing gradient problem. The model file as well as the app file are containerized in docker which is orchestrated through K3s, this is hosted on an EC2 instance. I am using Github Actions for CICD and Prometheus/Grafana as an observability layer.

### Motivation
I primarily built this to get more exposure to the following technologies/libraries:
1. Pytorch
2. AWS EC2
3. Kubernetes
4. Docker
5. Numpy
6. CI CD
7. Prometheus
8. Grafana
9. Sci Kit Learn

### Architecture
TODO

### Future Work
I enjoyed reading up on RNNs for this project, especially learning about LSTM. I want to try implementing my own version of LSTM and RNN from scratch. I would also like to extend this project when I have more technologies that I want to learn.