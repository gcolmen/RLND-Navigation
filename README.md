[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Banana navigation

### Introduction
This project builds a Reinforcement Learning agent based on a DQNN model that tries to solve the banana collection environment from Unity ML  (https://unity3d.com/machine-learning). 

### The environment
The environment is a 3D-box-like place where bananas fall down from the sky! There are two types of bananas: blue and yellow. Our agent must learn that yellow bananas are good and blue bananas are not.

The agent can perfom 4 different (and discrete) actions:
- Move forward
- Move backward
- Turn left
- Turn right

An agent navigating the environment looks like this:

![Trained Agent][image1]

The information received by the agent is a 37-dimensional array with data for agent's velocity and ray-based sensory data regarding the objects around it.

The agent will be rewared with +1 if it collects a yellow banana, and -1 if it's a blue one. Thus, the goal of our agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

### Environment solved
The task is episodic, and in order to solve the environment, the agent must get an average score of +15 (fifteen!) over 100 consecutive episodes. 


### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

3. This project uses Pytorch as ML platform to build the DQNN model (https://pytorch.org/).

### Instructions

There are three main files in this repository:
- `model.py`: This is where the DQNN model resides. 
- `dqn_agent.py`: This is where the logic for the agent is coded.
- `Navigation.ipnyb`: This is the main project file, where the environment is loaded and the agent is trained.

To train the agent you just need to execute each of the notebook cells in order. (Cell #3 is optional but gives an idea of how the agent randomly navigates the environment).