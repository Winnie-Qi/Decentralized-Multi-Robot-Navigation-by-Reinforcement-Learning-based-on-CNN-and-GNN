The baseline of this project is based on this [repository](https://github.com/proroklab/gnn_pathplanning).\
The scenario assumes local observation and distributed network, that is, each agent can only observe the world within a certain radius and can only communicate with other agents within this radius, and each agent makes motion decisions independently.\
Based on the above assumptions, the input structure of the network is shown below.\
<img src="https://github.com/Winnie-Qi/Decentralized-Multi-Robot-Navigation-by-Reinforcement-Learning-based-on-CNN-and-GNN/blob/main/pictures%20in%20readme/pic1.jpg" alt="drawing" width="30%"/>
For the agent S1, its observation is limited with the red box. Then, the first layer of the input is the position of the obstacles in the red box.