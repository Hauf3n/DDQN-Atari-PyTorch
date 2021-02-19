# DQN-Atari-PyTorch

 Implementation of (D)-DQN [(1)](https://arxiv.org/abs/1312.5602) [(2)](https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf) [(3)](https://arxiv.org/abs/1509.06461) by DeepMind.<br />
 
 Applied to the gym Breakout, Pong and SpaceInvaders environment. *NoFrameskip-v4
 
 # Results
 ![games](https://github.com/Hauf3n/DQN-Atari-PyTorch/blob/master/media/games.gif)
 [Youtube](https://youtu.be/dkcYvfiKYK0)<br /><br />
 
 # Training
 
 Due to computational resource constraints, i trained Breakout and SpaceInvaders for about 11-14 million steps. <br />
 The agents would become better given more training. 
 
 Training: Breakout <br />
 ![breakout](https://github.com/Hauf3n/DQN-Atari-PyTorch/blob/master/media/plot_breakout.png)<br />
 
 Training: Pong <br />
 ![breakout](https://github.com/Hauf3n/DQN-Atari-PyTorch/blob/master/media/plot_pong.png)<br />
 
 Training: SpaceInvaders <br />
 sparse rewards at ~(600-800) return. Often only one/two, fast moving targets left. Hard to optimize!
 ![breakout](https://github.com/Hauf3n/DQN-Atari-PyTorch/blob/master/media/plot_spaceinvaders.png)<br />
 

