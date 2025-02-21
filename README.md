# Self-Driving Car

## About the Project
This project uses **Deep Q-Network (DQN)** for training a self-driving car in **Pygame**. You can either train the AI or test its performance.  

## How to Use
### 1️⃣ **Training Mode**
- Set `render = False` in `main.py`  
- The program will start **training** using `ckpt.pth` (pretrained model)  

### 2️⃣ **Testing Mode**
- Set `render = True` in `main.py`  
- The AI will start running for **testing purposes**  

### 3️⃣ **Play Against AI**
- Run `game.py`  
- Controls:  
  - **WAD keys** (or **Arrow Keys**)  
  - No **Back/Down** button  

## Multiplayer Support
- **Maximum 3 players** (including AI)  
- To disable AI, set `is_ai_playing = False` in `game.py`  
- You can add up to **5 players** (but must modify code & repeat car colors)  

## Training Your Own AI
- Modify **reward settings** in `main.py` for better performance  
- Experimenting with different rewards can lead to interesting results. Just don't tweak it too much or the algorithm will go bonkers

## Notes
- If you add more players, **key mappings may get cluttered**  
- The AI is **almost perfect**—experiment with different settings for better results!  

---
**Enjoy the race! 🏎️💨**
