# 🚗 Smart Parking Allocation using DQN

## 📌 Project Description
This project implements a Smart Parking Allocation system using Deep Q-Learning (DQN), a Reinforcement Learning technique. The system intelligently learns and selects optimal parking slots based on environmental state and reward feedback, reducing congestion and improving parking efficiency in smart city environments.

---

## 🎯 Objective
To develop an AI-based parking management system that minimizes parking search time, reduces traffic congestion, and optimizes the allocation of parking spaces using reinforcement learning.

---

## 🚀 Features
- AI-powered parking slot allocation using Deep Q-Learning (DQN)
- Real-time decision making based on environment state
- Reward-based learning for optimal parking strategy
- Simulation of parking environment
- Improved efficiency in urban traffic management

---

## 🛠️ Tech Stack
- Python  
- Streamlit (for UI dashboard)  
- Reinforcement Learning (DQN)  
- NumPy / Pandas  
- OpenAI Gym / Gymnasium  

---

## 📊 System Workflow
1. Initialize parking environment  
2. Agent observes state of parking slots  
3. DQN model selects optimal action  
4. Reward is calculated based on efficiency  
5. Model learns and improves over time  

---

## ▶️ How to Run the Project
### 1. Install dependencies
```bash
pip install streamlit gymnasium torch numpy pandas
streamlit run app.py