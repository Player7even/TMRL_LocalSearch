# TMRL_LocalSearch
Erweiterung des Open Frameworks: TMRL, mit einem Local Search Algorithmus welcher die beste Rundenzeit des RL Agenten Frameblock basiert verbessert.


Unsere Aenderungen des Codes:
TMRL - main.py (Zeile 66 - 95: Replay funktion eingebunden, Parser Argument hinzugefuegt (Zeile 112)
TMRL - networking.py (Ab Zeile 939 alles hinzugefuegt)
TMRL - tm ->  tm_gym_interfaces.py (Neuanordnung der Interfaces, Einstellungen und Hilfsfunktionen Global angeordnet damit jedes Interface zugriff drauf hat, rewards in class TM2020InterfaceLidar (Zeile 246-258), Logging in class TM2020Interface (Ab Zeile 172))
TMRL - tm - utils -> compute_reward.py (def compute_reward: Mode Selection) 
