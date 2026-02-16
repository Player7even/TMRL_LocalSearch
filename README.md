# TMRL_LocalSearch
Erweiterung des Open Frameworks: TMRL, mit einem Local Search Algorithmus welcher die beste Rundenzeit des RL Agenten Frameblock basiert verbessert.

Unsere Änderungen des Codes:

1. TMRL - main.py (Zeile 66 - 95: Replayfunktion eingebunden, Parser Argument hinzugefügt (Zeile 112)

2. TMRL - networking.py (Ab Zeile 939 alles hinzugefügt)

3. TMRL - tm ->  tm_gym_interfaces.py (Neuanordnung der Interfaces, Einstellungen und Hilfsfunktionen Global angeordnet, damit jedes Interface Zugriff darauf hat, Rewards in class TM2020InterfaceLidar (Zeile 246-258), Logging in class TM2020Interface (Ab Zeile 172))

4. TMRL - tm - utils -> compute_reward.py (def compute_reward: Mode Selection eingebaut) 


Config Ordner in TMRL Data Überordner kopieren.
Hiermit ist es möglich die Lernrate, Episoden und Rewardfunktionen anzupassen.
