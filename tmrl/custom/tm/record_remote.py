import time
import pickle
import numpy as np
import os
from tmrl.custom.tm.utils.tools import TM2020OpenPlanetClient

# =========================================================
# KONFIGURATION
# =========================================================
OUTPUT_FILE = "reward.pkl"
SAVE_PATH = r"C:\Users\leons\TmrlData\reward"  # Pfad anpassen falls nötig!

def correct_data(data):
    """Korrigiert den 10^12 Bug direkt bei der Aufnahme."""
    # Data Struktur ist meist: [speed, ..., x(2), y(3), z(4), ...]
    data_list = list(data)
    
    x = float(data_list[2])
    y = float(data_list[3])
    z = float(data_list[4])
    speed = float(data_list[0])

    # Fix anwenden
    if abs(x) > 1e10: x /= 1e12
    if abs(y) > 1e10: y /= 1e12
    if abs(z) > 1e10: z /= 1e12
    if abs(speed) > 10000: speed /= 1e12

    # Zurückschreiben
    data_list[2] = x
    data_list[3] = y
    data_list[4] = z
    data_list[0] = speed
    
    return data_list

def record():
    client = TM2020OpenPlanetClient()
    trajectory = []
    
    print("--- RECORDING TOOL ---")
    print(f"1. Gehe in TrackMania an den Start.")
    print(f"2. Warte auf 'GO'.")
    print(f"3. Fahre die Strecke sauber ab.")
    print(f"4. Drücke STRG+C wenn du im Ziel bist, um zu speichern.")
    print("----------------------")
    
    input("Drücke ENTER um die Verbindung zu testen...")
    try:
        _ = client.retrieve_data()
        print("Verbindung OK!")
    except Exception as e:
        print(f"Fehler: {e}")
        print("Ist TrackMania offen und OpenPlanet geladen?")
        return

    print("Bereit? Aufnahme startet in 3 Sekunden...")
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    time.sleep(1)
    print(">>> GO! FAHRE LOS! <<<")

    try:
        while True:
            # Daten holen (ca. 20-50 Mal pro Sekunde, je nach Performance)
            data = client.retrieve_data()
            
            # Daten bereinigen (WICHTIG!)
            clean_data = correct_data(data)
            
            # Speichern
            trajectory.append(clean_data)
            
            # Kleines Feedback
            if len(trajectory) % 100 == 0:
                print(f"Aufnahme läuft... {len(trajectory)} Punkte")
            
            # Taktrate begrenzen (z.B. 20 Hz reicht locker für eine Reward-Linie)
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nAufnahme gestoppt!")
        
        if len(trajectory) == 0:
            print("Keine Daten aufgenommen.")
            return

        # Ordner erstellen
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)
            
        full_path = os.path.join(SAVE_PATH, OUTPUT_FILE)
        
        print(f"Speichere {len(trajectory)} Punkte in '{full_path}'...")
        
        with open(full_path, 'wb') as f:
            pickle.dump(trajectory, f)
            
        print("FERTIG! Du kannst jetzt das Training starten.")

if __name__ == "__main__":
    record()