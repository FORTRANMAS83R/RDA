# Etude de l'algorithme Range Doppler (RDA)

Ce projet présente une proposition d'implémentation de l'algorithme Range Doppler Algorithm (RDA) pour la formation d'image Radar à Synthèse d'Ouverture (RSO/SAR). Les données utilisées pour les tests sont issues de mesures fournies par [Fabrice Comblet](https://labsticc.fr/fr/annuaire/comblet-fabrice) (ENSTA).


## Architecture du code

Le fichier ``processing.py`` contient la chaîne de traitement RDA sous forme de classe, ainsi qu'une fonction ``process_and_visualize()`` permettant de lancer un traitement sur un jeu de données.

### Args: 

* conf : La configuration de l'acquisition au format suivant :
    - ``file_path`` : chemin vers les données à traiter ``[str]``
    - ``PRF`` : fréquence de répétition des impulsions en Hz ``[float]``
    - ``vp`` : vitesse de la plateforme en m/s ``[float]``
    - ``fc`` : fréquence de travail du radar en Hz ``[float]``
    - ``Tp`` : durée de l'impulsion en secondes ``[float]``
    - ``B0`` : bande passante du signal en Hz ``[float]``
    - ``theta`` : angle d'incidence du radar en degrés ``[float]``
    - ``Ro`` : distance de la plateforme au centre de l'image en mètres ``[float]``
    - ``window_r`` : ajout ou non d'un fenêtrage de Hanning en portée ``[bool]``
    - ``window_a`` : ajout ou non d'un fenêtrage de Hanning en azimut ``[bool]``
* ``dur`` : durée de l'acquisition en secondes ``[float]``
* ``visu_range`` (default=False) : active la visualisation d'une coupe en portée de l'image après compression en distance ``[bool]``

### Exemple de configuration
```python
config = {
    "file_path": "onepointtarget_3s.mat",
    "PRF": 300,
    "vp": 200,
    "fc": 4.5e9,
    "Tp": 0.25e-5,
    "B0": 100e6,
    "theta": 45,
    "Ro": 20e3, 
    "window_r": True, 
    "window_az": False
}
```
## Utilisation

Le notebook RDA.ipynb présente le fonctionnement de l'algorithme à travers la construction d'une image à partir de différents jeux de données.

1. Créer un environnement virtuel et l'activer :
```bash
   python -m venv env
   source env/bin/activate (Linux/MacOS)
   env\Scripts\activate (Windows)
```
2. Installer les dépendances du projet :
```bash
   pip install -r requirements.txt
```
3. Lancer le notebook :
 ```bash
   jupyter notebook RDA.ipynb
```

---

Ce code a été réalisé par [Mikael Franco](https://www.linkedin.com/in/mikael-franco/), étudiant en 3ème année à l'IMT Atlantique.