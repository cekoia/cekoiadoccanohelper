# principe
l'application permet d'aider aux annotations et générer les modèles

# initialisation de l'environnement
* créer un environnement virtuel
 virtualenv .venv
* installer les packages nécessaires
 pip install -r requirements
* lancer l'appli avec le débuggage visualstudio

# mise en production
* créer une application azure app service
* se connecter en ssh dessus et installer le package suivant : 
 apt-get install libgomp1
* déployer l'application avec visual studio code