from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

from config import get_llm_provider_name, get_provider
from providers import CustomOpenAI

load_dotenv()


context = [
    "---\ntitle: \"Configuration client X2GO:\"\ndate: 2022-09-26\n---\n\nLa configuration du client X2go ne fonctionne qu'avec une session de type MATE.\n\nLa configuration d'un serveur mandataire est nécessaire  pour une connexion depuis l'extérieur du laboratoire vers un serveur X2GO qui n'est pas directement autorisé en ssh.\n\nExemple de configuration:\n\n![](images/X2GO.png)\n"
]

# actual_output = ("The Great Wall of China is made entirely of gold and was built in a single year by the Ming Dynasty to store treasures.")
actual_output = """
1. Sélectionner l’environnement MATE dans X2Go
Ouvrez le client X2Go et créez (ou éditez) une connexion.
Dans l’onglet Session, choisissez MATE comme bureau.
Si l’option n’apparaît pas, assurez‑vous que le serveur possède le paquet mate-session (ou mate-desktop-environment).
Laissez les autres paramètres (port SSH, type de session) aux valeurs par défaut ou celles indiquées par votre administrateur.
2. Passer par un proxy externe
Option 1 : Tunnel SSH – SOCKS (recommandé)
bash

ssh -D 1080 -N -C user@proxy-serveur
-D 1080 : crée un proxy SOCKS local sur le port 1080.
-N : n’exécute aucune commande distante.
-C : active la compression (facultatif).
Dans X2Go → Paramètres → Proxy :
Type : SOCKS
Adresse : 127.0.0.1
Port : 1080
Option 2 : Proxy HTTP/HTTPS (cas d’un proxy d’entreprise)
Dans X2Go → Paramètres → Proxy :
Type : HTTP ou HTTPS
Adresse : proxy.entreprise.com (exemple)
Port : 8080
Ajoutez login / mot de passe si le proxy l’exige.
Note : certains proxys bloquent les connexions SSH ; privilégiez l’option 1 quand c’est possible.
Option 3 : Redirection de port via un relais SSH (VPN, serveur intermédiaire)
bash

ssh -L 2222:serveur-x2go:22 user@relais-serveur
Puis, dans X2Go, indiquez comme serveur : localhost:2222.
3. Vérifications & dépannage
Symptom	Action
Impossible de se connecter	- Testez le tunnel : telnet localhost 1080 (SOCKS) ou telnet localhost 2222 (port local). <br> - Vérifiez que le serveur X2Go accepte les connexions (pas de règle iptables/ufw).
Échec de la session MATE	- Installez les paquets sur le serveur : <br> sudo apt install mate-session mate-desktop-environment (Debian/Ubuntu). <br> - Redémarrez le service : sudo systemctl restart x2goserver.
Logs SSH	Sur le relais ou le serveur, consultez : journalctl -u sshd pour repérer d’éventuelles erreurs.
4. Point important
La question demandait explicitement comment combiner une session MATE avec un proxy externe. La réponse ci‑dessus réunit les deux exigences : choisir MATE dans le client X2Go et configurer le proxy (SSH → SOCKS recommandé, ou HTTP/HTTPS, ou redirection de port) afin de contourner les restrictions d’accès.
5. Résumé des étapes
MATE : sélectionnez‑le dans le client X2Go.
Proxy : créez un tunnel SSH SOCKS (ou utilisez HTTP/HTTPS ou un relais).
Test : vérifiez la connectivité du tunnel et les logs.
MATE installé : assurez‑vous que les paquets sont présents et redémarrez le serveur X2Go.
En suivant ces points, vous obtiendrez une connexion X2Go fonctionnelle, même lorsque le serveur n’est pas directement accessible depuis l’extérieur.
"""


test_case = LLMTestCase(
    input="How to set X2GO client for MATE sessions and required external proxy configuration?",
    actual_output=actual_output,
    context=context,
)

eval_llm = CustomOpenAI.from_provider(get_provider(get_llm_provider_name()))
halu_metric = HallucinationMetric(threshold=0.5, model=eval_llm)

##

halu_metric.measure(test_case)
print("Hallucination Metric:")
print("  Score: ", halu_metric.score)
print("  Reason: ", halu_metric.reason)
