# Au sujet des modeles utilisés

## ResNet18 seul (EuroSAT) — architecture de classification
Un ResNet, à la base, se termine par un global average pooling + une couche fully-connected qui sort N logits (1 par classe). build_resnet18_classifier prend le ResNet18 pré-entraîné et remplace juste cette dernière couche par une à 10 sorties (les classes EuroSAT). Entrée = image entière → sortie = 1 seul label. Rien d'autre.

## U-Net (avec un ResNet34 "dedans") — architecture de segmentation
U-Net a une structure en encodeur → décodeur :
- Encodeur : la partie qui compresse l'image en descendant (comme un ResNet classique) — c'est ici qu'on branche ResNet34 (sans sa couche finale de classification, juste les blocs convolutifs). Son rôle : extraire des features à plusieurs échelles.
- Décodeur : la partie symétrique qui remonte en résolution, avec des connexions directes ("skip connections") vers les couches de l'encodeur du même niveau, pour récupérer le détail spatial perdu par le downsampling. Cette partie n'a rien à voir avec ResNet — c'est la partie proprement "U-Net", initialisée aléatoirement, entraînée from scratch sur les 793 tuiles landuse.
- Sortie finale : une carte de la même résolution que l'entrée, avec 4 logits par pixel (smp.Unet(..., classes=4)).

C'est ce que fait segmentation_models_pytorch : ResNet34 n'est réutilisé que comme encodeur remplaçable, pas comme modèle complet — le vrai "modèle" ici, c'est U-Net, qui a juste besoin d'un backbone pour sa moitié descendante.

### Pourquoi 2 modèles dans le projet ?
Parce que ce sont deux tâches différentes : ResNet18/EuroSAT fait de la classification (répétition générale, prouve que le pipeline bout-en-bout marche) ; U-Net fait de la segmentation pixel (l'objectif réel du projet — des frontières précises pour calculer une surface, pas juste une étiquette par tuile de 64px).

### Est-ce que U-Net aurait pu utiliser ResNet18 comme encodeur ?
Oui, sans problème — build_unet(num_classes, device, encoder_name="resnet34") a un paramètre encoder_name, et segmentation_models_pytorch supporte "resnet18" comme n'importe quel autre backbone. Le choix de ResNet34 plutôt que ResNet18 est indépendant du choix fait pour EuroSAT — c'est un compromis propre au U-Net, documenté dans le docstring de models/unet_builder.py:10 : "resnet34, a good accuracy/speed tradeoff" (un peu plus profond que ResNet18, donc plus de capacité, tout en restant raisonnable en coût de calcul).

Et pour être précis : le ResNet18 fine-tuné sur EuroSAT et le ResNet34 encodeur du U-Net sont deux réseaux entraînés séparément, sur des tâches et des données différentes, sauvegardés dans deux checkpoints distincts (resnet18_eurosat_label_smoothing.pth vs unet_landuse.pth) — aucun poids n'est partagé entre les deux.