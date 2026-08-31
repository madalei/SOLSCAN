# Au sujet des modeles utilisés

## ResNet18 seul (EuroSAT) — architecture de classification
Un ResNet, à la base, se termine par un global average pooling + une couche fully-connected qui sort N logits (1 par classe). build_resnet18_classifier prend le ResNet18 pré-entraîné et remplace juste cette dernière couche par une à 10 sorties (les classes EuroSAT). Entrée = image entière → sortie = 1 seul label. Rien d'autre.

## U-Net (avec un ResNet34 "dedans") — architecture de segmentation
U-Net a une structure en encodeur → décodeur :
- Encodeur : la partie qui compresse l'image en descendant (comme un ResNet classique) — c'est ici qu'on branche ResNet34 (sans sa couche finale de classification, juste les blocs convolutifs). Son rôle : extraire des features à plusieurs échelles.
- Décodeur : la partie symétrique qui remonte en résolution, avec des connexions directes ("skip connections") vers les couches de l'encodeur du même niveau, pour récupérer le détail spatial perdu par le downsampling. Cette partie n'a rien à voir avec ResNet — c'est la partie proprement "U-Net", initialisée aléatoirement, entraînée from scratch sur les 793 tuiles landuse.
- Sortie finale : une carte de la même résolution que l'entrée, avec 4 logits par pixel (smp.Unet(..., classes=4)).

C'est ce que fait segmentation_models_pytorch : ResNet34 n'est réutilisé que comme encodeur remplaçable, pas comme modèle complet — le vrai "modèle" ici, c'est U-Net, qui a juste besoin d'un backbone pour sa moitié descendante.

## Pourquoi 2 modèles dans le projet ?
Parce que ce sont deux tâches différentes : ResNet18/EuroSAT fait de la classification (répétition générale, prouve que le pipeline bout-en-bout marche) ; U-Net fait de la segmentation pixel (l'objectif réel du projet — des frontières précises pour calculer une surface, pas juste une étiquette par tuile de 64px).

## Est-ce que U-Net aurait pu utiliser ResNet18 comme encodeur ?
Oui, sans problème — build_unet(num_classes, device, encoder_name="resnet34") a un paramètre encoder_name, et segmentation_models_pytorch supporte "resnet18" comme n'importe quel autre backbone. Le choix de ResNet34 plutôt que ResNet18 est indépendant du choix fait pour EuroSAT — c'est un compromis propre au U-Net, documenté dans le docstring de models/unet_builder.py:10 : "resnet34, a good accuracy/speed tradeoff" (un peu plus profond que ResNet18, donc plus de capacité, tout en restant raisonnable en coût de calcul).

Et pour être précis : le ResNet18 fine-tuné sur EuroSAT et le ResNet34 encodeur du U-Net sont deux réseaux entraînés séparément, sur des tâches et des données différentes, sauvegardés dans deux checkpoints distincts (resnet18_eurosat_label_smoothing.pth vs unet_landuse.pth) — aucun poids n'est partagé entre les deux.

## Principe de fonctionement de l'entrainement -> comment le model apprend?

### Segmentation (branche unet) 
Sur cette branche, le masque de vérité terrain n'est pas dessiné à la main. Il est généré automatiquement à partir de bases géographiques existantes, dans helpers/mask_rasterize.py (sur la branche unet) :

#### Fabrication du dataset d'entrainement pour U-NET

`notebooks/fetch_landuse_dataset.ipynb` orchestre tout : pour chaque AOI de la liste — récupère la scène Sentinel-2 (Planetary Computer), fetch les polygones via geo_fetch, les reprojette dans le CRS de la scène, les rasterise en un masque multi-classe (via helpers/mask_rasterize.py), puis découpe image et masque en tuiles fixes.

1. `helpers/landuse_aois.py` fournit la liste des AOI (bbox + département) à traiter.
2. `helpers/geo_fetch.py` On récupère des polygones géographiques déjà connus : parkings et zones industrielles via OpenStreetMap (Overpass API), friches via Cartofriches.

3. `helpers/mask_rasterize.py` On "brûle" (rasterize) ces polygones sur une grille de pixels alignée sur la tuile Sentinel-2 : chaque pixel reçoit l'ID de classe du polygone qui le contient (0=fond, 1=parking, 2=industriel, 3=friche).

4. Le résultat est écrit sur disque dans data/landuse/ :
data/landuse/images/{aoi}_tile_{row}_{col}.png — la tuile Sentinel-2 (RGB)
data/landuse/masks/{aoi}_tile_{row}_{col}.png — le masque correspondant (indices de classe 0-3 : fond/parking/industriel/friche)
data/landuse/masks_preview/ — une version coloriée du masque, juste pour l'inspection visuelle

5. En aval, helpers/segmentation_dataset.py (le Dataset PyTorch) charge ces paires image/masque, et notebooks/train_unet_landuse.ipynb + training/seg_engine.py s'en servent pour entraîner le U-Net (models/unet_builder.py).

Donc **la vérité terrain vient de données publiques déjà cartographiées**, pas d'annotation manuelle. C'est le "Option 1 / fine-tuning" évoqué dans ton CLAUDE.md, version pragmatique : au lieu d'annoter à la main dans QGIS, on réutilise des polygones qui existent déjà.

Le résultat : pour chaque image xxx.png dans data/landuse/images/, il existe un xxx.png correspondant dans data/landuse/masks/ — même dimensions, mais chaque pixel contient un entier (0-3) au lieu d'une couleur RGB.

### Comment le U-Net apprend à partir de ça
helpers/segmentation_dataset.py charge chaque paire (image, masque). Le U-Net (models/unet_builder.py) prend l'image en entrée et sort, pour chaque pixel, un score par classe (4 classes) — donc une sortie de forme (H, W, 4) au lieu d'une seule étiquette.

L'entraînement (training/seg_engine.py) compare pixel par pixel la prédiction du modèle au masque de vérité terrain, avec une loss (Dice + CrossEntropy). Le modèle ne "comprend" rien : la backprop ajuste les filtres convolutifs pour que les motifs visuels (couleur, texture, forme des toits, teinte du bitume vs terre nue...) qui co-occurrent statistiquement avec "industriel" dans les milliers d'exemples finissent par produire un score élevé sur la classe industrielle. C'est de la corrélation apprise sur des exemples étiquetés, pas de la reconnaissance sémantique.

Petit plus : l'encodeur est pré-entraîné sur ImageNet (encoder_weights="imagenet"), donc il ne part pas de zéro — il a déjà des filtres qui détectent bords/textures/formes génériques, et l'entraînement sur tes masques ne fait qu'adapter ces filtres au vocabulaire visuel du satellite.

#### Un piège réel que le projet a rencontré
Le fond (classe 0) domine massivement les pixels — parking et friche sont rares. Sans correction, le modèle apprend à prédire "fond partout" et obtient quand même ~99% de pixels corrects tout en étant inutile. C'est documenté dans docs/roadmap_segmentation.md et corrigé via compute_class_pixel_weights (pondération inverse de fréquence dans la loss) et le suivi de l'IoU par classe plutôt que l'accuracy globale, justement pour détecter ce problème.