Description et guideline globale du projet:

# Solarisation des Terrains artificialisés

## Objectif
Le projet consiste à développer un outil d'aide à la prospection photovoltaïque permettant d'identifier automatiquement, à partir d'images satellites, des zones déjà artificialisées susceptibles d'accueillir des projets solaires (friches industrielles, anciennes zones d'activité, terrains minéralisés, grandes surfaces inutilisées)

## Livrable
Le livrable final sera un prototype d'application capable d'analyser une zone géographique, de produire une cartographie des zones artificialisées détectées, d'estimer leur surface et de générer une liste de sites candidats à approfondir

## Personas

| Persona | Besoin |
| ---| --- |
| Développeur de projets photovoltaïques | Identifier rapidement des sites à fort potentiel. |
| Bureau d'études EnR | Automatiser la phase de pré-qualification avant les études détaillées. |
| Collectivité territoriale | Cartographier les surfaces artificialisées mobilisables pour la transition énergétique. |
| Analyste SIG / Géomaticien | Mettre à jour et enrichir les bases de données territoriales. |
| Asset manager / Gestionnaire de patrimoine immobilier | Identifier les actifs pouvant être valorisés par des projets photovoltaïques. |
| Conseiller en investissement immobilier (immobilier d'entreprise) | Évaluer le potentiel de création de valeur d'un actif avant acquisition. |

## Entrées / sorties du système
### Entrées
Le système prendra en entrée :
*   des images satellites ou aériennes d'une zone géographique donnée :
*   éventuellement des données géographiques complémentaires :
Les images seront découpées en tuiles, normalisées et préparées pour être utilisées par le modèle
### Sorties
L'application produira :
*   une carte de segmentation indiquant les zones artificialisées détectées ;
*   une estimation de surface en m² ou hectares ;
*   un classement des zones candidates

| Zone | Surface | Type détecté | Statut |
| ---| ---| ---| --- |
| Site A | 8,5 ha | Surface artificialisée | À étudier |
| Site B | 1,2 ha | Zone mixte | Faible potentiel |

##   

## Pipeline global
Images satellites
↓
Préparation des données (reprojection, découpage en tuiles,normalisation, augmentation)
↓
Modèle Deep Learning de segmentation(U-Net / DeepLab / SegFormer)
↓
Masque des zones artificialisées
↓
Post-traitement géospatial (polygonisation, calcul des surfaces, a voir..·)
↓
Application de règles métier simples(surface minimale, exclusion zones naturelles)
↓
Carte interactive

## Origine des données et préparation des jeux d'entraînement
Les données peuvent provenir de plusieurs sources ouvertes :
*   Images satellites Sentinel-2
*   Jeux de données d'occupation des sols existants, pouvant servir de base de pré-entraînement ou de comparaison ( ESA WorldCover ; EuroSAT etc )

Une phase d'adaptation des données sera nécessaire.
Deux stratégies sont possibles :
#### Option 1: fine-tuning sur données existantes
Utiliser un modèle pré-entraîné sur la segmentation satellite et adapter les dernières couches sur un petit jeu de données annoté.
Avantages :
*   moins de données nécessaires ;
*   temps d'entraînement réduit ;
*   meilleure faisabilité.

#### Option 2 : création d'un petit dataset spécifique
Créer manuellement des annotations sur quelques zones géographiques :
*   sélection d'images satellites ;
*   annotation des zones artificialisées avec un outil comme QGIS ou CVAT ;
*   création de masques de segmentation.

Ordre de grandeur réaliste : quelques centaines de tuiles annotées ; augmentation artificielle des données par rotation, inversion, variation lumineuse.

L'objectif est de démontrer une démarche complète de préparation, entraînement et évaluation.


## Exemples de Modèles préentrainés Deep Learning à comparer
L'objectif serait de comparer plusieurs familles de modèles pré-entraînés.

| Modèle | Famille | Intérêt |
| ---| ---| --- |
| U-Net | CNN segmentation | Référence historique, simple, rapide à adapter |
| DeepLabV3+ | CNN segmentation avancée | Très utilisé en télédétection, bon compromis précision/performance |
| SegFormer | Transformer vision | Modèle récent performant sur images satellite, moins dépendant des textures locales |
| Mask2Former | Transformer segmentation universelle | Plus complexe, potentiellement meilleure précision mais plus coûteux |

Approche recommandée :
1. établir une baseline avec U-Net ;
2. comparer avec DeepLabV3+ ;
3. tester SegFormer comme modèle final privilégié.
