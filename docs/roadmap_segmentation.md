# Feuille de route : segmentation multi-classe des zones artificialisées (U-Net)

Statut : implémentation abandonnée (step back) après un premier essai concluant sur le plan technique mais insuffisant sur le plan des résultats. Ce document capitalise ce qui a été appris pour une reprise future. Aucun code associé n'est présent dans le repo actuellement (retiré volontairement).

## 1. Objectif

Remplacer la segmentation binaire ("artificialisé" vs "non-artificialisé", basée sur ESA WorldCover) par une segmentation **multi-classe**, pour distinguer les types de zones réellement pertinents pour la prospection solaire :

| ID | Classe | Justification métier |
|---|---|---|
| 0 | Fond (non-artificialisé) | — |
| 1 | Parking (surface moyenne/grande) | Cible directe : loi APER (2023) impose des ombrières solaires sur les grands parkings |
| 2 | Industriel / commercial | Toitures/friches potentielles |
| 3 | Friche | Cible prioritaire : sites déjà délaissés, mobilisables sans destruction d'usage actif |

Le résidentiel individuel est explicitement hors périmètre. En cas de chevauchement de polygones, **friche est prioritaire** (un site industriel devenu friche doit être étiqueté friche).

## 2. Sources de données validées

Aucune de ces sources ne nécessite de compte/clé API — accessibles en HTTP simple.

### OpenStreetMap (Overpass API)
- Endpoint : `https://overpass-api.de/api/interpreter`
- Tags utilisés : `amenity=parking` (classe 1) ; `building=industrial|warehouse`, `landuse=industrial|commercial` (classe 2)
- **Piège technique** : le serveur renvoie `HTTP 406` si le header `User-Agent` par défaut de la librairie `requests` est utilisé — il faut un `User-Agent` explicite (ex. `"MonProjet/1.0 (contact)"`).
- Limite assumée : seuls les `way` (polygones simples) sont exploitables facilement ; les `relation` (multipolygones, ex. parkings avec îlots) sont plus complexes à reconstruire (gestion anneaux extérieurs/intérieurs) — non traités dans le premier essai.
- Filtre de surface minimale nécessaire côté parking (ex. ~1500 m², proxy du seuil loi APER) pour exclure les petites places résidentielles — se calcule facilement avec une formule de shoelace une fois les polygones reprojetés en CRS métrique, pas besoin de shapely.

### Cartofriches (Cerema, WFS)
- Endpoint : `https://www.geo2france.fr/geoserver/cerema/ows`, couche `cerema:cartofriche`
- Format : GeoJSON (`OUTPUTFORMAT=application/json`), champs utiles : `site_id` (préfixé par le code INSEE de la commune), `site_nom`, `site_type` (souvent `"inconnu"` — sans importance, seule la présence compte)
- **Piège technique** : le paramètre `BBOX` du WFS est peu fiable sur ce GeoServer (souci d'ordre d'axes lat/lon vs lon/lat). Solution robuste : filtrer par département via `CQL_FILTER=site_id LIKE '<préfixe_INSEE>%'`, récupérer tout le département (léger, ~1400 friches pour le Nord), puis filtrer côté client par intersection de bounding box avec l'AOI.
- Couverture confirmée : 5557 friches en Hauts-de-France, 1401 dans le Nord (59), 42 dans une AOI élargie autour de Dunkerque.

### Choix technique : pas de dépendance géo lourde
`rasterio.warp.transform_geom` (reprojection) + `rasterio.features.rasterize` (rasterisation multi-classe, avec priorité par ordre d'empilement des formes) suffisent. Pas besoin de `shapely`/`geopandas`/`pyproj`/`osmnx` — cohérent avec l'empreinte de dépendances existante du projet (déjà `rasterio`-only). Seul ajout nécessaire : `requests` (déjà présent en transitif via `pystac-client`, à passer en dépendance directe si repris).

## 3. Pipeline de préparation des données (structure validée)

1. Recherche + lecture d'une scène Sentinel-2 L2A sur l'AOI (Microsoft Planetary Computer STAC) — identique au pipeline binaire existant.
2. Fetch OSM (Overpass) + Cartofriches (WFS) sur la même AOI.
3. Reprojection de tous les polygones vers le CRS de la scène S2 (`rasterio.warp.transform_geom`).
4. Filtre de surface minimale sur les parkings.
5. Rasterisation en un seul mask multi-classe (`rasterio.features.rasterize`), formes empilées dans l'ordre de priorité (friche en dernier).
6. Découpage en tuiles fixes (ex. 256×256), sauvegarde images/masks en PNG. **Les masks stockent des indices de classe bruts (0..N-1), pas du 0-255** — pas visualisables tels quels dans un visualiseur standard, nécessite un remap couleur pour l'inspection visuelle.

## 4. Modèle et entraînement (structure validée)

- Modèle : `segmentation_models_pytorch.Unet(encoder_name="resnet34", encoder_weights="imagenet", classes=N)` — déjà générique, aucun changement nécessaire pour passer de 1 à N classes.
- Loss : `DiceLoss(mode="multiclass")` + `CrossEntropyLoss()` (au lieu de `mode="binary"` + `BCEWithLogitsLoss`).
- Métrique : IoU moyenné sur les classes (mean IoU) via `argmax` des logits — **mais toujours reporter l'IoU par classe séparément**, pas seulement la moyenne (voir §5, c'est ce qui a révélé le vrai problème).
- Chargement des masks : `PILToTensor()` (pas `ToTensor()`, qui diviserait les indices de classe par 255) + `.long()`.

## 5. Résultat du premier essai — diagnostic détaillé

Essai réalisé sur une AOI de ~475 km² autour de Dunkerque (66 tuiles générées, 46/9/11 train/val/test), 15 epochs.

**IoU par classe obtenu :**

| Classe | IoU |
|---|---|
| Fond | 0.888 |
| Industriel/commercial | 0.119 |
| Parking | 0.001 |
| Friche | 0.0002 |

**Causes identifiées (vérifiées empiriquement, pas des hypothèses) :**

1. **Déséquilibre de classe extrême, même localement.** Parking et friche ne représentent respectivement que 0.28% et 0.25% des pixels sur l'ensemble de l'AOI. Même dans les tuiles qui *contiennent* ces classes, elles restent minoritaires (médiane 0.27% pour parking, 0.53% pour friche par tuile). La loss n'étant pas pondérée par classe, le réseau trouve un minimum trivial ("prédire fond/industriel partout") sans jamais apprendre à distinguer ces classes rares.
2. **Objets minuscules à la résolution Sentinel-2 (10m/pixel).** Le seuil de 1500 m² pour un parking ne représente que ~15 pixels ; une friche de 2 ha ne fait que ~200 pixels sur une tuile de 65 536 pixels.
3. **Dataset trop petit et non diversifié.** 46 tuiles d'entraînement, toutes issues d'une seule ville (Dunkerque) — pas assez de diversité visuelle pour généraliser une signature "parking"/"friche".
4. **Nombre d'epochs insuffisant** compte tenu du signal faible disponible pour les classes rares.

Point de vérification confirmé par script (pas juste supposé) : les classes rares sont bien présentes dans les 3 splits (train/val/test), donc ce n'est pas un problème de fuite de split — c'est un problème de signal/poids dans la loss et de volume de données.

## 6. Prochaines étapes (priorisées)

### Doit être fait avant toute reprise sérieuse
1. **Pondérer la loss par classe** (`CrossEntropyLoss(weight=...)`, poids inversement proportionnels à la fréquence des classes) — levier le plus direct, ne nécessite pas plus de données, change juste `training/seg_engine.py`.
2. **Étendre le dataset à plusieurs AOI distinctes** (pas juste agrandir l'AOI actuelle — risque de sortir des limites d'une seule scène Sentinel-2/tuile MGRS ~110×110km). Zones candidates identifiées :
   - **Bassin minier du Nord-Pas-de-Calais** (Lens, Douai, Valenciennes, Béthune — départements 59 et 62) : zone la plus dense en friches de France (anciens sites miniers), cible directement la classe la plus rare.
   - **Calais / Boulogne-sur-Mer** (62) : port/industriel, diversifie parking et industriel.
   - Extension possible à d'autres régions (Le Havre, Marseille/Fos-sur-Mer...) si besoin de plus de diversité — nécessite de généraliser `FRICHE_DEPT_PREFIX` à une liste de départements plutôt qu'un seul.
   - Objectif indicatif : viser 1000+ tuiles pour avoir une masse critique sur les classes rares (actuellement 66).
   - Prévoir un préfixe de nom de tuile par AOI (`{aoi_label}_tile_{row}_{col}.png`) pour éviter les collisions entre zones.

### À évaluer ensuite
3. Gérer les `relation` OSM (multipolygones) actuellement ignorées.
4. Stratifier le split train/val/test par présence de classe rare plutôt qu'un `random_split` uniforme (actuellement pas un problème identifié, mais à surveiller si le dataset grossit de façon hétérogène entre AOI).
5. Envisager du sur-échantillonnage (oversampling) des tuiles contenant les classes rares, en complément de la pondération de la loss.
6. Réévaluer le nombre d'epochs une fois la loss pondérée et le dataset élargi (15 était clairement insuffisant pour ce niveau de difficulté).
7. Exporter une version colorée des masks (`data/.../masks_preview/`) pour l'inspection visuelle hors notebook (QGIS, Finder) — utile pour la validation qualité du dataset généré.

## 7. Fichiers de référence (supprimés, à recréer si repris)

Pour mémoire, la structure qui avait été mise en place et validée fonctionnellement (avant retrait) :
- `notebooks/fetch_landuse_dataset.ipynb` — fetch + rasterisation + tuilage
- `notebooks/train_unet_landuse.ipynb` — entraînement + visualisation + IoU par classe
- `helpers/segmentation_dataset.py` — `SegmentationTileDataset`, transforms image/mask
- `models/unet_builder.py` — `build_unet(num_classes, device)` (générique, réutilisable tel quel)
- `training/seg_engine.py` — `SegEngine` (loss Dice+CE, IoU par classe, boucle train/eval générique)
- `training/pipeline_configs.py` — entrée `UNET_CONFIG` (à réintroduire, `num_classes=4`)
- Données générées : `data/landuse/{images,masks}`, checkpoint `checkpoints/unet_landuse.pth`
