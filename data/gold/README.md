# Dataset gold

`gold_dataset.csv` est genere depuis `data/raw/` par :

```bash
uv run decrochage build-gold
```

Il contient :

- les capteurs joints temperature + pression ;
- la cible binaire `panne` ;
- les features temporelles anti-fuite (`shift(1)` avant `rolling`) ;
- uniquement les lignes exploitables apres jointure et suppression des valeurs manquantes.

Sur les donnees fournies dans ce starter, la commande genere 1896 lignes et un taux de panne
environ egal a 0.1055.
