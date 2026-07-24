# Donnees brutes Decrochage

Sources brutes fournies au debut du Sprint 3 :

- `capteurs_temperature.csv` : mesures temperature, separateur `;`, identifiants `MACH-01`.
- `capteurs_pression.tsv` : mesures pression, separateur tabulation, identifiants `MACH_01`.
- `releves_incidents.csv` : incidents, identifiants `M-06` / `M-2`, colonnes `date` + `time`.
- `machine.sql` : referentiel machines.

Ces fichiers sont volontairement heterogenes. Le package normalise les identifiants en `MACH-0N`
et construit le dataset gold via :

```bash
uv run decrochage build-gold
```
