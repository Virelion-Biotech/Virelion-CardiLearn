# GSE115665 reconciliation — Job 1

**Status:** RESOLVED (36/36 samples)
**Species:** Sus scrofa | **Assay:** bulk RNA-seq
**Source:** GEO SOFT characteristics (explicit `individual`, `treatment`, `tissue`, `surgery_days_after_birth`, `harvest_after_treatment`)
**Experimental unit:** individual_animal
**Subjects:** 25 distinct pigs

## Summary counts

| Field | Values |
|---|---|
| condition | sham 12, MI 24 |
| region | ANTE 12, IZ 12, BZ 12 |
| age at surgery | P2 18, P14 18 |
| timepoint | 1d 18, 7d 18 |
| regenerative_class | young_regenerative_window 18, older_nonregenerative 18 |

## Sample table

| GSM | subject_id | condition | region | age | timepoint | regen_class |
|---|---|---|---|---|---|---|
| GSM3186858 | Pig_1624 | sham | ANTE | P2 | 1d | young_regenerative_window |
| GSM3186859 | Pig_1738 | sham | ANTE | P2 | 7d | young_regenerative_window |
| GSM3186860 | Pig_1619 | MI | IZ | P2 | 1d | young_regenerative_window |
| GSM3186861 | Pig_1625 | sham | ANTE | P2 | 1d | young_regenerative_window |
| GSM3186862 | Pig_1619 | MI | BZ | P2 | 1d | young_regenerative_window |
| GSM3186863 | Pig_1623 | MI | IZ | P2 | 1d | young_regenerative_window |
| GSM3186864 | Pig_1623 | MI | BZ | P2 | 1d | young_regenerative_window |
| GSM3186865 | Pig_1680 | MI | IZ | P2 | 1d | young_regenerative_window |
| GSM3186866 | Pig_1680 | MI | BZ | P2 | 1d | young_regenerative_window |
| GSM3186867 | Pig_1682 | sham | ANTE | P2 | 1d | young_regenerative_window |
| GSM3186868 | Pig_1734 | MI | IZ | P2 | 7d | young_regenerative_window |
| GSM3186869 | Pig_1734 | MI | BZ | P2 | 7d | young_regenerative_window |
| GSM3186870 | Pig_1740 | MI | IZ | P2 | 7d | young_regenerative_window |
| GSM3186871 | Pig_1740 | MI | BZ | P2 | 7d | young_regenerative_window |
| GSM3186872 | Pig_1741 | MI | IZ | P2 | 7d | young_regenerative_window |
| GSM3186873 | Pig_1741 | MI | BZ | P2 | 7d | young_regenerative_window |
| GSM3186874 | Pig_1761 | MI | BZ | P2 | 7d | young_regenerative_window |
| GSM3186875 | Pig_1762 | MI | IZ | P2 | 7d | young_regenerative_window |
| GSM3186876 | Pig_1762 | MI | BZ | P2 | 7d | young_regenerative_window |
| GSM3186877 | Pig_1763 | MI | IZ | P2 | 7d | young_regenerative_window |
| GSM3186878 | Pig_1659 | MI | BZ | P14 | 7d | older_nonregenerative |
| GSM3186879 | Pig_1758 | sham | ANTE | P14 | 7d | older_nonregenerative |
| GSM3186880 | Pig_1759 | sham | ANTE | P14 | 7d | older_nonregenerative |
| GSM3186881 | Pig_1760 | sham | ANTE | P14 | 7d | older_nonregenerative |
| GSM3186882 | Pig_1633 | sham | ANTE | P14 | 1d | older_nonregenerative |
| GSM3186883 | Pig_1685 | sham | ANTE | P14 | 1d | older_nonregenerative |
| GSM3186884 | Pig_1686 | sham | ANTE | P14 | 1d | older_nonregenerative |
| GSM3186885 | Pig_1761 | MI | IZ | P2 | 7d | young_regenerative_window |
| GSM3186886 | Pig_1630 | MI | IZ | P14 | 1d | older_nonregenerative |
| GSM3186887 | Pig_1630 | MI | BZ | P14 | 1d | older_nonregenerative |
| GSM3186888 | Pig_1631 | MI | IZ | P14 | 1d | older_nonregenerative |
| GSM3186889 | Pig_1631 | MI | BZ | P14 | 1d | older_nonregenerative |
| GSM3186890 | Pig_1737 | sham | ANTE | P2 | 7d | young_regenerative_window |
| GSM3186891 | Pig_1684 | MI | IZ | P14 | 1d | older_nonregenerative |
| GSM3186892 | Pig_1684 | MI | BZ | P14 | 1d | older_nonregenerative |
| GSM3186893 | Pig_1739 | sham | ANTE | P2 | 7d | young_regenerative_window |

## Leakage rule

IZ and BZ from the same pig must not cross train/test boundaries. Split unit = `subject_id`.

## Note

Full machine-readable rows: `data/reconciled/GSE115665_samples.json`.
This is a reconciliation artifact only. `data/manifest.lock.json` is **not** produced until Jobs 1–4 pass audit.
